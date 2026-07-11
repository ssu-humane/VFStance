import argparse
from pathlib import Path

from src.batch_utils import (
    create_batch_request_file,
    download_batch_output,
    extract_text_from_response_row,
    load_json,
    load_jsonl,
    make_file_part,
    setup_client,
    submit_batch_job,
    upload_image,
    wait_for_batch_job,
    write_json,
    write_jsonl,
)
from src.utils import filter_articles_by_split, get_article_title


VFSTANCEGEN_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_CODE_ROOT = VFSTANCEGEN_ROOT.parent
DEFAULT_DATA_PATH = SUBMISSION_CODE_ROOT / "data" / "k-news-stance-mm-acl-arr.json"
DEFAULT_IMAGE_DIR = VFSTANCEGEN_ROOT / "outputs" / "generated_images" / "images"
DEFAULT_OUTPUT_DIR = VFSTANCEGEN_ROOT / "outputs" / "stance_predictions"
DEFAULT_ENV_PATH = SUBMISSION_CODE_ROOT / ".env"
DEFAULT_PROMPT_PATH = SUBMISSION_CODE_ROOT / "prompts" / "stage3_stance_prediction.txt"
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_REASONING_LEVEL = "LOW"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_OUTPUT_TOKENS = 16000


def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


def build_prompt(prompt_template, article):
    return prompt_template.format(
        issue=article.get("issue", ""),
        event_name=article.get("issue", ""),
        headline=get_article_title(article),
        title=get_article_title(article),
        article=article.get("article", ""),
        main_body=article.get("article", ""),
    )


def find_generated_image_path(image_dir, article_id):
    candidates = sorted(Path(image_dir).glob(f"id_{article_id}_image_*.png"))
    if candidates:
        return candidates[0]
    return None


def build_batch_request(article, prompt_template, uploaded_file, image_path, temperature, reasoning_level, max_output_tokens):
    return {
        "key": f"article_{article['id']}",
        "request": {
            "contents": [{
                "parts": [
                    {"text": build_prompt(prompt_template, article)},
                    make_file_part(uploaded_file, image_path),
                ]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "thinkingConfig": {"thinkingLevel": reasoning_level},
            },
        },
    }


def parse_batch_output(rows, article_lookup, model_name):
    results = []
    for row in rows:
        key = row.get("key", "")
        if not key.startswith("article_"):
            continue
        article_id = int(key.replace("article_", "", 1))
        article = article_lookup[article_id]
        raw_response = extract_text_from_response_row(row)
        results.append({
            "id": article_id,
            "issue": article.get("issue", ""),
            "headline": get_article_title(article),
            "gold": article.get("stance", ""),
            "prediction": raw_response.strip().lower(),
            "raw_response": raw_response,
            "resolved_image_path": article.get("_resolved_image_path"),
            "status": "ok",
            "model": model_name,
        })
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: predict stance with Gemini Batch API")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reasoning-level", type=str, default=DEFAULT_REASONING_LEVEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--job-name", type=str, default="vf-stage3-stance-prediction")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--max-wait-hours", type=int, default=25)
    parser.add_argument("--create-only", action="store_true")
    parser.add_argument("--monitor-only", type=str, default=None)
    parser.add_argument("--no-wait", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = setup_client(args.env_file)

    if args.monitor_only:
        batch_job = wait_for_batch_job(client, args.monitor_only, args.poll_interval, args.max_wait_hours)
        output_path = download_batch_output(client, batch_job, args.output_dir / "batch_output.jsonl")
        article_lookup = {int(key): value for key, value in load_json(args.output_dir / "batch_request_lookup.json").items()}
        rows = load_jsonl(output_path)
        results = parse_batch_output(rows, article_lookup, args.model)
        write_jsonl(args.output_dir / "results.jsonl", results)
        return

    prompt_template = load_prompt_template(args.prompt_file)
    articles = filter_articles_by_split(load_json(args.data_path), args.split)
    if args.limit is not None:
        articles = articles[: args.limit]

    requests = []
    article_lookup = {}
    uploaded_images = {}
    for article in articles:
        image_path = find_generated_image_path(args.image_dir, article["id"])
        if image_path is None:
            continue
        uploaded_file = upload_image(client, image_path)
        article_copy = dict(article)
        article_copy["_resolved_image_path"] = str(image_path)
        article_lookup[article["id"]] = article_copy
        uploaded_images[str(article["id"])] = {
            "name": uploaded_file.name,
            "uri": uploaded_file.uri,
            "mime_type": uploaded_file.mime_type,
            "image_path": str(image_path),
        }
        requests.append(build_batch_request(
            article_copy,
            prompt_template,
            uploaded_file,
            image_path,
            args.temperature,
            str(args.reasoning_level).upper(),
            args.max_output_tokens,
        ))

    write_json(args.output_dir / "batch_request_lookup.json", article_lookup)
    write_json(args.output_dir / "uploaded_images.json", uploaded_images)
    batch_file = create_batch_request_file(args.output_dir, "batch_requests.jsonl", requests)
    uploaded_file, batch_job = submit_batch_job(client, batch_file, args.model, args.job_name)
    write_json(args.output_dir / "batch_job.json", {
        "uploaded_file": uploaded_file.model_dump(),
        "batch_job": batch_job.model_dump(),
    })

    if args.create_only or args.no_wait:
        print(f"Created batch job: {batch_job.name}")
        return

    finished_batch = wait_for_batch_job(client, batch_job.name, args.poll_interval, args.max_wait_hours)
    output_path = download_batch_output(client, finished_batch, args.output_dir / "batch_output.jsonl")
    rows = load_jsonl(output_path)
    results = parse_batch_output(rows, article_lookup, args.model)
    write_jsonl(args.output_dir / "results.jsonl", results)
    print(f"Finished. output={args.output_dir / 'results.jsonl'}")
