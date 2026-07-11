import argparse
import json
from pathlib import Path

from src.batch_utils import (
    create_batch_request_file,
    download_batch_output,
    load_json,
    load_jsonl,
    setup_client,
    submit_batch_job,
    wait_for_batch_job,
    write_json,
    write_jsonl,
)
from src.utils import filter_articles_by_split, get_article_title, parse_json_object


VFSTANCEGEN_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_CODE_ROOT = VFSTANCEGEN_ROOT.parent
DEFAULT_DATA_PATH = SUBMISSION_CODE_ROOT / "data" / "k-news-stance-mm-acl-arr.json"
DEFAULT_OUTPUT_DIR = VFSTANCEGEN_ROOT / "outputs" / "annotations"
DEFAULT_ENV_PATH = SUBMISSION_CODE_ROOT / ".env"
DEFAULT_SYSTEM_PROMPT_PATH = SUBMISSION_CODE_ROOT / "prompts" / "stage1_visual_framing_annotation_system.txt"
DEFAULT_USER_PROMPT_PATH = SUBMISSION_CODE_ROOT / "prompts" / "stage1_visual_framing_annotation_user.txt"
DEFAULT_MODEL = "gemini-3-flash-preview"


def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


def build_user_prompt(prompt_template, article):
    return prompt_template.format(
        issue=article.get("issue", ""),
        headline=get_article_title(article),
        article=article.get("article", ""),
    )


def build_batch_request(article, system_prompt, user_prompt_template):
    return {
        "key": f"article_{article['id']}",
        "request": {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": build_user_prompt(user_prompt_template, article)}]}],
            "generationConfig": {
                "temperature": 1.0,
                "thinkingConfig": {"thinkingLevel": "LOW"},
                "responseMimeType": "application/json",
            },
        },
    }


def build_record(article, annotation, raw_response, model_name):
    return {
        "id": article["id"],
        "issue": article.get("issue", ""),
        "headline": get_article_title(article),
        "stance": article.get("stance", ""),
        "split": article.get("split", ""),
        "annotation": annotation,
        "raw_response": raw_response,
        "model": model_name,
    }


def parse_batch_output(output_rows, article_lookup, model_name, output_dir):
    records = []
    for row in output_rows:
        key = row.get("key", "")
        if not key.startswith("article_"):
            continue
        article_id = int(key.replace("article_", "", 1))
        article = article_lookup[article_id]
        raw_response = ""
        try:
            response = row.get("response") or {}
            candidates = response.get("candidates") or []
            for candidate in candidates:
                content = candidate.get("content") or {}
                for part in content.get("parts") or []:
                    if part.get("text"):
                        raw_response = str(part["text"]).strip()
                        break
                if raw_response:
                    break
            annotation = parse_json_object(raw_response)
            record = build_record(article, annotation, raw_response, model_name)
        except Exception as error:
            record = {
                "id": article_id,
                "issue": article.get("issue", ""),
                "headline": get_article_title(article),
                "stance": article.get("stance", ""),
                "split": article.get("split", ""),
                "annotation": None,
                "raw_response": raw_response,
                "model": model_name,
                "status": "error",
                "error": str(error),
            }
        write_json(Path(output_dir) / f"article_{article_id}.json", record)
        records.append(record)
    return records


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: annotate visual framing with Gemini Batch API")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--system-prompt-file", type=Path, default=DEFAULT_SYSTEM_PROMPT_PATH)
    parser.add_argument("--user-prompt-file", type=Path, default=DEFAULT_USER_PROMPT_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--job-name", type=str, default="vf-stage1-annotation")
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
        metadata = load_json(args.output_dir / "batch_request_lookup.json")
        article_lookup = {int(key): value for key, value in metadata.items()}
        rows = load_jsonl(output_path)
        records = parse_batch_output(rows, article_lookup, args.model, args.output_dir)
        write_jsonl(args.output_dir / "results.jsonl", records)
        return

    system_prompt = load_prompt_template(args.system_prompt_file)
    user_prompt_template = load_prompt_template(args.user_prompt_file)
    articles = filter_articles_by_split(load_json(args.data_path), args.split)
    if args.limit is not None:
        articles = articles[: args.limit]

    requests = [build_batch_request(article, system_prompt, user_prompt_template) for article in articles]
    batch_file = create_batch_request_file(args.output_dir, "batch_requests.jsonl", requests)
    article_lookup = {article["id"]: article for article in articles}
    write_json(args.output_dir / "batch_request_lookup.json", article_lookup)

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
    records = parse_batch_output(rows, article_lookup, args.model, args.output_dir)
    write_jsonl(args.output_dir / "results.jsonl", records)
    print(f"Finished. output={args.output_dir / 'results.jsonl'}")
