#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

SUBMISSION_CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SUBMISSION_CODE_ROOT / "vfstancegen"))

from src.batch_utils import (  # noqa: E402
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

DEFAULT_DATA_PATH = SUBMISSION_CODE_ROOT / "data" / "k-news-stance-mm-acl-arr.json"
DEFAULT_ENV_PATH = SUBMISSION_CODE_ROOT / ".env"
DEFAULT_ORIGINAL_IMAGE_DIR = SUBMISSION_CODE_ROOT / "data" / "original_images"
DEFAULT_OUTPUT_DIR = SUBMISSION_CODE_ROOT / "baseline" / "lvlm" / "outputs"
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_REASONING_LEVEL = "LOW"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_OUTPUT_TOKENS = 16000
LABELS = ("supportive", "neutral", "oppositional")


def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


def get_article_title(article):
    return article.get("headline") or article.get("title") or ""


def filter_articles_by_split(articles, split):
    if split == "all":
        return list(articles)
    return [article for article in articles if (article.get("split") or article.get("split_99")) == split]


def default_prompt_path(input_mode):
    prompt_dir = SUBMISSION_CODE_ROOT / "prompts"
    if input_mode == "textual":
        return prompt_dir / "baseline_text_only.txt"
    if input_mode == "visual":
        return prompt_dir / "baseline_image_only.txt"
    return prompt_dir / "baseline_multimodal.txt"


def build_prompt(prompt_template, article):
    return prompt_template.format(
        issue=article.get("issue", ""),
        event_name=article.get("issue", ""),
        headline=get_article_title(article),
        title=get_article_title(article),
        article=article.get("article", ""),
        main_body=article.get("article", ""),
    )


def find_image_path(image_dir, article_id):
    image_dir = Path(image_dir)
    patterns = [f"{article_id}_*.jpg", f"{article_id}_*.jpeg", f"{article_id}_*.png", f"{article_id}_*.*"]
    for pattern in patterns:
        matches = sorted(image_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def normalize_prediction(text):
    normalized = (text or "").strip().lower()
    if normalized in LABELS:
        return normalized
    for label in LABELS:
        if re.search(rf"\b{label}\b", normalized):
            return label
    return normalized


def build_batch_request(article, prompt_template, input_mode, uploaded_file, image_path, args):
    prompt_text = build_prompt(prompt_template, article)
    if input_mode == "textual":
        contents = [{"parts": [{"text": prompt_text}]}]
    elif input_mode == "visual":
        contents = [{
            "parts": [
                {"text": prompt_text},
                make_file_part(uploaded_file, image_path),
            ]
        }]
    else:
        contents = [{
            "parts": [
                {"text": prompt_text},
                make_file_part(uploaded_file, image_path),
            ]
        }]

    return {
        "key": f"article_{article['id']}",
        "request": {
            "contents": contents,
            "generationConfig": {
                "temperature": args.temperature,
                "maxOutputTokens": args.max_output_tokens,
                "thinkingConfig": {"thinkingLevel": str(args.reasoning_level).upper()},
            },
        },
    }


def parse_batch_output(rows, article_lookup, model_name, input_mode):
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
            "prediction": normalize_prediction(raw_response),
            "raw_response": raw_response,
            "resolved_image_path": article.get("_resolved_image_path"),
            "mode": input_mode,
            "status": "ok",
            "model": model_name,
        })
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline LVLM prediction with Gemini Batch API")
    parser.add_argument("--input-mode", type=str, default="multimodal", choices=["textual", "visual", "multimodal"])
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_ORIGINAL_IMAGE_DIR)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reasoning-level", type=str, default=DEFAULT_REASONING_LEVEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--job-name", type=str, default="baseline-lvlm")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--max-wait-hours", type=int, default=25)
    parser.add_argument("--create-only", action="store_true")
    parser.add_argument("--monitor-only", type=str, default=None)
    parser.add_argument("--no-wait", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    prompt_file = args.prompt_file or default_prompt_path(args.input_mode)
    output_dir = args.output_dir / args.input_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    client = setup_client(args.env_file)

    if args.monitor_only:
        batch_job = wait_for_batch_job(client, args.monitor_only, args.poll_interval, args.max_wait_hours)
        output_path = download_batch_output(client, batch_job, output_dir / "batch_output.jsonl")
        article_lookup = {int(key): value for key, value in load_json(output_dir / "batch_request_lookup.json").items()}
        rows = load_jsonl(output_path)
        results = parse_batch_output(rows, article_lookup, args.model, args.input_mode)
        write_jsonl(output_dir / "results.jsonl", results)
        return

    prompt_template = load_prompt_template(prompt_file)
    articles = filter_articles_by_split(load_json(args.data_path), args.split)
    if args.limit is not None:
        articles = articles[: args.limit]

    requests = []
    article_lookup = {}
    uploaded_images = {}
    for article in articles:
        uploaded_file = None
        image_path = None
        if args.input_mode in {"visual", "multimodal"}:
            image_path = find_image_path(args.image_dir, article["id"])
            if image_path is None:
                continue
            uploaded_file = upload_image(client, image_path)
            uploaded_images[str(article["id"])] = {
                "name": uploaded_file.name,
                "uri": uploaded_file.uri,
                "mime_type": uploaded_file.mime_type,
                "image_path": str(image_path),
            }
        article_copy = dict(article)
        article_copy["_resolved_image_path"] = str(image_path) if image_path else None
        article_lookup[article["id"]] = article_copy
        requests.append(build_batch_request(article_copy, prompt_template, args.input_mode, uploaded_file, image_path, args))

    write_json(output_dir / "batch_request_lookup.json", article_lookup)
    if uploaded_images:
        write_json(output_dir / "uploaded_images.json", uploaded_images)
    batch_file = create_batch_request_file(output_dir, "batch_requests.jsonl", requests)
    uploaded_file, batch_job = submit_batch_job(client, batch_file, args.model, args.job_name)
    write_json(output_dir / "batch_job.json", {
        "uploaded_file": uploaded_file.model_dump(),
        "batch_job": batch_job.model_dump(),
    })

    if args.create_only or args.no_wait:
        print(f"Created batch job: {batch_job.name}")
        return

    finished_batch = wait_for_batch_job(client, batch_job.name, args.poll_interval, args.max_wait_hours)
    output_path = download_batch_output(client, finished_batch, output_dir / "batch_output.jsonl")
    rows = load_jsonl(output_path)
    results = parse_batch_output(rows, article_lookup, args.model, args.input_mode)
    write_jsonl(output_dir / "results.jsonl", results)
    print(f"Finished. output={output_dir / 'results.jsonl'}")


if __name__ == "__main__":
    main()
