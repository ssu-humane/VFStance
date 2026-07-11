import argparse
from pathlib import Path

from src.batch_utils import (
    create_batch_request_file,
    download_batch_output,
    load_json,
    load_jsonl,
    save_images_from_response_row,
    setup_client,
    submit_batch_job,
    wait_for_batch_job,
    write_json,
    write_jsonl,
)
from src.prompts import T2I_PROMPT_TEMPLATE
from src.utils import write_text


VFSTANCEGEN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATION_DIR = VFSTANCEGEN_ROOT / "outputs" / "annotations"
DEFAULT_OUTPUT_DIR = VFSTANCEGEN_ROOT / "outputs" / "generated_images"
DEFAULT_ENV_PATH = VFSTANCEGEN_ROOT.parent / ".env"
DEFAULT_MODEL = "gemini-3.1-flash-image-preview"


def load_annotation_records(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix == ".jsonl":
            return load_jsonl(input_path)
        return [load_json(input_path)]
    return [load_json(path) for path in sorted(input_path.glob("article_*.json"))]


def stringify_subjects(subjects):
    if not subjects:
        return "none"
    return ", ".join(str(subject).strip() for subject in subjects if str(subject).strip())


def build_t2i_prompt(annotation):
    stylistic = annotation.get("stylistic_semiotic", {})
    denotative = annotation.get("denotative", {})
    return T2I_PROMPT_TEMPLATE.format(
        style=stylistic.get("style", ""),
        include_subjects=stringify_subjects(denotative.get("include_subjects", [])),
        composition=stylistic.get("composition", ""),
        angle=stylistic.get("angle", ""),
        distance=stylistic.get("distance", ""),
        saturation=stylistic.get("saturation", ""),
        luminosity=stylistic.get("luminosity", ""),
        exclude_subjects=stringify_subjects(denotative.get("exclude_subjects", [])),
    )


def build_prompt_record(stage1_record):
    return {
        "id": stage1_record["id"],
        "issue": stage1_record.get("issue", ""),
        "headline": stage1_record.get("headline", ""),
        "stance": stage1_record.get("stance", ""),
        "annotation": stage1_record["annotation"],
        "image_prompt": build_t2i_prompt(stage1_record["annotation"]),
    }


def build_batch_request(prompt_record):
    return {
        "key": f"article_{prompt_record['id']}",
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt_record["image_prompt"]}]}],
        },
    }


def parse_batch_output(output_rows, prompt_lookup, output_dir, model_name):
    prompts_dir = Path(output_dir) / "prompts"
    images_dir = Path(output_dir) / "images"
    records = []
    for row in output_rows:
        key = row.get("key", "")
        if not key.startswith("article_"):
            continue
        article_id = int(key.replace("article_", "", 1))
        prompt_record = prompt_lookup[article_id]
        image_paths = save_images_from_response_row(row, images_dir)
        record = {
            **prompt_record,
            "model": model_name,
            "image_paths": image_paths,
            "num_images": len(image_paths),
            "status": "generated" if image_paths else "no_image",
        }
        write_text(prompts_dir / f"article_{article_id}.txt", prompt_record["image_prompt"])
        write_json(prompts_dir / f"article_{article_id}.json", prompt_record)
        write_json(Path(output_dir) / f"article_{article_id}.json", record)
        records.append(record)
    return records


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: generate images with Gemini Batch API")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--job-name", type=str, default="vf-stage2-image-generation")
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
        prompt_lookup = {int(key): value for key, value in load_json(args.output_dir / "batch_request_lookup.json").items()}
        rows = load_jsonl(output_path)
        records = parse_batch_output(rows, prompt_lookup, args.output_dir, args.model)
        write_jsonl(args.output_dir / "results.jsonl", records)
        return

    records = load_annotation_records(args.input_path)
    if args.limit is not None:
        records = records[: args.limit]

    prompt_records = [build_prompt_record(record) for record in records]
    prompt_lookup = {record["id"]: record for record in prompt_records}
    write_json(args.output_dir / "batch_request_lookup.json", prompt_lookup)

    requests = [build_batch_request(record) for record in prompt_records]
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
    records = parse_batch_output(rows, prompt_lookup, args.output_dir, args.model)
    write_jsonl(args.output_dir / "results.jsonl", records)
    print(f"Finished. output={args.output_dir / 'results.jsonl'}")
