from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils.batch_api import (
    create_batch_input,
    parse_batch_output,
    save_batch_metadata,
    submit_batch,
    wait_for_batch,
)
from utils.data_utils import load_jsonl, load_runtime, save_json, save_jsonl, stage_output_path


MODEL_NAME = "gpt-4o-mini"
RTC_PROMPT = (
    "Given the following caption, please determine whether it is related to the target. "
    "Only respond with 'yes' or 'no'.\n"
    "Caption: {caption}\n"
    "Target: {issue}"
)
RSC_PROMPT = (
    "Given the following caption and text, please determine whether the stance of the "
    "caption towards the target is consistent with the original text. Only respond with "
    "'yes' or 'no'.\n"
    "Caption: {caption}\n"
    "Text: {text_combined}\n"
    "Target: {issue}"
)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_yes_no(value: object) -> str:
    candidate = normalize_text(value).lower()
    if candidate.startswith("yes") or "yes" in candidate:
        return "yes"
    if candidate.startswith("no") or "no" in candidate:
        return "no"
    return candidate


def load_source_rows(output_dir: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows_by_key: dict[tuple[str, int], dict[str, str]] = {}
    for split_name in ("train", "val", "test"):
        split_path = output_dir / f"{split_name}.jsonl"
        if not split_path.exists():
            continue
        for row in load_jsonl(split_path):
            rows_by_key[(split_name, int(row["id"]))] = {
                "issue": normalize_text(row["issue"]),
                "text_combined": normalize_text(row["text_combined"]),
            }
    return rows_by_key


def build_requests(vlm_rows: list[dict[str, Any]], source_rows: dict[tuple[str, int], dict[str, str]]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    for row in vlm_rows:
        row_id = int(row["id"])
        split = normalize_text(row["split"])
        prompt_id = normalize_text(row["prompt_id"])
        caption = normalize_text(row["caption"])
        source = source_rows[(split, row_id)]
        issue = source["issue"]
        text_combined = source["text_combined"]

        prompt_specs = {
            "rtc": RTC_PROMPT.format(caption=caption, issue=issue),
            "rsc": RSC_PROMPT.format(caption=caption, text_combined=text_combined, issue=issue),
        }

        for task_name, prompt_text in prompt_specs.items():
            custom_id = f"{split}:{row_id}:{prompt_id}:{task_name}"
            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": MODEL_NAME,
                        "temperature": 0,
                        "top_p": 1.0,
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": prompt_text,
                                    }
                                ],
                            }
                        ],
                    },
                }
            )

    return requests


def merge_results(vlm_rows: list[dict[str, Any]], parsed_results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    merged_rows: list[dict[str, Any]] = []

    for row in vlm_rows:
        row_id = int(row["id"])
        split = normalize_text(row["split"])
        prompt_id = normalize_text(row["prompt_id"])

        rtc_id = f"{split}:{row_id}:{prompt_id}:rtc"
        rsc_id = f"{split}:{row_id}:{prompt_id}:rsc"

        if rtc_id not in parsed_results:
            raise ValueError(f"Missing batch output for {rtc_id}")
        if rsc_id not in parsed_results:
            raise ValueError(f"Missing batch output for {rsc_id}")

        rtc_result = parsed_results[rtc_id]
        rsc_result = parsed_results[rsc_id]
        if rtc_result.get("error"):
            raise ValueError(f"Batch request failed for {rtc_id}: {rtc_result['error']}")
        if rsc_result.get("error"):
            raise ValueError(f"Batch request failed for {rsc_id}: {rsc_result['error']}")

        merged_rows.append(
            {
                "id": row_id,
                "split": split,
                "prompt_id": prompt_id,
                "rtc": normalize_yes_no(rtc_result["text"]),
                "rsc": normalize_yes_no(rsc_result["text"]),
            }
        )

    merged_rows.sort(key=lambda item: (item["split"], int(item["id"]), item["prompt_id"]))
    return merged_rows


def load_existing_batch_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return payload if isinstance(payload, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM captions with GPT-4o-mini via OpenAI Batch API.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Polling interval for batch completion")
    args = parser.parse_args()

    config, _ = load_runtime(args.config)
    output_dir = Path(config["output_dir"])

    vlm_rows = load_jsonl(output_dir / "vlm_evaluations.jsonl")
    source_rows = load_source_rows(output_dir)
    print(f"[step4c] Loaded {len(vlm_rows)} VLM evaluation rows")

    requests = build_requests(vlm_rows, source_rows)
    print(f"[step4c] Built {len(requests)} batch requests")

    batch_input_path = stage_output_path(config, "step4c_batch_input.jsonl")
    batch_meta_path = stage_output_path(config, "step4c_batch_meta.json")
    batch_output_path = stage_output_path(config, "step4c_batch_output.jsonl")
    batch_error_path = stage_output_path(config, "step4c_batch_error.jsonl")
    scores_output_path = stage_output_path(config, "scores_chatgpt.jsonl")

    existing_meta = load_existing_batch_metadata(batch_meta_path)
    submit_info: dict[str, Any]
    wait_info: dict[str, Any]

    if batch_output_path.exists():
        print(f"[step4c] Reusing existing batch output at {batch_output_path}")
        submit_info = existing_meta or {}
        wait_info = {
            "batch_id": submit_info.get("batch_id"),
            "status": submit_info.get("status", "completed"),
            "output_file": str(batch_output_path),
            "output_file_id": submit_info.get("output_file_id"),
            "error_file_id": submit_info.get("error_file_id"),
        }
    else:
        existing_batch_id = existing_meta.get("batch_id") if existing_meta else None
        if existing_batch_id:
            print(f"[step4c] Resuming existing batch {existing_batch_id} from {batch_meta_path}")
            submit_info = existing_meta
        else:
            create_batch_input(batch_input_path, requests)
            print(f"[step4c] Wrote batch input to {batch_input_path}")
            submit_info = submit_batch(batch_input_path)
            save_batch_metadata(batch_meta_path, submit_info)
            print(f"[step4c] Saved batch metadata to {batch_meta_path}")

        wait_info = wait_for_batch(
            submit_info["batch_id"],
            poll_seconds=args.poll_seconds,
            output_path=batch_output_path,
            error_path=batch_error_path,
        )
        save_batch_metadata(batch_meta_path, {**submit_info, **wait_info})
        print(f"[step4c] Updated batch metadata with final status={wait_info['status']}")

    if wait_info["status"] != "completed":
        raise RuntimeError(f"Batch job did not complete successfully: {wait_info}")

    parsed_results = parse_batch_output(batch_output_path)
    merged_rows = merge_results(vlm_rows, parsed_results)
    print(f"[step4c] Merged batch outputs into {len(merged_rows)} score rows")

    save_jsonl(scores_output_path, merged_rows)
    save_json(
        stage_output_path(config, "step4c_summary.json"),
        {
            "num_vlm_rows": len(vlm_rows),
            "num_batch_requests": len(requests),
            "batch_id": submit_info["batch_id"],
            "model": MODEL_NAME,
            "output_file": str(scores_output_path),
            "batch_output_file": str(batch_output_path),
        },
    )
    print(f"[step4c] Saved GPT evaluation scores to {scores_output_path}")


if __name__ == "__main__":
    main()
