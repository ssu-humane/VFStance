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


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_prediction_map(path: Path, value_key: str) -> dict[int, str]:
    rows = load_jsonl(path)
    return {int(row["id"]): normalize_text(row[value_key]) for row in rows}


def build_metadata_map(output_dir: Path) -> dict[tuple[str, int], dict[str, str]]:
    train_rows = load_jsonl(output_dir / "train.jsonl")
    val_rows = load_jsonl(output_dir / "val.jsonl")
    test_rows = load_jsonl(output_dir / "test.jsonl")
    train_sentiment = load_prediction_map(output_dir / "train_sentiment.jsonl", "predicted_sentiment")
    val_sentiment = load_prediction_map(output_dir / "val_sentiment.jsonl", "predicted_sentiment")
    test_sentiment = load_prediction_map(output_dir / "test_sentiment.jsonl", "predicted_sentiment")
    test_stance = load_prediction_map(output_dir / "test_stance_predictions.jsonl", "predicted_stance")

    metadata: dict[tuple[str, int], dict[str, str]] = {}

    for row in train_rows:
        row_id = int(row["id"])
        metadata[("train", row_id)] = {
            "issue": normalize_text(row["issue"]),
            "stance": normalize_text(row["stance"]),
            "sentiment": train_sentiment[row_id],
        }

    for row in val_rows:
        row_id = int(row["id"])
        metadata[("val", row_id)] = {
            "issue": normalize_text(row["issue"]),
            "stance": normalize_text(row["stance"]),
            "sentiment": val_sentiment[row_id],
        }

    for row in test_rows:
        row_id = int(row["id"])
        metadata[("test", row_id)] = {
            "issue": normalize_text(row["issue"]),
            "stance": test_stance[row_id],
            "sentiment": test_sentiment[row_id],
        }

    return metadata


def build_requests(prompt_rows: list[dict[str, Any]], metadata_map: dict[tuple[str, int], dict[str, str]], model: str) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    for row in prompt_rows:
        row_id = int(row["id"])
        split = normalize_text(row["split"])
        meta = metadata_map[(split, row_id)]
        issue = meta["issue"]
        stance = meta["stance"]
        sentiment = meta["sentiment"]

        prompt_specs = {
            "p6": (
                "Please expand the following news text into a detailed visual description. "
                "Describe the scene in terms of colors, objects, people, and other visual elements.\n"
                f"Text: {row['p1']}"
            ),
            "p7": (
                f"Please expand the following news text related to Target {issue} into a detailed visual description. "
                "Describe the scene in terms of colors, objects, people, and other visual elements. "
                "The generated image description should maintain the same stance towards the Target as the original text.\n"
                f"Text: {row['p2']}"
            ),
            "p8": (
                f"Please expand the following news text related to Target {issue} into a detailed visual description. "
                f"The text's stance on target {issue} is {stance}. Describe the scene in terms of colors, objects, people, and other visual elements. "
                "The generated image description should maintain the same stance towards the Target as the original text.\n"
                f"Text: {row['p3']}"
            ),
            "p9": (
                f"Please expand the following news text related to Target {issue} into a detailed visual description. "
                f"The text's stance on target {issue} is {stance}, and its sentiment is {sentiment}. Describe the scene in terms of colors, objects, people, and other visual elements. "
                "The generated image description should maintain the same stance towards the Target as the original text.\n"
                f"Text: {row['p4']}"
            ),
        }

        for prompt_key, user_text in prompt_specs.items():
            custom_id = f"{split}:{row_id}:{prompt_key}"
            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model,
                        "temperature": 0,
                        "top_p": 1.0,
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": user_text,
                                    }
                                ],
                            }
                        ],
                    },
                }
            )

    return requests


def merge_results(prompt_rows: list[dict[str, Any]], parsed_results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    merged_rows: list[dict[str, Any]] = []

    for row in prompt_rows:
        row_id = int(row["id"])
        split = normalize_text(row["split"])
        expanded = dict(row)
        for prompt_key in ("p6", "p7", "p8", "p9"):
            custom_id = f"{split}:{row_id}:{prompt_key}"
            if custom_id not in parsed_results:
                raise ValueError(f"Missing batch output for {custom_id}")
            result = parsed_results[custom_id]
            if result.get("error"):
                raise ValueError(f"Batch request failed for {custom_id}: {result['error']}")
            expanded[prompt_key] = normalize_text(result["text"])
        merged_rows.append(expanded)

    merged_rows.sort(key=lambda row: (row["split"], int(row["id"])))
    return merged_rows


def load_existing_batch_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return payload if isinstance(payload, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand p1-p4 into p6-p9 via OpenAI Batch API.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Polling interval for batch completion")
    args = parser.parse_args()

    config, _ = load_runtime(args.config)
    output_dir = Path(config["output_dir"])
    prompt_rows = load_jsonl(output_dir / "prompts_p1_p5.jsonl")
    metadata_map = build_metadata_map(output_dir)
    print(f"[step3b] Loaded {len(prompt_rows)} prompt rows")

    requests = build_requests(prompt_rows, metadata_map, model=config["chatgpt_model"])
    print(f"[step3b] Built {len(requests)} batch requests")

    batch_input_path = stage_output_path(config, "step3b_batch_input.jsonl")
    batch_meta_path = stage_output_path(config, "step3b_batch_meta.json")
    batch_output_path = stage_output_path(config, "step3b_batch_output.jsonl")
    batch_error_path = stage_output_path(config, "step3b_batch_error.jsonl")

    existing_meta = load_existing_batch_metadata(batch_meta_path)
    submit_info: dict[str, Any]
    wait_info: dict[str, Any]

    if batch_output_path.exists():
        print(f"[step3b] Reusing existing batch output at {batch_output_path}")
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
            print(f"[step3b] Resuming existing batch {existing_batch_id} from {batch_meta_path}")
            submit_info = existing_meta
        else:
            create_batch_input(batch_input_path, requests)
            print(f"[step3b] Wrote batch input to {batch_input_path}")
            submit_info = submit_batch(batch_input_path)
            save_batch_metadata(batch_meta_path, submit_info)
            print(f"[step3b] Saved batch metadata to {batch_meta_path}")

        wait_info = wait_for_batch(
            submit_info["batch_id"],
            poll_seconds=args.poll_seconds,
            output_path=batch_output_path,
            error_path=batch_error_path,
        )
        save_batch_metadata(batch_meta_path, {**submit_info, **wait_info})
        print(f"[step3b] Updated batch metadata with final status={wait_info['status']}")

    if wait_info["status"] != "completed":
        raise RuntimeError(f"Batch job did not complete successfully: {wait_info}")

    parsed_results = parse_batch_output(batch_output_path)
    merged_rows = merge_results(prompt_rows, parsed_results)
    print(f"[step3b] Merged batch outputs into {len(merged_rows)} prompt rows")

    save_jsonl(stage_output_path(config, "prompts_all.jsonl"), merged_rows)
    save_json(
        stage_output_path(config, "step3b_summary.json"),
        {
            "num_prompt_rows": len(prompt_rows),
            "num_batch_requests": len(requests),
            "batch_id": submit_info["batch_id"],
            "output_file": str(stage_output_path(config, "prompts_all.jsonl")),
            "batch_output_file": str(batch_output_path),
        },
    )
    print(f"[step3b] Saved final prompts to {stage_output_path(config, 'prompts_all.jsonl')}")


if __name__ == "__main__":
    main()
