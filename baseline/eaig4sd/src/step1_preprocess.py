from __future__ import annotations

import argparse
from collections import Counter

from utils.data_utils import load_dataset, load_runtime, save_json, save_jsonl, stage_output_path

VALID_SPLITS = {"train", "test"}
VALID_STANCES = {"supportive", "neutral", "oppositional"}


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_output_row(record: dict, row_index: int) -> dict:
    split = normalize_text(record.get("split")).lower()
    if split == "validation":
        split = "val"
    if split not in VALID_SPLITS | {"val"}:
        raise ValueError(f"Invalid split at row {row_index}: {record.get('split')}")

    stance = normalize_text(record.get("stance")).lower()
    if stance not in VALID_STANCES:
        raise ValueError(f"Invalid stance at row {row_index}: {record.get('stance')}")

    raw_id = record.get("id")
    if not isinstance(raw_id, int):
        raise ValueError(f"Invalid id at row {row_index}: {raw_id}")

    headline = normalize_text(record.get("headline"))
    main_body = normalize_text(record.get("article"))
    text_combined = f"{headline} {main_body}".strip()

    return {
        "id": raw_id,
        "issue": normalize_text(record.get("issue")),
        "headline": headline,
        "text_combined": text_combined,
        "stance": stance,
        "split": split,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess k-news stance multimodal dataset.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config, _ = load_runtime(args.config)
    records = load_dataset(config)

    split_counter = Counter()
    train_rows = []
    val_rows = []
    test_rows = []

    for index, record in enumerate(records, start=1):
        output_row = build_output_row(record, index)
        split = output_row["split"]
        split_counter[split] += 1
        if split == "train":
            train_rows.append(output_row)
        elif split == "val":
            val_rows.append(output_row)
        else:
            test_rows.append(output_row)

    save_jsonl(stage_output_path(config, "train.jsonl"), train_rows)
    save_jsonl(stage_output_path(config, "val.jsonl"), val_rows)
    save_jsonl(stage_output_path(config, "test.jsonl"), test_rows)
    save_json(
        stage_output_path(config, "step1_summary.json"),
        {
            "num_records": len(train_rows) + len(val_rows) + len(test_rows),
            "splits": dict(split_counter),
            "output_files": {
                "train": str(stage_output_path(config, "train.jsonl")),
                "val": str(stage_output_path(config, "val.jsonl")),
                "test": str(stage_output_path(config, "test.jsonl")),
            },
            "text_policy": "Stored full concatenated headline + article without truncation.",
        },
    )


if __name__ == "__main__":
    main()
