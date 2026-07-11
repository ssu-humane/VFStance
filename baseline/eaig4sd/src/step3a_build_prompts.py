from __future__ import annotations

import argparse
from pathlib import Path

from utils.data_utils import load_jsonl, load_runtime, save_json, save_jsonl, stage_output_path


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_split_rows(output_dir: Path, split: str) -> list[dict]:
    return load_jsonl(output_dir / f"{split}.jsonl")


def load_prediction_map(path: Path, value_key: str) -> dict[int, str]:
    rows = load_jsonl(path)
    return {int(row["id"]): normalize_text(row[value_key]) for row in rows}


def build_prompts(row: dict, stance: str, sentiment: str) -> dict:
    issue = normalize_text(row["issue"])
    text_combined = normalize_text(row["text_combined"])
    return {
        "p1": text_combined,
        "p2": f"{issue}: {text_combined}",
        "p3": f"{stance} {issue}: {text_combined}",
        "p4": f"{sentiment} and {stance} {issue}: {text_combined}",
        "p5": f"{stance} {issue}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EAIG4SD prompts p1-p5.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config, _ = load_runtime(args.config)
    output_dir = Path(config["output_dir"])

    train_rows = load_split_rows(output_dir, "train")
    val_rows = load_split_rows(output_dir, "val")
    test_rows = load_split_rows(output_dir, "test")

    train_sentiment_map = load_prediction_map(output_dir / "train_sentiment.jsonl", "predicted_sentiment")
    val_sentiment_map = load_prediction_map(output_dir / "val_sentiment.jsonl", "predicted_sentiment")
    test_sentiment_map = load_prediction_map(output_dir / "test_sentiment.jsonl", "predicted_sentiment")
    test_stance_map = load_prediction_map(output_dir / "test_stance_predictions.jsonl", "predicted_stance")

    prompt_rows = []

    for row in train_rows:
        row_id = int(row["id"])
        if row_id not in train_sentiment_map:
            raise ValueError(f"Missing predicted sentiment for train id={row_id}")
        stance = normalize_text(row["stance"])
        sentiment = train_sentiment_map[row_id]
        prompts = build_prompts(row, stance=stance, sentiment=sentiment)
        prompt_rows.append(
            {
                "id": row_id,
                "split": "train",
                **prompts,
            }
        )

    for row in val_rows:
        row_id = int(row["id"])
        if row_id not in val_sentiment_map:
            raise ValueError(f"Missing predicted sentiment for val id={row_id}")
        stance = normalize_text(row["stance"])
        sentiment = val_sentiment_map[row_id]
        prompts = build_prompts(row, stance=stance, sentiment=sentiment)
        prompt_rows.append(
            {
                "id": row_id,
                "split": "val",
                **prompts,
            }
        )

    for row in test_rows:
        row_id = int(row["id"])
        if row_id not in test_sentiment_map:
            raise ValueError(f"Missing predicted sentiment for test id={row_id}")
        if row_id not in test_stance_map:
            raise ValueError(f"Missing predicted stance for test id={row_id}")
        stance = test_stance_map[row_id]
        sentiment = test_sentiment_map[row_id]
        prompts = build_prompts(row, stance=stance, sentiment=sentiment)
        prompt_rows.append(
            {
                "id": row_id,
                "split": "test",
                **prompts,
            }
        )

    prompt_rows.sort(key=lambda row: (row["split"], row["id"]))
    save_jsonl(stage_output_path(config, "prompts_p1_p5.jsonl"), prompt_rows)
    save_json(
        stage_output_path(config, "step3a_summary.json"),
        {
            "num_prompt_rows": len(prompt_rows),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "output_file": str(stage_output_path(config, "prompts_p1_p5.jsonl")),
        },
    )


if __name__ == "__main__":
    main()
