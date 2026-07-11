from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from utils.data_utils import load_jsonl, load_runtime, save_jsonl


VALID_SENTIMENTS = ("positive", "negative", "neutral")
VALID_SPLITS = ("train", "val", "test")


def build_prompt(text_combined: str) -> str:
    return (
        "What is the overall sentiment tone of the following text?\n"
        "Respond with exactly one word: positive, negative, or neutral.\n\n"
        f"Text: {text_combined}"
    )


def normalize_sentiment(text: str) -> str:
    candidate = (text or "").strip().lower()
    for label in VALID_SENTIMENTS:
        if label in candidate:
            return label
    return "neutral"


def load_rows(config: dict) -> tuple[list[dict], list[dict], list[dict]]:
    output_dir = Path(config["output_dir"])
    train_rows = load_jsonl(output_dir / "train.jsonl")
    val_rows = load_jsonl(output_dir / "val.jsonl")
    test_rows = load_jsonl(output_dir / "test.jsonl")
    return train_rows, val_rows, test_rows


def load_model_and_processor(config: dict, adapter_dir: Path) -> tuple[torch.nn.Module, AutoProcessor, int]:
    model_name = config["vlm_model"]
    processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True, use_fast=False)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    if hasattr(model, "generation_config") and hasattr(model.generation_config, "temperature"):
        model.generation_config.temperature = None

    max_length = int(config["stage1"]["max_length"])
    return model, processor, max_length


@torch.inference_mode()
def predict_rows(
    model: torch.nn.Module,
    processor: AutoProcessor,
    rows: list[dict],
    max_length: int,
    device: torch.device,
) -> list[dict]:
    tokenizer = processor.tokenizer
    model = model.to(device)
    predictions = []

    for row in rows:
        prompt = f"User: {build_prompt(row['text_combined'])}\nAssistant:"
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generated = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].size(1)
        decoded = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
        predictions.append(
            {
                "id": row["id"],
                "predicted_sentiment": normalize_sentiment(decoded),
            }
        )

    predictions.sort(key=lambda row: row["id"])
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict zero-shot sentiment with the Step 2a LoRA-adapted model.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to run: train,val,test",
    )
    args = parser.parse_args()

    config, root_dir = load_runtime(args.config)
    if "max_length" not in config["stage1"]:
        raise ValueError("stage1.max_length must be set in config.yaml before running sentiment prediction.")

    adapter_dir = root_dir / "models" / "stance_predictor"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    requested_splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    invalid_splits = [split for split in requested_splits if split not in VALID_SPLITS]
    if invalid_splits:
        raise ValueError(f"Invalid splits: {invalid_splits}. Valid options are {VALID_SPLITS}")

    train_rows, val_rows, test_rows = load_rows(config)
    split_rows = {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
    }
    model, processor, max_length = load_model_and_processor(config, adapter_dir)
    device = torch.device(config["stage1"]["device"])

    for split in requested_splits:
        rows = split_rows[split]
        print(f"[step2-sentiment] Starting {split} inference for {len(rows)} examples on {device}")
        predictions = predict_rows(model, processor, rows, max_length, device)
        output_path = Path(config["output_dir"]) / f"{split}_sentiment.jsonl"
        print(f"[step2-sentiment] Writing {split} predictions to {output_path}")
        save_jsonl(output_path, predictions)

    print("[step2-sentiment] Done")


if __name__ == "__main__":
    main()
