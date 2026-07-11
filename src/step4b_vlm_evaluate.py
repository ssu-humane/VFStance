from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from utils.data_utils import load_jsonl, load_runtime, save_json, save_jsonl


VALID_SPLITS = {"train", "val", "test"}
PROMPT_KEYS = [f"p{i}" for i in range(1, 10)]
CAPTION_PROMPT = (
    "Please provide a caption for the image that includes details about the scene, "
    "people, actions, expressions, and background. If there is any text in the image, "
    "please incorporate that into the caption as well."
)
RTI_PROMPT = (
    "Given the text, target and the image, whether the image is related to the target, "
    "reply with 'yes' or 'no'.\n"
    "Text: {text_combined}\n"
    "Target: {issue}"
)
RSI_PROMPT = (
    "Given the text, target and the image, whether the image can reflect the stance "
    "of the text towards the target, reply with 'yes' or 'no'.\n"
    "Text: {text_combined}\n"
    "Target: {issue}"
)


def load_source_rows(config: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    output_dir = Path(config["output_dir"])
    source_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for split_name in ("train", "val", "test"):
        for row in load_jsonl(output_dir / f"{split_name}.jsonl"):
            source_rows[(split_name, int(row["id"]))] = row
    return source_rows


def select_rows(rows: list[dict[str, Any]], splits: set[str], shard: int, num_shards: int) -> list[dict[str, Any]]:
    filtered = [row for row in rows if str(row.get("split", "")).strip() in splits]
    filtered.sort(key=lambda row: (str(row["split"]), int(row["id"])))
    return [row for index, row in enumerate(filtered) if index % num_shards == shard]


def get_article_dir(image_dir: Path, row: dict[str, Any]) -> Path:
    split = str(row["split"]).strip()
    article_id = str(row.get("article_id", row["id"]))
    return image_dir / split / article_id


def load_completed_rows(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    if not path.exists():
        return {}

    completed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in load_jsonl(path):
        key = (str(row["split"]).strip(), int(row["id"]), str(row["prompt_id"]).strip())
        if row.get("caption") and row.get("rti") and row.get("rsi"):
            completed[key] = row
    return completed


def build_tasks(rows: list[dict[str, Any]], image_dir: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in rows:
        article_dir = get_article_dir(image_dir, row)
        for prompt_id in PROMPT_KEYS:
            tasks.append(
                {
                    "id": int(row["id"]),
                    "split": str(row["split"]).strip(),
                    "prompt_id": prompt_id,
                    "image_path": article_dir / f"{prompt_id}.png",
                }
            )
    return tasks


def load_model_and_processor(model_name: str, device: str) -> tuple[torch.nn.Module, AutoProcessor]:
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    return model, processor


def run_vlm(
    model: torch.nn.Module,
    processor: AutoProcessor,
    image_path: Path,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image file: {image_path}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    decoded = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return decoded.strip()


def normalize_yes_no(text: str) -> str:
    candidate = (text or "").strip().lower()
    if candidate.startswith("yes"):
        return "yes"
    if candidate.startswith("no"):
        return "no"
    if "yes" in candidate:
        return "yes"
    if "no" in candidate:
        return "no"
    return candidate


def merge_shard_outputs(output_dir: Path) -> int:
    rows_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for shard_path in sorted(output_dir.glob("vlm_evaluations_shard*.jsonl")):
        for row in load_jsonl(shard_path):
            key = (str(row["split"]).strip(), int(row["id"]), str(row["prompt_id"]).strip())
            rows_by_key[key] = row

    merged_rows = [rows_by_key[key] for key in sorted(rows_by_key, key=lambda item: (item[0], item[1], item[2]))]
    save_jsonl(output_dir / "vlm_evaluations.jsonl", merged_rows)
    return len(merged_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated images with Qwen2.5-VL-7B-Instruct.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--gpu", type=int, required=True, help="CUDA device index to use")
    parser.add_argument("--shard", type=int, required=True, help="Shard index for this worker")
    parser.add_argument("--num-shards", type=int, default=3, help="Total number of shards")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated splits to process")
    args = parser.parse_args()

    if args.shard < 0 or args.shard >= args.num_shards:
        raise ValueError(f"shard must be in [0, {args.num_shards - 1}]")

    requested_splits = {split.strip() for split in args.splits.split(",") if split.strip()}
    invalid_splits = requested_splits - VALID_SPLITS
    if invalid_splits:
        raise ValueError(f"Invalid splits requested: {sorted(invalid_splits)}")

    config, _ = load_runtime(args.config)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    output_dir = Path(config["output_dir"])
    image_dir = Path(config["image_dir"])

    source_rows = load_source_rows(config)
    prompt_rows = load_jsonl(output_dir / "prompts_all.jsonl")
    shard_rows = select_rows(prompt_rows, requested_splits, shard=args.shard, num_shards=args.num_shards)
    shard_output_path = output_dir / f"vlm_evaluations_shard{args.shard}.jsonl"

    completed = load_completed_rows(shard_output_path)
    all_tasks = build_tasks(shard_rows, image_dir=image_dir)
    pending_tasks = [task for task in all_tasks if (task["split"], task["id"], task["prompt_id"]) not in completed]

    print(f"[step4b] Device={device} shard={args.shard}/{args.num_shards} splits={sorted(requested_splits)}")
    print(f"[step4b] Assigned {len(shard_rows)} rows, {len(all_tasks)} prompt-image tasks, pending {len(pending_tasks)}")

    model, processor = load_model_and_processor(config["vlm_model"], device=device)
    results = list(completed.values())

    progress = tqdm(pending_tasks, desc="Running VLM evaluation", unit="image")
    for task in progress:
        source_row = source_rows[(task["split"], task["id"])]
        text_combined = str(source_row["text_combined"]).strip()
        issue = str(source_row["issue"]).strip()
        image_path = Path(task["image_path"])

        caption = run_vlm(model, processor, image_path, CAPTION_PROMPT, device=device, max_new_tokens=128)
        rti = normalize_yes_no(
            run_vlm(
                model,
                processor,
                image_path,
                RTI_PROMPT.format(text_combined=text_combined, issue=issue),
                device=device,
                max_new_tokens=8,
            )
        )
        rsi = normalize_yes_no(
            run_vlm(
                model,
                processor,
                image_path,
                RSI_PROMPT.format(text_combined=text_combined, issue=issue),
                device=device,
                max_new_tokens=8,
            )
        )

        results.append(
            {
                "id": task["id"],
                "split": task["split"],
                "prompt_id": task["prompt_id"],
                "caption": caption,
                "rti": rti,
                "rsi": rsi,
            }
        )
        results.sort(key=lambda row: (str(row["split"]), int(row["id"]), str(row["prompt_id"])))
        save_jsonl(shard_output_path, results)
        progress.set_postfix(done=len(results), remaining=len(all_tasks) - len(results))

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    merged_count = merge_shard_outputs(output_dir)
    summary_path = output_dir / f"step4b_summary_shard{args.shard}.json"
    save_json(
        summary_path,
        {
            "vlm_model": config["vlm_model"],
            "device": device,
            "shard": args.shard,
            "num_shards": args.num_shards,
            "splits": sorted(requested_splits),
            "assigned_rows": len(shard_rows),
            "assigned_tasks": len(all_tasks),
            "completed_tasks": len(results),
            "merged_rows": merged_count,
            "shard_output_file": str(shard_output_path),
            "merged_output_file": str(output_dir / "vlm_evaluations.jsonl"),
        },
    )
    print(f"[step4b] Finished shard {args.shard}. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
