from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, CLIPModel

from utils.data_utils import load_jsonl, load_runtime, save_json, save_jsonl, stage_output_path


VALID_SPLITS = {"train", "val", "test"}
PROMPT_KEYS = [f"p{i}" for i in range(1, 10)]


def select_rows(rows: list[dict[str, Any]], splits: set[str], shard: int, num_shards: int) -> list[dict[str, Any]]:
    filtered = [row for row in rows if str(row.get("split", "")).strip() in splits]
    filtered.sort(key=lambda row: (str(row["split"]), int(row["id"])))
    return [row for index, row in enumerate(filtered) if index % num_shards == shard]


def load_completed_scores(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}

    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in load_jsonl(path):
        split = str(row.get("split", "")).strip()
        row_id = int(row["id"])
        scores = row.get("scores") or {}
        if all(key in scores for key in PROMPT_KEYS):
            completed[(split, row_id)] = row
    return completed


def resolve_text(row: dict[str, Any]) -> str:
    prompt_text = row.get("p1")
    if isinstance(prompt_text, str) and prompt_text.strip():
        return prompt_text.strip()

    raise ValueError(f"Missing p1 text for id={row['id']} split={row['split']}")


def get_article_dir(image_dir: Path, row: dict[str, Any]) -> Path:
    split = str(row["split"]).strip()
    article_id = str(row.get("article_id", row["id"]))
    return image_dir / split / article_id


def get_image_paths(image_dir: Path, row: dict[str, Any]) -> dict[str, Path]:
    article_dir = get_article_dir(image_dir, row)
    return {prompt_key: article_dir / f"{prompt_key}.png" for prompt_key in PROMPT_KEYS}


def load_clip_model(model_name: str, device: str) -> tuple[CLIPModel, AutoProcessor]:
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = CLIPModel.from_pretrained(model_name, torch_dtype=dtype)
    processor = AutoProcessor.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, processor


def encode_text(model: CLIPModel, processor: AutoProcessor, text: str, device: str) -> torch.Tensor:
    inputs = processor(text=[text], padding=True, return_tensors="pt", truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        features = model.get_text_features(**inputs)
    return F.normalize(features, dim=-1)


def encode_images(model: CLIPModel, processor: AutoProcessor, image_paths: dict[str, Path], device: str) -> tuple[list[str], torch.Tensor]:
    ordered_keys = list(image_paths.keys())
    missing = [str(path) for path in image_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing candidate images: {missing[:3]}{' ...' if len(missing) > 3 else ''}")

    images = [Image.open(image_paths[key]).convert("RGB") for key in ordered_keys]
    try:
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
        return ordered_keys, F.normalize(features, dim=-1)
    finally:
        for image in images:
            image.close()


def score_row(model: CLIPModel, processor: AutoProcessor, row: dict[str, Any], image_dir: Path, device: str) -> dict[str, Any]:
    text_feature = encode_text(model, processor, resolve_text(row), device)
    ordered_keys, image_features = encode_images(model, processor, get_image_paths(image_dir, row), device)
    similarities = torch.matmul(image_features, text_feature.T).squeeze(-1).detach().cpu().tolist()

    if not isinstance(similarities, list):
        similarities = [float(similarities)]

    return {
        "id": int(row["id"]),
        "split": str(row["split"]).strip(),
        "scores": {key: float(score) for key, score in zip(ordered_keys, similarities)},
    }


def merge_shard_outputs(output_dir: Path) -> int:
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for shard_path in sorted(output_dir.glob("scores_sr_shard*.jsonl")):
        for row in load_jsonl(shard_path):
            key = (str(row["split"]).strip(), int(row["id"]))
            rows_by_key[key] = row

    merged_rows = [rows_by_key[key] for key in sorted(rows_by_key, key=lambda item: (item[0], item[1]))]
    save_jsonl(output_dir / "scores_sr.jsonl", merged_rows)
    return len(merged_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CLIP cosine similarity between text and generated images.")
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
    prompts_path = output_dir / "prompts_all.jsonl"

    all_rows = load_jsonl(prompts_path)
    shard_rows = select_rows(all_rows, requested_splits, shard=args.shard, num_shards=args.num_shards)

    shard_output_path = output_dir / f"scores_sr_shard{args.shard}.jsonl"
    completed_scores = load_completed_scores(shard_output_path)
    pending_rows = [row for row in shard_rows if (str(row["split"]).strip(), int(row["id"])) not in completed_scores]

    print(f"[step4a] Device={device} shard={args.shard}/{args.num_shards} splits={sorted(requested_splits)}")
    print(f"[step4a] Loaded {len(all_rows)} total rows, assigned {len(shard_rows)} rows, pending {len(pending_rows)}")

    model, processor = load_clip_model(config["clip_model"], device=device)
    results = list(completed_scores.values())

    progress = tqdm(pending_rows, desc="Computing CLIP SR", unit="article")
    for row in progress:
        result = score_row(model, processor, row, image_dir=image_dir, device=device)
        results.append(result)
        results.sort(key=lambda item: (str(item["split"]), int(item["id"])))
        save_jsonl(shard_output_path, results)
        progress.set_postfix(done=len(results), remaining=len(shard_rows) - len(results))

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    merged_count = merge_shard_outputs(output_dir)
    summary_path = output_dir / f"step4a_summary_shard{args.shard}.json"
    save_json(
        summary_path,
        {
            "clip_model": config["clip_model"],
            "device": device,
            "shard": args.shard,
            "num_shards": args.num_shards,
            "splits": sorted(requested_splits),
            "assigned_rows": len(shard_rows),
            "completed_rows": len(results),
            "merged_rows": merged_count,
            "shard_output_file": str(shard_output_path),
            "merged_output_file": str(output_dir / "scores_sr.jsonl"),
        },
    )
    print(f"[step4a] Finished shard {args.shard}. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
