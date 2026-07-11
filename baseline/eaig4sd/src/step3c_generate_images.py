from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login, snapshot_download
from tqdm.auto import tqdm

from utils.data_utils import getenv_required, load_jsonl, load_runtime, save_json, stage_output_path


VALID_SPLITS = {"train", "val", "test"}


def resolve_model_path(root_dir: Path, config: dict) -> str:
    local_model_dir = root_dir / "models" / "sd3-medium-diffusers"
    if local_model_dir.exists():
        return str(local_model_dir)
    return config["t2i_model"]


def ensure_local_model(root_dir: Path, config: dict, model_path: str) -> str:
    local_model_dir = root_dir / "models" / "sd3-medium-diffusers"
    if local_model_dir.exists():
        return str(local_model_dir)
    if Path(model_path).exists():
        return model_path

    hf_token = getenv_required("HF_TOKEN")
    local_model_dir.mkdir(parents=True, exist_ok=True)
    print("[step3c] Local SD3 model not found. Downloading from Hugging Face first.")
    login(token=hf_token)
    snapshot_download(
        repo_id=config["t2i_model"],
        local_dir=str(local_model_dir),
        local_dir_use_symlinks=False,
        token=hf_token,
        resume_download=True,
    )
    print(f"[step3c] Download complete: {local_model_dir}")
    return str(local_model_dir)


def load_pipeline(model_path: str, device: str) -> StableDiffusion3Pipeline:
    hf_token = getenv_required("HF_TOKEN")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        token=hf_token,
        local_files_only=Path(model_path).exists(),
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def select_rows(rows: list[dict], splits: set[str], shard: int, num_shards: int) -> list[dict]:
    filtered = [row for row in rows if str(row.get("split", "")).strip() in splits]
    filtered.sort(key=lambda row: (str(row["split"]), int(row["id"])))
    return [row for index, row in enumerate(filtered) if index % num_shards == shard]


def expected_output_paths(image_dir: Path, split: str, article_id: int) -> dict[str, Path]:
    article_dir = image_dir / split / str(article_id)
    article_dir.mkdir(parents=True, exist_ok=True)
    return {f"p{i}": article_dir / f"p{i}.png" for i in range(1, 10)}


def generate_missing_images(
    pipeline: StableDiffusion3Pipeline,
    rows: list[dict],
    image_dir: Path,
    inference_steps: int,
    guidance_scale: float,
) -> dict[str, int]:
    generated = 0
    skipped = 0

    progress = tqdm(rows, desc="Generating SD3 images", unit="article")
    for row in progress:
        article_id = int(row["id"])
        split = str(row["split"]).strip()
        output_paths = expected_output_paths(image_dir, split, article_id)

        for prompt_key, output_path in output_paths.items():
            prompt_text = str(row.get(prompt_key, "")).strip()
            if not prompt_text:
                raise ValueError(f"Missing prompt {prompt_key} for id={article_id} split={split}")
            if output_path.exists():
                skipped += 1
                continue

            result = pipeline(
                prompt=prompt_text,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
            )
            image = result.images[0]
            image.save(output_path)
            generated += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        progress.set_postfix(generated=generated, skipped=skipped)

    return {"generated": generated, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate p1-p9 images with Stable Diffusion 3.")
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

    config, root_dir = load_runtime(args.config)
    device = f"cuda:{args.gpu}"
    prompts_path = stage_output_path(config, "prompts_all.jsonl")
    all_rows = load_jsonl(prompts_path)
    shard_rows = select_rows(all_rows, requested_splits, shard=args.shard, num_shards=args.num_shards)
    image_dir = Path(config["image_dir"])

    model_path = resolve_model_path(root_dir, config)
    model_path = ensure_local_model(root_dir, config, model_path)
    print(f"[step3c] Using model source: {model_path}")
    print(f"[step3c] Device={device} shard={args.shard}/{args.num_shards} splits={sorted(requested_splits)}")
    print(f"[step3c] Loaded {len(all_rows)} total prompt rows, assigned {len(shard_rows)} rows to this shard")

    pipeline = load_pipeline(model_path, device=device)
    stats = generate_missing_images(
        pipeline=pipeline,
        rows=shard_rows,
        image_dir=image_dir,
        inference_steps=int(config["stage2"]["sd3_inference_steps"]),
        guidance_scale=float(config["stage2"]["sd3_guidance_scale"]),
    )

    summary_path = stage_output_path(config, f"step3c_summary_shard{args.shard}.json")
    save_json(
        summary_path,
        {
            "model_path": model_path,
            "device": device,
            "shard": args.shard,
            "num_shards": args.num_shards,
            "splits": sorted(requested_splits),
            "assigned_rows": len(shard_rows),
            **stats,
        },
    )
    print(f"[step3c] Finished shard {args.shard}. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
