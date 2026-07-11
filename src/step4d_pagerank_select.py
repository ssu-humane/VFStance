from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel

from utils.data_utils import load_jsonl, load_runtime, save_json, save_jsonl, stage_output_path


PROMPT_KEYS = [f"p{i}" for i in range(1, 10)]


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def yes_no_score(value: object) -> float:
    return 5.0 if normalize_text(value).lower() == "yes" else 0.0


def load_score_maps(output_dir: Path) -> tuple[dict[tuple[str, int], dict[str, float]], dict[tuple[str, int, str], dict[str, str]], dict[tuple[str, int, str], dict[str, str]]]:
    sr_map: dict[tuple[str, int], dict[str, float]] = {}
    for row in load_jsonl(output_dir / "scores_sr.jsonl"):
        sr_map[(normalize_text(row["split"]), int(row["id"]))] = {key: float(value) for key, value in (row.get("scores") or {}).items()}

    vlm_map: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in load_jsonl(output_dir / "vlm_evaluations.jsonl"):
        vlm_map[(normalize_text(row["split"]), int(row["id"]), normalize_text(row["prompt_id"]))] = {
            "rti": normalize_text(row.get("rti")),
            "rsi": normalize_text(row.get("rsi")),
        }

    chatgpt_map: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in load_jsonl(output_dir / "scores_chatgpt.jsonl"):
        chatgpt_map[(normalize_text(row["split"]), int(row["id"]), normalize_text(row["prompt_id"]))] = {
            "rtc": normalize_text(row.get("rtc")),
            "rsc": normalize_text(row.get("rsc")),
        }

    return sr_map, vlm_map, chatgpt_map


def load_clip_model(model_name: str, device: str) -> tuple[CLIPModel, AutoProcessor]:
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = CLIPModel.from_pretrained(model_name, torch_dtype=dtype)
    processor = AutoProcessor.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, processor


def get_article_dir(image_dir: Path, row: dict[str, Any]) -> Path:
    split = normalize_text(row["split"])
    article_id = str(row.get("article_id", row["id"]))
    return image_dir / split / article_id


def encode_images(model: CLIPModel, processor: AutoProcessor, image_paths: list[Path], device: str) -> torch.Tensor:
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing candidate images: {missing[:3]}{' ...' if len(missing) > 3 else ''}")

    images = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
        return F.normalize(features, dim=-1)
    finally:
        for image in images:
            image.close()


def build_transition_matrix(image_features: torch.Tensor) -> torch.Tensor:
    similarity = torch.matmul(image_features, image_features.T)
    similarity = similarity.clone()
    similarity.fill_diagonal_(0.0)
    similarity = torch.clamp(similarity, min=0.0)

    row_sums = similarity.sum(dim=1, keepdim=True)
    zero_rows = row_sums.squeeze(1) <= 0
    if zero_rows.any():
        similarity[zero_rows] = 1.0
        similarity[zero_rows, zero_rows] = 0.0
        row_sums = similarity.sum(dim=1, keepdim=True)

    return similarity / row_sums


def run_pagerank(initial_scores: torch.Tensor, transition: torch.Tensor, damping: float, threshold: float, max_iter: int) -> torch.Tensor:
    pr = initial_scores / initial_scores.sum()
    teleport = pr.clone()

    for _ in range(max_iter):
        new_pr = (1.0 - damping) * teleport + damping * torch.matmul(transition.T, pr)
        if torch.max(torch.abs(new_pr - pr)).item() < threshold:
            return new_pr
        pr = new_pr

    return pr


def select_prompt_for_row(
    row: dict[str, Any],
    sr_scores: dict[str, float],
    vlm_map: dict[tuple[str, int, str], dict[str, str]],
    chatgpt_map: dict[tuple[str, int, str], dict[str, str]],
    image_dir: Path,
    model: CLIPModel,
    processor: AutoProcessor,
    device: str,
    damping: float,
    threshold: float,
    max_iter: int,
) -> dict[str, Any]:
    split = normalize_text(row["split"])
    row_id = int(row["id"])
    article_dir = get_article_dir(image_dir, row)

    sr_values: list[float] = []
    st_values: list[float] = []
    ss_values: list[float] = []
    image_paths: list[Path] = []

    for prompt_id in PROMPT_KEYS:
        vlm = vlm_map[(split, row_id, prompt_id)]
        chatgpt = chatgpt_map[(split, row_id, prompt_id)]
        sr = float(sr_scores[prompt_id])
        st = yes_no_score(vlm["rti"]) + yes_no_score(chatgpt["rtc"])
        ss = yes_no_score(vlm["rsi"]) + yes_no_score(chatgpt["rsc"])

        sr_values.append(sr)
        st_values.append(st)
        ss_values.append(ss)
        image_paths.append(article_dir / f"{prompt_id}.png")

    image_features = encode_images(model, processor, image_paths, device=device)
    initial_scores = torch.tensor(
        [(sr + st + ss) / 3.0 for sr, st, ss in zip(sr_values, st_values, ss_values)],
        dtype=torch.float32,
        device=image_features.device,
    )
    transition = build_transition_matrix(image_features.float())
    pagerank_scores = run_pagerank(
        initial_scores=initial_scores,
        transition=transition,
        damping=damping,
        threshold=threshold,
        max_iter=max_iter,
    )

    best_index = int(torch.argmax(pagerank_scores).item())
    selected_prompt = PROMPT_KEYS[best_index]
    return {
        "id": row_id,
        "split": split,
        "selected_prompt": selected_prompt,
        "score": float(pagerank_scores[best_index].item()),
        "source_image": str(image_paths[best_index]),
    }


def copy_selected_image(source_path: Path, target_dir: Path, split: str, row_id: int) -> Path:
    destination = target_dir / split / f"{row_id}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best generated image per article using CLIP-weighted PageRank.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config, _ = load_runtime(args.config)
    output_dir = Path(config["output_dir"])
    image_dir = Path(config["image_dir"])
    selected_image_dir = image_dir.parent / "images_selected"
    device = config["stage3"]["device"] if torch.cuda.is_available() else "cpu"
    damping = float(config["stage3"]["pagerank_damping"])
    threshold = float(config["stage3"]["pagerank_threshold"])
    max_iter = int(config["stage3"]["pagerank_max_iter"])

    prompt_rows = load_jsonl(output_dir / "prompts_all.jsonl")
    sr_map, vlm_map, chatgpt_map = load_score_maps(output_dir)
    print(f"[step4d] Loaded {len(prompt_rows)} prompt rows")

    model, processor = load_clip_model(config["clip_model"], device=device)
    selected_rows: list[dict[str, Any]] = []
    compatibility_rows: list[dict[str, Any]] = []

    for row in prompt_rows:
        split = normalize_text(row["split"])
        row_id = int(row["id"])
        selection = select_prompt_for_row(
            row=row,
            sr_scores=sr_map[(split, row_id)],
            vlm_map=vlm_map,
            chatgpt_map=chatgpt_map,
            image_dir=image_dir,
            model=model,
            processor=processor,
            device=device,
            damping=damping,
            threshold=threshold,
            max_iter=max_iter,
        )
        destination = copy_selected_image(Path(selection["source_image"]), selected_image_dir, split, row_id)
        selected_rows.append(
            {
                "id": row_id,
                "split": split,
                "selected_prompt": selection["selected_prompt"],
                "score": selection["score"],
            }
        )
        compatibility_rows.append(
            {
                "article_id": row_id,
                "split": split,
                "selected_prompt": selection["selected_prompt"],
                "selected_image": str(destination.relative_to(output_dir.parent)),
                "score": selection["score"],
            }
        )

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    selected_rows.sort(key=lambda row: (row["split"], int(row["id"])))
    compatibility_rows.sort(key=lambda row: (row["split"], int(row["article_id"])))

    save_jsonl(output_dir / "selected_images.jsonl", selected_rows)
    save_json(output_dir / "step4d_selected_images.json", compatibility_rows)
    save_json(
        output_dir / "step4d_summary.json",
        {
            "num_articles": len(selected_rows),
            "selected_output_file": str(output_dir / "selected_images.jsonl"),
            "compatibility_output_file": str(output_dir / "step4d_selected_images.json"),
            "selected_image_dir": str(selected_image_dir),
            "device": device,
            "damping": damping,
            "threshold": threshold,
            "max_iter": max_iter,
        },
    )
    print(f"[step4d] Saved selected prompts to {output_dir / 'selected_images.jsonl'}")
    print(f"[step4d] Copied selected images to {selected_image_dir}")


if __name__ == "__main__":
    main()
