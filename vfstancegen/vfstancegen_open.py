#!/usr/bin/env python3
"""VFStanceGen with open-weight models (full pipeline).

This example script runs the entire VFStanceGen pipeline with open-weight
models, demonstrating that the framework is training-free and modular:

  Stage 1 (visual-framing annotation)              : OpenGVLab/InternVL3-14B-Instruct
  Stage 2 (stance-aware image generation)          : stabilityai/stable-diffusion-3.5-large
  Stage 3 (stance prediction)                      : OpenGVLab/InternVL3-14B-Instruct

It reuses the same prompts, data format, and output conventions as the
Gemini-based scripts (stage1_annotate_visual_framing.py, stage2_generate_image.py,
stage3_predict_stance.py), replacing the Batch API calls with local inference.

Usage:
  python vfstancegen_open.py --stage all           # run stages 1 -> 2 -> 3
  python vfstancegen_open.py --stage 1             # run a single stage
"""

import argparse
import gc
import sys
from pathlib import Path

import torch

VFSTANCEGEN_ROOT = Path(__file__).resolve().parent
SUBMISSION_CODE_ROOT = VFSTANCEGEN_ROOT.parent
sys.path.insert(0, str(VFSTANCEGEN_ROOT))

from src.prompts import T2I_PROMPT_TEMPLATE  # noqa: E402
from src.utils import (  # noqa: E402
    filter_articles_by_split,
    get_article_title,
    load_json,
    load_jsonl,
    parse_json_object,
    write_json,
    write_jsonl,
)

DEFAULT_DATA_PATH = SUBMISSION_CODE_ROOT / "data" / "k-news-stance-mm-acl-arr.json"
DEFAULT_OUTPUT_ROOT = VFSTANCEGEN_ROOT / "outputs" / "open"
DEFAULT_STAGE1_SYSTEM_PROMPT = SUBMISSION_CODE_ROOT / "prompts" / "stage1_visual_framing_annotation_system.txt"
DEFAULT_STAGE1_USER_PROMPT = SUBMISSION_CODE_ROOT / "prompts" / "stage1_visual_framing_annotation_user.txt"
DEFAULT_STAGE3_PROMPT = SUBMISSION_CODE_ROOT / "prompts" / "stage3_stance_prediction.txt"
DEFAULT_LVLM = "OpenGVLab/InternVL3-14B-Instruct"
DEFAULT_T2I = "stabilityai/stable-diffusion-3.5-large"
DEFAULT_ENV_PATH = SUBMISSION_CODE_ROOT / ".env"
STANCE_LABELS = ("supportive", "neutral", "oppositional")


def load_hf_token(env_file):
    """Load HF_TOKEN from a .env file and export it for gated model downloads."""
    import os
    from dotenv import load_dotenv

    load_dotenv(env_file)
    token = os.getenv("HF_TOKEN")
    if token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        os.environ.setdefault("HF_HUB_TOKEN", token)
    return token


def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


def release_gpu(*objects):
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_lvlm(model_name, dtype):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, processor


@torch.inference_mode()
def chat_lvlm(model, processor, system_prompt, user_prompt, image=None,
              max_new_tokens=1024, temperature=0.0):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    user_content = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": user_prompt})
    messages.append({"role": "user", "content": user_content})

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    do_sample = temperature is not None and temperature > 0
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
    )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------
# Stage 1: visual-framing annotation (open LVLM as the annotator)
# --------------------------------------------------------------------------

def run_stage1(args):
    output_dir = args.output_root / "annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = load_prompt_template(args.stage1_system_prompt)
    user_template = load_prompt_template(args.stage1_user_prompt)
    articles = filter_articles_by_split(load_json(args.data_path), args.split)
    if args.limit is not None:
        articles = articles[: args.limit]

    model, processor = load_lvlm(args.lvlm, torch.bfloat16)
    records = []
    for index, article in enumerate(articles, start=1):
        user_prompt = user_template.format(
            issue=article.get("issue", ""),
            headline=get_article_title(article),
            article=article.get("article", ""),
        )
        annotation, raw_response, error = None, "", None
        for _ in range(args.max_retries + 1):
            raw_response = chat_lvlm(
                model, processor, system_prompt, user_prompt,
                max_new_tokens=args.stage1_max_new_tokens, temperature=args.temperature,
            )
            try:
                annotation = parse_json_object(raw_response)
                error = None
                break
            except ValueError as parse_error:
                error = str(parse_error)

        record = {
            "id": article["id"],
            "issue": article.get("issue", ""),
            "headline": get_article_title(article),
            "stance": article.get("stance", ""),
            "split": article.get("split", ""),
            "annotation": annotation,
            "raw_response": raw_response,
            "model": args.lvlm,
        }
        if error:
            record["status"] = "error"
            record["error"] = error
        write_json(output_dir / f"article_{article['id']}.json", record)
        records.append(record)
        print(f"[stage1] {index}/{len(articles)} id={article['id']}"
              f"{' (parse error)' if error else ''}")

    write_jsonl(output_dir / "results.jsonl", records)
    release_gpu(model, processor)
    print(f"[stage1] done. output={output_dir / 'results.jsonl'}")


# --------------------------------------------------------------------------
# Stage 2: stance-aware image generation (open T2I)
# --------------------------------------------------------------------------

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


def run_stage2(args):
    from diffusers import StableDiffusion3Pipeline

    annotation_path = args.output_root / "annotations" / "results.jsonl"
    output_dir = args.output_root / "generated_images"
    images_dir = output_dir / "images"
    prompts_dir = output_dir / "prompts"
    images_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    records = [r for r in load_jsonl(annotation_path) if r.get("annotation")]
    if args.limit is not None:
        records = records[: args.limit]

    pipeline = StableDiffusion3Pipeline.from_pretrained(args.t2i, torch_dtype=torch.bfloat16)
    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
    outputs = []
    for index, record in enumerate(records, start=1):
        prompt = build_t2i_prompt(record["annotation"])
        image = pipeline(
            prompt=prompt,
            num_inference_steps=args.t2i_steps,
            guidance_scale=args.t2i_guidance,
            generator=generator,
        ).images[0]
        image_path = images_dir / f"id_{record['id']}_image_1.png"
        image.save(image_path)
        prompt_record = {
            "id": record["id"],
            "issue": record.get("issue", ""),
            "headline": record.get("headline", ""),
            "stance": record.get("stance", ""),
            "annotation": record["annotation"],
            "image_prompt": prompt,
            "image_path": str(image_path),
            "model": args.t2i,
        }
        write_json(prompts_dir / f"article_{record['id']}.json", prompt_record)
        outputs.append(prompt_record)
        print(f"[stage2] {index}/{len(records)} id={record['id']}")

    write_jsonl(output_dir / "results.jsonl", outputs)
    release_gpu(pipeline)
    print(f"[stage2] done. images={images_dir}")


# --------------------------------------------------------------------------
# Stage 3: stance prediction (open LVLM as the detector)
# --------------------------------------------------------------------------

def normalize_prediction(raw_response):
    lowered = raw_response.strip().lower()
    for label in STANCE_LABELS:
        if label in lowered:
            return label
    return lowered


def run_stage3(args):
    from PIL import Image

    prompt_template = load_prompt_template(args.stage3_prompt)
    image_dir = args.output_root / "generated_images" / "images"
    output_dir = args.output_root / "stance_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    articles = filter_articles_by_split(load_json(args.data_path), args.split)
    if args.limit is not None:
        articles = articles[: args.limit]

    model, processor = load_lvlm(args.lvlm, torch.bfloat16)
    results = []
    for index, article in enumerate(articles, start=1):
        image_candidates = sorted(image_dir.glob(f"id_{article['id']}_image_*.png"))
        if not image_candidates:
            print(f"[stage3] {index}/{len(articles)} id={article['id']} skipped (no image)")
            continue
        image = Image.open(image_candidates[0]).convert("RGB")
        prompt = prompt_template.format(
            issue=article.get("issue", ""),
            event_name=article.get("issue", ""),
            headline=get_article_title(article),
            title=get_article_title(article),
            article=article.get("article", ""),
            main_body=article.get("article", ""),
        )
        raw_response = chat_lvlm(
            model, processor, None, prompt, image=image,
            max_new_tokens=args.stage3_max_new_tokens, temperature=args.temperature,
        )
        results.append({
            "id": article["id"],
            "issue": article.get("issue", ""),
            "headline": get_article_title(article),
            "gold": article.get("stance", ""),
            "prediction": normalize_prediction(raw_response),
            "raw_response": raw_response,
            "model": args.lvlm,
        })
        print(f"[stage3] {index}/{len(articles)} id={article['id']} -> {results[-1]['prediction']}")

    write_jsonl(output_dir / "results.jsonl", results)
    release_gpu(model, processor)
    print(f"[stage3] done. output={output_dir / 'results.jsonl'}")


# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="VFStanceGen full pipeline with open-weight models"
    )
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lvlm", type=str, default=DEFAULT_LVLM,
                        help="Open-weight LVLM for Stages 1 and 3")
    parser.add_argument("--t2i", type=str, default=DEFAULT_T2I,
                        help="Open-weight T2I model for Stage 2")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH,
                        help=".env file containing HF_TOKEN for gated models")
    parser.add_argument("--stage1-system-prompt", type=Path, default=DEFAULT_STAGE1_SYSTEM_PROMPT)
    parser.add_argument("--stage1-user-prompt", type=Path, default=DEFAULT_STAGE1_USER_PROMPT)
    parser.add_argument("--stage3-prompt", type=Path, default=DEFAULT_STAGE3_PROMPT)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=1,
                        help="Stage 1 retries when the response is not valid JSON")
    parser.add_argument("--stage1-max-new-tokens", type=int, default=1024)
    parser.add_argument("--stage3-max-new-tokens", type=int, default=16)
    parser.add_argument("--t2i-steps", type=int, default=28)
    parser.add_argument("--t2i-guidance", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    load_hf_token(args.env_file)
    stages = ["1", "2", "3"] if args.stage == "all" else [args.stage]
    runners = {"1": run_stage1, "2": run_stage2, "3": run_stage3}
    for stage in stages:
        runners[stage](args)


if __name__ == "__main__":
    main()
