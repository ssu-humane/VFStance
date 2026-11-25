#!/usr/bin/env python3
"""Stage 3: detect the stance of an article toward its target issue.

``--input-mode multimodal`` (VFStance) gives the LVLM the issue, the article, and
the Stage 2 generated image. ``--input-mode text`` (VFStance (Text)) skips image
generation and puts the Stage 1 spec into the prompt as text, rendered through the
same Stage 2 template.

The model is asked for a single word out of supportive / neutral / oppositional.
Open-weight models do not always obey that instruction, so ``normalize_prediction``
extracts the label from the raw response.
"""

import argparse
import json
import re
from pathlib import Path

from .backends import add_backend_args, load_text_backend_for
from .utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_IMAGE_DIR,
    DEFAULT_PREDICTION_PATH,
    DEFAULT_STAGE1_PATH,
    STAGE2_TEMPLATE_PATH,
    STAGE3_PROMPT_PATH,
    STAGE3_TEXT_PROMPT_PATH,
    STANCE_LABELS,
    build_t2i_prompt,
    fill_slots,
    get_article_stance,
    get_article_title,
    image_path,
    load_articles,
    load_jsonl,
    read_text,
)

DEFAULT_MAX_TOKENS = {"api": 16000, "open": 16}
METHOD_NAMES = {"multimodal": "VFStance", "text": "VFStance (Text)"}


def normalize_prediction(raw_response):
    """Extract a stance label from a possibly verbose response."""
    normalized = (raw_response or "").strip().lower()
    if normalized in STANCE_LABELS:
        return normalized
    matches = [
        label
        for label in STANCE_LABELS
        if re.search(rf"\b{re.escape(label)}\b", normalized)
    ]
    return matches[0] if len(matches) == 1 else normalized


def build_prompt(prompt_template, article, visual_description=""):
    return fill_slots(
        prompt_template,
        issue=article.get("issue", ""),
        headline=get_article_title(article),
        article=article.get("article", ""),
        visual_description=visual_description,
    )


def load_visual_descriptions(stage1_path, template_path):
    """Render the Stage 1 specs into the visual descriptions used by text mode."""
    template = Path(template_path).read_text(encoding="utf-8").strip()
    descriptions = {}
    for record in load_jsonl(stage1_path):
        annotation = record.get("annotation")
        if isinstance(annotation, dict):
            descriptions[str(record["id"])] = build_t2i_prompt(
                annotation, template, image_generation=False
            )
    return descriptions


def resolve_inputs(args, articles):
    """Map each article id to its generated image path or visual description."""
    if args.input_mode == "multimodal":
        resolved = {
            str(article["id"]): image_path(args.image_dir, article["id"]) for article in articles
        }
        missing = [article_id for article_id, path in resolved.items() if not path.is_file()]
        if missing:
            preview = ", ".join(missing[:10])
            raise FileNotFoundError(
                f"Missing {len(missing)} Stage 2 image(s) in {args.image_dir} "
                f"(first article IDs: {preview}). Run Stage 2 first."
            )
        return resolved

    descriptions = load_visual_descriptions(args.stage1_path, args.template_file)
    missing = [str(article["id"]) for article in articles if str(article["id"]) not in descriptions]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"Missing valid Stage 1 specs for {len(missing)} article(s) in {args.stage1_path} "
            f"(first article IDs: {preview}). Run Stage 1 first."
        )
    return descriptions


def parse_args():
    parser = argparse.ArgumentParser(description="VFStance Stage 3: stance detection")
    add_backend_args(parser, "text")
    parser.add_argument(
        "--input-mode",
        choices=["multimodal", "text"],
        default="multimodal",
        help="multimodal: article + generated image (VFStance); "
             "text: article + Stage 1 spec as text (VFStance (Text)).",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR,
                        help="multimodal mode: directory with the Stage 2 images.")
    parser.add_argument("--stage1-path", type=Path, default=DEFAULT_STAGE1_PATH,
                        help="text mode: Stage 1 output (JSONL).")
    parser.add_argument("--template-file", type=Path, default=STAGE2_TEMPLATE_PATH,
                        help="text mode: template used to render the spec as text.")
    parser.add_argument("--prompt-file", type=Path, default=None,
                        help="Defaults to the prompt file matching --input-mode.")
    parser.add_argument("--output", type=Path, default=DEFAULT_PREDICTION_PATH)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="Predict the first N articles.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.prompt_file is None:
        args.prompt_file = (
            STAGE3_TEXT_PROMPT_PATH if args.input_mode == "text" else STAGE3_PROMPT_PATH
        )
    return args


def main():
    args = parse_args()
    prompt_template = read_text(args.prompt_file)
    articles = load_articles(args.data_path, args.split, args.limit)
    resolved = resolve_inputs(args, articles)
    max_tokens = args.max_new_tokens or DEFAULT_MAX_TOKENS[args.backend]
    method = METHOD_NAMES[args.input_mode]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    invalid = 0
    with load_text_backend_for(
        args.backend,
        args.model,
        temperature=args.temperature,
        max_tokens=max_tokens,
        seed=args.seed,
    ) as backend:
        print(
            f"[stage3] backend={args.backend} model={backend.model_name} method={method} "
            f"articles={len(articles)} output={args.output}",
            flush=True,
        )
        with open(args.output, "w", encoding="utf-8") as output_file:
            for index, article in enumerate(articles, start=1):
                article_id = str(article["id"])
                image = None
                visual_description = ""
                resolved_image_path = None
                if args.input_mode == "multimodal":
                    from PIL import Image

                    resolved_image_path = str(resolved[article_id])
                    with Image.open(resolved_image_path) as source:
                        image = source.convert("RGB").copy()
                else:
                    visual_description = resolved[article_id]

                prompt = build_prompt(prompt_template, article, visual_description)
                try:
                    raw_response = backend.generate(prompt, image=image)
                finally:
                    if image is not None:
                        image.close()

                prediction = normalize_prediction(raw_response)
                if prediction not in STANCE_LABELS:
                    invalid += 1
                record = {
                    "id": article["id"],
                    "issue": article.get("issue", ""),
                    "headline": get_article_title(article),
                    "gold": get_article_stance(article),
                    "prediction": prediction,
                    "raw_response": raw_response,
                    "resolved_image_path": resolved_image_path,
                    "visual_description": visual_description or None,
                    "input_mode": args.input_mode,
                    "method": method,
                    "model": backend.model_name,
                    "status": "ok" if prediction in STANCE_LABELS else "invalid_prediction",
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
                print(
                    f"[stage3] {index}/{len(articles)} id={article_id} "
                    f"gold={record['gold']} prediction={prediction}",
                    flush=True,
                )

    print(
        f"[stage3] done. {len(articles)} articles, {invalid} invalid prediction(s). "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
