#!/usr/bin/env python3
"""Stage 2: turn a Stage 1 spec into a stance-aware generated image.

Only the stylistic-semiotic and denotative levels of the spec are used -- the
eight features style, composition, angle, distance, saturation, luminosity,
include_subjects, and exclude_subjects -- and they are filled into the slots of
the rule-based template in
``prompts/stage2_stance_aware_image_generation_template.txt``. No LLM is involved
in this stage.

The resulting T2I prompt is rendered by the text-to-image model and saved as
``images/id_{id}_image_1.png``.
"""

import argparse
import json
from pathlib import Path

from .backends import add_backend_args, load_image_backend
from .utils import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_STAGE1_PATH,
    DEFAULT_STAGE2_PATH,
    STAGE2_TEMPLATE_PATH,
    build_t2i_prompt,
    image_path,
    load_jsonl,
)


def load_stage1_records(path, limit=None):
    """Load Stage 1 records that carry a usable spec."""
    records = load_jsonl(path)
    usable = [record for record in records if isinstance(record.get("annotation"), dict)]
    skipped = len(records) - len(usable)
    if skipped:
        print(f"[stage2] skipping {skipped} Stage 1 record(s) without a valid spec", flush=True)
    if limit is not None:
        usable = usable[:limit]
    if not usable:
        raise ValueError(f"No Stage 1 records with a valid spec in {path}")
    return usable


def build_record(stage1_record, prompt, saved_path, model_name):
    return {
        "id": stage1_record["id"],
        "issue": stage1_record.get("issue", ""),
        "headline": stage1_record.get("headline", ""),
        "stance": stage1_record.get("stance", ""),
        "annotation": stage1_record["annotation"],
        "image_prompt": prompt,
        "image_path": str(saved_path),
        "model": model_name,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="VFStance Stage 2: stance-aware image generation"
    )
    add_backend_args(parser, "image")
    parser.add_argument("--input", type=Path, default=DEFAULT_STAGE1_PATH,
                        help="Stage 1 output (JSONL).")
    parser.add_argument("--template-file", type=Path, default=STAGE2_TEMPLATE_PATH)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_STAGE2_PATH,
                        help="JSONL log of T2I prompts and saved image paths.")
    parser.add_argument("--limit", type=int, default=None, help="Generate the first N images.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Keep images that already exist instead of regenerating them.")
    parser.add_argument("--steps", type=int, default=28, help="Open backend: diffusion steps.")
    parser.add_argument("--guidance", type=float, default=4.5,
                        help="Open backend: classifier-free guidance scale.")
    parser.add_argument("--t5-max-length", type=int, default=256,
                        help="Open backend: max_sequence_length for the T5 text encoder.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def backend_kwargs(args):
    if args.backend == "api":
        return {}
    return {
        "steps": args.steps,
        "guidance": args.guidance,
        "t5_max_length": args.t5_max_length,
        "seed": args.seed,
    }


def main():
    args = parse_args()
    template = Path(args.template_file).read_text(encoding="utf-8").strip()
    records = load_stage1_records(args.input, args.limit)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with load_image_backend(args.backend, args.model, **backend_kwargs(args)) as backend:
        print(
            f"[stage2] backend={args.backend} model={backend.model_name} "
            f"prompts={len(records)} images={args.image_dir}",
            flush=True,
        )
        with open(args.output, "w", encoding="utf-8") as output_file:
            for index, stage1_record in enumerate(records, start=1):
                article_id = stage1_record["id"]
                prompt = build_t2i_prompt(stage1_record["annotation"], template)
                saved_path = image_path(args.image_dir, article_id)
                if args.skip_existing and saved_path.is_file():
                    print(f"[stage2] {index}/{len(records)} id={article_id} (exists)", flush=True)
                else:
                    image = backend.generate_image(prompt)
                    image.save(saved_path)
                    print(f"[stage2] {index}/{len(records)} id={article_id}", flush=True)
                record = build_record(stage1_record, prompt, saved_path, backend.model_name)
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()

    print(f"[stage2] done. images={args.image_dir} log={args.output}", flush=True)


if __name__ == "__main__":
    main()
