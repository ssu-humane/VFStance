#!/usr/bin/env python3
"""Stage 1: annotate the visual framing of a news article.

The annotator sees the target issue, the headline, and the article body only --
no zero-shot stance prediction is supplied -- and returns a JSON spec filling the
four levels of the visual framing schema (ideological, connotative,
stylistic-semiotic, denotative).

Open-weight models occasionally break the JSON format, so every parse failure is
retried up to ``--max-retries`` times.
"""

import argparse
import json
from pathlib import Path

from .backends import add_backend_args, load_text_backend_for
from .utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_STAGE1_PATH,
    STAGE1_PROMPT_PATH,
    fill_slots,
    get_article_stance,
    get_article_title,
    load_articles,
    parse_json_object,
    read_text,
)

DEFAULT_MAX_TOKENS = {"api": 16000, "open": 16000}


def build_prompt(prompt_template, article):
    return fill_slots(
        prompt_template,
        issue=article.get("issue", ""),
        headline=get_article_title(article),
        article=article.get("article", ""),
    )


def annotate_article(backend, prompt, max_retries):
    """Return (annotation, raw_response, error) retrying on invalid JSON."""
    raw_response = ""
    error = None
    for attempt in range(max_retries + 1):
        raw_response = backend.generate(prompt)
        try:
            return parse_json_object(raw_response), raw_response, None
        except ValueError as parse_error:
            error = str(parse_error)
            if attempt < max_retries:
                print(
                    f"[stage1] invalid JSON (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{error}. retrying",
                    flush=True,
                )
    return None, raw_response, error


def build_record(article, annotation, raw_response, model_name, error=None):
    record = {
        "id": article["id"],
        "issue": article.get("issue", ""),
        "headline": get_article_title(article),
        "stance": get_article_stance(article),
        "split": article.get("split", ""),
        "annotation": annotation,
        "raw_response": raw_response,
        "model": model_name,
    }
    if error:
        record["status"] = "error"
        record["error"] = error
    return record


def parse_args():
    parser = argparse.ArgumentParser(
        description="VFStance Stage 1: visual framing annotation"
    )
    add_backend_args(parser, "text")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--prompt-file", type=Path, default=STAGE1_PROMPT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_STAGE1_PATH)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="Annotate the first N articles.")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Extra attempts for responses that are not valid JSON.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative")
    return args


def main():
    args = parse_args()
    prompt_template = read_text(args.prompt_file)
    articles = load_articles(args.data_path, args.split, args.limit)
    max_tokens = args.max_new_tokens or DEFAULT_MAX_TOKENS[args.backend]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with load_text_backend_for(
        args.backend,
        args.model,
        temperature=args.temperature,
        max_tokens=max_tokens,
        seed=args.seed,
        response_mime_type="application/json",
    ) as backend:
        model_name = backend.model_name
        print(
            f"[stage1] backend={args.backend} model={model_name} "
            f"articles={len(articles)} output={args.output}",
            flush=True,
        )
        with open(args.output, "w", encoding="utf-8") as output_file:
            for index, article in enumerate(articles, start=1):
                prompt = build_prompt(prompt_template, article)
                annotation, raw_response, error = annotate_article(
                    backend, prompt, args.max_retries
                )
                if error:
                    failures += 1
                record = build_record(article, annotation, raw_response, model_name, error)
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
                print(f"[stage1] {index}/{len(articles)} id={article['id']}", flush=True)

    print(
        f"[stage1] done. {len(articles)} articles, {failures} parse failures. "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
