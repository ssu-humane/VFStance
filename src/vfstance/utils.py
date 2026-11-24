"""Shared paths, dataset helpers, and prompt utilities for the VFStance pipeline."""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "prompts"
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "k_news_stance_mm" / "ko" / "k-news-stance-mm.json"

DEFAULT_STAGE1_PATH = OUTPUT_DIR / "stage1.jsonl"
DEFAULT_STAGE2_PATH = OUTPUT_DIR / "stage2.jsonl"
DEFAULT_IMAGE_DIR = OUTPUT_DIR / "images"
DEFAULT_PREDICTION_PATH = OUTPUT_DIR / "predictions.jsonl"

STAGE1_PROMPT_PATH = PROMPT_DIR / "stage1_visual_framing_annotation.txt"
STAGE2_TEMPLATE_PATH = PROMPT_DIR / "stage2_stance_aware_image_generation_template.txt"
STAGE3_PROMPT_PATH = PROMPT_DIR / "stage3_stance_detection.txt"
STAGE3_TEXT_PROMPT_PATH = PROMPT_DIR / "stage3_stance_detection_text.txt"

STANCE_LABELS = ("supportive", "neutral", "oppositional")

# Stage 2 uses only the stylistic-semiotic and denotative levels of the Stage 1 spec.
STYLISTIC_SEMIOTIC_FEATURES = (
    "style",
    "composition",
    "angle",
    "distance",
    "saturation",
    "luminosity",
)
DENOTATIVE_FEATURES = ("include_subjects", "exclude_subjects")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_text(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


def get_article_title(article):
    return article.get("headline") or ""


def get_article_stance(article):
    """Return the public-schema stance while accepting legacy experiment files."""
    return article.get("article_stance") or article.get("stance") or ""


def get_article_split(article):
    return article.get("split") or ""


def filter_articles_by_split(articles, split):
    if split == "all":
        return list(articles)
    return [article for article in articles if get_article_split(article) == split]


def load_articles(data_path, split="test", limit=None):
    articles = filter_articles_by_split(load_json(data_path), split)
    if limit is not None:
        articles = articles[:limit]
    if not articles:
        raise ValueError(f"No articles found for split={split!r} in {data_path}")
    return articles


def parse_json_object(text):
    """Parse the first JSON object in a model response, tolerating code fences."""
    if not text:
        raise ValueError("Empty model response")

    stripped = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    candidates = []
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())
    candidates.append(stripped)

    brace_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Model response was not a valid JSON object")


def fill_slots(template, **values):
    """Replace only the named ``{slot}`` placeholders in a prompt template.

    ``str.format`` cannot be used here: the prompts contain literal JSON braces,
    and a filled-in value may itself contain braces.
    """
    if not values:
        return template
    pattern = re.compile(r"\{(" + "|".join(re.escape(key) for key in values) + r")\}")
    return pattern.sub(lambda match: str(values[match.group(1)]), template)


def image_filename(article_id, image_index=1):
    return f"id_{article_id}_image_{image_index}.png"


def image_path(image_dir, article_id, image_index=1):
    return Path(image_dir) / image_filename(article_id, image_index)


def stringify_subjects(subjects):
    if not subjects:
        return "none"
    joined = ", ".join(str(subject).strip() for subject in subjects if str(subject).strip())
    return joined or "none"


def build_t2i_prompt(annotation, template):
    """Fill the rule-based Stage 2 template with the eight visual-framing features.

    Only the stylistic-semiotic and denotative levels of the Stage 1 spec are used;
    no LLM is involved in this step.
    """
    stylistic = annotation.get("stylistic_semiotic") or {}
    denotative = annotation.get("denotative") or {}
    slots = {feature: stylistic.get(feature, "") for feature in STYLISTIC_SEMIOTIC_FEATURES}
    for feature in DENOTATIVE_FEATURES:
        slots[feature] = stringify_subjects(denotative.get(feature, []))
    return fill_slots(template, **slots)
