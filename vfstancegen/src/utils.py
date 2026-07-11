import json
import mimetypes
import re
from pathlib import Path


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


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def get_article_title(article):
    return article.get("headline") or ""


def get_article_split(article):
    return article.get("split") or ""


def filter_articles_by_split(articles, split):
    if split == "all":
        return list(articles)
    return [article for article in articles if get_article_split(article) == split]


def guess_mime_type(path):
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "image/png"


def extract_response_text(response):
    text = getattr(response, "text", None)
    if text:
        return text.strip()

    response_dict = response.model_dump()
    candidates = response_dict.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if "text" in part and part["text"]:
                return str(part["text"]).strip()
    return ""


def parse_json_object(text):
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
