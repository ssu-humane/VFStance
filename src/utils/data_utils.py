from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config_file = Path(config_path) if config_path else ROOT_DIR / "config.yaml"
    with config_file.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    return resolve_paths(config, config_file.parent)


def resolve_paths(config: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    resolved = dict(config)
    for key in ("data_path", "env_path", "output_dir", "image_dir"):
        value = resolved.get(key)
        if value:
            resolved[key] = str((config_dir / value).resolve()) if not Path(value).is_absolute() else value
    return resolved


def load_runtime(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path)
    env_path = Path(config["env_path"])
    if env_path.exists():
        load_dotenv(env_path)
    ensure_directories(config)
    return config, ROOT_DIR


def ensure_directories(config: dict[str, Any]) -> None:
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    image_dir = Path(config["image_dir"])
    for relative in ("", "train", "val", "test"):
        (image_dir / relative).mkdir(parents=True, exist_ok=True)
    selected_image_dir = image_dir.parent / "images_selected"
    for relative in ("", "train", "val", "test", "test_as_original"):
        (selected_image_dir / relative).mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "models" / "stance_predictor").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "models" / "multimodal_detector").mkdir(parents=True, exist_ok=True)


def load_dataset(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_path = Path(config["data_path"])
    with data_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "items", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported dataset structure in {data_path}")


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows = []
    with source.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_article_id(record: dict[str, Any], fallback_index: int) -> str:
    article_id = record.get("article_id") or record.get("id") or record.get("news_id")
    return str(article_id) if article_id is not None else f"{fallback_index:03d}"


def infer_split(record: dict[str, Any]) -> str:
    split = str(record.get("split", "train")).lower()
    return split if split in {"train", "val", "test"} else "train"


def split_sentences(text: str, max_sentences: int) -> list[str]:
    if not text:
        return []
    candidates = [chunk.strip() for chunk in text.replace("\n", " ").split(".") if chunk.strip()]
    return candidates[:max_sentences]


def stage_output_path(config: dict[str, Any], filename: str) -> Path:
    return Path(config["output_dir"]) / filename


def getenv_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Required environment variable is missing: {name}")
    return value
