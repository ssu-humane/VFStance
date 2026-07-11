from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

LABELS = ["oppositional", "neutral", "supportive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
IMAGE_EXTENSION_PRIORITY = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]


def submission_root(anchor_file: str | Path) -> Path:
    return Path(anchor_file).resolve().parents[3]


def default_dataset_path(anchor_file: str | Path) -> Path:
    return submission_root(anchor_file) / "data" / "k-news-stance-mm-acl-arr.json"


def default_image_dir(anchor_file: str | Path) -> Path:
    return submission_root(anchor_file) / "data" / "images"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def save_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_logger(output_dir: str | Path, name: str) -> logging.Logger:
    output_dir = ensure_dir(output_dir)
    logger = logging.getLogger(f"{name}_{output_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def resolve_image_file(image_dir: str | Path, article_id: int) -> str | None:
    image_dir = Path(image_dir)
    for extension in IMAGE_EXTENSION_PRIORITY:
        candidate = image_dir / f"{article_id}_1{extension}"
        if candidate.exists():
            return str(candidate)
    matches = sorted(image_dir.glob(f"{article_id}_1.*"))
    if matches:
        return str(matches[0])
    return None


def compute_metrics(gold_ids: Sequence[int], pred_ids: Sequence[int]) -> Dict[str, Any]:
    labels = [LABEL_TO_ID[label] for label in LABELS]
    class_f1 = f1_score(gold_ids, pred_ids, labels=labels, average=None, zero_division=0)
    matrix = confusion_matrix(gold_ids, pred_ids, labels=labels)
    return {
        "accuracy": float(accuracy_score(gold_ids, pred_ids)),
        "macro_f1": float(f1_score(gold_ids, pred_ids, labels=labels, average="macro", zero_division=0)),
        "f1_oppositional": float(class_f1[LABEL_TO_ID["oppositional"]]),
        "f1_neutral": float(class_f1[LABEL_TO_ID["neutral"]]),
        "f1_supportive": float(class_f1[LABEL_TO_ID["supportive"]]),
        "confusion_matrix": {"labels": LABELS, "matrix": matrix.tolist()},
    }
