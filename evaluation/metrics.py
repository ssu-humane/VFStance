#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score


LABELS = ["supportive", "neutral", "oppositional"]
LABEL_MAP = {
    "supportive": "supportive",
    "neutral": "neutral",
    "oppositional": "oppositional",
    "지지적": "supportive",
    "중립적": "neutral",
    "비판적": "oppositional",
}


def normalize_label(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return LABEL_MAP.get(text, text.lower())


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_results_path(path):
    path = Path(path)
    if path.is_dir():
        for candidate in ("results.jsonl", "predictions.jsonl", "test_results.jsonl", "test_predictions.jsonl"):
            candidate_path = path / candidate
            if candidate_path.exists():
                return candidate_path
        return path / "results.jsonl"
    return path


def extract_gold_label(row):
    for key in ("gold", "stance", "label", "target"):
        value = row.get(key)
        normalized = normalize_label(value)
        if normalized:
            return normalized
    return ""


def extract_pred_label(row):
    for key in ("prediction", "pred", "output"):
        value = row.get(key)
        normalized = normalize_label(value)
        if normalized:
            return normalized
    return ""


def collect_labels(rows):
    gold = []
    pred = []
    kept_rows = []
    skipped = 0
    for row in rows:
        gold_label = extract_gold_label(row)
        pred_label = extract_pred_label(row)
        if gold_label not in LABELS or pred_label not in LABELS:
            skipped += 1
            continue
        gold.append(gold_label)
        pred.append(pred_label)
        kept_rows.append(row)
    return gold, pred, kept_rows, skipped


def safe_standard_error(values):
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return math.sqrt(variance) / math.sqrt(n)


def compute_metrics_from_labels(gold, pred):
    accuracy = float(accuracy_score(gold, pred))
    macro_f1 = float(f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0))
    per_class = f1_score(gold, pred, labels=LABELS, average=None, zero_division=0)
    correctness = [1.0 if g == p else 0.0 for g, p in zip(gold, pred)]
    return {
        "n": len(gold),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": {label: float(score) for label, score in zip(LABELS, per_class)},
        "standard_error": {
            "accuracy": safe_standard_error(correctness),
        },
    }


def compute_metrics(rows):
    gold, pred, kept_rows, skipped = collect_labels(rows)
    if not gold:
        raise ValueError("No valid labeled rows found in results")
    metrics = compute_metrics_from_labels(gold, pred)
    metrics["skipped_rows"] = skipped
    metrics["valid_rows"] = len(kept_rows)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics for one results.jsonl")
    parser.add_argument("results_path", type=Path, help="Path to results.jsonl or a directory containing it")
    args = parser.parse_args()

    results_path = resolve_results_path(args.results_path)
    rows = load_jsonl(results_path)
    metrics = compute_metrics(rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
