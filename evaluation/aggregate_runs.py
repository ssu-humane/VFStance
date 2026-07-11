#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

from metrics import LABELS, compute_metrics, load_jsonl, resolve_results_path


def summarize_metric(values):
    n = len(values)
    mean = sum(values) / n
    if n <= 1:
        std = 0.0
        se = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (n - 1)
        std = math.sqrt(variance)
        se = std / math.sqrt(n)
    return {
        "mean": mean,
        "std": std,
        "se": se,
        "n_runs": n,
    }


def aggregate_run_metrics(run_metrics):
    accuracy_values = [metric["accuracy"] for metric in run_metrics]
    macro_f1_values = [metric["macro_f1"] for metric in run_metrics]
    per_class_values = {
        label: [metric["per_class_f1"][label] for metric in run_metrics]
        for label in LABELS
    }
    return {
        "accuracy": summarize_metric(accuracy_values),
        "macro_f1": summarize_metric(macro_f1_values),
        "per_class_f1": {
            label: summarize_metric(values)
            for label, values in per_class_values.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics across multiple run outputs")
    parser.add_argument("run_paths", nargs="+", type=Path, help="Run directories or results.jsonl files")
    parser.add_argument("--output-file", type=Path, default=None)
    args = parser.parse_args()

    run_summaries = []
    for run_path in args.run_paths:
        results_path = resolve_results_path(run_path)
        rows = load_jsonl(results_path)
        metrics = compute_metrics(rows)
        run_summaries.append({
            "run_path": str(results_path),
            "metrics": metrics,
        })

    aggregate = aggregate_run_metrics([summary["metrics"] for summary in run_summaries])
    payload = {
        "runs": run_summaries,
        "aggregate": aggregate,
    }

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
