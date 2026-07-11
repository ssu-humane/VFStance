#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from metrics import LABELS, compute_metrics, load_jsonl, resolve_results_path


def collect_run_metrics(run_paths):
    run_metrics = []
    for run_path in run_paths:
        results_path = resolve_results_path(run_path)
        rows = load_jsonl(results_path)
        run_metrics.append({
            "run_path": str(results_path),
            "metrics": compute_metrics(rows),
        })
    return run_metrics


def select_metric(run_metrics, metric_name):
    if metric_name == "accuracy":
        return [run["metrics"]["accuracy"] for run in run_metrics]
    if metric_name == "macro_f1":
        return [run["metrics"]["macro_f1"] for run in run_metrics]
    prefix = "per_class_f1."
    if metric_name.startswith(prefix):
        label = metric_name[len(prefix):]
        if label not in LABELS:
            raise ValueError(f"Unsupported class label for per_class_f1: {label}")
        return [run["metrics"]["per_class_f1"][label] for run in run_metrics]
    raise ValueError(f"Unsupported metric: {metric_name}")


def main():
    try:
        from scipy.stats import mannwhitneyu
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scipy is required for Mann–Whitney U test") from exc

    parser = argparse.ArgumentParser(description="Run Mann–Whitney U significance test across run metrics")
    parser.add_argument("--group-a", nargs="+", required=True, type=Path, help="Run directories or results.jsonl files for group A")
    parser.add_argument("--group-b", nargs="+", required=True, type=Path, help="Run directories or results.jsonl files for group B")
    parser.add_argument("--metric", type=str, default="macro_f1", choices=[
        "accuracy",
        "macro_f1",
        "per_class_f1.supportive",
        "per_class_f1.neutral",
        "per_class_f1.oppositional",
    ])
    parser.add_argument("--alternative", type=str, default="two-sided", choices=["two-sided", "less", "greater"])
    args = parser.parse_args()

    group_a_runs = collect_run_metrics(args.group_a)
    group_b_runs = collect_run_metrics(args.group_b)
    group_a_values = select_metric(group_a_runs, args.metric)
    group_b_values = select_metric(group_b_runs, args.metric)

    statistic, p_value = mannwhitneyu(group_a_values, group_b_values, alternative=args.alternative)
    payload = {
        "metric": args.metric,
        "alternative": args.alternative,
        "group_a": {
            "n_runs": len(group_a_values),
            "values": group_a_values,
            "runs": [run["run_path"] for run in group_a_runs],
        },
        "group_b": {
            "n_runs": len(group_b_values),
            "values": group_b_values,
            "runs": [run["run_path"] for run in group_b_runs],
        },
        "mann_whitney_u": {
            "statistic": float(statistic),
            "p_value": float(p_value),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
