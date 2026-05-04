#!/usr/bin/env python
"""Aggregate probe metrics into flat CSV/JSON summaries."""

import json
import sys
from pathlib import Path

import pandas as pd


def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args(argv)


def flatten_metrics(payload):
    base = {
        "dataset_path": payload.get("dataset_path", ""),
        "task": payload.get("task"),
        "probe_type": payload.get("probe_type"),
        "message_variant": payload.get("message_variant"),
        "variant_label": payload.get("variant_label"),
        "wolf_id": payload.get("wolf_id"),
        "split_seed": payload.get("split_seed"),
        "n_train_samples": payload.get("n_train_samples"),
        "n_val_samples": payload.get("n_val_samples"),
        "n_test_samples": payload.get("n_test_samples"),
        "n_train_episodes": payload.get("n_train_episodes"),
        "n_val_episodes": payload.get("n_val_episodes"),
        "n_test_episodes": payload.get("n_test_episodes"),
    }
    rows = []
    for metric_name, metric_value in payload.get("metrics", {}).items():
        row = dict(base)
        row["metric"] = metric_name
        row["value"] = metric_value
        rows.append(row)
    return rows


def main(argv):
    args = parse_args(argv)
    probe_root = Path(args.probe_root).expanduser().resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = probe_root
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_files = sorted(probe_root.rglob("metrics.json"))
    if not metric_files:
        raise SystemExit(f"No metrics.json files found under {probe_root}")

    all_rows = []
    for metric_file in metric_files:
        payload = json.loads(metric_file.read_text())
        all_rows.extend(flatten_metrics(payload))

    all_runs = pd.DataFrame(all_rows).sort_values(
        by=["task", "wolf_id", "variant_label", "metric"]
    )
    all_runs.to_csv(output_dir / "probe_all_runs.csv", index=False)

    summary = (
        all_runs.groupby(
            ["task", "probe_type", "wolf_id", "variant_label", "metric"],
            dropna=False,
        )["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_runs"})
    )
    summary.to_csv(output_dir / "probe_summary.csv", index=False)
    (output_dir / "probe_summary.json").write_text(summary.to_json(orient="records", indent=2))

    pivot = summary.pivot_table(
        index=["task", "wolf_id", "metric"],
        columns="variant_label",
        values="mean",
        aggfunc="first",
    ).reset_index()

    if "marginal" in pivot.columns:
        for col in list(pivot.columns):
            if isinstance(col, str) and col.startswith("random_seed"):
                pivot["trained_minus_" + col] = pivot.get("trained", float("nan")) - pivot[col]
        pivot["trained_minus_marginal"] = pivot.get("trained", float("nan")) - pivot["marginal"]

    pivot.to_csv(output_dir / "F2a_table.csv", index=False)
    print(f"saved summaries under {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
