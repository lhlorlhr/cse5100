#!/usr/bin/env python
"""Aggregate and plot communication-activity sweeps for MPE simple_tag."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plot_mpe_comm_activity_sweeps.py. "
        f"Import failed with: {exc!r}"
    )


OPPONENTS = ["random", "heuristic"]
METRICS = ["mean_predator_return", "mean_capture_step", "capture_rate", "mean_collision_steps"]
ACTIVITY_FLOOR = 1e-4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("onpolicy/scripts/results/MPE/simple_tag/mappo"),
        help="Directory containing step1_full_* and step2_dim*_lambda* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to ROOT/comm_activity_sweeps.",
    )
    return parser.parse_args()


def parse_lambda(path):
    match = re.search(r"lambda([0-9]+(?:p[0-9]+)?)", path.name)
    if match is None:
        raise ValueError(f"Could not parse lambda from {path}")
    return float(match.group(1).replace("p", "."))


def parse_dim(path):
    match = re.search(r"dim(\d+)", path.name)
    if match is None:
        raise ValueError(f"Could not parse communication dimension from {path}")
    return int(match.group(1))


def scalar_from_summary(summary, metric):
    if not summary:
        return np.nan
    latest_step = max(summary.keys(), key=lambda step: int(step))
    value = summary[latest_step].get(metric)
    if value is None:
        return np.nan
    if isinstance(value, dict):
        return float(value.get(metric, np.nan))
    return float(value)


def read_run(run_dir):
    metrics_path = run_dir / "fixed_opponent_eval" / "fixed_opponent_metrics.json"
    summary_path = run_dir / "logs" / "summary.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics_payload = json.load(handle)
    with summary_path.open("r", encoding="utf-8") as handle:
        log_summary = json.load(handle)

    row = {
        "seed": int(metrics_payload.get("seed", re.search(r"run(\d+)", run_dir.name).group(1))),
        "run": int(re.search(r"run(\d+)", run_dir.name).group(1)),
    }
    for opponent in OPPONENTS:
        opponent_result = metrics_payload["results"][opponent]
        for metric in METRICS:
            row[f"{opponent}_{metric}"] = float(opponent_result[metric])

    speak_rate = scalar_from_summary(log_summary, "comm/speak_rate")
    if np.isnan(speak_rate):
        speak_rate = scalar_from_summary(log_summary, "agent0/mean_step_comm_activity")
    row["speak_rate"] = speak_rate
    row["token0_share"] = scalar_from_summary(log_summary, "comm/vocab_usage_full_token0")
    row["comm_entropy_active"] = scalar_from_summary(log_summary, "comm/comm_entropy_active")
    return row


def read_step1_rows(root):
    rows = []
    for exp_dir in sorted(root.glob("step1_full_comm8_lambda*"), key=parse_lambda):
        penalty = parse_lambda(exp_dir)
        for run_dir in sorted(exp_dir.glob("run*")):
            metrics_path = run_dir / "fixed_opponent_eval" / "fixed_opponent_metrics.json"
            summary_path = run_dir / "logs" / "summary.json"
            if not metrics_path.exists() or not summary_path.exists():
                continue
            row = read_run(run_dir)
            row.update({"family": "step1_comm8", "dim": 8, "lambda": penalty, "experiment": exp_dir.name})
            rows.append(row)
    return pd.DataFrame(rows)


def read_step2_rows(root):
    rows = []
    for exp_dir in sorted(root.glob("step2_dim*_lambda*"), key=lambda path: (parse_dim(path), parse_lambda(path))):
        dim = parse_dim(exp_dir)
        penalty = parse_lambda(exp_dir)
        for run_dir in sorted(exp_dir.glob("run*")):
            metrics_path = run_dir / "fixed_opponent_eval" / "fixed_opponent_metrics.json"
            summary_path = run_dir / "logs" / "summary.json"
            if not metrics_path.exists() or not summary_path.exists():
                continue
            row = read_run(run_dir)
            row.update({"family": "step2_dim_lambda", "dim": dim, "lambda": penalty, "experiment": exp_dir.name})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(rows, group_columns):
    numeric_columns = [
        column
        for column in rows.columns
        if column not in set(group_columns) | {"family", "experiment"}
        and pd.api.types.is_numeric_dtype(rows[column])
    ]
    pieces = []
    for column in numeric_columns:
        stats = rows.groupby(group_columns, sort=True)[column].agg(["mean", "std", "count"]).reset_index()
        stats["metric"] = column
        stats = stats.rename(columns={"mean": "mean", "std": "std", "count": "n"})
        pieces.append(stats)
    summary = pd.concat(pieces, ignore_index=True)
    summary["std"] = summary["std"].fillna(0.0)
    return summary[group_columns + ["metric", "mean", "std", "n"]]


def wide_summary(summary, index_columns):
    mean_table = summary.pivot_table(index=index_columns, columns="metric", values="mean", aggfunc="first")
    std_table = summary.pivot_table(index=index_columns, columns="metric", values="std", aggfunc="first")
    mean_table = mean_table.add_suffix("_mean")
    std_table = std_table.add_suffix("_std")
    return pd.concat([mean_table, std_table], axis=1).reset_index()


def format_lambda(value):
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:g}"


def style_matplotlib():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def plot_comm8_sweep(step1_wide, step1_rows, output_dir):
    style_matplotlib()
    data = step1_wide.sort_values("lambda").copy()
    data["activity_plot"] = data["speak_rate_mean"].clip(lower=ACTIVITY_FLOOR)
    data["activity_std_plot"] = data["speak_rate_std"].fillna(0.0)

    lambda_positions = np.arange(len(data))
    lambda_labels = [format_lambda(value) for value in data["lambda"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    ax = axes[0]
    ax.errorbar(
        lambda_positions,
        data["activity_plot"],
        yerr=data["activity_std_plot"],
        marker="o",
        markersize=4,
        linewidth=1.8,
        capsize=3,
        color="#4C78A8",
    )
    ax.set_yscale("log")
    ax.set_ylim(ACTIVITY_FLOOR * 0.7, 1.0)
    ax.set_xticks(lambda_positions)
    ax.set_xticklabels(lambda_labels)
    ax.set_xlabel("Penalty $\\lambda$")
    ax.set_ylabel("Communication activity")
    ax.set_title("Activity collapse")
    ax.grid(axis="y", alpha=0.3, which="both")

    ax = axes[1]
    seed_rows = step1_rows.copy()
    seed_rows["activity_plot"] = seed_rows["speak_rate"].clip(lower=ACTIVITY_FLOOR)
    ax.scatter(
        seed_rows["activity_plot"],
        seed_rows["heuristic_mean_predator_return"],
        s=14,
        color="#9E9E9E",
        alpha=0.55,
        edgecolor="none",
        label="seed",
        zorder=1,
    )
    ax.plot(
        data["activity_plot"],
        data["heuristic_mean_predator_return_mean"],
        marker="o",
        markersize=4,
        linewidth=1.5,
        color="#F58518",
        label="mean",
        zorder=2,
    )
    for _, row in data.iterrows():
        text = f"$\\lambda$={format_lambda(row['lambda'])}"
        xytext = {
            0.0: (-24, 9),
            0.5: (6, 8),
            0.71: (7, -18),
            1.0: (7, 8),
            1.41: (8, -17),
            2.0: (-30, 10),
            10.0: (7, -24),
        }.get(float(row["lambda"]), (5, 6))
        if row["lambda"] >= 10:
            text = "$\\lambda$=10\n(silent)"
        ax.annotate(
            text,
            (row["activity_plot"], row["heuristic_mean_predator_return_mean"]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set_xlim(ACTIVITY_FLOOR * 0.7, 1.0)
    ax.set_xlabel("Communication activity")
    ax.set_ylabel("Heuristic-prey return")
    ax.set_title("Performance--activity trade-off")
    ax.grid(axis="both", alpha=0.3, which="both")
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout(pad=0.7)
    for suffix in ["pdf", "png"]:
        fig.savefig(output_dir / f"F_activity_comm8_lambda_sweep.{suffix}", dpi=300)
    plt.close(fig)


def plot_dim_ablation(step2_wide, output_dir):
    style_matplotlib()
    dims = sorted(step2_wide["dim"].unique())
    colors = {2: "#4C78A8", 8: "#F58518", 32: "#54A24B"}
    markers = {2: "o", 8: "s", 32: "^"}

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharex=True)
    for dim in dims:
        dim_data = step2_wide[step2_wide["dim"] == dim].sort_values("lambda").copy()
        dim_data["activity_plot"] = dim_data["speak_rate_mean"].clip(lower=ACTIVITY_FLOOR)

        axes[0].plot(
            dim_data["lambda"],
            dim_data["activity_plot"],
            marker=markers.get(dim, "o"),
            markersize=4,
            linewidth=1.8,
            color=colors.get(dim),
            label=f"$d={dim}$",
        )
        axes[1].plot(
            dim_data["lambda"],
            dim_data["heuristic_mean_predator_return_mean"],
            marker=markers.get(dim, "o"),
            markersize=4,
            linewidth=1.8,
            color=colors.get(dim),
            label=f"$d={dim}$",
        )

    axes[0].set_yscale("log")
    axes[0].set_ylim(ACTIVITY_FLOOR * 0.7, 1.0)
    axes[0].set_xlabel("Penalty $\\lambda$")
    axes[0].set_ylabel("Communication activity")
    axes[0].set_title("Activity by channel size")
    axes[0].grid(axis="y", alpha=0.3, which="both")

    axes[1].set_xlabel("Penalty $\\lambda$")
    axes[1].set_ylabel("Heuristic-prey return")
    axes[1].set_title("Fixed-opponent performance")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(frameon=False, loc="best")

    for ax in axes:
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])

    fig.tight_layout(pad=0.7)
    for suffix in ["pdf", "png"]:
        fig.savefig(output_dir / f"F_dim_lambda_activity_ablation.{suffix}", dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else root / "comm_activity_sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)

    step1_rows = read_step1_rows(root)
    step2_rows = read_step2_rows(root)
    if step1_rows.empty:
        raise SystemExit(f"No step1_full_comm8_lambda* fixed-opponent outputs found under {root}")
    if step2_rows.empty:
        raise SystemExit(f"No step2_dim*_lambda* fixed-opponent outputs found under {root}")

    step1_summary = summarize(step1_rows, ["lambda"])
    step2_summary = summarize(step2_rows, ["dim", "lambda"])
    step1_wide = wide_summary(step1_summary, ["lambda"])
    step2_wide = wide_summary(step2_summary, ["dim", "lambda"])

    step1_rows.to_csv(output_dir / "comm8_activity_sweep_seed_metrics.csv", index=False)
    step1_wide.to_csv(output_dir / "comm8_activity_sweep_summary.csv", index=False)
    step2_rows.to_csv(output_dir / "dim_lambda_activity_ablation_seed_metrics.csv", index=False)
    step2_wide.to_csv(output_dir / "dim_lambda_activity_ablation_summary.csv", index=False)

    plot_comm8_sweep(step1_wide, step1_rows, output_dir)
    plot_dim_ablation(step2_wide, output_dir)
    print(f"saved communication activity sweep outputs under {output_dir}")


if __name__ == "__main__":
    main()
