#!/usr/bin/env python
"""Aggregate and plot fixed-opponent bandwidth-sweep results for MPE simple_tag."""

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
        "matplotlib is required for plot_mpe_bandwidth_sweep.py. "
        f"Import failed with: {exc!r}"
    )


DEFAULT_DIMS = [0, 2, 4, 8, 16, 32, 64]
OPPONENTS = ["random", "heuristic"]
METRICS = ["mean_predator_return", "mean_capture_step", "capture_rate", "mean_collision_steps"]
OPPONENT_LABELS = {
    "random": "Random prey",
    "heuristic": "Heuristic prey",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("onpolicy/scripts/results/MPE/simple_tag/mappo"),
        help="Directory containing base_reward_* run folders.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=DEFAULT_DIMS,
        help="Communication dimensions to include. Use 0 for no communication.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to ROOT/base_reward_bandwidth_sweep.",
    )
    return parser.parse_args()


def run_dir_for_dim(root, dim):
    if dim == 0:
        return root / "base_reward_nocomm"
    return root / f"base_reward_comm{dim}"


def read_rows(root, dims):
    rows = []
    for dim in dims:
        exp_dir = run_dir_for_dim(root, dim)
        for metrics_path in sorted(exp_dir.glob("run*/fixed_opponent_eval/fixed_opponent_metrics.json")):
            run_match = re.search(r"run(\d+)", str(metrics_path))
            if run_match is None:
                continue
            with metrics_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            seed = int(payload.get("seed", run_match.group(1)))
            for opponent in OPPONENTS:
                result = payload["results"][opponent]
                row = {
                    "dim": dim,
                    "seed": seed,
                    "run": int(run_match.group(1)),
                    "opponent": opponent,
                }
                for metric in METRICS:
                    row[metric] = float(result[metric])
                rows.append(row)
    return pd.DataFrame(rows)


def aggregate(rows):
    grouped = rows.groupby(["dim", "opponent"], sort=True)
    pieces = []
    for metric in METRICS:
        stats = grouped[metric].agg(["mean", "std", "count"]).reset_index()
        stats["metric"] = metric
        stats = stats.rename(columns={"count": "n"})
        pieces.append(stats)
    long = pd.concat(pieces, ignore_index=True)
    long["std"] = long["std"].fillna(0.0)
    return long[["dim", "opponent", "metric", "mean", "std", "n"]]


def write_latex_table(summary, output_path):
    table = summary.pivot_table(
        index="dim",
        columns=["opponent", "metric"],
        values=["mean", "std"],
        aggfunc="first",
    )
    base = table.loc[0]

    def fmt_mean_std(dim, opponent, metric):
        mean = table.loc[dim, ("mean", opponent, metric)]
        std = table.loc[dim, ("std", opponent, metric)]
        return rf"\({mean:.1f} \pm {std:.1f}\)"

    def fmt_delta(dim, opponent):
        if dim == 0:
            return "--"
        value = (
            table.loc[dim, ("mean", opponent, "mean_predator_return")]
            - base[("mean", opponent, "mean_predator_return")]
        )
        return rf"\({value:+.1f}\)"

    random_means = table[("mean", "random", "mean_predator_return")]
    heuristic_means = table[("mean", "heuristic", "mean_predator_return")]
    best_random = float(random_means.max())
    best_heuristic = float(heuristic_means.max())

    lines = [
        r"\begin{tabular}{rcccc}",
        r"\toprule",
        r"\(d\) & Random return & \(\Delta\) vs. \(d=0\) & Heuristic return & \(\Delta\) vs. \(d=0\) \\",
        r"\midrule",
    ]
    for dim in sorted(summary["dim"].unique()):
        random_cell = fmt_mean_std(dim, "random", "mean_predator_return")
        heuristic_cell = fmt_mean_std(dim, "heuristic", "mean_predator_return")
        if np.isclose(table.loc[dim, ("mean", "random", "mean_predator_return")], best_random):
            random_cell = rf"\(\mathbf{{{table.loc[dim, ('mean', 'random', 'mean_predator_return')]:.1f} \pm {table.loc[dim, ('std', 'random', 'mean_predator_return')]:.1f}}}\)"
        if np.isclose(table.loc[dim, ("mean", "heuristic", "mean_predator_return")], best_heuristic):
            heuristic_cell = rf"\(\mathbf{{{table.loc[dim, ('mean', 'heuristic', 'mean_predator_return')]:.1f} \pm {table.loc[dim, ('std', 'heuristic', 'mean_predator_return')]:.1f}}}\)"
        lines.append(
            f"{dim} & {random_cell} & {fmt_delta(dim, 'random')} & "
            f"{heuristic_cell} & {fmt_delta(dim, 'heuristic')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_summary(summary, output_dir):
    dims = sorted(summary["dim"].unique())
    x = np.arange(len(dims))
    colors = {
        "random": "#4C78A8",
        "heuristic": "#F58518",
    }

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
    for col, opponent in enumerate(OPPONENTS):
        opponent_summary = summary[summary["opponent"] == opponent]
        for row_idx, metric in enumerate(["mean_predator_return", "mean_capture_step"]):
            ax = axes[row_idx, col]
            metric_summary = opponent_summary[opponent_summary["metric"] == metric].set_index("dim")
            means = [metric_summary.loc[dim, "mean"] for dim in dims]
            stds = [metric_summary.loc[dim, "std"] for dim in dims]
            ax.errorbar(
                x,
                means,
                yerr=stds,
                marker="o",
                markersize=5,
                linewidth=2,
                capsize=3,
                color=colors[opponent],
            )
            ax.grid(axis="y", alpha=0.25)
            if row_idx == 0:
                ax.set_title(OPPONENT_LABELS[opponent])
                ax.set_ylabel("Predator return")
            else:
                ax.set_ylabel("First-capture step")
                ax.set_xlabel("Communication dimension $d$")
                ax.set_xticks(x)
                ax.set_xticklabels([str(dim) for dim in dims])

    fig.tight_layout(pad=0.6)
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"F1_bandwidth_sweep_return_capture_step.{suffix}", dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else root / "base_reward_bandwidth_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(root, args.dims)
    if rows.empty:
        raise SystemExit(f"No fixed-opponent metrics found under {root}")
    expected = len(args.dims) * len(OPPONENTS) * 3
    if len(rows) != expected:
        print(f"warning: expected {expected} rows for 3 seeds, found {len(rows)}")

    summary = aggregate(rows)
    rows.to_csv(output_dir / "bandwidth_sweep_seed_metrics.csv", index=False)
    summary.to_csv(output_dir / "bandwidth_sweep_summary.csv", index=False)
    write_latex_table(summary, output_dir / "bandwidth_sweep_table.tex")
    plot_summary(summary, output_dir)
    print(f"saved bandwidth sweep outputs under {output_dir}")


if __name__ == "__main__":
    main()
