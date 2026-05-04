#!/usr/bin/env python
"""Aggregate and plot message cutoff-intervention results for MPE simple_tag."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plot_mpe_cutoff_intervention.py. "
        f"Import failed with: {exc!r}"
    )


DEFAULT_DIMS = [2, 4, 8, 16, 32, 64]
OPPONENTS = ["random", "heuristic"]
INTERVENTIONS = ["zero", "random", "permuted", "noise"]
METRICS = ["mean_predator_return", "mean_capture_step", "mean_collision_steps"]

OPPONENT_LABELS = {
    "random": "Random prey",
    "heuristic": "Heuristic prey",
}
INTERVENTION_LABELS = {
    "zero": "Zero",
    "random": "Random replay",
    "permuted": "Permuted",
    "noise": "Noise",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("onpolicy/scripts/results/MPE/simple_tag/mappo"),
        help="Directory containing base_reward_comm* run folders.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=DEFAULT_DIMS,
        help="Communication dimensions to include.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to ROOT/base_reward_cutoff_intervention.",
    )
    return parser.parse_args()


def read_delta_rows(root, dims):
    rows = []
    for dim in dims:
        exp_dir = root / f"base_reward_comm{dim}"
        for summary_path in sorted(exp_dir.glob("run*/cutoff_intervention_eval/cutoff_intervention_summary.json")):
            run_match = re.search(r"run(\d+)", str(summary_path))
            if run_match is None:
                continue
            with summary_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            seed = int(payload.get("seed", run_match.group(1)))
            for opponent in OPPONENTS:
                vanilla = payload["results"][opponent]["vanilla"]["summary"]
                for intervention in INTERVENTIONS:
                    summary = payload["results"][opponent][intervention]["summary"]
                    row = {
                        "dim": dim,
                        "seed": seed,
                        "run": int(run_match.group(1)),
                        "opponent": opponent,
                        "intervention": intervention,
                    }
                    for metric in METRICS:
                        row[f"delta_{metric}"] = float(summary[metric] - vanilla[metric])
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize(rows, group_cols):
    metric_cols = [f"delta_{metric}" for metric in METRICS]
    parts = []
    for metric_col in metric_cols:
        stats = rows.groupby(group_cols, sort=True)[metric_col].agg(["mean", "std", "count"]).reset_index()
        stats["std"] = stats["std"].fillna(0.0)
        stats["metric"] = metric_col.replace("delta_", "")
        stats = stats.rename(columns={"count": "n"})
        parts.append(stats)
    return pd.concat(parts, ignore_index=True)[group_cols + ["metric", "mean", "std", "n"]]


def write_latex_table(overall, output_path):
    table = overall.pivot_table(
        index=["opponent", "intervention"],
        columns="metric",
        values=["mean", "std"],
        aggfunc="first",
    )

    def return_cell(opponent, intervention):
        mean = table.loc[(opponent, intervention), ("mean", "mean_predator_return")]
        std = table.loc[(opponent, intervention), ("std", "mean_predator_return")]
        return rf"\({mean:+.1f} \pm {std:.1f}\)"

    def mean_cell(opponent, intervention, metric):
        mean = table.loc[(opponent, intervention), ("mean", metric)]
        return rf"\({mean:+.2f}\)"

    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Opponent & Intervention & \(\Delta\) return & \(\Delta\) capture step & \(\Delta\) collision \\",
        r"\midrule",
    ]
    for opponent_idx, opponent in enumerate(OPPONENTS):
        if opponent_idx > 0:
            lines.append(r"\midrule")
        for intervention in INTERVENTIONS:
            lines.append(
                f"{opponent.capitalize()} & {INTERVENTION_LABELS[intervention]} & "
                f"{return_cell(opponent, intervention)} & "
                f"{mean_cell(opponent, intervention, 'mean_capture_step')} & "
                f"{mean_cell(opponent, intervention, 'mean_collision_steps')} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_heatmap(by_dim, output_dir):
    metric = "mean_predator_return"
    heatmap_rows = []
    for opponent in OPPONENTS:
        sub = by_dim[(by_dim["opponent"] == opponent) & (by_dim["metric"] == metric)]
        pivot = sub.pivot(index="dim", columns="intervention", values="mean").loc[:, INTERVENTIONS]
        heatmap_rows.append(pivot)

    max_abs = max(float(np.nanmax(np.abs(pivot.to_numpy()))) for pivot in heatmap_rows)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig = plt.figure(figsize=(8.0, 3.3))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.045], wspace=0.12)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    cax = fig.add_subplot(grid[0, 2])
    image = None
    for ax, opponent, pivot in zip(axes, OPPONENTS, heatmap_rows):
        image = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_title(OPPONENT_LABELS[opponent])
        ax.set_xticks(np.arange(len(INTERVENTIONS)))
        ax.set_xticklabels([INTERVENTION_LABELS[name] for name in INTERVENTIONS], rotation=30, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(dim) for dim in pivot.index])
        ax.set_xlabel("Intervention")
        if ax is axes[0]:
            ax.set_ylabel("Communication dimension $d$")

        for row_idx in range(pivot.shape[0]):
            for col_idx in range(pivot.shape[1]):
                value = float(pivot.iloc[row_idx, col_idx])
                text_color = "white" if abs(value) > 0.55 * max_abs else "black"
                ax.text(col_idx, row_idx, f"{value:+.0f}", ha="center", va="center", color=text_color, fontsize=8)

    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(r"$\Delta$ predator return")
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.28, top=0.88)
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"F_intervention_return_heatmap.{suffix}", dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else root / "base_reward_cutoff_intervention"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_delta_rows(root, args.dims)
    if rows.empty:
        raise SystemExit(f"No cutoff-intervention summaries found under {root}")
    expected = len(args.dims) * len(OPPONENTS) * len(INTERVENTIONS) * 3
    if len(rows) != expected:
        print(f"warning: expected {expected} rows for 3 seeds, found {len(rows)}")

    overall = summarize(rows, ["opponent", "intervention"])
    by_dim = summarize(rows, ["dim", "opponent", "intervention"])

    rows.to_csv(output_dir / "cutoff_intervention_checkpoint_deltas.csv", index=False)
    overall.to_csv(output_dir / "cutoff_intervention_overall_summary.csv", index=False)
    by_dim.to_csv(output_dir / "cutoff_intervention_by_dim_summary.csv", index=False)
    write_latex_table(overall, output_dir / "cutoff_intervention_table.tex")
    plot_heatmap(by_dim, output_dir)
    print(f"saved cutoff-intervention outputs under {output_dir}")


if __name__ == "__main__":
    main()
