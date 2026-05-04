#!/usr/bin/env python
"""Plot cross-model role occupancy for simple_tag role analysis."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUNS = [
    (
        "No-comm",
        "base_reward_nocomm/run1/role_analysis",
        ["vanilla"],
    ),
    (
        "Comm-8",
        "base_reward_comm8/run1/role_analysis",
        ["vanilla", "zero"],
    ),
    (
        "Comm-32",
        "base_reward_comm32/run1/role_analysis",
        ["vanilla", "zero"],
    ),
]

ROLE_ORDER = ["Blocker", "Chaser", "Flanker"]
ROLE_COLORS = {
    "Blocker": "#4C78A8",
    "Chaser": "#F58518",
    "Flanker": "#54A24B",
}
INTERVENTION_LABELS = {
    "vanilla": "Vanilla",
    "zero": "Zero message",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mappo_dir",
        type=Path,
        default=Path("onpolicy/scripts/results/MPE/simple_tag/mappo"),
        help="Directory containing base_reward_* run folders.",
    )
    parser.add_argument(
        "--state",
        choices=["pre", "post"],
        default="post",
        help="Role-analysis state view to plot.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("onpolicy/scripts/results/MPE/simple_tag/mappo/base_reward_role_analysis"),
    )
    parser.add_argument(
        "--stem",
        default="role_occupancy_cross_model",
        help="Output file stem. Writes both .pdf and .png.",
    )
    return parser.parse_args()


def load_occupancy(mappo_dir, state):
    frames = []
    for model_label, relative_dir, interventions in DEFAULT_RUNS:
        path = (
            mappo_dir
            / relative_dir
            / f"role_analysis_outputs_{state}"
            / "role_occupancy.csv"
        )
        df = pd.read_csv(path)
        df = df[df["intervention"].isin(interventions)].copy()
        df["model"] = model_label
        df["condition"] = df["intervention"].map(INTERVENTION_LABELS)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def condition_order():
    return [
        ("No-comm", "Vanilla"),
        ("Comm-8", "Vanilla"),
        ("Comm-8", "Zero message"),
        ("Comm-32", "Vanilla"),
        ("Comm-32", "Zero message"),
    ]


def make_plot(df, output_dir, stem, state):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(9.8, 3.2))

    bar_width = 0.22
    within_gap = 0.04
    group_gap = 0.42
    x_positions = []
    x_labels = []
    group_centers = []
    group_labels = []
    x = 0.0

    for model, condition in condition_order():
        group_start = x
        for wolf_id in [0, 1, 2]:
            subset = df[
                (df["model"] == model)
                & (df["condition"] == condition)
                & (df["wolf_id"] == wolf_id)
            ]
            bottom = 0.0
            for role in ROLE_ORDER:
                value = float(subset.loc[subset["role_name"] == role, "probability"].iloc[0])
                ax.bar(
                    x,
                    value,
                    width=bar_width,
                    bottom=bottom,
                    color=ROLE_COLORS[role],
                    edgecolor="white",
                    linewidth=0.5,
                    label=role if not x_positions else None,
                )
                bottom += value
            x_positions.append(x)
            x_labels.append(f"W{wolf_id}")
            x += bar_width + within_gap
        group_centers.append((group_start + x - bar_width - within_gap) / 2.0)
        group_labels.append(f"{model}\n{condition}")
        x += group_gap

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Role occupancy")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for center, label in zip(group_centers, group_labels):
        ax.text(
            center,
            -0.18,
            label,
            ha="center",
            va="top",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
        )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles[: len(ROLE_ORDER)],
        labels[: len(ROLE_ORDER)],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(ROLE_ORDER),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.28, top=0.82)
    pdf_path = output_dir / f"{stem}_{state}.pdf"
    png_path = output_dir / f"{stem}_{state}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main():
    args = parse_args()
    df = load_occupancy(args.mappo_dir, args.state)
    make_plot(df, args.output_dir, args.stem, args.state)


if __name__ == "__main__":
    main()
