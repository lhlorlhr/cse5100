#!/usr/bin/env python
"""Plot summary figures for MPE communication probing."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plot_mpe_probe_figures.py. "
        f"Import failed with: {exc!r}"
    )


HEATMAP_TASK_TO_LABEL_KEY = {
    "quadrant": "quadrant",
    "is_self_closest": "is_self_closest",
}

F2A_REPORT_SPECS = [
    {
        "task": "sheep_rel_xy",
        "metric": "r2",
        "label": "Sheep rel. position\n$R^2$",
    },
    {
        "task": "quadrant",
        "metric": "balanced_accuracy",
        "label": "Sheep quadrant\nbalanced acc.",
    },
    {
        "task": "is_self_closest",
        "metric": "auroc",
        "label": "Self is closest\nAUROC",
    },
    {
        "task": "closest_wolf_id",
        "metric": "balanced_accuracy",
        "label": "Closest wolf ID\nbalanced acc.",
    },
]

VARIANT_STYLE = {
    "trained": {"label": "trained", "color": "#4C78A8"},
    "random-init": {"label": "random-init", "color": "#F58518"},
    "marginal": {"label": "marginal", "color": "#54A24B"},
}


def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_summary", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--info_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args(argv)


def coerce_wolf_id(value):
    if pd.isna(value):
        return None
    return int(value)


def aggregate_variant(values_df, variant_name, per_seed_random=False):
    if values_df.empty:
        return None

    if per_seed_random:
        seed_means = values_df.groupby("variant_label")["mean"].mean()
        if seed_means.empty:
            return None
        std = float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0
        return float(seed_means.mean()), std, int(len(seed_means))

    values = values_df["mean"].dropna()
    if values.empty:
        return None
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean()), std, int(len(values))


def build_f2a_report(summary):
    rows = []
    for spec in F2A_REPORT_SPECS:
        task_rows = summary[
            (summary["task"] == spec["task"])
            & (summary["metric"] == spec["metric"])
        ].copy()
        for variant in ["trained", "random-init", "marginal"]:
            if variant == "random-init":
                variant_rows = task_rows[
                    task_rows["variant_label"].astype(str).str.startswith("random_seed")
                ]
                aggregate = aggregate_variant(variant_rows, variant, per_seed_random=True)
            else:
                variant_rows = task_rows[task_rows["variant_label"] == variant]
                aggregate = aggregate_variant(variant_rows, variant)

            if aggregate is None:
                continue
            mean, std, n = aggregate
            rows.append(
                {
                    "task": spec["task"],
                    "metric": spec["metric"],
                    "target": spec["task"],
                    "plot_label": spec["label"],
                    "variant": variant,
                    "mean": mean,
                    "std": std,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def main(argv):
    args = parse_args(argv)
    probe_summary_path = Path(args.probe_summary).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    info_dir = Path(args.info_dir).expanduser().resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = probe_summary_path.parent.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(probe_summary_path)
    dataset = np.load(dataset_path, allow_pickle=False)
    entropy_df = pd.read_csv(info_dir / "token_entropy.csv")

    f2a_report = build_f2a_report(summary)
    if f2a_report.empty:
        raise SystemExit("No F2a report rows found.")
    f2a_report.to_csv(output_dir / "F2a_probe_scores_summary.csv", index=False)

    variant_order = ["trained", "random-init", "marginal"]
    plot_labels = [spec["label"] for spec in F2A_REPORT_SPECS]
    x = np.arange(len(plot_labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for idx, variant in enumerate(variant_order):
        sub = f2a_report[f2a_report["variant"] == variant].set_index("plot_label")
        means = [sub.loc[label, "mean"] if label in sub.index else np.nan for label in plot_labels]
        stds = [sub.loc[label, "std"] if label in sub.index else 0.0 for label in plot_labels]
        ax.bar(
            x + (idx - 1) * width,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=VARIANT_STYLE[variant]["label"],
            color=VARIANT_STYLE[variant]["color"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels)
    ax.set_ylabel("Probe score")
    min_score = float(np.nanmin(f2a_report["mean"] - f2a_report["std"]))
    lower_ylim = min(-0.03, min_score - 0.01)
    ax.set_ylim(lower_ylim, 0.75)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("F2a: Linear probe scores by message variant")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "F2a_probe_accuracy.png", dpi=200)
    plt.close(fig)

    heatmap_candidates = []
    pivot_rows = summary.copy()
    wolf_compare = pivot_rows["wolf_id"].fillna(-1).astype(int)
    for _, row in pivot_rows.iterrows():
        if row["variant_label"] != "trained":
            continue
        if row["task"] not in HEATMAP_TASK_TO_LABEL_KEY:
            continue
        if row["metric"] not in {"balanced_accuracy", "auroc"}:
            continue
        row_wolf = -1 if pd.isna(row["wolf_id"]) else int(row["wolf_id"])
        selector = (
            (pivot_rows["task"] == row["task"])
            & (pivot_rows["metric"] == row["metric"])
            & (wolf_compare == row_wolf)
        )
        marginal_rows = pivot_rows[selector & (pivot_rows["variant_label"] == "marginal")]
        if marginal_rows.empty:
            continue
        signal = float(row["mean"] - marginal_rows.iloc[0]["mean"])
        heatmap_candidates.append((signal, row["task"], coerce_wolf_id(row["wolf_id"])))

    if heatmap_candidates:
        heatmap_candidates.sort(reverse=True)
        _, best_task, best_wolf = heatmap_candidates[0]
        tokens = np.argmax(dataset["msg_trained"], axis=-1)
        labels = np.asarray(dataset[HEATMAP_TASK_TO_LABEL_KEY[best_task]][:, best_wolf], dtype=np.int64)
        comm_dim = dataset["msg_trained"].shape[-1]
        num_classes = int(labels.max()) + 1
        mat = np.zeros((comm_dim, num_classes), dtype=np.float64)
        for token_id, label in zip(tokens[:, best_wolf], labels):
            mat[int(token_id), int(label)] += 1.0
        row_totals = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, np.maximum(row_totals, 1.0))

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(mat, cmap="hot", aspect="auto")
        ax.set_xlabel(best_task)
        ax.set_ylabel("Token ID")
        ax.set_title(f"F2b: P(class|token) for {best_task}, wolf{best_wolf}")
        ax.set_yticks(np.arange(comm_dim))
        ax.set_xticks(np.arange(num_classes))
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "F2b_heatmap.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(entropy_df["wolf_id"].astype(str), entropy_df["entropy_bits"])
    ax.axhline(entropy_df["max_entropy_bits"].iloc[0], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Wolf ID")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("F2c: Token entropy by wolf")
    fig.tight_layout()
    fig.savefig(output_dir / "F2c_token_entropy.png", dpi=200)
    plt.close(fig)

    print(f"saved figures under {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
