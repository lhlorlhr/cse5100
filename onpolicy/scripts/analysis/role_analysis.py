#!/usr/bin/env python
"""Offline  emergence analysis for simple_tag trajectories."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "distance_rank",
    "distance",
    "front_back",
    "lateral_signed",
    "lateral_abs",
]


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_path", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Defaults to <trajectory_dir>/_analysis_outputs.",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="post",
        choices=["pre", "post"],
        help="Which stored state to use for  descriptors.",
    )
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--tsne_sample", type=int, default=5000)
    parser.add_argument("--skip_tsne", action="store_true")
    return parser.parse_args(args)


def load_payload(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "records" in payload:
        return payload
    return {"metadata": {}, "records": payload}


def normalize(vector, eps=1e-8):
    norm = np.linalg.norm(vector)
    if norm < eps:
        return np.zeros_like(vector)
    return vector / norm


def cross2d(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def message_to_token(message):
    message = np.asarray(message, dtype=np.float32).reshape(-1)
    if message.size == 0 or np.allclose(message, 0.0):
        return -1
    return int(np.argmax(message))


def compute_samples(record, state):
    wolf_positions = np.asarray(record[f"{state}_wolf_positions"], dtype=np.float32)
    sheep_positions = np.asarray(record[f"{state}_sheep_positions"], dtype=np.float32)
    if sheep_positions.ndim == 2:
        sheep_position = sheep_positions[0]
    else:
        sheep_position = sheep_positions

    centroid = wolf_positions.mean(axis=0)
    forward = normalize(sheep_position - centroid)
    distances = np.linalg.norm(wolf_positions - sheep_position[None, :], axis=1)
    ranks = distances.argsort().argsort() + 1

    policy_messages = np.asarray(record["policy_messages"], dtype=np.float32)
    applied_messages = np.asarray(record["applied_messages"], dtype=np.float32)

    samples = []
    for wolf_id in range(wolf_positions.shape[0]):
        rel = wolf_positions[wolf_id] - sheep_position
        front_back = float(np.dot(rel, forward))
        lateral_signed = cross2d(forward, rel)
        lateral_abs = abs(lateral_signed)

        sample = {
            "episode": int(record["episode"]),
            "episode_seed": int(record["episode_seed"]),
            "step": int(record["step"]),
            "opponent_policy": record["opponent_policy"],
            "intervention": record["intervention"],
            "wolf_id": int(wolf_id),
            "distance_rank": float(ranks[wolf_id]),
            "distance": float(distances[wolf_id]),
            "front_back": front_back,
            "lateral_signed": lateral_signed,
            "lateral_abs": float(lateral_abs),
            "policy_message_token": message_to_token(policy_messages[wolf_id]),
            "applied_message_token": message_to_token(applied_messages[wolf_id]),
            "policy_message": policy_messages[wolf_id],
            "applied_message": applied_messages[wolf_id],
        }
        samples.append(sample)
    return samples


def records_to_dataframe(records, state):
    samples = []
    for record in records:
        samples.extend(compute_samples(record, state))
    if not samples:
        raise ValueError("No samples found in trajectory payload.")
    return pd.DataFrame(samples)


def normalized_inertia(x_scaled, labels, centers):
    labels = np.asarray(labels, dtype=np.int64)
    within = np.sum((x_scaled - centers[labels]) ** 2)
    total = np.sum((x_scaled - x_scaled.mean(axis=0, keepdims=True)) ** 2)
    if total <= 1e-12:
        return 0.0
    return float(within / total)


def safe_cluster_metrics(x_scaled, labels, centers):
    unique_labels = np.unique(labels)
    result = {
        "n_samples": int(x_scaled.shape[0]),
        "n_clusters_present": int(unique_labels.size),
        "normalized_inertia": normalized_inertia(x_scaled, labels, centers),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    if x_scaled.shape[0] <= unique_labels.size or unique_labels.size < 2:
        return result
    result["silhouette"] = float(silhouette_score(x_scaled, labels))
    result["davies_bouldin"] = float(davies_bouldin_score(x_scaled, labels))
    result["calinski_harabasz"] = float(calinski_harabasz_score(x_scaled, labels))
    return result


def infer__names(cluster_summary):
    vanilla_summary = cluster_summary.copy()
    chaser_cluster = int(
        vanilla_summary.sort_values(["distance_rank", "distance"]).iloc[0]["_cluster"]
    )

    remaining = vanilla_summary[vanilla_summary["_cluster"] != chaser_cluster]
    _map = {chaser_cluster: "Chaser"}
    if not remaining.empty:
        flanker_cluster = int(remaining.sort_values("lateral_abs", ascending=False).iloc[0]["_cluster"])
        _map[flanker_cluster] = "Flanker"
        for cluster_id in vanilla_summary["_cluster"]:
            cluster_id = int(cluster_id)
            if cluster_id not in _map:
                _map[cluster_id] = "Blocker"
    return _map


def compute_transition_table(df):
    rows = []
    group_columns = ["opponent_policy", "intervention", "wolf_id", "episode"]
    for group_key, group in df.sort_values("step").groupby(group_columns):
        labels = group["_name"].to_numpy()
        for prev_, next_ in zip(labels[:-1], labels[1:]):
            rows.append(
                {
                    "opponent_policy": group_key[0],
                    "intervention": group_key[1],
                    "wolf_id": int(group_key[2]),
                    "from_": prev_,
                    "to_": next_,
                    "count": 1,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["opponent_policy", "intervention", "wolf_id", "from_", "to_", "count", "probability"])

    transitions = pd.DataFrame(rows)
    transitions = (
        transitions.groupby(["opponent_policy", "intervention", "wolf_id", "from_", "to_"], as_index=False)
        ["count"]
        .sum()
    )
    totals = transitions.groupby(["opponent_policy", "intervention", "wolf_id", "from_"])["count"].transform("sum")
    transitions["probability"] = transitions["count"] / totals
    return transitions


def compute_message__stats(df):
    vanilla = df[df["intervention"] == "vanilla"].copy()
    if vanilla.empty:
        return {}, pd.DataFrame()

    has_policy_messages = bool((vanilla["policy_message_token"] >= 0).any())
    if not has_policy_messages:
        return {
            "has_policy_messages": False,
            "mutual_info_policy_message_": None,
            "normalized_mutual_info_policy_message_": None,
            "n_vanilla_samples": int(len(vanilla)),
            "n_policy_message_tokens": 0,
            "n_s": int(vanilla["_name"].nunique()),
        }, pd.DataFrame()

    stats = {
        "has_policy_messages": True,
        "mutual_info_policy_message_": float(
            mutual_info_score(vanilla["policy_message_token"], vanilla["_name"])
        ),
        "normalized_mutual_info_policy_message_": float(
            normalized_mutual_info_score(vanilla["policy_message_token"], vanilla["_name"])
        ),
        "n_vanilla_samples": int(len(vanilla)),
        "n_policy_message_tokens": int(vanilla["policy_message_token"].nunique()),
        "n_s": int(vanilla["_name"].nunique()),
    }
    contingency = pd.crosstab(
        vanilla["policy_message_token"],
        vanilla["_name"],
        normalize="index",
    )
    return stats, contingency


def make_plots(df, contingency, output_dir, random_seed, tsne_sample, skip_tsne):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping plots.")
        return

    for intervention, group in df.groupby("intervention"):
        fig, ax = plt.subplots(figsize=(6, 5))
        for _name, _group in group.groupby("_name"):
            ax.scatter(
                _group["front_back"],
                _group["lateral_signed"],
                s=5,
                alpha=0.35,
                label=_name,
            )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("front_back")
        ax.set_ylabel("lateral_signed")
        ax.set_title(f" descriptor scatter: {intervention}")
        ax.legend(markerscale=3)
        fig.tight_layout()
        fig.savefig(output_dir / f"scatter_{intervention}.png", dpi=180)
        plt.close(fig)

    occupancy = (
        df.groupby(["intervention", "wolf_id", "_name"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = occupancy.groupby(["intervention", "wolf_id"])["count"].transform("sum")
    occupancy["probability"] = occupancy["count"] / totals
    pivot = occupancy.pivot_table(
        index=["intervention", "wolf_id"],
        columns="_name",
        values="probability",
        fill_value=0.0,
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel(" occupancy")
    ax.set_title(" occupancy by wolf")
    fig.tight_layout()
    fig.savefig(output_dir / "_occupancy.png", dpi=180)
    plt.close(fig)

    if not contingency.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        image = ax.imshow(contingency.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(contingency.columns)))
        ax.set_xticklabels(contingency.columns, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(contingency.index)))
        ax.set_yticklabels(contingency.index)
        ax.set_xlabel("")
        ax.set_ylabel("policy message token")
        ax.set_title("P( | message token), vanilla")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "message__contingency.png", dpi=180)
        plt.close(fig)

    if skip_tsne:
        return

    vanilla = df[df["intervention"] == "vanilla"].copy()
    if vanilla.empty:
        return
    messages = np.stack(vanilla["policy_message"].to_numpy())
    if len(messages) < 3 or messages.shape[1] == 0:
        return

    rng = np.random.default_rng(random_seed)
    sample_size = min(int(tsne_sample), len(messages))
    sample_idx = rng.choice(len(messages), size=sample_size, replace=False)
    sampled_messages = messages[sample_idx]
    sampled_s = vanilla.iloc[sample_idx]["_name"].to_numpy()
    perplexity = max(2, min(30, sample_size // 3))
    if sample_size <= perplexity:
        return

    embedding = TSNE(
        n_components=2,
        random_state=random_seed,
        init="random",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(sampled_messages)

    fig, ax = plt.subplots(figsize=(6, 5))
    for _name in np.unique(sampled_s):
        mask = sampled_s == _name
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.5, label=_name)
    ax.set_title("t-SNE of vanilla policy messages")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(output_dir / "message_tsne_vanilla.png", dpi=180)
    plt.close(fig)


def main(args):
    all_args = parse_args(args)
    trajectory_path = Path(all_args.trajectory_path).expanduser().resolve()
    output_dir = (
        Path(all_args.output_dir).expanduser().resolve()
        if all_args.output_dir
        else trajectory_path.parent / "_analysis_outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_payload(trajectory_path)
    df = records_to_dataframe(payload["records"], all_args.state)

    vanilla_mask = df["intervention"] == "vanilla"
    if not vanilla_mask.any():
        raise ValueError("The trajectory file must include vanilla records to define the  space.")

    x_all = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    x_vanilla = df.loc[vanilla_mask, FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_vanilla_scaled = scaler.fit_transform(x_vanilla)
    x_all_scaled = scaler.transform(x_all)

    kmeans = KMeans(
        n_clusters=all_args.n_clusters,
        random_state=all_args.random_seed,
        n_init=50,
    )
    kmeans.fit(x_vanilla_scaled)
    df["_cluster"] = kmeans.predict(x_all_scaled).astype(int)

    vanilla_cluster_summary = (
        df.loc[vanilla_mask]
        .groupby("_cluster")[FEATURE_COLUMNS]
        .mean()
        .reset_index()
    )
    _map = infer__names(vanilla_cluster_summary)
    df["_name"] = df["_cluster"].map(_map)

    metrics = {}
    for (opponent_policy, intervention), group in df.groupby(["opponent_policy", "intervention"]):
        group_idx = group.index.to_numpy()
        labels = df.loc[group_idx, "_cluster"].to_numpy(dtype=np.int64)
        metrics[f"{opponent_policy}/{intervention}"] = safe_cluster_metrics(
            x_all_scaled[group_idx],
            labels,
            kmeans.cluster_centers_,
        )

    cluster_summary = (
        df.groupby(["intervention", "_cluster", "_name"])[FEATURE_COLUMNS]
        .mean()
        .reset_index()
    )
    occupancy = (
        df.groupby(["opponent_policy", "intervention", "wolf_id", "_name"])
        .size()
        .rename("count")
        .reset_index()
    )
    occupancy["probability"] = occupancy["count"] / occupancy.groupby(
        ["opponent_policy", "intervention", "wolf_id"]
    )["count"].transform("sum")

    transitions = compute_transition_table(df)
    message__stats, contingency = compute_message__stats(df)

    df_to_save = df.drop(columns=["policy_message", "applied_message"])
    df_to_save.to_csv(output_dir / "_samples.csv", index=False)
    cluster_summary.to_csv(output_dir / "cluster_summary.csv", index=False)
    occupancy.to_csv(output_dir / "_occupancy.csv", index=False)
    transitions.to_csv(output_dir / "_transitions.csv", index=False)
    contingency.to_csv(output_dir / "message__contingency.csv")

    report = {
        "trajectory_path": str(trajectory_path),
        "state": all_args.state,
        "n_samples": int(len(df)),
        "features": list(FEATURE_COLUMNS),
        "_map": {str(k): v for k, v in _map.items()},
        "metrics": metrics,
        "message__stats": message__stats,
        "metadata": payload.get("metadata", {}),
    }
    with open(output_dir / "_analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    make_plots(
        df,
        contingency,
        output_dir,
        all_args.random_seed,
        all_args.tsne_sample,
        all_args.skip_tsne,
    )

    print(f"Saved  analysis outputs to: {output_dir}")
    print(json.dumps({"metrics": metrics, "message__stats": message__stats}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
