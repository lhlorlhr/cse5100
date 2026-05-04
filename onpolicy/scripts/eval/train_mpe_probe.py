#!/usr/bin/env python
"""Train simple linear probes on collected MPE communication datasets."""

import json
import sys
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required for train_mpe_probe.py. "
        f"Import failed with: {exc!r}"
    )


TASK_SPECS = {
    "sheep_rel_xy": {"kind": "regression", "joint": False},
    "quadrant": {"kind": "multiclass", "joint": False},
    "is_self_closest": {"kind": "binary", "joint": False},
    "closest_wolf_id": {"kind": "multiclass", "joint": True},
}


def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--probe_type", type=str, default="linear", choices=["linear"])
    parser.add_argument(
        "--message_variant",
        type=str,
        required=True,
        choices=["trained", "random", "marginal"],
    )
    parser.add_argument(
        "--random_policy_seed",
        type=int,
        default=None,
        help="Required when --message_variant random.",
    )
    parser.add_argument(
        "--wolf_id",
        type=int,
        default=None,
        help="Required for per-wolf tasks and unused for joint tasks.",
    )
    parser.add_argument("--split_seed", type=int, default=1)
    parser.add_argument("--marginal_seed", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args(argv)


def resolve_output_dir(args, variant_label):
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if args.output_dir:
        root = Path(args.output_dir).expanduser().resolve()
    else:
        root = dataset_path.parent.parent
    wolf_part = "joint"
    if args.wolf_id is not None:
        wolf_part = f"wolf{args.wolf_id}"
    output_dir = root / "probes" / args.probe_type / args.task / wolf_part / variant_label
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_variant_label(args):
    if args.message_variant == "random":
        if args.random_policy_seed is None:
            raise ValueError("--random_policy_seed is required when --message_variant random.")
        return f"random_seed{args.random_policy_seed}"
    return args.message_variant


def load_variant_matrix(data, args):
    if args.message_variant == "trained":
        return np.asarray(data["msg_trained"], dtype=np.float32)

    if args.message_variant == "random":
        key = f"msg_random_seed{args.random_policy_seed}"
        if key not in data:
            raise KeyError(f"{key} not found in dataset.")
        return np.asarray(data[key], dtype=np.float32)

    trained = np.asarray(data["msg_trained"], dtype=np.float32)
    trained_tokens = np.argmax(trained, axis=-1)
    rng = np.random.default_rng(args.marginal_seed)
    marginal = np.zeros_like(trained)
    comm_dim = trained.shape[-1]
    for wolf_idx in range(trained.shape[1]):
        counts = np.bincount(trained_tokens[:, wolf_idx], minlength=comm_dim).astype(np.float64)
        probs = counts / max(counts.sum(), 1.0)
        sampled = rng.choice(comm_dim, size=trained.shape[0], p=probs)
        marginal[np.arange(trained.shape[0]), wolf_idx, sampled] = 1.0
    return marginal


def split_episodes(episode_ids, split_seed, train_ratio, val_ratio):
    unique_episodes = np.unique(episode_ids)
    rng = np.random.default_rng(split_seed)
    shuffled = unique_episodes.copy()
    rng.shuffle(shuffled)

    n_episodes = len(shuffled)
    n_train = int(round(n_episodes * train_ratio))
    n_val = int(round(n_episodes * val_ratio))
    n_train = min(max(n_train, 1), max(n_episodes - 2, 1))
    n_val = min(max(n_val, 1), max(n_episodes - n_train - 1, 1))
    n_test = n_episodes - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train = max(n_train - 1, 1)

    train_eps = shuffled[:n_train]
    val_eps = shuffled[n_train:n_train + n_val]
    test_eps = shuffled[n_train + n_val:]
    return train_eps, val_eps, test_eps


def select_rows(X, y, episode_ids, allowed_episodes):
    mask = np.isin(episode_ids, allowed_episodes)
    return X[mask], y[mask]


def build_xy(data, args):
    variant_matrix = load_variant_matrix(data, args)
    spec = TASK_SPECS[args.task]

    if spec["joint"]:
        X = variant_matrix.reshape(variant_matrix.shape[0], -1)
        y = np.asarray(data[args.task], dtype=np.int64)
        return X, y

    if args.wolf_id is None:
        raise ValueError(f"--wolf_id is required for task={args.task}.")

    X = variant_matrix[:, args.wolf_id, :]
    if args.task == "sheep_rel_xy":
        y = np.asarray(data["sheep_rel_xy"][:, args.wolf_id, :], dtype=np.float32)
    else:
        y = np.asarray(data[args.task][:, args.wolf_id], dtype=np.int64)
    return X, y


def fit_and_eval(task, X_train, y_train, X_test, y_test):
    spec = TASK_SPECS[task]
    if spec["kind"] == "regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "mse": float(mean_squared_error(y_test, y_pred)),
        }
        return model, y_pred, metrics

    classifier = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        multi_class="auto",
    )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }

    if spec["kind"] == "binary":
        y_prob = classifier.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) > 1:
            metrics["auroc"] = float(roc_auc_score(y_test, y_prob))
        else:
            metrics["auroc"] = float("nan")
    return classifier, y_pred, metrics


def main(argv):
    args = parse_args(argv)
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    data = np.load(dataset_path, allow_pickle=False)
    variant_label = build_variant_label(args)
    output_dir = resolve_output_dir(args, variant_label)

    X, y = build_xy(data, args)
    episode_ids = np.asarray(data["episode_id"], dtype=np.int64)
    train_eps, val_eps, test_eps = split_episodes(
        episode_ids,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    X_train, y_train = select_rows(X, y, episode_ids, train_eps)
    X_val, y_val = select_rows(X, y, episode_ids, val_eps)
    X_test, y_test = select_rows(X, y, episode_ids, test_eps)

    model, y_pred, metrics = fit_and_eval(args.task, X_train, y_train, X_test, y_test)

    metrics_payload = {
        "dataset_path": str(dataset_path),
        "task": args.task,
        "probe_type": args.probe_type,
        "message_variant": args.message_variant,
        "variant_label": variant_label,
        "wolf_id": args.wolf_id,
        "split_seed": int(args.split_seed),
        "marginal_seed": int(args.marginal_seed),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "n_train_samples": int(X_train.shape[0]),
        "n_val_samples": int(X_val.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_train_episodes": int(len(train_eps)),
        "n_val_episodes": int(len(val_eps)),
        "n_test_episodes": int(len(test_eps)),
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    np.savez_compressed(
        output_dir / "predictions.npz",
        y_test=y_test,
        y_pred=y_pred,
        test_episode_ids=test_eps,
    )

    config_payload = {
        "task": args.task,
        "probe_type": args.probe_type,
        "message_variant": args.message_variant,
        "variant_label": variant_label,
        "random_policy_seed": args.random_policy_seed,
        "wolf_id": args.wolf_id,
        "split_seed": args.split_seed,
        "marginal_seed": args.marginal_seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "feature_dim": int(X.shape[-1]),
    }
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
