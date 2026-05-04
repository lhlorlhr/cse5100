#!/usr/bin/env python
"""Compute entropy, token marginals, mutual information, and co-occurrence tables."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import mutual_info_score
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required for analyze_mpe_probe_info.py. "
        f"Import failed with: {exc!r}"
    )


CONCEPT_SPECS = {
    "quadrant": 4,
    "is_self_closest": 2,
    "distance_bin": 3,
}


def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args(argv)


def compute_entropy(probs):
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def main(argv):
    args = parse_args(argv)
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = dataset_path.parent.parent / "info"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(dataset_path, allow_pickle=False)
    msg_trained = np.asarray(data["msg_trained"], dtype=np.float32)
    tokens = np.argmax(msg_trained, axis=-1)
    n_wolves = tokens.shape[1]
    comm_dim = msg_trained.shape[-1]

    entropy_rows = []
    marginal_payload = []
    for wolf_idx in range(n_wolves):
        counts = np.bincount(tokens[:, wolf_idx], minlength=comm_dim).astype(np.float64)
        probs = counts / max(counts.sum(), 1.0)
        entropy_bits = compute_entropy(probs)
        entropy_rows.append(
            {
                "wolf_id": wolf_idx,
                "entropy_bits": entropy_bits,
                "max_entropy_bits": float(np.log2(comm_dim)),
            }
        )
        marginal_payload.append(
            {
                "wolf_id": wolf_idx,
                "counts": counts.astype(int).tolist(),
                "probs": [float(x) for x in probs],
            }
        )

    pd.DataFrame(entropy_rows).to_csv(output_dir / "token_entropy.csv", index=False)
    (output_dir / "token_marginals.json").write_text(json.dumps(marginal_payload, indent=2))

    mi_rows = []
    for wolf_idx in range(n_wolves):
        for concept, num_classes in CONCEPT_SPECS.items():
            labels = np.asarray(data[concept][:, wolf_idx], dtype=np.int64)
            mi_nat = mutual_info_score(tokens[:, wolf_idx], labels)
            mi_rows.append(
                {
                    "wolf_id": wolf_idx,
                    "concept": concept,
                    "mi_nats": float(mi_nat),
                    "mi_bits": float(mi_nat / np.log(2.0)),
                }
            )

            counts = np.zeros((comm_dim, num_classes), dtype=np.float64)
            for token_id, label in zip(tokens[:, wolf_idx], labels):
                counts[int(token_id), int(label)] += 1.0
            row_totals = counts.sum(axis=1, keepdims=True)
            conditional = np.divide(
                counts,
                np.maximum(row_totals, 1.0),
                out=np.zeros_like(counts),
                where=row_totals > 0.0,
            )
            df = pd.DataFrame(
                conditional,
                columns=[f"class_{idx}" for idx in range(num_classes)],
            )
            df.insert(0, "token_id", np.arange(comm_dim))
            df.to_csv(output_dir / f"cooccurrence_{concept}_wolf{wolf_idx}.csv", index=False)

    pd.DataFrame(mi_rows).to_csv(output_dir / "mutual_information.csv", index=False)
    print(f"saved information analysis under {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
