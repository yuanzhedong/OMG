"""Generate a tiny synthetic *materialized* OMG dataset + matching stats file.

This exists only to exercise the real training loop end-to-end before the
official OMG-Data / checkpoints are released. The numbers are random; do NOT
use any motion-quality output from a run on this data.

Layout written under --root:
    <root>/<split>/summary.json
    <root>/<split>/shard_00000.npz
    <root>/<split>/shard_00000.json

It also writes a normalization stats file (mean=0, std=1 over 125 dims) to
--stats so the 125D representation can load without the real dataset.

Shapes follow the 125D representation config (seq=60, num_prev_states=10,
feat_dim=125) and the materialized format in omg.data.materialized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Default set the MaterializedG1MotionDataset loads for training, plus the
# masks every sample carries.
TENSOR_KEYS = (
    "fps",
    "audio_features",
    "human_motion",
    "motion_features",
    "history_features",
    "canon_root_pos",
    "canon_root_quat",
    "has_text",
)
MASK_KEYS = ("valid", "has_audio", "has_human_motion")


def _write_split(split_root: Path, num_samples: int, shard_size: int, seq: int, hist: int, feat: int, rng: np.random.Generator) -> None:
    split_root.mkdir(parents=True, exist_ok=True)
    num_shards = (num_samples + shard_size - 1) // shard_size
    written = 0
    for shard_id in range(num_shards):
        n = min(shard_size, num_samples - written)
        arrays = {
            "fps": np.full((n,), 30.0, dtype=np.float32),
            "audio_features": rng.standard_normal((n, seq, 35)).astype(np.float32),
            "human_motion": rng.standard_normal((n, seq, 66)).astype(np.float32),
            "motion_features": rng.standard_normal((n, seq, feat)).astype(np.float32),
            "history_features": rng.standard_normal((n, hist, feat)).astype(np.float32),
            "canon_root_pos": rng.standard_normal((n, 1, 3)).astype(np.float32),
            "canon_root_quat": np.tile(np.asarray([1, 0, 0, 0], np.float32), (n, 1, 1)),
            "has_text": np.ones((n,), dtype=np.bool_),
            "mask__valid": np.ones((n, seq), dtype=np.bool_),
            "mask__has_audio": np.ones((n, seq), dtype=np.bool_),
            "mask__has_human_motion": np.ones((n, seq), dtype=np.bool_),
        }
        np.savez(split_root / f"shard_{shard_id:05d}.npz", **arrays)
        metadata = [
            {"caption": f"synthetic smoke clip {written + i}", "meta": {"dataset": "smoke", "index": written + i}}
            for i in range(n)
        ]
        (split_root / f"shard_{shard_id:05d}.json").write_text(json.dumps(metadata), encoding="utf-8")
        written += n
    summary = {"samples": int(num_samples), "shards": int(num_shards), "shard_size": int(shard_size)}
    (split_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    print(f"[smoke] wrote {split_root} samples={num_samples} shards={num_shards} shard_size={shard_size}")


def _write_stats(stats_path: Path, feat: int) -> None:
    stats = {
        "feature": "synthetic-smoke",
        "rotation_representation": "rot6d",
        "split": "train",
        "count": 1,
        "num_prev_states": 10,
        "canonical_frame_idx": 9,
        "sequence_length": 60,
        "mean": [0.0] * feat,
        "std": [1.0] * feat,
        "default_root_pos": [0.0, 0.0, 0.75],
        "default_root_quat": [1.0, 0.0, 0.0, 0.0],
        "default_joint_dof": [0.0] * 29,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats) + "\n", encoding="utf-8")
    print(f"[smoke] wrote stats {stats_path} (feat_dim={feat})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/_smoke/materialized/smoke_dataset"))
    parser.add_argument("--stats", type=Path, default=Path("assets/stats/g1_125d_stats.json"))
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--val-samples", type=int, default=4)
    parser.add_argument("--test-samples", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument("--seq", type=int, default=60)
    parser.add_argument("--hist", type=int, default=10)
    parser.add_argument("--feat", type=int, default=125)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-stats", action="store_true", help="skip writing the stats file")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    _write_split(args.root / "train", args.train_samples, args.shard_size, args.seq, args.hist, args.feat, rng)
    _write_split(args.root / "val", args.val_samples, args.shard_size, args.seq, args.hist, args.feat, rng)
    _write_split(args.root / "test", args.test_samples, args.shard_size, args.seq, args.hist, args.feat, rng)
    if not args.no_stats:
        _write_stats(args.stats, args.feat)


if __name__ == "__main__":
    main()
