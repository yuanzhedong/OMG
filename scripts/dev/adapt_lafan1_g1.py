"""Adapt the Unitree LAFAN1->G1 retargeting dataset into OMG source format.

Source: lvhaidong/LAFAN1_Retargeting_Dataset  (g1/*.csv, 30 FPS)
  Each CSV row = root_joint(X Y Z QX QY QZ QW) + 29 G1 joints, in the SAME
  joint order as omg.robots.g1.constants.G1_JOINT_NAMES.

OMG qpos_36 = root_pos(3) + root_quat(W X Y Z)(4) + joint_dof(29).
So the only transform is reordering the root quaternion xyzw -> wxyz; the
29 joints map 1:1.

Output layout (OMG UnifiedG1MotionIndex):
  <out>/g1/<clip>.npz          key "qpos" (T,36) float32, key "fps" 30.0
  <out>/labels/<clip>.json     {"caption": "<text>"}
  <out>/info.yaml              {train: [...], val: [...], test: [...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from omg.robots.g1.kinematics import G1Kinematics

# Map LAFAN1 action category (filename prefix before _subject) to a caption.
CATEGORY_CAPTIONS = {
    "walk": "a person walks",
    "run": "a person runs",
    "sprint": "a person sprints",
    "dance": "a person is dancing",
    "jumps": "a person jumps",
    "fight": "a person fights",
    "fightAndSports": "a person does martial arts and sports moves",
    "fallAndGetUp": "a person falls down and gets back up",
}


def _category(stem: str) -> str:
    head = stem.split("_subject")[0]
    return head.rstrip("0123456789")


def _caption(stem: str) -> str:
    cat = _category(stem)
    return CATEGORY_CAPTIONS.get(cat, f"a person performs {cat}")


def _csv_to_qpos36(csv_path: Path) -> np.ndarray:
    a = np.loadtxt(csv_path, delimiter=",").astype(np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[1] != 36:
        raise ValueError(f"{csv_path}: expected 36 cols, got {a.shape[1]}")
    pos = a[:, 0:3]
    qx, qy, qz, qw = a[:, 3], a[:, 4], a[:, 5], a[:, 6]
    quat_wxyz = np.stack([qw, qx, qy, qz], axis=1)  # OMG wants wxyz
    joints = a[:, 7:36]
    return np.concatenate([pos, quat_wxyz, joints], axis=1).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("/scratch/user/yzdong/OMG-Data/raw/lafan1_g1/g1"))
    p.add_argument("--out", type=Path, default=Path("/scratch/user/yzdong/OMG-Data/omg_data/lafan1/g1"))
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--val-every", type=int, default=8, help="every Nth clip -> val")
    p.add_argument("--test-every", type=int, default=8, help="every Nth clip (offset) -> test")
    args = p.parse_args()

    dataset_dir = args.out.parent  # .../lafan1
    g1_dir = args.out
    labels_dir = dataset_dir / "labels"
    g1_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(args.src.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no CSVs under {args.src}")

    kin = G1Kinematics()  # default assets/robots/g1/g1_kinematics.json

    splits = {"train": [], "val": [], "test": []}
    for i, csv in enumerate(csvs):
        stem = csv.stem
        qpos = _csv_to_qpos36(csv)
        # Precompute forward kinematics (OMG source npz ships these).
        with torch.no_grad():
            fk = kin.forward_kinematics(torch.from_numpy(qpos).float())
        body_pos_w = fk["body_pos_w"].cpu().numpy().astype(np.float32)   # (T,30,3)
        body_quat_w = fk["body_quat_w"].cpu().numpy().astype(np.float32)  # (T,30,4)
        np.savez(
            g1_dir / f"{stem}.npz",
            qpos=qpos,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            fps=np.float32(args.fps),
        )
        (labels_dir / f"{stem}.json").write_text(
            json.dumps({"caption": _caption(stem)}), encoding="utf-8"
        )
        # hold out a few clips for val/test, keep everything also trainable
        if i % args.val_every == 0:
            splits["val"].append(stem)
        elif i % args.test_every == args.test_every - 1:
            splits["test"].append(stem)
        splits["train"].append(stem)

    # ensure val/test non-empty
    if not splits["val"]:
        splits["val"].append(splits["train"][0])
    if not splits["test"]:
        splits["test"].append(splits["train"][-1])

    (dataset_dir / "info.yaml").write_text(yaml.safe_dump(splits, sort_keys=False), encoding="utf-8")

    frames = sum(int(np.load(g1_dir / f"{s}.npz")["qpos"].shape[0]) for s in splits["train"])
    print(f"[adapt] clips={len(csvs)} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"[adapt] total train frames={frames} (~{frames/args.fps/60:.1f} min @ {args.fps:.0f}fps)")
    print(f"[adapt] wrote -> {dataset_dir}")
    cats = sorted({_category(c.stem) for c in csvs})
    print(f"[adapt] categories: {cats}")


if __name__ == "__main__":
    main()
