"""Adapt the BONES-SEED Unitree-G1 subset into OMG source format.

Source CSV (36 cols, 120 fps): Frame, root_translate{X,Y,Z} (cm),
root_rotate{X,Y,Z} (Euler degrees), then 29 <joint>_dof (degrees) in
omg.robots.g1.constants.G1_JOINT_NAMES order.

OMG qpos_36 = root_pos(m, 3) + root_quat(wxyz, 4) + joint_dof(rad, 29).
Captions come from the temporal-label JSONL (per-segment descriptions).

Output (UnifiedG1MotionIndex layout) under <out>:
  g1/<name>.npz       qpos (T,36) + body_pos_w + body_quat_w + fps(30)
  labels/<name>.json  {"caption": ..., "segments":[{start_time,end_time,action}]}
  info.yaml           {train:[...], val:[...], test:[...]}
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "src")
from omg.robots.g1.kinematics import G1Kinematics

SRC_FPS, DST_FPS = 120.0, 30.0
STRIDE = int(SRC_FPS / DST_FPS)  # 4


def csv_to_qpos36(csv_path: Path) -> np.ndarray:
    a = np.loadtxt(csv_path, delimiter=",", skiprows=1).astype(np.float64)
    if a.ndim == 1:
        a = a[None, :]
    a = a[::STRIDE]  # 120 -> 30 fps
    pos = a[:, 1:4] / 100.0                       # cm -> m
    quat_xyzw = R.from_euler("XYZ", a[:, 4:7], degrees=True).as_quat()  # xyzw
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]], axis=1)
    joints = np.deg2rad(a[:, 7:36])               # deg -> rad
    return np.concatenate([pos, quat_wxyz, joints], axis=1).astype(np.float32)


def load_labels(jsonl: Path) -> dict[str, list[dict]]:
    out = {}
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[d["filename"]] = d.get("events", [])
    return out


def caption_for(stem: str, labels: dict[str, list[dict]]) -> tuple[str, list[dict]]:
    # match CSV stem to JSONL filename (handle trailing _M variant)
    ev = labels.get(stem) or labels.get(stem[:-2] if stem.endswith("_M") else stem + "_M")
    if not ev:
        # fall back to a readable name from the stem
        name = stem.split("__")[0].replace("_", " ").strip()
        return (f"a person {name}" if name else "a person moves"), []
    segs = [{"start_time": float(e["start_time"]), "end_time": float(e["end_time"]),
             "action": str(e["description"]).strip()} for e in ev]
    summary = segs[0]["action"] if segs else ""
    return summary, segs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("/scratch/user/yzdong/OMG-Data/raw/bones_seed/g1/csv"))
    p.add_argument("--jsonl", type=Path, default=Path("/scratch/user/yzdong/OMG-Data/raw/bones_seed/metadata/seed_metadata_v002_temporal_labels.jsonl"))
    p.add_argument("--out", type=Path, default=Path("/scratch/user/yzdong/OMG-Data/omg_data/bones_seed"))
    p.add_argument("--limit", type=int, default=4000, help="number of clips to adapt")
    p.add_argument("--min-frames", type=int, default=45, help="skip clips shorter than this (post-resample)")
    p.add_argument("--validate", action="store_true", help="convert a few, print FK upright check, exit")
    args = p.parse_args()

    kin = G1Kinematics()
    csvs = sorted(args.src.rglob("*.csv"))
    print(f"[bones] found {len(csvs)} csvs; using up to {args.limit}")

    if args.validate:
        for c in csvs[:3]:
            q = csv_to_qpos36(c)
            bpw = kin.forward_kinematics(torch.from_numpy(q).float())["body_pos_w"].numpy()
            zmin, zmax = float(bpw[..., 2].min()), float(bpw[..., 2].max())
            print(f"  {c.name}: frames={len(q)} rootZ={q[0,2]:.3f} bodyZ[min={zmin:.3f} max={zmax:.3f}] "
                  f"-> {'UPRIGHT (feet~0, head~1.3)' if zmin < 0.15 and zmax > 1.0 else 'CHECK euler convention'}")
        return

    g1_dir, lbl_dir = args.out / "g1", args.out / "labels"
    g1_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.jsonl)
    splits = {"train": [], "val": [], "test": []}
    kept = frames_total = 0
    for i, c in enumerate(csvs):
        if kept >= args.limit:
            break
        try:
            q = csv_to_qpos36(c)
        except Exception:
            continue
        if len(q) < args.min_frames or not np.isfinite(q).all():
            continue
        stem = c.stem
        fk = kin.forward_kinematics(torch.from_numpy(q).float())
        np.savez(g1_dir / f"{stem}.npz", qpos=q,
                 body_pos_w=fk["body_pos_w"].numpy().astype(np.float32),
                 body_quat_w=fk["body_quat_w"].numpy().astype(np.float32),
                 fps=np.float32(DST_FPS))
        cap, segs = caption_for(stem, labels)
        (lbl_dir / f"{stem}.json").write_text(json.dumps({"caption": cap, "segments": segs}))
        (splits["val"] if kept % 25 == 0 else splits["test"] if kept % 25 == 1 else splits["train"]).append(stem)
        kept += 1; frames_total += len(q)
        if kept % 500 == 0:
            print(f"[bones] {kept} clips, {frames_total/DST_FPS/60:.0f} min so far")
    import yaml
    (args.out / "info.yaml").write_text(yaml.safe_dump(splits, sort_keys=False))
    print(f"[bones] DONE: {kept} clips, ~{frames_total/DST_FPS/60:.0f} min "
          f"(train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}) -> {args.out}")


if __name__ == "__main__":
    main()
