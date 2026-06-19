#!/usr/bin/env bash
# Execute a generated OMG reference through the HoloMotion tracker (physics) and
# render a kinematic-reference | physics-executed side-by-side video.
# Usage: scripts/dev/track_and_render.sh <slug> <model_dir> <exp>
#   e.g. scripts/dev/track_and_render.sh a_person_runs outputs_generate_full/a_person_runs/100m/lafan1_100m_scaling
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv-cu128/bin/activate; set -a; source .env; set +a
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=""
SLUG="${1:?slug}"; REFDIR="${2:?reference dir}"
ONNX="/scratch/user/yzdong/OMG-models/holomotion_dl/HoloMotion_motion_tracking_model_v1.3.1/exported/motion_tracking_model.onnx"
REF="$REFDIR/reference_motion.npz"
OUT="outputs_tracker/$SLUG"

python -m omg.cli.pipeline.main --mode tracker-only \
  --seed-motion "$REF" --holomotion-onnx "$ONNX" --tracker-providers CPUExecutionProvider \
  --num-frames 150 --output-root "$OUT" >/dev/null 2>&1

python3 - "$OUT" "$SLUG" <<'PY'
import sys, glob, numpy as np, torch, mujoco, cv2, os
from omg.robots.g1.kinematics import G1Kinematics
from omg.render.mujoco import render_qpos_frames
out, slug = sys.argv[1], sys.argv[2]
d = np.load(glob.glob(f"{out}/**/holomotion_rollout.npz", recursive=True)[0])
ref, ex = d["reference_qpos_36"].astype(np.float32), d["executed_qpos_36"].astype(np.float32)
kin = G1Kinematics(); xml = "assets/holomotion/g1_29dof/scene_29dof.xml"
def render(q):
    m = mujoco.MjModel.from_xml_path(xml); data = mujoco.MjData(m)
    bpw = kin.forward_kinematics(torch.from_numpy(q).float())["body_pos_w"].numpy()
    return render_qpos_frames(q, bpw, kin, m, data, width=640, height=640, follow_mode="xy",
                             title="", camera_distance_scale=0.6)
fr_ref, fr_ex = render(ref), render(ex)
fell = "FELL" if ex[:,2].min() < 0.4 else "upright"
def lab(img, txt, c):
    img = img.copy(); cv2.rectangle(img,(0,0),(img.shape[1],30),(20,20,20),-1)
    cv2.putText(img,txt,(8,21),cv2.FONT_HERSHEY_SIMPLEX,0.55,c,1); return img
n = min(len(fr_ref), len(fr_ex)); vw = None; path = f"{out}/tracker_sidebyside_{slug}.mp4"
for i in range(n):
    a = lab(cv2.cvtColor(fr_ref[i],cv2.COLOR_RGB2BGR),"Kinematic reference (brain)",(120,200,255))
    b = lab(cv2.cvtColor(fr_ex[i],cv2.COLOR_RGB2BGR),"HoloMotion physics execution",(120,255,120))
    f = cv2.hconcat([a,b])
    if vw is None: vw = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'),30,(f.shape[1],f.shape[0]))
    vw.write(f)
vw.release()
print(f"{slug}: executed travel={np.hypot(ex[:,0].ptp(),ex[:,1].ptp()):.2f}m z_min={ex[:,2].min():.3f} ({fell}) -> {path}")
PY
