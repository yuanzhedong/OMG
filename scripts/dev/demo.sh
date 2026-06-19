#!/usr/bin/env bash
# OMG live demo: text prompt -> diffusion brain (reference) -> HoloMotion tracker
# (physics) -> side-by-side video. One command.
#
#   scripts/dev/demo.sh "a person walks forward"
#
# Output: outputs_demo/<slug>/demo_<slug>.mp4  (left=kinematic reference, right=physics)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv-cu128/bin/activate; set -a; source .env; set +a
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false OMG_DATASET_CACHE_DIR=/tmp/omg_cache_lafan

TEXT="${1:-a person walks forward}"
CK="${CK:-outputs/lafan1_100m_scaling/checkpoints/sstep=004000.ckpt}"   # 100M full-loss (grounded)
ONNX="/scratch/user/yzdong/OMG-models/holomotion_dl/HoloMotion_motion_tracking_model_v1.3.1/exported/motion_tracking_model.onnx"
SLUG=$(echo "$TEXT" | tr ' ' '_' | tr -cd '[:alnum:]_' | cut -c1-40)
OUT="outputs_demo/$SLUG"; mkdir -p "$OUT"

echo "[demo] 1/3 generating reference for: \"$TEXT\""
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -m omg.cli.generation.generate --exp 100m --ckpt_path "$CK" --text "$TEXT" \
  --num_frames 150 --history_val_index 0 --cfg_text_scale 3.0 --seed 0 \
  --output_root "$OUT/gen" data=omg_data_lafan model.use_audio=false model.use_human_motion=false >/dev/null 2>&1
REF=$(find "$OUT/gen" -name reference_motion.npz | head -1)

echo "[demo] 2/3 executing under physics (HoloMotion tracker, CPU)"
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl \
python -m omg.cli.pipeline.main --mode tracker-only --seed-motion "$REF" --holomotion-onnx "$ONNX" \
  --tracker-providers CPUExecutionProvider --num-frames 150 --output-root "$OUT/track" >/dev/null 2>&1
ROLL=$(find "$OUT/track" -name holomotion_rollout.npz | head -1)

echo "[demo] 3/3 rendering side-by-side video"
MUJOCO_GL=egl python3 - "$ROLL" "$OUT/demo_${SLUG}.mp4" "$TEXT" <<'PY'
import sys, numpy as np, torch, mujoco, cv2
from omg.robots.g1.kinematics import G1Kinematics
from omg.render.mujoco import render_qpos_frames
roll, out, text = sys.argv[1], sys.argv[2], sys.argv[3]
d = np.load(roll); ref, ex = d["reference_qpos_36"].astype(np.float32), d["executed_qpos_36"].astype(np.float32)
kin = G1Kinematics(); xml = "assets/holomotion/g1_29dof/scene_29dof.xml"
def render(q):
    m = mujoco.MjModel.from_xml_path(xml); data = mujoco.MjData(m)
    bpw = kin.forward_kinematics(torch.from_numpy(q).float())["body_pos_w"].numpy()
    return render_qpos_frames(q, bpw, kin, m, data, width=560, height=560, follow_mode="xy", title="", camera_distance_scale=0.62)
fr_ref, fr_ex = render(ref), render(ex)
upright = "UPRIGHT" if ex[:,2].min() > 0.4 else "FELL"
def lab(im, t, c):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR).copy(); cv2.rectangle(im,(0,0),(im.shape[1],30),(20,20,20),-1)
    cv2.putText(im, t, (8,21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1); return im
n = min(len(fr_ref), len(fr_ex)); vw = None
for i in range(n):
    a = lab(fr_ref[i], "kinematic reference (brain)", (120,200,255))
    b = lab(fr_ex[i], f"physics execution ({upright})", (120,255,120))
    top = cv2.hconcat([a,b]); banner = np.full((38, top.shape[1], 3), 30, np.uint8)
    cv2.putText(banner, f'OMG: "{text}"', (10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    f = cv2.vconcat([banner, top])
    if vw is None: vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), 30, (f.shape[1], f.shape[0]))
    vw.write(f)
vw.release()
print(f"[demo] done -> {out}  (physics: {upright})")
PY
