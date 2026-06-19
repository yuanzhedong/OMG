#!/usr/bin/env bash
# Generate G1 motion from a text prompt and render a MuJoCo video.
# Usage: scripts/dev/render_demo.sh "a person walks forward" [num_frames]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv-cu128/bin/activate
set -a; source .env; set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0            # idle Blackwell PRO 6000 for inference/render
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl
export OMG_DATASET_CACHE_DIR=/tmp/omg_cache_lafan

TEXT="${1:-a person walks forward}"
NUM_FRAMES="${2:-120}"
EXP_NAME=lafan1_50m_demo
CKPT="${CKPT:-outputs/${EXP_NAME}/checkpoints/last.ckpt}"
SLUG=$(echo "$TEXT" | tr ' ' '_' | tr -cd '[:alnum:]_' | cut -c1-40)

python -m omg.cli.generation.generate \
  --exp 50m --ckpt_path "$CKPT" \
  --text "$TEXT" --num_frames "$NUM_FRAMES" \
  --history_val_index 0 \
  --cfg_text_scale 3.0 --seed 0 \
  --render_video --camera_view iso --follow_mode xy --scene_preset studio \
  --width 1280 --height 720 --fps 30 --title "OMG: $TEXT" \
  --output_root "outputs_generate/$SLUG" \
  data=omg_data_lafan model.use_audio=false model.use_human_motion=false
echo "[render] done -> outputs_generate/$SLUG"
