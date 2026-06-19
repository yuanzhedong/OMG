#!/usr/bin/env bash
# Train 50M OMG text-to-motion on real LAFAN1->G1 data.
# Runs on the 2x idle Blackwell RTX PRO 6000 (sm_120) via the cu128 venv,
# leaving the 4x RTX 4090 free for the existing RSL-RL jobs.
# Resumable: re-run to continue from last.ckpt if present.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv-cu128/bin/activate
set -a; source .env; set +a
# Address GPUs by physical PCI order so indices match nvidia-smi.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,3            # the two Blackwell PRO 6000 (idle)
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false
export OMG_DATASET_CACHE_DIR=/tmp/omg_cache_lafan

EXP_NAME=lafan1_50m_demo
CKPT="outputs/${EXP_NAME}/checkpoints/last.ckpt"
RESUME=""
[[ -f "$CKPT" ]] && RESUME="ckpt_path=$CKPT" && echo "[train] resuming from $CKPT"

exec python -m omg.cli.generation.train \
  exp=50m data=omg_data_lafan trainer=2gpu logger=wandb \
  exp_name="$EXP_NAME" \
  model.use_audio=false model.use_human_motion=false \
  data.loader_opts.train.batch_size=192 \
  data.loader_opts.train.num_workers=8 \
  data.loader_opts.train.prefetch_factor=4 \
  trainer.max_steps=60000 \
  trainer.val_check_interval=2000 \
  callbacks.checkpoint.every_n_train_steps=1000 \
  $RESUME
