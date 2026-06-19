#!/usr/bin/env bash
# Train 100M full-loss OMG DiT on the BONES-SEED subset (real captions, ~8.7h of
# motion) on the 2x Blackwell GPUs. Uses a bones-specific stats file so the
# LAFAN1 demo model keeps working. Resumable from last.ckpt.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."/..
source .venv-cu128/bin/activate
set -a; source .env; set +a
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,3
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false
export OMG_DATASET_CACHE_DIR=/tmp/omg_cache_bones

EXP_NAME=bones_seed_40k_100m
CKPT="outputs/${EXP_NAME}/checkpoints/last.ckpt"
RESUME=""; [[ -f "$CKPT" ]] && RESUME="ckpt_path=$CKPT"

exec python -m omg.cli.generation.train \
  exp=100m data=omg_data_bones40k trainer=2gpu logger=wandb \
  exp_name="$EXP_NAME" \
  representation.stats_path=assets/stats/g1_125d_bones40k_stats.json \
  model.use_audio=false model.use_human_motion=false \
  callbacks.divergence_guard.max_loss=200 \
  callbacks.divergence_guard.max_grad_norm=100000 \
  data.loader_opts.train.batch_size=64 \
  data.loader_opts.train.num_workers=10 \
  data.loader_opts.train.prefetch_factor=4 \
  trainer.max_steps=60000 \
  trainer.val_check_interval=2500 \
  callbacks.checkpoint.every_n_train_steps=2500 \
  $RESUME
