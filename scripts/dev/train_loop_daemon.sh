#!/usr/bin/env bash
# Continuous training daemon — keeps the 2x Blackwell GPUs busy with back-to-back
# OMG runs (cycling model sizes as a scaling sweep). Runs until stopped.
#
#   start:  nohup bash scripts/dev/train_loop_daemon.sh > /tmp/train_loop.log 2>&1 &
#   stop:   touch /tmp/omg_train_loop.stop      (stops after the current run finishes)
#   resume: rm -f /tmp/omg_train_loop.stop && restart
#
# Each loop run logs to W&B (project omg). Checkpoints are pruned after each run
# (metrics live in W&B) so disk stays bounded — these are "keep-hot" runs, not
# the demo model (outputs/lafan1_50m_demo is never touched).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO="$(pwd)"
STOP=/tmp/omg_train_loop.stop
SIZES=(50m 100m 300m)   # 500m/1b dropped: impractically slow + 1b diverges on 40 clips
i=0

echo "[loop] daemon start $(date) pid=$$"
while [ ! -f "$STOP" ]; do
  # Don't double-book GPUs: wait for any existing OMG training to finish.
  while pgrep -f "omg.cli.generation.train" >/dev/null 2>&1; do
    [ -f "$STOP" ] && break
    sleep 30
  done
  [ -f "$STOP" ] && break

  size=${SIZES[$((i % ${#SIZES[@]}))]}
  name="lafan1_${size}_loop_$(date +%m%d_%H%M)"
  echo "[loop] === launching $name ($size) at $(date) ==="
  bash -c 'source .venv-cu128/bin/activate; set -a; source .env; set +a; \
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,3 PYTHONPATH=src TOKENIZERS_PARALLELISM=false OMG_DATASET_CACHE_DIR=/tmp/omg_cache_lafan; \
    python -m omg.cli.generation.train exp='"$size"' data=omg_data_lafan trainer=2gpu logger=wandb exp_name='"$name"' \
      model.use_audio=false model.use_human_motion=false \
      data.loader_opts.train.batch_size=64 data.loader_opts.train.num_workers=8 data.loader_opts.train.prefetch_factor=4 \
      trainer.max_steps=15000 trainer.val_check_interval=5000 \
      callbacks.checkpoint.every_n_train_steps=5000 callbacks.checkpoint.save_top_k=0' \
    >> /tmp/train_loop_run.log 2>&1 || echo "[loop] run $name exited non-zero"

  echo "[loop] === $name finished at $(date); pruning checkpoints ==="
  rm -rf "$REPO/outputs/${name}/checkpoints" 2>/dev/null || true
  i=$((i + 1))
done
echo "[loop] daemon stopped $(date) (found $STOP)"
