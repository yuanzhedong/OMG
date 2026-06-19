#!/usr/bin/env bash
# One-shot guard: at the target time (default Friday 08:00), stop the continuous
# training daemon AND free the Blackwell GPUs for the live demo.
# Only touches the OMG training loop — never the user's rsl_rl jobs or render/generate.
#
#   start: nohup bash scripts/dev/stop_loop_friday.sh > /tmp/stop_guard.log 2>&1 &
#   override time: nohup bash scripts/dev/stop_loop_friday.sh "2026-06-19 07:30:00" &
#   cancel: pkill -f stop_loop_friday.sh
set -uo pipefail
TARGET_STR="${1:-2026-06-19 08:00:00}"
TARGET=$(date -d "$TARGET_STR" +%s)
STOP=/tmp/omg_train_loop.stop

echo "[guard] will stop the training loop at: $TARGET_STR  (now $(date))"
while [ "$(date +%s)" -lt "$TARGET" ]; do
  sleep 300
done

echo "[guard] $(date): TARGET reached — stopping training loop and freeing GPUs"
touch "$STOP"                                            # daemon won't launch new runs
pkill -f "scripts/dev/train_loop_daemon.sh" 2>/dev/null || true   # stop the daemon itself
pkill -f "omg.cli.generation.train" 2>/dev/null || true # kill in-flight loop training (NOT rsl_rl, NOT generate)
sleep 15
echo "[guard] $(date): done. Blackwell GPU state:"
CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader | sed -n '1p;4p'
echo "[guard] To re-enable training later: rm -f $STOP && relaunch scripts/dev/train_loop_daemon.sh"
