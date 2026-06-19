#!/usr/bin/env bash
# One-shot OMG reproduction pipeline: stats -> train (-> export -> benchmark).
#
# Loads .env automatically (copy from .env.example and fill WANDB_API_KEY).
# Requires the official OMG-Data + (for benchmarks) the evaluator checkpoint,
# which are not yet released as of this writing -- the script checks and tells
# you what is missing instead of failing cryptically.
#
# Usage:
#   scripts/reproduce.sh                       # full pipeline at defaults
#   EXP=100m TRAINER=4gpu scripts/reproduce.sh # bigger model, 4 GPUs
#   STAGES="stats train" scripts/reproduce.sh  # only run some stages
#
# Stages: stats train export benchmark   (default: "stats train")
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Load .env --------------------------------------------------------------
if [[ -f .env ]]; then
  echo "[repro] loading .env"
  set -a; # shellcheck disable=SC1091
  source .env; set +a
else
  echo "[repro] WARN: no .env found (cp .env.example .env). Continuing with shell env."
fi

# --- Config (override via environment) --------------------------------------
export PYTHONPATH="${PYTHONPATH:-src}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
EXP="${EXP:-50m}"
DATA="${DATA:-omg_data_materialized}"
TRAINER="${TRAINER:-4gpu}"
LOGGER="${LOGGER:-wandb}"
EXP_NAME="${EXP_NAME:-${EXP}_repro}"
STAGES="${STAGES:-stats train}"
STATS_OUT="${STATS_OUT:-assets/stats/g1_125d_stats.json}"
CKPT="${CKPT:-outputs/${EXP_NAME}/checkpoints/last.ckpt}"
ONNX_OUT="${ONNX_OUT:-models/generation/onnx/${EXP}/last_denoiser_step.onnx}"
EVALUATOR_CKPT="${EVALUATOR_CKPT:-models/evaluator/pretrained.ckpt}"
PY="${PY:-python}"

echo "[repro] EXP=$EXP DATA=$DATA TRAINER=$TRAINER LOGGER=$LOGGER EXP_NAME=$EXP_NAME STAGES='$STAGES'"

# wandb sanity: warn if logging to wandb without a key
if [[ "$LOGGER" == "wandb" && -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-online}" == "online" ]]; then
  echo "[repro] WARN: LOGGER=wandb but WANDB_API_KEY is empty. Set it in .env or use LOGGER=none / WANDB_MODE=offline."
fi

have_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# --- Stage: compute normalization stats -------------------------------------
if have_stage stats; then
  echo "[repro] === stats ==="
  $PY -m omg.cli.generation.compute_stats \
    --data-config configs/generation/data/omg_data.yaml \
    --representation-config configs/generation/representation/125d.yaml \
    --paths-config configs/generation/paths/default.yaml \
    --output "$STATS_OUT"
  echo "[repro] wrote stats -> $STATS_OUT"
fi

# --- Stage: train -----------------------------------------------------------
if have_stage train; then
  echo "[repro] === train ==="
  $PY -m omg.cli.generation.train \
    exp="$EXP" data="$DATA" trainer="$TRAINER" logger="$LOGGER" \
    exp_name="$EXP_NAME"
fi

# --- Stage: export ONNX -----------------------------------------------------
if have_stage export; then
  echo "[repro] === export ==="
  [[ -f "$CKPT" ]] || { echo "[repro] ERROR: checkpoint not found: $CKPT"; exit 1; }
  $PY -m omg.cli.generation.export_onnx \
    --exp "$EXP" --ckpt_path "$CKPT" --output "$ONNX_OUT" \
    --batch_size 2 --device cuda
  echo "[repro] wrote onnx -> $ONNX_OUT"
fi

# --- Stage: benchmark -------------------------------------------------------
if have_stage benchmark; then
  echo "[repro] === benchmark ==="
  [[ -f "$CKPT" ]] || { echo "[repro] ERROR: checkpoint not found: $CKPT"; exit 1; }
  if [[ ! -f "$EVALUATOR_CKPT" ]]; then
    echo "[repro] ERROR: evaluator checkpoint not found: $EVALUATOR_CKPT (unreleased)."
    echo "[repro]        Download OMG-Evaluator and set EVALUATOR_CKPT, then rerun STAGES=benchmark."
    exit 1
  fi
  $PY -m omg.cli.generation.benchmark text \
    --exp "$EXP" --ckpt_path "$CKPT" \
    --evaluator_checkpoint "$EVALUATOR_CKPT" \
    --output_dir "outputs/benchmarks/${EXP_NAME}_text"
fi

echo "[repro] done."
