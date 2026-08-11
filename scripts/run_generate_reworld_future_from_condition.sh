#!/usr/bin/env bash
# Generate future frames from scene_*_window_000_conditioning.mp4 with ReWorld IG DiT.
# Set COND_DIR, DIT_CKPT, OUTPUT_ROOT, LTX_ROOT (and optional RESOURCE_GPU / torchrun env).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

COND_DIR="${COND_DIR:-/path/to/conditioning_videos}"
DIT_CKPT="${DIT_CKPT:-/path/to/dualflow_dit_best_loss.safetensors}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/reworld_ig_future_from_condition}"
LTX_ROOT="${LTX_ROOT:-/path/to/LTX-Video-0.9.5}"
TRANSFORMER_CFG="${TRANSFORMER_CFG:-${REPO_ROOT}/configs/dualflow/transformer_vae_dualflow_config.json}"

MAX_SCENES="${MAX_SCENES:-0}"
IG_SCALE="${IG_SCALE:-1.4}"
IG_BLOCK="${IG_BLOCK:-8}"
NPROC="${RESOURCE_GPU:-1}"

EXTRA=()
if [[ "${MAX_SCENES}" != "0" ]]; then
  EXTRA+=(--max-scenes "${MAX_SCENES}")
fi

CMD=(
  scripts/generate_reworld_future_from_condition.py
  --conditioning-dir "${COND_DIR}"
  --dit-checkpoint "${DIT_CKPT}"
  --output-root "${OUTPUT_ROOT}"
  --transformer-config "${TRANSFORMER_CFG}"
  --vae-model-source "${LTX_ROOT}"
  --video-width "${VIDEO_WIDTH:-768}"
  --video-height "${VIDEO_HEIGHT:-384}"
  --total-source-frames 33
  --condition-source-frames 9
  --fps 8
  --num-inference-steps 30
  --ig-scale "${IG_SCALE}"
  --ig-block "${IG_BLOCK}"
  "${EXTRA[@]}"
)

if [[ "${NPROC}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC}" \
           --nnodes="${WORLD_SIZE:-1}" \
           --node_rank="${RANK:-0}" \
           --rdzv_endpoint="${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-29500}" \
           "${CMD[@]}"
else
  python "${CMD[@]}"
fi
