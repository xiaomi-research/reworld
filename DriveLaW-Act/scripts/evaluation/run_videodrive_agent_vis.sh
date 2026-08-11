#!/usr/bin/env bash
# DriveLaW-Act planning trajectory visualization (BEV + front camera).
# Assumes cluster sets RESOURCE_GPU / WORLD_SIZE / RANK / MASTER_ADDR / MASTER_PORT.
#
# Outputs: traj_plot_vis_drivelaw_act/{token}_vis_traj.png
#   left  = front cam + agent(red) / human(green) trajectories
#   right = BEV map + trajectories

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/dataset/maps"
export NAVSIM_EXP_ROOT="/path/to/DriveLaW-Act"
export NAVSIM_DEVKIT_ROOT="/path/to/DriveLaW-Act"
export OPENSCENE_DATA_ROOT="/path/to/dataset"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH}"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_TIMEOUT=1800000

cd "${NAVSIM_DEVKIT_ROOT}"

# Planning checkpoint is set inside this YAML (diffusion_model.model_path).
CONFIG_FILE="${CONFIG_FILE:-navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-traj_plot_vis_drivelaw_act}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
# Optional: SCORE_CSV=/path/to/pdm_scores.csv  (visualize score==1 tokens)
SCORE_CSV="${SCORE_CSV:-}"
# Optional: TOKENS=token1,token2
TOKENS="${TOKENS:-}"

EXTRA_ARGS=()
if [[ -n "${SCORE_CSV}" ]]; then
  EXTRA_ARGS+=(--score-csv "${SCORE_CSV}")
fi
if [[ -n "${TOKENS}" ]]; then
  EXTRA_ARGS+=(--tokens "${TOKENS}")
fi

torchrun --nproc_per_node=$RESOURCE_GPU \
         --nnodes=$WORLD_SIZE \
         --node_rank=$RANK \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/plt_all_vis.py" \
    --config-file "${CONFIG_FILE}" \
    --split test \
    --scene-filter navtest \
    --output-dir "${OUTPUT_DIR}" \
    --max-samples "${MAX_SAMPLES}" \
    "${EXTRA_ARGS[@]}" \
    > videodrive_planning_vis.log 2>&1
