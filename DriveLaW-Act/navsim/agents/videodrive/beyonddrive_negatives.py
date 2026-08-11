"""BeyondDrive hard-negative loading for ReWorld Stage-3 RDE.

Expected layout under ``negative_samples_path`` (BeyondDrive-compatible pool):

    {negative_samples_path}/{scene_token}.pkl

Each pickle is a dict with at least:

    - ``pdm_score_matrix``: (N, 7) float array — columns
      0=NC, 1=DAC, 2=EP, 3=TTC, 4=comfort, 5=DDC, 6=overall PDMS.
    - ``pred_trajectorys``: (N, L, 3) proposal trajectories in ego SE2 (x, y, heading),
      same frame as ``targets['trajectory']``.

Obtain the pool from BeyondDrive (https://github.com/wjl2244/BeyondDrive): download
their released negatives or run ``scripts/evaluation/run_generate_negative_samples_pool.sh``.

Hard negatives are proposals whose submetric (``beyonddrive_submetric_index``,
default -1 = overall PDMS) is below ``beyonddrive_submetric_threshold``; the
closest-to-expert unsafe proposal is selected per scene.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Columns from BeyondDrive PDM scoring (NC, DAC, EP, TTC, comfort, DDC, PDMS):
PDM_SCORE_MATRIX_COLUMNS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)


def resolve_beyonddrive_submetric_index(args: Any | None = None) -> int:
    """Match BeyondDrive ``SUBMETRIC_INDEX`` env (default -1 = overall PDMS column)."""
    if args is not None and getattr(args, "beyonddrive_submetric_index", None) is not None:
        return int(args.beyonddrive_submetric_index)
    return int(os.environ.get("SUBMETRIC_INDEX", "-1"))


def resolve_beyonddrive_submetric_threshold(args: Any | None = None) -> float:
    if args is not None and getattr(args, "beyonddrive_submetric_threshold", None) is not None:
        return float(args.beyonddrive_submetric_threshold)
    return 0.6


def beyonddrive_training_enabled(args: Any | None) -> bool:
    return bool(getattr(args, "use_beyonddrive", False))


def negative_samples_dir(args: Any | None) -> Path | None:
    raw = getattr(args, "negative_samples_path", None) if args is not None else None
    if not raw:
        return None
    path = Path(str(raw))
    return path


def load_negative_sample_pkl(pkl_path: Path) -> dict[str, Any]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def select_hard_negative_trajectory(
    token_trajs_pdms: dict[str, Any],
    human_trajectory: torch.Tensor,
    *,
    submetric_index: int,
    submetric_threshold: float = 0.6,
) -> tuple[torch.Tensor, bool]:
    """
    Official BeyondDrive selection (CacheOnlyDataset._load_scene_with_token):

    1. Filter proposals with ``pdm_score_matrix[:, submetric_index] < threshold``.
    2. Among filtered unsafe proposals, pick the one closest to expert (L2 over waypoints).

    Returns:
        negative_trajectory: (L, 3) float32 in ego SE2 (same space as ``targets['trajectory']``).
        has_valid_negative: False when no unsafe proposal exists (tensor is zeros).
    """
    expert = human_trajectory.detach().float()
    if expert.ndim != 2:
        raise ValueError(f"human_trajectory must be (L, 3), got {tuple(expert.shape)}")

    score_matrix = np.asarray(token_trajs_pdms["pdm_score_matrix"])
    pred_trajectorys = token_trajs_pdms["pred_trajectorys"]
    if pred_trajectorys is None or len(pred_trajectorys) == 0:
        return expert.new_zeros(expert.shape), False

    pred_trajectorys = np.asarray(pred_trajectorys)
    if pred_trajectorys.ndim != 3:
        raise ValueError(
            f"pred_trajectorys must be (N, L, C), got {pred_trajectorys.shape}"
        )

    valid_index = score_matrix[:, int(submetric_index)] < float(submetric_threshold)
    filtered = pred_trajectorys[valid_index]
    if filtered.shape[0] == 0:
        return expert.new_zeros(expert.shape), False

    filtered_t = torch.from_numpy(filtered).to(device=expert.device, dtype=torch.float32)
    # Same reduction as official: mean over waypoint dims, then argmin over candidates.
    dist = (filtered_t - expert.unsqueeze(0)).pow(2).sum(dim=-1).mean(dim=-1)
    min_index = int(torch.argmin(dist).item())
    return filtered_t[min_index], True


def attach_hard_negative_to_targets(
    targets: dict[str, torch.Tensor],
    *,
    token: str,
    negative_samples_path: Path,
    submetric_index: int,
    submetric_threshold: float,
) -> None:
    """Load ``{token}.pkl`` and write ``targets['negative_trajectory']`` (+ token)."""
    pkl_path = negative_samples_path / f"{token}.pkl"
    if not pkl_path.is_file():
        raise FileNotFoundError(
            f"BeyondDrive negative sample not found: {pkl_path}. "
            "Download the official pool or run negative-sample generation first."
        )

    token_trajs_pdms = load_negative_sample_pkl(pkl_path)
    if "trajectory" not in targets:
        raise KeyError("targets must contain 'trajectory' for BeyondDrive hard-negative selection.")

    negative_trajectory, _ = select_hard_negative_trajectory(
        token_trajs_pdms,
        targets["trajectory"],
        submetric_index=submetric_index,
        submetric_threshold=submetric_threshold,
    )
    targets["negative_trajectory"] = negative_trajectory
    targets["token"] = token
