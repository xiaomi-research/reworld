"""BeyondDrive RDE loss (aligned with BeyondDrive-main transfuser_loss / meanfuser_model)."""

from __future__ import annotations

from typing import Any

import torch

from navsim.agents.videodrive.beyonddrive_traj_utils import diff_traj


def lambda_rde_loss(args: Any) -> float:
    return float(getattr(args, "rde_loss_weight", 0.0) or 0.0)


def beyonddrive_use_delta_traj(args: Any) -> bool:
    """Official LTFv7 / MeanFuser use delta representation (TransfuserConfig.use_delta_traj=True)."""
    return bool(getattr(args, "beyonddrive_use_delta_traj", True))


def flow_matching_x0_actions(
    noisy_actions: torch.Tensor,
    pred_velocity: torch.Tensor,
    t_actions: torch.Tensor,
) -> torch.Tensor:
    """
    Rectified-flow x0 estimate for linear path ``noisy = (1-t)*x0 + t*noise``.

    With target velocity ``v = noise - x0`` (VideoDrive training target), holds exactly:
    ``x0 = noisy - t * v`` when ``v == pred_velocity``.
    """
    if noisy_actions.shape != pred_velocity.shape:
        raise ValueError(
            f"noisy_actions {tuple(noisy_actions.shape)} != pred_velocity {tuple(pred_velocity.shape)}"
        )
    t = t_actions.view(-1, 1, 1).to(device=noisy_actions.device, dtype=noisy_actions.dtype)
    return noisy_actions - t * pred_velocity


def compute_beyonddrive_rde_loss(
    pred_trajectory_se2: torch.Tensor,
    negative_trajectory_se2: torch.Tensor,
    *,
    use_delta_traj: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Official BeyondDrive RDE (transfuser_loss.py / meanfuser_model.py / multimodal_loss.py).

    transfuser (LTFv7, use_delta_traj=True)::
        negative_index = (targets['negative_trajectory'].sum(-1).sum(-1) > 0)
        delta_neg = diff_traj(targets['negative_trajectory'])
        rde = ((pred_delta - delta_neg).abs().mean(-1).mean(-1))[negative_index].mean()
        rde = -rde

    DiffusionDrive (absolute)::
        rde = ((best_reg - neg).abs().mean(-1).mean(-1))[negative_index].mean(); reg += 5 * -rde

    Returns:
        rde_loss: scalar (negated mean L1, add with ``+ rde_loss_weight * rde_loss``).
        l1_distance: per-sample distance before negation.
        valid_mask: official ``negative_index``.
    """
    if pred_trajectory_se2.shape != negative_trajectory_se2.shape:
        raise ValueError(
            "pred/negative trajectory shape mismatch: "
            f"{tuple(pred_trajectory_se2.shape)} vs {tuple(negative_trajectory_se2.shape)}"
        )

    valid_mask = negative_trajectory_se2.sum(-1).sum(-1) > 0

    if use_delta_traj:
        pred_repr = diff_traj(pred_trajectory_se2)
        neg_repr = diff_traj(negative_trajectory_se2)
    else:
        pred_repr = pred_trajectory_se2
        neg_repr = negative_trajectory_se2

    l1_distance = (pred_repr - neg_repr).abs().mean(-1).mean(-1)

    if not valid_mask.any():
        zero = pred_trajectory_se2.new_zeros(())
        return zero, l1_distance, valid_mask

    rde_loss = -l1_distance[valid_mask].mean()
    return rde_loss, l1_distance, valid_mask
