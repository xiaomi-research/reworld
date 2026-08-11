"""Resolve cross-modal alignment block indices and loss knobs from planning yaml / args."""

from __future__ import annotations

from typing import Any


def _bool_arg(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "on")
    return bool(v)


def resolve_cross_modal_align_blocks(args: Any) -> list[int]:
    raw = getattr(args, "cross_modal_align_blocks", None)
    if raw is None:
        return []
    if isinstance(raw, int):
        return [int(raw)]
    return [int(b) for b in list(raw)]


def lambda_cross_modal_align(args: Any) -> float:
    return float(getattr(args, "lambda_cross_modal_align", 0.0) or 0.0)


def cross_modal_align_grad_video(args: Any) -> bool:
    """When True, cross-modal align loss backprops into Video DiT (no detach on video hidden)."""
    return _bool_arg(getattr(args, "cross_modal_align_grad_video", False))


def cross_modal_align_stopgrad_video(args: Any) -> bool:
    """Inverse of ``cross_modal_align_grad_video`` (default: stop-grad on video)."""
    return not cross_modal_align_grad_video(args)


def cross_modal_align_loss_type(args: Any) -> str:
    return str(getattr(args, "cross_modal_align_loss_type", "cosine")).lower()


def cross_modal_align_timestep_max(args: Any) -> float | None:
    """If set, apply align loss only when action noise t <= this value."""
    val = getattr(args, "cross_modal_align_timestep_max", None)
    if val is None:
        return None
    return float(val)


def cross_modal_align_training_enabled(args: Any, model: Any) -> bool:
    """Whether attention-weighted cross-modal align loss should run."""
    from navsim.agents.videodrive.ig_setup import is_planning_transformer_with_ig

    train_mode = str(getattr(args, "train_mode", "")).strip().strip("'\"")
    return (
        lambda_cross_modal_align(args) > 0.0
        and bool(resolve_cross_modal_align_blocks(args))
        and is_planning_transformer_with_ig(model)
        and _bool_arg(getattr(args, "return_action", False))
        and train_mode in ("all", "action_only", "action_full")
    )
