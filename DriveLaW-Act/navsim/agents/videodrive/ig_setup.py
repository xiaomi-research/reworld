"""Bootstrap planning aux modules (cross-modal projector) around checkpoint load."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import torch.nn as nn
from safetensors import safe_open

from navsim.agents.videodrive.ig_config import resolve_cross_modal_align_blocks

logger = logging.getLogger(__name__)

_CROSS_MODAL_LEGACY_PREFIX = "cross_modal_projector."
_CROSS_MODAL_PER_BLOCK_KEY = re.compile(r"(?:^|\.)cross_modal_projectors\.(\d+)\.")


def unwrap_training_module(m: nn.Module) -> nn.Module:
    """Strip DeepSpeed / DDP / torch.compile wrappers for type checks and attribute access."""
    inner = m
    while True:
        if hasattr(inner, "module"):
            inner = inner.module
            continue
        if hasattr(inner, "_orig_mod"):
            inner = inner._orig_mod
            continue
        break
    return inner


def is_planning_transformer_with_ig(m: nn.Module) -> bool:
    """Detect planning transformer with cross-modal projectors (checkpoint class name kept for compat)."""
    inner = unwrap_training_module(m)
    if type(inner).__name__ != "LTXVideoTransformer3DModelWithIG":
        return False
    return hasattr(inner, "cross_modal_projectors") and hasattr(inner, "ensure_cross_modal_projectors")


def _strip_checkpoint_prefix(key: str) -> str:
    for prefix in ("transformer.", "diffusion_model."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def list_checkpoint_state_dict_keys(checkpoint_path: str | None) -> list[str]:
    """List tensor keys in a safetensors checkpoint (file or diffusers folder)."""
    if not checkpoint_path:
        return []
    path = str(checkpoint_path).strip()
    if not path:
        return []

    if os.path.isdir(path):
        index_json = os.path.join(path, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.isfile(index_json):
            from navsim.agents.videodrive.utils.model_utils import load_index_file

            return list(load_index_file(index_json).keys())
        single = os.path.join(path, "diffusion_pytorch_model.safetensors")
        if os.path.isfile(single):
            path = single
        else:
            return []

    if not os.path.isfile(path) or not path.endswith(".safetensors"):
        return []

    with safe_open(path, framework="pt") as f:
        return list(f.keys())


def cross_modal_blocks_in_checkpoint_keys(keys: list[str]) -> list[int]:
    blocks: set[int] = set()
    for key in keys:
        norm = _strip_checkpoint_prefix(key)
        m = _CROSS_MODAL_PER_BLOCK_KEY.search(norm)
        if m:
            blocks.add(int(m.group(1)))
    return sorted(blocks)


def cross_modal_legacy_projector_in_checkpoint_keys(keys: list[str]) -> bool:
    for key in keys:
        norm = _strip_checkpoint_prefix(key)
        if norm.startswith(_CROSS_MODAL_LEGACY_PREFIX):
            return True
    return False


def cross_modal_projector_in_checkpoint_keys(keys: list[str]) -> bool:
    """True if checkpoint has per-block or legacy cross-modal projector weights."""
    return bool(cross_modal_blocks_in_checkpoint_keys(keys)) or cross_modal_legacy_projector_in_checkpoint_keys(
        keys
    )


def _resolve_cross_modal_blocks(args: Any | None, ckpt_keys: list[str]) -> list[int]:
    blocks: set[int] = set(cross_modal_blocks_in_checkpoint_keys(ckpt_keys))
    if args is not None:
        blocks.update(int(b) for b in resolve_cross_modal_align_blocks(args))
    return sorted(blocks)


def ensure_planning_aux_modules_before_load(
    model: nn.Module,
    args: Any | None = None,
    checkpoint_path: str | None = None,
) -> None:
    """Create aux modules *before* ``load_checkpoints`` so their tensors can be restored."""
    if not is_planning_transformer_with_ig(model):
        return

    ckpt_keys = list_checkpoint_state_dict_keys(checkpoint_path)
    need_cross_modal = cross_modal_projector_in_checkpoint_keys(ckpt_keys)

    if args is not None:
        if resolve_cross_modal_align_blocks(args) or getattr(args, "lambda_cross_modal_align", 0):
            need_cross_modal = True

    cm_blocks = _resolve_cross_modal_blocks(args, ckpt_keys)
    if need_cross_modal and cm_blocks and hasattr(model, "ensure_cross_modal_projectors"):
        model.ensure_cross_modal_projectors(cm_blocks)
        logger.info(
            "Pre-load: ensured cross_modal_projectors for blocks %s (checkpoint=%s)",
            cm_blocks,
            bool(ckpt_keys),
        )


def setup_cross_modal_from_args(model: nn.Module, args: Any) -> bool:
    """Ensure per-block cross-modal projectors; load or migrate checkpoint weights."""
    if not is_planning_transformer_with_ig(model):
        return False
    if not hasattr(model, "ensure_cross_modal_projectors"):
        return False

    blocks = resolve_cross_modal_align_blocks(args)
    if not blocks and not getattr(args, "lambda_cross_modal_align", 0):
        return False

    ckpt = None
    dm = getattr(args, "diffusion_model", None)
    if isinstance(dm, dict):
        ckpt = dm.get("model_path")
    else:
        ckpt = getattr(dm, "model_path", None) if dm is not None else None

    if not blocks:
        blocks = cross_modal_blocks_in_checkpoint_keys(list_checkpoint_state_dict_keys(ckpt))
    if not blocks:
        return False

    from navsim.agents.videodrive.cross_modal_align import bootstrap_cross_modal_projectors_from_checkpoint

    loaded = bootstrap_cross_modal_projectors_from_checkpoint(model, ckpt, blocks)
    logger.info(
        "Cross-modal projectors: blocks=%s checkpoint_loaded=%s path=%s",
        blocks,
        loaded,
        ckpt,
    )
    return True
