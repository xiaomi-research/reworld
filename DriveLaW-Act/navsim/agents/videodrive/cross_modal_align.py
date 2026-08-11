"""Attention-weighted cross-modal alignment (planning training only)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossModalProjector(nn.Module):
    """Project detached video readout (video_dim) into action hidden space."""

    def __init__(self, video_dim: int, action_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(video_dim)
        self.proj = nn.Linear(video_dim, action_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


def _attn_heads_and_dim(attn: nn.Module) -> tuple[int, int]:
    heads = int(getattr(attn, "heads"))
    inner = int(attn.to_q.out_features)
    dim_head = inner // heads
    return heads, dim_head


def compute_cross_attn_probs_for_align(
    attn: nn.Module,
    q_hs: torch.Tensor,
    kv_hs: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Softmax attention probs for alignment; mirrors action ``attn2`` Q/K path.

    ``kv_hs`` must already be detached by the caller when Video DiT should not
    receive align gradients; pass live ``encoder_hidden_states`` to allow video grads.

    Returns:
        attn_probs: (B, H, L, S)
    """
    heads, dim_head = _attn_heads_and_dim(attn)

    q = attn.to_q(q_hs)
    k = attn.to_k(kv_hs)
    q = attn.norm_q(q)
    k = attn.norm_k(k)

    q = q.unflatten(-1, (heads, dim_head)).transpose(1, 2)
    k = k.unflatten(-1, (heads, dim_head)).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim_head)

    if attention_mask is not None:
        if attention_mask.ndim == 2:
            mask = attention_mask.unsqueeze(1).unsqueeze(1)
        elif attention_mask.ndim == 3:
            mask = attention_mask.unsqueeze(1)
        else:
            mask = attention_mask
        scores = scores + mask.to(dtype=scores.dtype, device=scores.device)

    return scores.softmax(dim=-1)


def weighted_video_readout(
    attn_probs: torch.Tensor,
    video_hs_det: torch.Tensor,
) -> torch.Tensor:
    """Token-level readout: (B,H,L,S) x (B,S,Dv) -> (B,L,Dv)."""
    weights = attn_probs.mean(dim=1)
    return torch.bmm(weights, video_hs_det)


def token_cosine_align_loss(h_act: torch.Tensor, z_vid: torch.Tensor) -> torch.Tensor:
    """Per action-token cosine loss, averaged over tokens and batch."""
    h = F.normalize(h_act, dim=-1)
    z = F.normalize(z_vid, dim=-1)
    return (1.0 - (h * z).sum(dim=-1)).mean()


def symmetric_token_cosine_align_loss(h_act: torch.Tensor, z_vid: torch.Tensor) -> torch.Tensor:
    """Token + global pooled cosine align (stronger scene-level signal)."""
    tok = token_cosine_align_loss(h_act, z_vid)
    h = F.normalize(h_act.mean(dim=1), dim=-1)
    z = F.normalize(z_vid.mean(dim=1), dim=-1)
    glob = (1.0 - (h * z).sum(dim=-1)).mean()
    return tok + 0.5 * glob


def cross_modal_align_loss_from_pack(
    pack: dict[str, torch.Tensor],
    projector: CrossModalProjector,
    *,
    loss_type: str = "cosine",
) -> torch.Tensor:
    """Compute align loss from one supervised block pack."""
    readout = weighted_video_readout(pack["attn_probs_align"], pack["video_hs"])
    z_vid = projector(readout)
    if loss_type == "symmetric":
        return symmetric_token_cosine_align_loss(pack["h_act"], z_vid)
    return token_cosine_align_loss(pack["h_act"], z_vid)


def _block_key(block_idx: int) -> str:
    return str(int(block_idx))


def _load_checkpoint_blob(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    path = Path(checkpoint_path)
    if path.is_dir():
        index_json = path / "diffusion_pytorch_model.safetensors.index.json"
        if index_json.is_file():
            from navsim.agents.videodrive.utils.model_utils import load_index_file

            return load_index_file(str(index_json))
        single = path / "diffusion_pytorch_model.safetensors"
        if single.is_file():
            path = single
        else:
            return {}
    if not path.is_file() or path.suffix != ".safetensors":
        return {}
    return load_file(str(path))


def _projector_state_from_prefix(
    blob: dict[str, torch.Tensor],
    key_prefix: str,
) -> dict[str, torch.Tensor] | None:
    out: dict[str, torch.Tensor] = {}
    for suffix in ("norm.weight", "norm.bias", "proj.weight"):
        tensor = None
        for key in (f"{key_prefix}{suffix}", suffix):
            if key in blob:
                tensor = blob[key]
                break
        if tensor is None:
            return None
        out[suffix] = tensor
    return out


def _apply_projector_state(projector: CrossModalProjector, state: dict[str, torch.Tensor]) -> bool:
    norm_w, norm_b, proj_w = state["norm.weight"], state["norm.bias"], state["proj.weight"]
    if (
        tuple(norm_w.shape) != tuple(projector.norm.weight.shape)
        or tuple(norm_b.shape) != tuple(projector.norm.bias.shape)
        or tuple(proj_w.shape) != tuple(projector.proj.weight.shape)
    ):
        return False
    projector.norm.weight.data.copy_(
        norm_w.to(device=projector.norm.weight.device, dtype=projector.norm.weight.dtype)
    )
    projector.norm.bias.data.copy_(
        norm_b.to(device=projector.norm.bias.device, dtype=projector.norm.bias.dtype)
    )
    projector.proj.weight.data.copy_(
        proj_w.to(device=projector.proj.weight.device, dtype=projector.proj.weight.dtype)
    )
    return True


def _copy_projector_tensors(
    projector: CrossModalProjector,
    blob: dict[str, torch.Tensor],
    *,
    key_prefix: str,
) -> bool:
    state = _projector_state_from_prefix(blob, key_prefix)
    if state is None:
        return False
    return _apply_projector_state(projector, state)


def _legacy_projector_prefixes() -> tuple[str, ...]:
    return (
        "cross_modal_projector.",
        "transformer.cross_modal_projector.",
        "diffusion_model.cross_modal_projector.",
    )


def _per_block_projector_prefixes(block_idx: int) -> tuple[str, ...]:
    key = _block_key(block_idx)
    return (
        f"cross_modal_projectors.{key}.",
        f"transformer.cross_modal_projectors.{key}.",
        f"diffusion_model.cross_modal_projectors.{key}.",
        f"cross_modal_projectors.{block_idx}.",
        f"transformer.cross_modal_projectors.{block_idx}.",
        f"diffusion_model.cross_modal_projectors.{block_idx}.",
    )


def bootstrap_cross_modal_projectors_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path | None,
    block_indices: list[int],
) -> bool:
    """Load per-block projectors from checkpoint.

    - New format ``cross_modal_projectors.{block}.*``: load each block independently.
    - Legacy single ``cross_modal_projector.*``: copy into every supervise block missing
      a per-block entry.
    """
    if not hasattr(model, "ensure_cross_modal_projectors"):
        return False

    blocks = sorted({int(b) for b in block_indices})
    if not blocks:
        return False

    model.ensure_cross_modal_projectors(blocks)
    wpath = (str(checkpoint_path) if checkpoint_path else "").strip()
    if not wpath:
        return False

    blob = _load_checkpoint_blob(wpath)
    if not blob:
        return False

    loaded_blocks: list[int] = []
    for bi in blocks:
        head = model.cross_modal_projectors[_block_key(bi)]
        for prefix in _per_block_projector_prefixes(bi):
            if _copy_projector_tensors(head, blob, key_prefix=prefix):
                loaded_blocks.append(bi)
                break

    legacy_loaded = False
    legacy_state: dict[str, torch.Tensor] | None = None
    for prefix in _legacy_projector_prefixes():
        legacy_state = _projector_state_from_prefix(blob, prefix)
        if legacy_state is not None:
            break

    if legacy_state is not None:
        missing = [bi for bi in blocks if bi not in loaded_blocks]
        for bi in missing:
            _apply_projector_state(model.cross_modal_projectors[_block_key(bi)], legacy_state)
        legacy_loaded = bool(missing)
        if legacy_loaded:
            logger.info(
                "Cross-modal projectors: copied legacy cross_modal_projector -> blocks %s",
                missing,
            )

    if loaded_blocks:
        logger.info(
            "Cross-modal projectors: loaded per-block weights for blocks %s",
            sorted(set(loaded_blocks)),
        )
    return bool(loaded_blocks) or legacy_loaded
