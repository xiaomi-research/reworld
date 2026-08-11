"""Internal Guidance (IG; arXiv:2512.24176): training auxiliary head + inference-time extrapolation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from dualflow import logger
from dualflow.ltx.transformer_ltx_ig import is_ltx_transformer_with_native_ig


def register_block_output_hooks_multi(
    transformer: nn.Module,
    block_indices: list[int],
) -> tuple[dict[int, list[Tensor]], list[Any]]:
    """Capture transformer block outputs for legacy IG linear readout."""
    captures: dict[int, list[Tensor]] = {int(i): [] for i in block_indices}
    handles: list[Any] = []

    def _make_hook(idx: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            h = output[0] if isinstance(output, tuple) else output
            captures[idx].append(h)

        return hook

    blocks = transformer.transformer_blocks
    for idx in block_indices:
        handles.append(blocks[int(idx)].register_forward_hook(_make_hook(int(idx))))
    return captures, handles


def pop_latest_captures(captures_map: dict[int, list[Tensor]]) -> dict[int, Tensor]:
    return {k: v[-1] for k, v in captures_map.items() if v}


def dit_inner_dim(transformer: nn.Module) -> int:
    c = transformer.config
    return int(c.num_attention_heads * c.attention_head_dim)


def ig_velocity_out_channels(*, dit_latent_mode: str, c_bridge: int, c_vae: int) -> int:
    """Channel width of the main velocity head (vae_only: ``c_vae``)."""
    del dit_latent_mode, c_bridge  # kept for call-site compatibility
    return int(c_vae)


def combine_internal_guidance_predictions(
    pred_f: Tensor,
    pred_i: Tensor,
    *,
    ig_scale: float,
    scheduler_timestep: Tensor,
    ig_min_timestep: float | None,
) -> Tensor:
    """ Eq. (5) style extrapolation on the model output used by the scheduler (velocity field here).

    ``pred_w = pred_i + w * (pred_f - pred_i)``. When ``ig_min_timestep`` is set (guidance-interval style,
    paper §4.2), use ``w = 1`` on low-noise steps (scheduler ``t`` below threshold) and ``w = ig_scale``
    otherwise.
    """
    w = float(ig_scale)
    if ig_min_timestep is not None:
        t_val = float(scheduler_timestep.flatten()[0].item())
        if t_val < float(ig_min_timestep):
            w = 1.0
    if abs(w - 1.0) < 1e-6:
        return pred_f
    return pred_i + w * (pred_f - pred_i)


def predict_velocity_with_internal_guidance(
    transformer: nn.Module,
    *,
    ig_head: nn.Linear | None,
    ig_block_idx: int | None,
    ig_scale: float,
    ig_min_timestep: float | None,
    scheduler_timestep: Tensor,
    **transformer_kw: Any,
) -> Tensor:
    """Single DiT forward; native IG uses one forward with dual velocity; legacy uses hook + ``ig_head``."""
    native = is_ltx_transformer_with_native_ig(transformer)
    want_extrap = ig_block_idx is not None and abs(float(ig_scale) - 1.0) > 1e-6

    base_kw = {
        k: v
        for k, v in transformer_kw.items()
        if k not in (
            "return_dict",
            "return_intermediate_velocity",
            "intermediate_block_idx",
            "return_intermediate_velocities",
            "intermediate_block_indices",
        )
    }

    if native and want_extrap:
        out = transformer(
            **base_kw,
            return_dict=False,
            return_intermediate_velocity=True,
            intermediate_block_idx=int(ig_block_idx),
        )
        noise_pred_f, noise_pred_i = out[0], out[1]
        return combine_internal_guidance_predictions(
            noise_pred_f,
            noise_pred_i,
            ig_scale=float(ig_scale),
            scheduler_timestep=scheduler_timestep,
            ig_min_timestep=ig_min_timestep,
        )

    if not want_extrap:
        out = transformer(**transformer_kw)
        return out[0] if isinstance(out, tuple) else out

    if ig_head is None:
        logger.warning(
            "internal_guidance: extrapolation requested but model has no native IG branch and ig_head is None — "
            "using vanilla velocity."
        )
        out = transformer(**transformer_kw)
        return out[0] if isinstance(out, tuple) else out

    cmap, handles = register_block_output_hooks_multi(transformer, [int(ig_block_idx)])
    try:
        out = transformer(**transformer_kw)
        noise_pred_f = out[0] if isinstance(out, tuple) else out
        h_mid = pop_latest_captures(cmap)[int(ig_block_idx)]
        noise_pred_i = ig_head(h_mid.to(dtype=noise_pred_f.dtype))
        return combine_internal_guidance_predictions(
            noise_pred_f,
            noise_pred_i,
            ig_scale=float(ig_scale),
            scheduler_timestep=scheduler_timestep,
            ig_min_timestep=ig_min_timestep,
        )
    finally:
        for h in handles:
            h.remove()


def load_optional_ig_head_linear(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Linear | None:
    """Load ``ig_head`` from a trainer ``dualflow_dit`` ``.safetensors`` if present."""
    from safetensors.torch import load_file

    p = Path(checkpoint_path)
    if not p.is_file():
        return None
    blob = load_file(str(p))
    prefix = "ig_head."
    keys = sorted(k for k in blob if k.startswith(prefix))
    if not keys:
        return None
    w_k = prefix + "weight"
    b_k = prefix + "bias"
    if w_k not in blob:
        return None
    weight = blob[w_k]
    bias = blob[b_k] if b_k in blob else torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
    oc, ic = int(weight.shape[0]), int(weight.shape[1])
    lin = nn.Linear(ic, oc, bias=True)
    lin.weight.data.copy_(weight)
    lin.bias.data.copy_(bias)
    lin.to(device=device, dtype=dtype)
    lin.eval()
    return lin


def copy_ig_head_from_safetensors_into(module: nn.Linear, checkpoint_path: str | Path) -> bool:
    """Load ``ig_head.{weight,bias}`` from a trainer ``.safetensors`` into an existing ``nn.Linear``.

    Returns True if tensors were present and shapes matched. Does nothing and returns False otherwise.
    """
    from safetensors.torch import load_file

    p = Path(checkpoint_path)
    if not p.is_file() or p.suffix not in (".safetensors", ".sft"):
        return False
    blob = load_file(str(p))
    prefix = "ig_head."
    w_k, b_k = prefix + "weight", prefix + "bias"
    if w_k not in blob:
        return False
    weight = blob[w_k]
    if tuple(weight.shape) != tuple(module.weight.shape):
        return False
    bias = blob.get(b_k)
    module.weight.data.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
    if bias is not None:
        if tuple(bias.shape) != tuple(module.bias.shape):
            return False
        module.bias.data.copy_(bias.to(device=module.bias.device, dtype=module.bias.dtype))
    else:
        module.bias.zero_()
    return True
