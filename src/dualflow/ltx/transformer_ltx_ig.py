"""LTX DiT with Internal Guidance (IG): per-block auxiliary velocity heads + final head.

Matches the **dual-head** layout from the official SiT implementation in
`Internal-Guidance <https://github.com/CVL-UESTC/Internal-Guidance>`__:
each intermediate block has its own ``ig_scale_shift_table`` + ``ig_norm_out`` + ``ig_proj_out``.

Paper / reference: `Guiding a Diffusion Transformer with the Internal Dynamics of Itself` (IG).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel

try:
    from diffusers.utils import apply_lora_scale
except ImportError:
    try:
        from diffusers.utils.peft_utils import apply_lora_scale
    except ImportError:

        def apply_lora_scale(_attention_kwargs_name: str = "attention_kwargs"):
            """Older diffusers: ``apply_lora_scale`` missing; LoRA scaling is a no-op."""

            def _decorator(forward_fn):
                return forward_fn

            return _decorator


def is_ltx_transformer_with_native_ig(m: nn.Module) -> bool:
    return isinstance(m, LTXVideoTransformer3DModelWithIG)


def _block_key(block_idx: int) -> str:
    return str(int(block_idx))


class IgOutputHead(nn.Module):
    """One IG velocity head (same structure as the legacy single ``ig_*`` stack)."""

    def __init__(self, inner_dim: int, out_ch: int) -> None:
        super().__init__()
        self.ig_scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.ig_norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.ig_proj_out = nn.Linear(inner_dim, out_ch)

    @torch.no_grad()
    def init_from_main(self, scale_shift_table: torch.Tensor, proj_out: nn.Linear) -> None:
        self.ig_scale_shift_table.copy_(scale_shift_table)
        self.ig_proj_out.weight.copy_(proj_out.weight)
        self.ig_proj_out.bias.copy_(proj_out.bias)

    def velocity_from_hidden(
        self, hidden_states: torch.Tensor, embedded_timestep: torch.Tensor
    ) -> torch.Tensor:
        scale_shift_values = self.ig_scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        h = self.ig_norm_out(hidden_states)
        h = h * (1 + scale) + shift
        return self.ig_proj_out(h)


class LTXVideoTransformer3DModelWithIG(LTXVideoTransformer3DModel):
    """``LTXVideoTransformer3DModel`` with per-block IG output stacks."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ig_heads = nn.ModuleDict()

    def ensure_ig_heads(self, block_indices: list[int]) -> None:
        """Create ``IgOutputHead`` modules for each block index (idempotent)."""
        inner_dim = self.proj_out.in_features
        out_ch = self.proj_out.out_features
        # Match device/dtype of the already-placed backbone (heads may be created after .to(cuda)).
        ref = self.proj_out.weight
        for bi in sorted({int(b) for b in block_indices}):
            key = _block_key(bi)
            if key not in self.ig_heads:
                head = IgOutputHead(inner_dim, out_ch)
                self.ig_heads[key] = head.to(device=ref.device, dtype=ref.dtype)

    @torch.no_grad()
    def init_ig_auxiliary_from_main(self, block_indices: list[int] | None = None) -> None:
        """Initialize IG head(s) from the main output head."""
        if block_indices is None:
            block_indices = [int(k) for k in self.ig_heads.keys()]
        self.ensure_ig_heads(block_indices)
        for bi in block_indices:
            self.ig_heads[_block_key(bi)].init_from_main(self.scale_shift_table, self.proj_out)

    def _velocity_from_hidden_main(self, hidden_states: torch.Tensor) -> torch.Tensor:
        embedded = self._last_embedded_timestep
        scale_shift_values = self.scale_shift_table[None, None] + embedded[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        h = self.norm_out(hidden_states)
        h = h * (1 + scale) + shift
        return self.proj_out(h)

    def _velocity_from_hidden_ig(self, hidden_states: torch.Tensor, block_idx: int) -> torch.Tensor:
        head = self.ig_heads[_block_key(block_idx)]
        return head.velocity_from_hidden(hidden_states, self._last_embedded_timestep)

    def _validate_block_idx(self, block_idx: int) -> None:
        n_blocks = len(self.transformer_blocks)
        if block_idx < 0 or block_idx >= n_blocks:
            raise ValueError(
                f"block index must be in [0, {n_blocks - 1}], got {block_idx}"
            )

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        rope_interpolation_scale: tuple[float, float, float] | torch.Tensor | None = None,
        video_coords: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        return_intermediate_velocity: bool = False,
        intermediate_block_idx: int | None = None,
        return_intermediate_velocities: bool = False,
        intermediate_block_indices: list[int] | None = None,
    ) -> Transformer2DModelOutput | tuple:
        multi = return_intermediate_velocities
        single = return_intermediate_velocity
        if multi and single:
            raise ValueError("Use either return_intermediate_velocity or return_intermediate_velocities, not both.")
        if multi:
            if not intermediate_block_indices:
                raise ValueError(
                    "return_intermediate_velocities=True requires non-empty intermediate_block_indices"
                )
            for bi in intermediate_block_indices:
                self._validate_block_idx(int(bi))
                if _block_key(int(bi)) not in self.ig_heads:
                    raise ValueError(
                        f"IG head for block {bi} missing; call ensure_ig_heads first."
                    )
        if single:
            if intermediate_block_idx is None:
                raise ValueError("return_intermediate_velocity=True requires intermediate_block_idx")
            self._validate_block_idx(int(intermediate_block_idx))
            if _block_key(int(intermediate_block_idx)) not in self.ig_heads:
                raise ValueError(
                    f"IG head for block {intermediate_block_idx} missing; call ensure_ig_heads first."
                )

        image_rotary_emb = self.rope(hidden_states, num_frames, height, width, rope_interpolation_scale, video_coords)

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.size(0)
        hidden_states = self.proj_in(hidden_states)

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        self._last_embedded_timestep = embedded_timestep

        velocity_intermediate: torch.Tensor | None = None
        velocities_multi: dict[int, torch.Tensor] = {}
        want_blocks: set[int] = set()
        if multi:
            want_blocks = {int(b) for b in intermediate_block_indices}
        elif single:
            want_blocks = {int(intermediate_block_idx)}

        n_blocks = len(self.transformer_blocks)

        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    encoder_attention_mask,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                )

            if i in want_blocks:
                v = self._velocity_from_hidden_ig(hidden_states, i)
                if multi:
                    velocities_multi[i] = v
                else:
                    velocity_intermediate = v

        output = self._velocity_from_hidden_main(hidden_states)

        if multi:
            missing = want_blocks - set(velocities_multi.keys())
            if missing:
                raise RuntimeError(
                    f"intermediate_block_indices {sorted(missing)} never matched any block (blocks={n_blocks})"
                )
            if not return_dict:
                return (output, velocities_multi)
            raise ValueError(
                "return_intermediate_velocities with return_dict=True is not supported; use return_dict=False"
            )

        if single:
            if velocity_intermediate is None:
                raise RuntimeError(
                    f"intermediate_block_idx={intermediate_block_idx} never matched any block "
                    f"(blocks={n_blocks})"
                )
            if not return_dict:
                return (output, velocity_intermediate)
            raise ValueError(
                "return_intermediate_velocity with return_dict=True is not supported; use return_dict=False"
            )

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def _load_ig_head_tensors_into(
    head: IgOutputHead,
    blob: dict[str, torch.Tensor],
    *,
    key_prefix: str,
) -> bool:
    def _get(*candidates: str) -> torch.Tensor | None:
        for k in candidates:
            full = f"{key_prefix}{k}" if key_prefix else k
            if full in blob:
                return blob[full]
            if k in blob:
                return blob[k]
        return None

    ok_table = False
    t = _get("ig_scale_shift_table")
    if t is not None and tuple(t.shape) == tuple(head.ig_scale_shift_table.shape):
        head.ig_scale_shift_table.data.copy_(
            t.to(device=head.ig_scale_shift_table.device, dtype=head.ig_scale_shift_table.dtype)
        )
        ok_table = True

    ok_proj = False
    w = _get("ig_proj_out.weight")
    if w is not None and tuple(w.shape) == tuple(head.ig_proj_out.weight.shape):
        head.ig_proj_out.weight.data.copy_(
            w.to(device=head.ig_proj_out.weight.device, dtype=head.ig_proj_out.weight.dtype)
        )
        ok_proj = True
    b = _get("ig_proj_out.bias")
    if b is not None and ok_proj and tuple(b.shape) == tuple(head.ig_proj_out.bias.shape):
        head.ig_proj_out.bias.data.copy_(
            b.to(device=head.ig_proj_out.bias.device, dtype=head.ig_proj_out.bias.dtype)
        )

    return ok_table and ok_proj


def load_auxiliary_ig_branch_from_safetensors(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    block_idx: int | None = None,
) -> bool:
    """Load IG tensors for one block from ``.safetensors``.

    Tries per-block keys ``ig_heads.{idx}.*`` / ``transformer.ig_heads.{idx}.*``, then legacy flat ``ig_*``.
    """
    if not is_ltx_transformer_with_native_ig(model) or not checkpoint_path:
        return False
    from safetensors.torch import load_file

    p = Path(checkpoint_path)
    if not p.is_file() or p.suffix not in (".safetensors", ".sft"):
        return False
    blob = load_file(str(p))

    if block_idx is None:
        if len(model.ig_heads) == 1:
            block_idx = int(next(iter(model.ig_heads.keys())))
        else:
            return False

    bi = int(block_idx)
    model.ensure_ig_heads([bi])
    head = model.ig_heads[_block_key(bi)]

    for prefix in (
        f"transformer.ig_heads.{bi}.",
        f"ig_heads.{bi}.",
        f"transformer.ig_heads.{_block_key(bi)}.",
        f"ig_heads.{_block_key(bi)}.",
    ):
        if _load_ig_head_tensors_into(head, blob, key_prefix=prefix):
            return True

    return _load_ig_head_tensors_into(
        head,
        blob,
        key_prefix="transformer." if "transformer.ig_scale_shift_table" in blob else "",
    ) or _load_ig_head_tensors_into(head, blob, key_prefix="")


def ingest_legacy_external_ig_head_into_auxiliary(
    model: nn.Module,
    checkpoint_path: str | Path | None,
    *,
    block_idx: int | None = None,
) -> bool:
    """Copy legacy ``ig_head.{weight,bias}`` into ``ig_proj_out`` for one block if shapes match."""
    if not is_ltx_transformer_with_native_ig(model) or not checkpoint_path:
        return False
    from safetensors.torch import load_file

    p = Path(checkpoint_path)
    if not p.is_file() or p.suffix not in (".safetensors", ".sft"):
        return False
    blob = load_file(str(p))
    w_key = "transformer.ig_head.weight" if "transformer.ig_head.weight" in blob else "ig_head.weight"
    if w_key not in blob:
        return False

    if block_idx is None:
        if len(model.ig_heads) == 1:
            block_idx = int(next(iter(model.ig_heads.keys())))
        else:
            block_idx = 0

    bi = int(block_idx)
    model.ensure_ig_heads([bi])
    head = model.ig_heads[_block_key(bi)]

    w = blob[w_key]
    if tuple(w.shape) != tuple(head.ig_proj_out.weight.shape):
        return False
    head.ig_proj_out.weight.data.copy_(
        w.to(device=head.ig_proj_out.weight.device, dtype=head.ig_proj_out.weight.dtype)
    )
    b_key = "transformer.ig_head.bias" if "transformer.ig_head.bias" in blob else "ig_head.bias"
    if b_key in blob:
        b = blob[b_key]
        if tuple(b.shape) == tuple(head.ig_proj_out.bias.shape):
            head.ig_proj_out.bias.data.copy_(
                b.to(device=head.ig_proj_out.bias.device, dtype=head.ig_proj_out.bias.dtype)
            )
    with torch.no_grad():
        head.ig_scale_shift_table.copy_(model.scale_shift_table)
    return True


def bootstrap_ig_heads_from_checkpoint(
    model: LTXVideoTransformer3DModelWithIG,
    checkpoint_path: str | Path | None,
    block_indices: list[int],
    *,
    legacy_block_idx: int | None = None,
) -> tuple[bool, bool]:
    """Ensure per-block heads, load from checkpoint, init remaining from main.

    Returns ``(any_loaded_from_ckpt, any_legacy_migrated)``.
    """
    blocks = sorted({int(b) for b in block_indices})
    model.ensure_ig_heads(blocks)
    wpath = (str(checkpoint_path) if checkpoint_path else "").strip()
    loaded_blocks: set[int] = set()
    any_legacy = False
    legacy_target = int(legacy_block_idx if legacy_block_idx is not None else blocks[0])

    if wpath:
        for bi in blocks:
            if load_auxiliary_ig_branch_from_safetensors(model, wpath, block_idx=bi):
                loaded_blocks.add(bi)
        if not loaded_blocks:
            any_legacy = ingest_legacy_external_ig_head_into_auxiliary(
                model, wpath, block_idx=legacy_target
            )
            if any_legacy:
                loaded_blocks.add(legacy_target)
                if len(blocks) > 1:
                    src = model.ig_heads[_block_key(legacy_target)].state_dict()
                    for bi in blocks:
                        if bi == legacy_target:
                            continue
                        model.ig_heads[_block_key(bi)].load_state_dict(src)
                        loaded_blocks.add(bi)

    for bi in blocks:
        if bi not in loaded_blocks:
            model.ig_heads[_block_key(bi)].init_from_main(model.scale_shift_table, model.proj_out)

    return bool(loaded_blocks), any_legacy
