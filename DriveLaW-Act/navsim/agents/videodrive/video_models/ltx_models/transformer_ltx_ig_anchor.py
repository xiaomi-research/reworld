# Planning DiT with action expert + per-block cross-modal alignment projectors.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.models.modeling_outputs import Transformer2DModelOutput

from navsim.agents.videodrive.cross_modal_align import CrossModalProjector, _block_key
from navsim.agents.videodrive.video_models.diffusion_planner.diffusion_planner import (
    preprocessing_action_states,
)
from navsim.agents.videodrive.video_models.ltx_models.transformer_ltx_anchor import (
    LTXVideoTransformer3DModel,
)


class LTXVideoTransformer3DModelWithIG(LTXVideoTransformer3DModel):
    """``LTXVideoTransformer3DModel`` (action expert) + per-block cross-modal projectors."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cross_modal_projectors = nn.ModuleDict()

    def ensure_cross_modal_projectors(self, block_indices: list[int]) -> nn.ModuleDict:
        if not getattr(self, "action_expert", False):
            raise ValueError("cross_modal_projectors require action_expert=True")
        video_dim = self.transformer_blocks[0].attn1.to_q.in_features
        for bi in sorted({int(b) for b in block_indices}):
            key = _block_key(bi)
            if key not in self.cross_modal_projectors:
                self.cross_modal_projectors[key] = CrossModalProjector(
                    video_dim, self.action_inner_dim
                )
        return self.cross_modal_projectors

    def _run_action_block(
        self,
        block_idx: int,
        *,
        action_hidden_states: torch.Tensor,
        final_hidden_states: torch.Tensor,
        action_temb_fused: torch.Tensor,
        action_rotary_emb: torch.Tensor,
        collect_cross_modal_align: bool = False,
        cross_modal_align_stopgrad_video: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        out = self.action_blocks[block_idx](
            hidden_states=action_hidden_states,
            encoder_hidden_states=final_hidden_states,
            temb=action_temb_fused,
            rotary_emb=action_rotary_emb,
            collect_cross_modal_align=collect_cross_modal_align,
            cross_modal_align_stopgrad_video=cross_modal_align_stopgrad_video,
        )
        if collect_cross_modal_align:
            action_hidden_states, align_pack = out
            return action_hidden_states, align_pack
        return out, None

    def _validate_block_idx(self, block_idx: int) -> None:
        n_blocks = len(self.transformer_blocks)
        if block_idx < 0 or block_idx >= n_blocks:
            raise ValueError(f"block index must be in [0, {n_blocks - 1}], got {block_idx}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Union[Tuple[float, float, float], torch.Tensor]] = None,
        video_coords: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        action_states: torch.Tensor = None,
        action_timestep: torch.LongTensor = None,
        return_video: bool = True,
        return_action: bool = False,
        store_buffer: bool = False,
        video_states_buffer=None,
        video_attention_mask: torch.Tensor = None,
        context_dict: Optional[Dict[str, torch.Tensor]] = None,
        return_cross_modal_align: bool = False,
        cross_modal_block_indices: Optional[List[int]] = None,
        cross_modal_align_stopgrad_video: bool = True,
    ):
        align_blocks: set[int] = set()
        if return_cross_modal_align:
            if not return_action:
                raise ValueError("cross-modal align requires return_action=True")
            if not cross_modal_block_indices:
                raise ValueError(
                    "return_cross_modal_align=True requires non-empty cross_modal_block_indices"
                )
            for bi in cross_modal_block_indices:
                self._validate_block_idx(int(bi))
            align_blocks = {int(b) for b in cross_modal_block_indices}

        cross_modal_packs: dict[int, dict[str, Any]] = {}

        if return_video or store_buffer:
            if store_buffer:
                video_states_buffer = []
            image_rotary_emb = self.rope(
                hidden_states, num_frames, height, width, rope_interpolation_scale, video_coords
            )

            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            if video_attention_mask is not None and video_attention_mask.ndim == 2:
                video_attention_mask = (1 - video_attention_mask.to(hidden_states.dtype)) * -10000.0
                video_attention_mask = video_attention_mask.unsqueeze(0)

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

        if return_action:
            if video_states_buffer is None:
                assert store_buffer or return_video
            action_temb_fused, action_embedded_timestep, action_rotary_emb, action_hidden_states = (
                preprocessing_action_states(self, action_states, action_timestep, context_dict)
            )
            self._last_action_embedded_timestep = action_embedded_timestep

        for block_idx, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                if return_video or store_buffer:
                    hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        encoder_attention_mask,
                    )
                    if store_buffer:
                        video_states_buffer.append(hidden_states.clone())
                else:
                    hidden_states = video_states_buffer[block_idx]

                if return_action:
                    B_video = hidden_states.shape[0]
                    B_action = action_hidden_states.shape[0]
                    assert B_action % B_video == 0
                    K = B_action // B_video
                    final_hidden_states = hidden_states.repeat_interleave(K, dim=0)
                    collect_align = block_idx in align_blocks
                    action_hidden_states, align_pack = self._run_action_block(
                        block_idx,
                        action_hidden_states=action_hidden_states,
                        final_hidden_states=final_hidden_states,
                        action_temb_fused=action_temb_fused,
                        action_rotary_emb=action_rotary_emb,
                        collect_cross_modal_align=collect_align,
                        cross_modal_align_stopgrad_video=cross_modal_align_stopgrad_video,
                    )
                    if align_pack is not None:
                        cross_modal_packs[block_idx] = align_pack
            else:
                if return_video or store_buffer:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                    if store_buffer:
                        video_states_buffer.append(hidden_states.clone())
                else:
                    hidden_states = video_states_buffer[block_idx]

                if return_action:
                    B_video = hidden_states.shape[0]
                    B_action = action_hidden_states.shape[0]
                    assert B_action % B_video == 0
                    K = B_action // B_video
                    final_hidden_states = hidden_states.repeat_interleave(K, dim=0)
                    collect_align = block_idx in align_blocks
                    action_hidden_states, align_pack = self._run_action_block(
                        block_idx,
                        action_hidden_states=action_hidden_states,
                        final_hidden_states=final_hidden_states,
                        action_temb_fused=action_temb_fused,
                        action_rotary_emb=action_rotary_emb,
                        collect_cross_modal_align=collect_align,
                        cross_modal_align_stopgrad_video=cross_modal_align_stopgrad_video,
                    )
                    if align_pack is not None:
                        cross_modal_packs[block_idx] = align_pack

        final_output: dict[str, Any] = {}

        if store_buffer:
            final_output["video_states_buffer"] = video_states_buffer

        if return_video:
            scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
            shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
            h = self.norm_out(hidden_states)
            h = h * (1 + scale) + shift
            final_output["video"] = self.proj_out(h)

        if return_action:
            if self.action_final_embeddings:
                action_scale_shift_values = (
                    self.action_scale_shift_table[None, None] + action_embedded_timestep[:, :, None]
                )
                action_shift, action_scale = action_scale_shift_values[:, :, 0], action_scale_shift_values[:, :, 1]
                action_hidden_states = self.action_norm_out(action_hidden_states)
                action_hidden_states = action_hidden_states * (1 + action_scale) + action_shift
            else:
                action_hidden_states = self.action_norm_out(action_hidden_states)
                action_hidden_states = self.action_proj_extra(action_hidden_states)

            action_output = self.action_proj_out(action_hidden_states)
            final_output["action"] = action_output
            if hasattr(self, "action_cls_head"):
                pooled = action_hidden_states.mean(dim=1)
                final_output["action_logits"] = self.action_cls_head(pooled)

        if cross_modal_packs:
            final_output["cross_modal"] = cross_modal_packs

        if not return_dict:
            return (final_output,)
        return Transformer2DModelOutput(sample=final_output)
