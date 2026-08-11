"""Synchronous vae_only denoising with LTX FlowMatchEulerDiscreteScheduler."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from diffusers.utils import export_to_video

from dualflow import logger
from dualflow.dual_conditioning import expand_timesteps_with_conditioning
from dualflow.internal_guidance import predict_velocity_with_internal_guidance
from dualflow.ltx.latents import decode_video, get_rope_scale_factors
from dualflow.ltx.model_loader import load_scheduler
from dualflow.model_utils import null_text_embeddings

if TYPE_CHECKING:
    from diffusers import AutoencoderKLLTXVideo


def _normalize_ig_sampling_kwargs(
    ig_head: torch.nn.Linear | None,
    ig_block_idx: int | None,
    ig_scale: float,
    ig_min_timestep: float | None,
) -> tuple[torch.nn.Linear | None, int | None, float, float | None]:
    """Native IG uses ``ig_head=None`` with a non-None ``ig_block_idx``; legacy linear readout uses both."""
    bi = ig_block_idx if ig_block_idx is not None else None
    if ig_head is None:
        return None, bi, float(ig_scale), ig_min_timestep
    return ig_head, bi if bi is not None else 0, float(ig_scale), ig_min_timestep


def _timestep_ids_for_transformer(
    t_step: torch.Tensor,
    *,
    seq: int,
    device: torch.device,
    conditioning_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Integer timestep per token (0–1000), same convention as stage-2 training."""
    tid = int(round(float(t_step.flatten()[0].item())))
    tid = max(0, min(1000, tid))
    tid_vec = torch.tensor([tid], device=device, dtype=torch.long)
    if conditioning_mask is None:
        return torch.full((1, seq), tid, device=device, dtype=torch.long)
    return expand_timesteps_with_conditioning(conditioning_mask, tid_vec)


def _per_token_timesteps_for_scheduler_step(
    t_step: torch.Tensor,
    conditioning_mask: torch.Tensor | None,
    *,
    seq: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Float timesteps per token for ``scheduler.step(..., per_token_timesteps=...)``."""
    if conditioning_mask is None:
        return None
    t_val = float(t_step.flatten()[0].item())
    base = torch.full((1, seq), t_val, device=device, dtype=torch.float32)
    cm = conditioning_mask.float()
    return torch.min(base, (1.0 - cm) * 1000.0)


def _apply_vae_latent_anchor(
    z_pix: torch.Tensor,
    z_pix_anchor: torch.Tensor | None,
    conditioning_mask: torch.Tensor | None,
) -> torch.Tensor:
    if conditioning_mask is None or z_pix_anchor is None:
        return z_pix
    m = conditioning_mask.unsqueeze(-1)
    z_pix_anchor = z_pix_anchor.to(dtype=z_pix.dtype)
    return torch.where(m.expand_as(z_pix), z_pix_anchor, z_pix)


def run_vae_latent_sampling(
    transformer: torch.nn.Module,
    vae: AutoencoderKLLTXVideo,
    *,
    c_vae: int,
    nf: int,
    nh: int,
    nw: int,
    num_inference_steps: int,
    vae_model_source: str,
    fps: float,
    caption_channels: int,
    null_text_seq_len: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    output_video_path: Path | str | None = None,
    conditioning_mask: torch.Tensor | None = None,
    z_pix_anchor: torch.Tensor | None = None,
    ig_head: torch.nn.Linear | None = None,
    ig_block_idx: int | None = None,
    ig_scale: float = 1.0,
    ig_min_timestep: float | None = None,
) -> Path | torch.Tensor:
    """VAE-latent-only denoising with the same scheduler contract as the LTX pipeline."""
    torch.manual_seed(seed)
    seq = nf * nh * nw
    rope = get_rope_scale_factors(fps)
    enc, mask = null_text_embeddings(1, null_text_seq_len, caption_channels, device, dtype)

    if z_pix_anchor is not None:
        z_pix_anchor = z_pix_anchor.to(device=device, dtype=dtype)

    z_pix = torch.randn(1, seq, c_vae, device=device, dtype=dtype)
    if conditioning_mask is not None and z_pix_anchor is not None:
        m = conditioning_mask.unsqueeze(-1)
        z_pix = torch.where(m.expand_as(z_pix), z_pix_anchor, z_pix)

    scheduler = load_scheduler(vae_model_source)
    scheduler.set_timesteps(num_inference_steps, device=device)

    transformer.eval()
    ig_h, ig_bi, ig_sc, ig_min = _normalize_ig_sampling_kwargs(ig_head, ig_block_idx, ig_scale, ig_min_timestep)

    for t in scheduler.timesteps:
        t = t.to(device=device)
        ts = _timestep_ids_for_transformer(t, seq=seq, device=device, conditioning_mask=conditioning_mask)
        per_tok = _per_token_timesteps_for_scheduler_step(
            t, conditioning_mask, seq=seq, device=device
        )
        with torch.no_grad():
            noise_pred = predict_velocity_with_internal_guidance(
                transformer,
                ig_head=ig_h,
                ig_block_idx=ig_bi,
                ig_scale=ig_sc,
                ig_min_timestep=ig_min,
                scheduler_timestep=t,
                hidden_states=z_pix,
                encoder_hidden_states=enc,
                encoder_attention_mask=mask,
                timestep=ts,
                num_frames=nf,
                height=nh,
                width=nw,
                rope_interpolation_scale=rope,
                video_coords=None,
                return_dict=False,
            )
        z_pix = scheduler.step(
            -noise_pred,
            t,
            z_pix,
            per_token_timesteps=per_tok,
            return_dict=False,
        )[0]
        z_pix = z_pix.to(dtype=dtype)
        z_pix = _apply_vae_latent_anchor(z_pix, z_pix_anchor, conditioning_mask)

    if output_video_path is None:
        return z_pix

    out_p = Path(output_video_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    vid = decode_video(vae, z_pix, num_frames=nf, height=nh, width=nw, device=device, dtype=dtype)
    if vid.dim() == 5:
        vid = vid[0]
    vid_fhwc = (
        vid.permute(1, 2, 3, 0).contiguous().clamp(0, 1).float().cpu().detach().numpy()
    )
    export_to_video(vid_fhwc, str(out_p), fps=int(fps))
    logger.info("vae-only dualflow: saved sample video to %s", out_p)
    return out_p
