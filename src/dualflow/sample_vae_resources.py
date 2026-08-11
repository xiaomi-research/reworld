"""Load RGB conditioning clips and encode VAE latents for vae_only sampling."""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import AutoencoderKLLTXVideo
from torch import Tensor

from dualflow.dual_conditioning import rgb_frames_to_latent_frames
from dualflow.ltx.latents import encode_video
from dualflow.ltx.video_io import crop_video, read_video_leading_frames, resize_video
from dualflow.train_configs import DualFlowSampleConfig


def load_condition_bcfhw(
    *,
    video_path: str | Path,
    condition_source_frames: int,
    video_width: int,
    video_height: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, float]:
    """Load the leading ``condition_source_frames`` RGB frames as ``[B,C,F,H,W]`` in [0,1]."""
    frames, fps = read_video_leading_frames(video_path, condition_source_frames)
    frames = resize_video(frames, video_width, video_height)
    frames = crop_video(frames, video_width, video_height)
    bcfhw = frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous().to(device, dtype=dtype)
    return bcfhw, float(fps)


def place_condition_latents_in_full(
    z_cond: Tensor,
    *,
    nf: int,
    nh: int,
    nw: int,
) -> Tensor:
    """Copy packed condition latents into the leading tokens of a full ``nf*nh*nw`` canvas."""
    b, l_cond, c = z_cond.shape
    full_len = int(nf) * int(nh) * int(nw)
    full = torch.zeros(b, full_len, c, device=z_cond.device, dtype=z_cond.dtype)
    n = min(int(l_cond), full_len)
    full[:, :n] = z_cond[:, :n]
    return full


def encode_condition_anchor_latents(
    vae: AutoencoderKLLTXVideo,
    bcfhw: Tensor,
    *,
    nf: int,
    nh: int,
    nw: int,
    condition_source_frames: int,
    temporal_compression: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Encode a short condition RGB clip and embed it into a full-length latent canvas.

    Only the first ``rgb_frames_to_latent_frames(condition_source_frames)`` latent frames are
    meaningful; the rest of the canvas is zeros (denoising uses the conditioning mask).
    """
    enc = encode_video(vae, bcfhw, device=device, dtype=dtype)
    z_cond = enc["latents"].to(dtype=dtype)
    lf_exp = rgb_frames_to_latent_frames(condition_source_frames, temporal_compression)
    lf_got = int(enc["num_frames"])
    if lf_got != lf_exp:
        raise RuntimeError(
            f"condition encode got {lf_got} latent frames, expected {lf_exp} "
            f"from condition_source_frames={condition_source_frames}, temporal_compression={temporal_compression}"
        )
    if int(enc["height"]) != nh or int(enc["width"]) != nw:
        raise RuntimeError(
            f"condition latent spatial grid ({enc['height']},{enc['width']}) != ({nh},{nw})"
        )
    return place_condition_latents_in_full(z_cond, nf=nf, nh=nh, nw=nw)


def load_and_encode_condition_anchor(
    cfg: DualFlowSampleConfig,
    video_path: str | Path,
    vae: AutoencoderKLLTXVideo,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, float]:
    """Convenience: load leading condition frames → full-canvas ``z_pix_anchor`` + fps."""
    bcfhw, fps = load_condition_bcfhw(
        video_path=video_path,
        condition_source_frames=cfg.condition_source_frames,
        video_width=cfg.video_width,
        video_height=cfg.video_height,
        device=device,
        dtype=dtype,
    )
    z_anchor = encode_condition_anchor_latents(
        vae,
        bcfhw,
        nf=cfg.latent_frames,
        nh=cfg.latent_height,
        nw=cfg.latent_width,
        condition_source_frames=cfg.condition_source_frames,
        temporal_compression=cfg.temporal_compression,
        device=device,
        dtype=dtype,
    )
    return z_anchor, fps


# Back-compat aliases used by older call sites
def load_bcfhw_from_video_path(
    cfg: DualFlowSampleConfig,
    video_path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, float]:
    return load_condition_bcfhw(
        video_path=video_path,
        condition_source_frames=cfg.condition_source_frames,
        video_width=cfg.video_width,
        video_height=cfg.video_height,
        device=device,
        dtype=dtype,
    )


def encode_vae_latents_only_from_bcfhw(
    vae: AutoencoderKLLTXVideo,
    bcfhw: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, int, int, int]:
    """Encode ``[B,C,F,H,W]`` RGB to packed VAE latents (short condition clip length)."""
    enc = encode_video(vae, bcfhw, device=device, dtype=dtype)
    z_pix = enc["latents"].to(dtype=dtype)
    return z_pix, int(enc["num_frames"]), int(enc["height"]), int(enc["width"])
