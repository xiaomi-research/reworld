"""LTX VAE latent pack/encode/decode and ROPE helpers (vendored for the DualFlow package)."""

from __future__ import annotations

import torch
from diffusers import AutoencoderKLLTXVideo
from torch import Tensor


def pack_latents(
    latents: Tensor,
    spatial_patch_size: int = 1,
    temporal_patch_size: int = 1,
) -> Tensor:
    """Reshapes latents [B,C,F,H,W] into patches and flattens to sequence form [B,L,D]."""
    b, c, f, h, w = latents.shape
    latents = latents.reshape(
        b,
        -1,
        f // temporal_patch_size,
        temporal_patch_size,
        h // spatial_patch_size,
        spatial_patch_size,
        w // spatial_patch_size,
        spatial_patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def unpack_packed_latents_bcfhw(
    packed: Tensor,
    num_frames: int,
    height: int,
    width: int,
) -> Tensor:
    """Inverse of ``pack_latents`` (patch 1,1 only): [B, L, C] -> [B, C, F, H, W] with L=F*H*W."""
    b, l, c = packed.shape
    exp = int(num_frames) * int(height) * int(width)
    if l != exp:
        raise ValueError(
            f"packed length L={l} != F*H*W={exp} (num_frames={num_frames}, height={height}, width={width})"
        )
    x = packed.view(b, int(num_frames), int(height), int(width), c)
    return x.permute(0, 4, 1, 2, 3).contiguous()


def _unpack_latents(
    latents: Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> Tensor:
    """Inverse of ``pack_latents``; matches ``diffusers`` ``LTXPipeline._unpack_latents``."""
    batch_size = latents.size(0)
    latents = latents.reshape(
        batch_size,
        num_frames,
        height,
        width,
        -1,
        patch_size_t,
        patch_size,
        patch_size,
    )
    return latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)


def _normalize_latents(latents: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    mean = mean.view(1, -1, 1, 1, 1).repeat(latents.shape[0], 1, 1, 1, 1).to(latents.device, latents.dtype)
    std = std.view(1, -1, 1, 1, 1).repeat(latents.shape[0], 1, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - mean) / std


def encode_video(
    vae: AutoencoderKLLTXVideo,
    image_or_video: Tensor,
    patch_size: int = 1,
    patch_size_t: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
    *,
    pixel_values_in_zero_one: bool = True,
) -> dict[str, Tensor | int]:
    """Encode to packed latents. Input must be ``[B,C,F,H,W]`` (diffusers ``AutoencoderKLLTXVideo`` layout).

    ``AutoencoderKLLTXVideo`` matches diffusers pipelines: **pixels are expected in ``[-1, 1]``** (see
    ``VaeImageProcessor.normalize``). By default we map ``[0, 1]`` → ``[-1, 1]`` via ``2*x - 1``, same as
    ``ltxv_trainer`` datasets / ``VaeImageProcessor``. Set ``pixel_values_in_zero_one=False`` if you already
    pass ``[-1, 1]``.
    """
    device = device or vae.device
    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert image_or_video.ndim == 5, f"Expected 5D tensor, got {image_or_video.ndim}D tensor"
    image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
    if pixel_values_in_zero_one:
        image_or_video = image_or_video * 2.0 - 1.0
    latents = vae.encode(image_or_video).latent_dist.sample(generator=generator)
    latents = latents.to(dtype=dtype)
    _, _, num_frames, height, width = latents.shape
    latents = _normalize_latents(latents, vae.latents_mean, vae.latents_std)
    latents = pack_latents(latents, patch_size, patch_size_t)
    return {"latents": latents, "num_frames": num_frames, "height": height, "width": width}


def decode_video(  # noqa: PLR0913
    vae: AutoencoderKLLTXVideo,
    latents: Tensor,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    patch_size: int = 1,
    patch_size_t: int = 1,
    decode_timestep: float = 0.0,
    decode_noise_scale: float | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    device = device or vae.device
    latents = latents.to(device=device, dtype=vae.dtype)
    if latents.dim() == 1:
        latents = latents.unsqueeze(0)
    elif latents.dim() == 2:
        latents = latents.unsqueeze(0)
    if latents.dim() == 3:
        latents = _unpack_latents(latents, num_frames, height, width, patch_size, patch_size_t)
    elif latents.dim() == 5:
        pass
    else:
        raise ValueError(
            f"decode_video expects packed [B, L, D] or unpacked [B, C, F, H, W]; got shape {tuple(latents.shape)}"
        )
    latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents * latents_std / vae.config.scaling_factor + latents_mean
    if decode_noise_scale is None:
        decode_noise_scale = decode_timestep
    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
    decode_noise_scale = torch.tensor([decode_noise_scale], device=device, dtype=latents.dtype).view(1, 1, 1, 1, 1)
    latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise
    timestep = torch.tensor([decode_timestep], device=device, dtype=latents.dtype)
    video = vae.decode(latents, timestep, return_dict=False)[0]
    video *= 0.5
    video += 0.5
    return video.to(dtype=dtype) if dtype is not None else video


def get_rope_scale_factors(fps: float) -> list[float]:
    if fps <= 0:
        raise ValueError("FPS must be a positive number.")
    temporal_compression_ratio = 8.0
    spatial_compression_ratio = 32.0
    return [
        temporal_compression_ratio / fps,
        spatial_compression_ratio,
        spatial_compression_ratio,
    ]
