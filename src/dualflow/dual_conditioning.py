"""Conditioning mask on packed latent tokens (first K latent frames; no skip_condition_frames)."""

from __future__ import annotations

import random

import torch
from torch import Tensor


def rgb_frames_to_latent_frames(num_rgb_frames: int, temporal_compression: int = 8) -> int:
    if num_rgb_frames < 1:
        return 0
    return (num_rgb_frames - 1) // temporal_compression + 1


def conditioning_mask_packed(
    batch_size: int,
    seq_len: int,
    num_latent_frames_to_condition: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
) -> Tensor:
    """True = conditioning token (keep clean, zero timestep in DiT, mask flow/REPA)."""
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    per_t = latent_height * latent_width
    n_tokens = min(seq_len, max(0, num_latent_frames_to_condition) * per_t)
    mask[:, :n_tokens] = True
    return mask


def maybe_conditioning_mask(
    *,
    batch_size: int,
    seq_len: int,
    condition_source_rgb_frames: int,
    latent_height: int,
    latent_width: int,
    temporal_compression: int,
    first_frame_conditioning_p: float,
    device: torch.device,
) -> Tensor:
    """With probability `first_frame_conditioning_p`, mark first latent frames (from condition RGB count)."""
    if first_frame_conditioning_p <= 0.0 or random.random() > first_frame_conditioning_p:
        return torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    lf = rgb_frames_to_latent_frames(condition_source_rgb_frames, temporal_compression)
    return conditioning_mask_packed(
        batch_size,
        seq_len,
        lf,
        latent_height,
        latent_width,
        device,
    )


def expand_timesteps_with_conditioning(
    conditioning_mask: Tensor,
    sampled_timestep_values: Tensor,
) -> Tensor:
    """Match `training_strategies.TrainingStrategy._create_timesteps_from_conditioning_mask`."""
    expanded = sampled_timestep_values.unsqueeze(1).expand_as(conditioning_mask)
    zeros = torch.zeros_like(expanded, dtype=sampled_timestep_values.dtype)
    return torch.where(conditioning_mask, zeros, expanded)
