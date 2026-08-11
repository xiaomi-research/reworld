"""Minimal dataset for vae_only DiT smoke tests."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class DualLatentDummyDataset(Dataset):
    """Random packed VAE latents with correct shapes for vae_only training."""

    def __init__(
        self,
        *,
        batch_items: int,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        c_vae: int,
        fps: float = 24.0,
    ) -> None:
        self.batch_items = batch_items
        self.latent_frames = latent_frames
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.c_vae = c_vae
        self.fps = fps
        self.seq_len = latent_frames * latent_height * latent_width

    def __len__(self) -> int:
        return self.batch_items

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | float]:
        return {
            "pix_latents": torch.randn(self.seq_len, self.c_vae),
            "num_frames": self.latent_frames,
            "height": self.latent_height,
            "width": self.latent_width,
            "fps": self.fps,
        }
