"""Contract for DriveLaW-Act: world latents produced by DualFlow encoders (reference shapes only)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldLatentShapes:
    """Expected tensor layouts when exchanging `z_bridge` / `z_vae` with planning."""

    # Packed sequence form (matches LTX transformer after pack_latents, patch 1×1×1)
    z_bridge_packed: tuple[int, int, int]  # (B, L, C_bridge)
    z_vae_packed: tuple[int, int, int]  # (B, L, C_vae)
    # Unpacked VAE latent grid (before pack), for custom decoders
    z_vae_grid: tuple[int, int, int, int, int]  # (B, C_vae, T_lat, H_lat, W_lat)
    z_bridge_grid: tuple[int, int, int, int, int]  # (B, C_bridge, T_lat, H_lat, W_lat)


def world_latent_shapes(
    *,
    batch: int,
    total_rgb_frames: int,
    video_height: int,
    video_width: int,
    c_bridge: int,
    c_vae: int,
    temporal_compression: int = 8,
    spatial_compression: int = 32,
) -> WorldLatentShapes:
    t_lat = (total_rgb_frames - 1) // temporal_compression + 1
    h_lat = video_height // spatial_compression
    w_lat = video_width // spatial_compression
    seq = t_lat * h_lat * w_lat
    return WorldLatentShapes(
        z_bridge_packed=(batch, seq, c_bridge),
        z_vae_packed=(batch, seq, c_vae),
        z_vae_grid=(batch, c_vae, t_lat, h_lat, w_lat),
        z_bridge_grid=(batch, c_bridge, t_lat, h_lat, w_lat),
    )
