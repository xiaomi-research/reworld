"""MP4 sampling for vae_only DualFlow DiT with optional conditioning and IG."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from dualflow import logger
from dualflow.async_dualflow_inference import run_vae_latent_sampling
from dualflow.dual_conditioning import conditioning_mask_packed, rgb_frames_to_latent_frames
from dualflow.internal_guidance import load_optional_ig_head_linear
from dualflow.ltx.latents import decode_video
from dualflow.ltx.model_loader import load_vae
from dualflow.model_utils import (
    load_ltx_transformer_from_json_channel_override,
    load_ltx_transformer_weights,
    load_ltx_transformer_weights_prefixed,
)
from dualflow.sample_vae_resources import load_and_encode_condition_anchor
from dualflow.train_configs import DualFlowSampleConfig, resolve_ig_sampling_block, resolve_ig_train_blocks

_WINDOW000_RE = re.compile(r"^scene_(\d+)_window_000_conditioning\.mp4$")


def _load_dit_checkpoint(model: torch.nn.Module, checkpoint: str) -> None:
    blob = load_file(checkpoint)
    if any(k.startswith("transformer.") for k in blob):
        load_ltx_transformer_weights_prefixed(model, checkpoint, "transformer.")
    elif any(k.startswith("transformer_joint.") for k in blob):
        load_ltx_transformer_weights_prefixed(model, checkpoint, "transformer_joint.")
    else:
        load_ltx_transformer_weights(model, checkpoint)


def _list_window000_videos(cond_dir: Path) -> list[Path]:
    return sorted(p for p in cond_dir.iterdir() if p.is_file() and _WINDOW000_RE.match(p.name))


def _resolve_conditioning_paths(cfg: DualFlowSampleConfig) -> list[Path]:
    cond_dir = (cfg.conditioning_dir or "").strip()
    if cond_dir:
        d = Path(cond_dir)
        if not d.is_dir():
            raise FileNotFoundError(f"conditioning_dir not found: {d}")
        paths = _list_window000_videos(d)
        if cfg.max_scenes > 0:
            paths = paths[: cfg.max_scenes]
        if not paths:
            raise FileNotFoundError(f"No scene_*_window_000_conditioning.mp4 under {d}")
        return paths
    return [Path(p) for p in cfg.sample_conditioning_video_paths if str(p).strip()]


def _output_paths_for_conditioning(base: Path, n: int) -> list[Path]:
    if n <= 0:
        return []
    if n == 1:
        return [base]
    parent = base.parent
    stem = base.stem
    suf = base.suffix
    return [parent / f"{stem}_cond_{i:02d}{suf}" for i in range(n)]


def _save_frames_png(frames_fhwc: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(frames_fhwc.shape[0]):
        rgb = (frames_fhwc[i] * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(out_dir / f"frame_{i:05d}.png")


def _export_mp4(frames_fhwc: np.ndarray, path: Path, fps: float) -> None:
    from diffusers.utils import export_to_video

    path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames_fhwc, str(path), fps=int(fps))


def _build_ig_kwargs(dit: torch.nn.Module, cfg: DualFlowSampleConfig, device, dtype) -> dict:
    from dualflow.ltx.transformer_ltx_ig import (
        bootstrap_ig_heads_from_checkpoint,
        is_ltx_transformer_with_native_ig,
    )

    ig_sampling_block = resolve_ig_sampling_block(cfg)
    ig_head_blocks = sorted({*resolve_ig_train_blocks(cfg), ig_sampling_block})

    if is_ltx_transformer_with_native_ig(dit):
        bootstrap_ig_heads_from_checkpoint(
            dit,
            cfg.dit_checkpoint,
            ig_head_blocks,
            legacy_block_idx=ig_sampling_block,
        )
        logger.info(
            "sample_dualflow: IG heads ready (blocks=%s, sampling_block=%s).",
            ig_head_blocks,
            ig_sampling_block,
        )

    if abs(float(cfg.internal_guidance_sampling_scale) - 1.0) <= 1e-6:
        return {}

    if is_ltx_transformer_with_native_ig(dit):
        return {
            "ig_head": None,
            "ig_block_idx": int(ig_sampling_block),
            "ig_scale": float(cfg.internal_guidance_sampling_scale),
            "ig_min_timestep": cfg.internal_guidance_sampling_min_timestep,
        }
    ig_head_loaded = load_optional_ig_head_linear(cfg.dit_checkpoint, device=device, dtype=dtype)
    if ig_head_loaded is None:
        logger.warning(
            "sample_dualflow: internal_guidance_sampling_scale=%s but no IG head in checkpoint — vanilla.",
            cfg.internal_guidance_sampling_scale,
        )
        return {}
    return {
        "ig_head": ig_head_loaded,
        "ig_block_idx": int(ig_sampling_block),
        "ig_scale": float(cfg.internal_guidance_sampling_scale),
        "ig_min_timestep": cfg.internal_guidance_sampling_min_timestep,
    }


def sample_dualflow_async(cfg: DualFlowSampleConfig) -> list[Path]:
    """Run vae_only sampling from a YAML-backed ``DualFlowSampleConfig``; returns output paths."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32

    nf = cfg.latent_frames
    nh = cfg.latent_height
    nw = cfg.latent_width

    dit = load_ltx_transformer_from_json_channel_override(
        cfg.transformer_config,
        in_channels=cfg.c_vae,
        out_channels=cfg.c_vae,
        torch_dtype=dtype,
    ).to(device).eval()
    _load_dit_checkpoint(dit, cfg.dit_checkpoint)
    ig_kw = _build_ig_kwargs(dit, cfg, device, dtype)
    dit.to(device)

    vae = load_vae(cfg.vae_model_source, dtype=dtype)
    vae.to(device).eval()

    cond_paths = _resolve_conditioning_paths(cfg)
    use_cond = len(cond_paths) > 0
    batch_root = (cfg.output_root or "").strip()
    outs: list[Path] = []

    if use_cond:
        lf = rgb_frames_to_latent_frames(cfg.condition_source_frames, cfg.temporal_compression)
        cond_mask_base = conditioning_mask_packed(1, nf * nh * nw, lf, nh, nw, device)
        future_n = cfg.total_source_frames - cfg.condition_source_frames

        if batch_root:
            out_root = Path(batch_root)
            vid_dir = out_root / "videos"
            frame_root = out_root / "frames"
            vid_dir.mkdir(parents=True, exist_ok=True)
            frame_root.mkdir(parents=True, exist_ok=True)
        else:
            target_paths = _output_paths_for_conditioning(Path(cfg.output_video_path), len(cond_paths))
            wi = 0

        for ci, ref in enumerate(cond_paths):
            if not ref.is_file():
                logger.warning("sample_dualflow: conditioning video not found: %s — skip.", ref)
                continue
            try:
                z_pix_a, fps_rd = load_and_encode_condition_anchor(cfg, ref, vae, device, dtype)
                sample_fps = float(fps_rd) if fps_rd > 0 else float(cfg.fps)

                if batch_root:
                    stem = ref.stem.replace("_conditioning", "")
                    mp4_33 = vid_dir / f"{stem}_gen33.mp4"
                    mp4_fut = vid_dir / f"{stem}_future{future_n}.mp4"
                    frames_dir = frame_root / stem
                    z_pix = run_vae_latent_sampling(
                        dit,
                        vae,
                        c_vae=cfg.c_vae,
                        nf=nf,
                        nh=nh,
                        nw=nw,
                        num_inference_steps=cfg.num_inference_steps,
                        vae_model_source=cfg.vae_model_source,
                        fps=sample_fps,
                        caption_channels=int(dit.config.caption_channels),
                        null_text_seq_len=cfg.null_text_seq_len,
                        seed=cfg.seed + ci * 17_389,
                        device=device,
                        dtype=dtype,
                        output_video_path=None,
                        conditioning_mask=cond_mask_base,
                        z_pix_anchor=z_pix_a,
                        **ig_kw,
                    )
                    assert isinstance(z_pix, torch.Tensor)
                    vid = decode_video(
                        vae, z_pix, num_frames=nf, height=nh, width=nw, device=device, dtype=dtype
                    )
                    if vid.dim() == 5:
                        vid = vid[0]
                    vid_fhwc = (
                        vid.permute(1, 2, 3, 0).contiguous().clamp(0, 1).float().cpu().detach().numpy()
                    )
                    future = vid_fhwc[cfg.condition_source_frames :]
                    _export_mp4(vid_fhwc, mp4_33, sample_fps)
                    _export_mp4(future, mp4_fut, sample_fps)
                    _save_frames_png(future, frames_dir)
                    outs.append(mp4_33)
                    logger.info("sample_dualflow: batch ok %s -> %s", stem, mp4_33)
                else:
                    vid_out = target_paths[wi]
                    p = run_vae_latent_sampling(
                        dit,
                        vae,
                        c_vae=cfg.c_vae,
                        nf=nf,
                        nh=nh,
                        nw=nw,
                        num_inference_steps=cfg.num_inference_steps,
                        vae_model_source=cfg.vae_model_source,
                        fps=sample_fps,
                        caption_channels=int(dit.config.caption_channels),
                        null_text_seq_len=cfg.null_text_seq_len,
                        seed=cfg.seed + ci * 17_389,
                        device=device,
                        dtype=dtype,
                        output_video_path=vid_out,
                        conditioning_mask=cond_mask_base,
                        z_pix_anchor=z_pix_a,
                        **ig_kw,
                    )
                    outs.append(Path(p))
                    wi += 1
            except Exception as e:
                logger.warning("sample_dualflow: conditional sample failed for %s (%s).", ref, e)
    else:
        p = run_vae_latent_sampling(
            dit,
            vae,
            c_vae=cfg.c_vae,
            nf=nf,
            nh=nh,
            nw=nw,
            num_inference_steps=cfg.num_inference_steps,
            vae_model_source=cfg.vae_model_source,
            fps=cfg.fps,
            caption_channels=int(dit.config.caption_channels),
            null_text_seq_len=cfg.null_text_seq_len,
            seed=cfg.seed,
            device=device,
            dtype=dtype,
            output_video_path=cfg.output_video_path,
            **ig_kw,
        )
        outs.append(Path(p))
    return outs
