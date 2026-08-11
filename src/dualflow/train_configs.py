"""YAML-friendly configs for DualFlow vae_only DiT training and sampling."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def resolve_ig_train_blocks(cfg: BaseModel) -> list[int]:
    """DiT block indices with per-block IG heads used during training."""
    blocks = list(getattr(cfg, "internal_guidance_dit_blocks", None) or [])
    if blocks:
        return sorted({int(b) for b in blocks})
    return [int(getattr(cfg, "internal_guidance_dit_block", 1))]


def resolve_ig_sampling_block(cfg: BaseModel) -> int:
    """DiT block index used for inference-time IG extrapolation."""
    sb = getattr(cfg, "internal_guidance_sampling_block", None)
    if sb is not None:
        return int(sb)
    return max(resolve_ig_train_blocks(cfg))


class DualFlowCheckpointConfig(ConfigBaseModel):
    """Intermediate checkpoints during stage-2 training (LTX ``checkpoints``-style)."""

    interval: int = Field(
        default=0,
        ge=0,
        description=(
            "Save ``dualflow_dit_step_XXXXXXX.safetensors`` every N optimizer steps. "
            "0 = only write ``dualflow_dit.safetensors`` at the end of training."
        ),
    )
    keep_last_n: int = Field(
        default=-1,
        ge=-1,
        description="Keep only the N newest step checkpoints on disk; -1 keeps all ``step_*`` files.",
    )
    save_best_loss: bool = Field(
        default=False,
        description=(
            "If True, save ``dualflow_dit_best_loss.safetensors`` when the training loss metric improves "
            "(same reduced mean loss as logs). Overwrites in place; not pruned by ``keep_last_n``."
        ),
    )
    best_loss_ema_decay: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "EMA decay for the loss used to pick best (0 = compare raw step loss; e.g. 0.99 smooths spikes)."
        ),
    )


class DualFlowValidationTrainConfig(ConfigBaseModel):
    """Periodic validation loss on a held-out video folder (same preprocessing as training)."""

    interval: int = Field(
        default=0,
        ge=0,
        description="Run validation (average FM+IG loss on val clips) every N steps. 0 = disabled.",
    )
    video_root: str | None = Field(
        default=None,
        description="Directory of validation videos (same layout as training ``video_root``). Required if ``interval`` > 0.",
    )
    num_batches: int = Field(default=1, ge=1, description="Number of val batches to average each run.")

    @model_validator(mode="after")
    def _coerce_validation_when_no_video_root(self) -> DualFlowValidationTrainConfig:
        if self.interval > 0 and not self.video_root:
            warnings.warn(
                "validation.interval > 0 but validation.video_root is unset; "
                "disabling periodic validation (interval coerced to 0). "
                "Set validation.video_root to a folder of validation videos to enable.",
                UserWarning,
                stacklevel=2,
            )
            return self.model_copy(update={"interval": 0})
        return self


class DualFlowDiTTrainConfig(ConfigBaseModel):
    """Stage 2: LTX DiT on VAE latents only with optional Internal Guidance."""

    dit_latent_mode: Literal["vae_only"] = Field(
        default="vae_only",
        description="Only ``z_pix`` VAE latents (``in_channels = out_channels = c_vae``).",
    )
    transformer_config: str = Field(
        description=(
            "LTX ``config.json`` path. For ``vae_only``, I/O channels are overridden to ``c_vae`` at load time."
        ),
    )
    transformer_weights: str | None = Field(
        default=None,
        description=(
            "Optional LTX DiT init (``.safetensors`` with keys ``transformer.*``, legacy ``transformer_joint.*``, "
            "or a flat state dict). Applied after the model is built with I/O = ``c_vae``."
        ),
    )

    vae_model_source: str = Field(default="LTXV_2B_0.9.5")
    c_vae: int = Field(default=128)

    transport_use_lognorm: bool = Field(
        default=True,
        description=(
            "Match SFD ``transport.use_lognorm``: if True, sample base t via logit-normal "
            "(Gaussian + sigmoid); if False, uniform on [0,1]."
        ),
    )
    transport_train_eps: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Reserved for SFD ``train_eps`` interval; velocity+LINEAR SFD training keeps [0,1].",
    )
    transport_lognorm_mu: float = Field(default=0.0, description="SFD logit-normal mu when shift_lg is False.")
    transport_lognorm_sigma: float = Field(default=1.0, gt=0.0, description="SFD logit-normal sigma.")
    transport_shift_lg: bool = Field(
        default=False,
        description="If True, use shifted_mu for logit-normal (SFD transport.shift_lg).",
    )
    transport_shifted_mu: float = Field(default=0.0, description="SFD shifted log-normal mean when shift_lg is True.")

    lambda_internal_guidance: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Internal Guidance auxiliary loss (IG; arXiv:2512.24176): MSE of intermediate velocity vs "
            "the same flow target as the final head. 0 = no IG loss."
        ),
    )
    lambda_internal_guidance_warmup_steps: int = Field(default=0, ge=0)
    internal_guidance_dit_blocks: list[int] = Field(
        default_factory=list,
        description=(
            "``transformer_blocks`` indices for IG training; each block has its own IG head. "
            "IG losses are averaged across blocks. If empty, uses ``internal_guidance_dit_block``."
        ),
    )
    internal_guidance_dit_block: int = Field(
        default=1,
        ge=0,
        description="Single-block IG index when ``internal_guidance_dit_blocks`` is empty.",
    )
    internal_guidance_sampling_block: int | None = Field(
        default=None,
        ge=0,
        description=(
            "``transformer_blocks`` index for inference IG extrapolation. "
            "``None`` = ``max(internal_guidance_dit_blocks)`` (deepest training block)."
        ),
    )
    internal_guidance_sampling_scale: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Inference-time IG extrapolation strength w from Eq. (5): "
            "pred_w = pred_i + w*(pred_f - pred_i). 1.0 disables extrapolation."
        ),
    )
    internal_guidance_sampling_min_timestep: float | None = Field(
        default=None,
        description=(
            "Guidance-interval style IG: when set, force w=1 on scheduler steps "
            "with timestep value below this threshold."
        ),
    )

    cross_attention_dim: int = Field(default=2048)
    null_text_seq_len: int = Field(default=1)

    learning_rate: float = Field(default=1e-5)
    max_grad_norm: float = Field(
        default=0.0,
        ge=0.0,
        description="If > 0, clip total gradient norm of DiT before optimizer.step. 0 = disabled.",
    )
    steps: int = Field(default=1000)
    batch_size: int = Field(default=1)
    output_dir: str = Field(default="outputs/dualflow_dit")
    seed: int = Field(default=42)
    dtype: Literal["bf16", "fp32"] = Field(default="bf16")

    use_dummy_data: bool = Field(default=False)
    video_root: str | None = Field(default=None)
    total_source_frames: int = Field(
        default=33,
        ge=1,
        description=(
            "Number of RGB frames per clip for training, validation, and training-time samples. "
            "``latent_frames = (total_source_frames-1)//temporal_compression+1``."
        ),
    )
    video_width: int = Field(default=512, ge=32)
    video_height: int = Field(default=512, ge=32)

    condition_source_frames: int = Field(
        default=9,
        ge=1,
        description="Leading RGB frames used to define conditioned latent tokens.",
    )
    first_frame_conditioning_p: float = Field(default=1.0, ge=0.0, le=1.0)
    temporal_compression: int = Field(default=8, ge=1)
    spatial_compression: int = Field(
        default=32,
        ge=1,
        description="Latent H/W vs video (must match LTX VAE).",
    )

    num_dataloader_workers: int = Field(default=4, ge=0)

    sample_every_n_steps: int = Field(
        default=0,
        ge=0,
        description=(
            "If >0 and main process: every N steps run vae_only sampling and save MP4 under output_dir/samples/. "
            "0 = disabled."
        ),
    )
    sample_num_inference_steps: int = Field(default=30, ge=2)
    train_sample_fps: float = Field(
        default=24.0,
        gt=0.0,
        description="FPS for RoPE + MP4 export when not using a conditioning file.",
    )
    sample_conditioning_video_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Optional conditioning MP4s for periodic training samples. Only the leading "
            "``condition_source_frames`` RGB frames are encoded as anchors."
        ),
    )

    save_loss_csv: bool = Field(
        default=True,
        description="If True (main process): record per-step loss breakdown to CSV under output_dir.",
    )
    save_loss_plot: bool = Field(
        default=True,
        description="If True: save PNG of total training loss vs step when the CSV is flushed.",
    )
    save_training_config_record: bool = Field(
        default=True,
        description="If True (main process): write resolved training config to JSON under output_dir at train start.",
    )
    training_config_record_filename: str = Field(
        default="dualflow_dit_train_config.json",
        description="Filename (under output_dir) for the training config snapshot.",
    )
    loss_csv_filename: str = Field(default="dualflow_dit_loss.csv")
    loss_plot_filename: str = Field(default="dualflow_dit_loss.png")

    log_every_n_steps: int = Field(
        default=100,
        ge=1,
        description="Log training loss on the main process every N steps.",
    )
    checkpoints: DualFlowCheckpointConfig = Field(
        default_factory=DualFlowCheckpointConfig,
        description="Periodic checkpoints (LTX ``checkpoints.interval`` / ``keep_last_n``).",
    )

    validation: DualFlowValidationTrainConfig = Field(
        default_factory=DualFlowValidationTrainConfig,
        description="Periodic validation loss on a separate folder.",
    )

    @model_validator(mode="after")
    def _validate_online(self) -> DualFlowDiTTrainConfig:
        if self.validation.interval > 0 and self.use_dummy_data:
            raise ValueError("validation.interval > 0 is not compatible with use_dummy_data=true (no val clips).")
        if not self.use_dummy_data and not self.video_root:
            raise ValueError("video_root is required when use_dummy_data=false.")
        tc_path = (self.transformer_config or "").strip()
        if tc_path:
            p = Path(tc_path).expanduser()
            if p.is_file():
                try:
                    tcd = json.loads(p.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass
                else:
                    ic = tcd.get("in_channels")
                    oc = tcd.get("out_channels")
                    if ic is not None and oc is not None:
                        exp = int(self.c_vae)
                        if int(ic) != exp or int(oc) != exp:
                            warnings.warn(
                                f"transformer_config ({p}): in_channels/out_channels are {ic}/{oc} but "
                                f"vae_only overrides to c_vae={exp} at load time.",
                                UserWarning,
                                stacklevel=2,
                            )
        ig_blocks = resolve_ig_train_blocks(self)
        if self.lambda_internal_guidance > 0.0 and not ig_blocks:
            raise ValueError("lambda_internal_guidance > 0 requires at least one IG training block.")
        return self

    @property
    def latent_width(self) -> int:
        return self.video_width // self.spatial_compression

    @property
    def latent_height(self) -> int:
        return self.video_height // self.spatial_compression

    @property
    def latent_frames(self) -> int:
        return (self.total_source_frames - 1) // self.temporal_compression + 1


class DualFlowSampleConfig(ConfigBaseModel):
    """MP4 sampling for vae_only DiT with optional conditioning videos and IG extrapolation."""

    dit_latent_mode: Literal["vae_only"] = Field(
        default="vae_only",
        description="VAE-latent-only DiT (``c_vae`` I/O channels).",
    )
    transformer_config: str
    dit_checkpoint: str
    vae_model_source: str = "LTXV_2B_0.9.5"

    c_vae: int = 128
    video_width: int = 512
    video_height: int = 512
    total_source_frames: int = 33
    temporal_compression: int = Field(8, ge=1)
    spatial_compression: int = Field(32, ge=1)
    condition_source_frames: int = Field(
        default=9,
        ge=1,
        description="Leading RGB frames → conditioned latent tokens.",
    )
    sample_conditioning_video_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit conditioning MP4 paths. Only leading ``condition_source_frames`` are used. "
            "Ignored when ``conditioning_dir`` is set."
        ),
    )
    conditioning_dir: str | None = Field(
        default=None,
        description=(
            "If set, discover ``scene_*_window_000_conditioning.mp4`` under this folder "
            "(batch inference). Takes precedence over ``sample_conditioning_video_paths``."
        ),
    )
    max_scenes: int = Field(
        default=0,
        ge=0,
        description="When using ``conditioning_dir``, cap number of scenes (0 = all).",
    )
    output_root: str | None = Field(
        default=None,
        description=(
            "Batch output root: writes ``videos/`` and ``frames/``. "
            "If unset, uses ``output_video_path`` for a single/list export."
        ),
    )

    cross_attention_dim: int = 2048
    null_text_seq_len: int = 1
    num_inference_steps: int = 30
    dtype: Literal["bf16", "fp32"] = "bf16"
    seed: int = 42
    internal_guidance_dit_blocks: list[int] = Field(
        default_factory=list,
        description="IG training blocks (must match Stage-1). Empty -> ``internal_guidance_dit_block``.",
    )
    internal_guidance_dit_block: int = Field(
        default=1,
        ge=0,
        description="Single IG block when ``internal_guidance_dit_blocks`` is empty.",
    )
    internal_guidance_sampling_block: int | None = Field(
        default=None,
        ge=0,
        description="IG extrapolation block at inference. ``None`` = deepest training block.",
    )
    internal_guidance_sampling_scale: float = Field(
        default=1.0,
        ge=0.0,
        description="IG extrapolation w (Eq. 5): pred_w = pred_i + w*(pred_f - pred_i). 1.0 = vanilla sampling.",
    )
    internal_guidance_sampling_min_timestep: float | None = Field(
        default=None,
        description="Optional: apply scale w only when scheduler timestep ≥ this value.",
    )
    output_video_path: str = "outputs/dualflow_sample.mp4"
    fps: float = Field(
        default=8.0,
        gt=0.0,
        description="FPS for RoPE + MP4 export.",
    )

    @model_validator(mode="after")
    def _validate_sample_sources(self) -> DualFlowSampleConfig:
        has_dir = bool((self.conditioning_dir or "").strip())
        has_paths = any(str(p).strip() for p in self.sample_conditioning_video_paths)
        if not has_dir and not has_paths:
            # Unconditional sampling is allowed (noise-only).
            return self
        return self

    @property
    def latent_width(self) -> int:
        return self.video_width // self.spatial_compression

    @property
    def latent_height(self) -> int:
        return self.video_height // self.spatial_compression

    @property
    def latent_frames(self) -> int:
        return (self.total_source_frames - 1) // self.temporal_compression + 1
