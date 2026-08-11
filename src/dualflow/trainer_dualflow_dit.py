"""Stage-2 trainer: vae_only LTX DiT with flow matching and optional Internal Guidance."""

from __future__ import annotations

import contextlib
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dualflow import logger
from dualflow.async_dualflow_inference import run_vae_latent_sampling
from dualflow.datasets_dual import DualLatentDummyDataset
from dualflow.dual_conditioning import (
    conditioning_mask_packed,
    expand_timesteps_with_conditioning,
    maybe_conditioning_mask,
    rgb_frames_to_latent_frames,
)
from dualflow.ltx.latents import encode_video, get_rope_scale_factors
from dualflow.ltx.model_loader import load_vae
from dualflow.model_utils import (
    load_ltx_transformer_from_json_channel_override,
    load_ltx_transformer_weights,
    load_ltx_transformer_weights_prefixed,
    null_text_embeddings,
    safetensors_torch_load_device,
)
from dualflow.online_dataset import OnlineVideoFolderDataset
from dualflow.timestep_async import (
    continuous_t_to_sigma,
    continuous_t_to_timestep_id,
    sample_t_base_continuous,
)
from dualflow.train_configs import DualFlowDiTTrainConfig, resolve_ig_sampling_block, resolve_ig_train_blocks


def _dual_collate_vae_only(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pix_latents": torch.stack([b["pix_latents"] for b in batch], dim=0),
        "num_frames": int(batch[0]["num_frames"]),
        "height": int(batch[0]["height"]),
        "width": int(batch[0]["width"]),
        "fps": float(batch[0]["fps"]),
    }


def _online_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
        "fps": float(batch[0]["fps"]),
    }


def _load_stage2_transformer_weights(model: torch.nn.Module, path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"transformer_weights not found: {path}")
    if str(p).endswith((".safetensors", ".sft")):
        blob = load_file(str(p))
        if any(k.startswith("transformer.") for k in blob):
            load_ltx_transformer_weights_prefixed(model, p, "transformer.")
        elif any(k.startswith("transformer_joint.") for k in blob):
            load_ltx_transformer_weights_prefixed(model, p, "transformer_joint.")
        else:
            load_ltx_transformer_weights(model, p)
    else:
        load_ltx_transformer_weights(model, p)


def _flow_mse_masked(pred: torch.Tensor, target: torch.Tensor, conditioning_mask: torch.Tensor) -> torch.Tensor:
    """Average MSE on tokens where conditioning_mask is False."""
    m = (~conditioning_mask).float().unsqueeze(-1)
    err = (pred - target).pow(2) * m
    denom = m.sum() * pred.shape[-1] + 1e-8
    return err.sum() / denom


_DIT_LOSS_CSV_FIELDS = (
    "step",
    "loss",
    "loss_fm",
    "loss_ig",
    "lambda_ig_eff",
    "loss_ig_weighted",
)


def _reduce_scalar_for_log(accelerator: Accelerator, t: torch.Tensor) -> float:
    x = t.detach()
    if x.ndim > 0:
        x = x.mean()
    if accelerator.num_processes <= 1:
        return float(x.item())
    red = getattr(accelerator, "reduce", None)
    if red is None:
        return float(x.item())
    return float(red(x, reduction="mean").item())


def _write_dualflow_dit_training_config_record(out_dir: Path, cfg: DualFlowDiTTrainConfig) -> None:
    name = cfg.training_config_record_filename.strip() or "dualflow_dit_train_config.json"
    path = out_dir / name
    payload: dict[str, Any] = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg.model_dump(mode="json"),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("dualflow_dit: wrote training config record -> %s", path)


def _write_dualflow_dit_loss_artifacts(
    out_dir: Path,
    rows: list[dict[str, int | float | str]],
    *,
    save_plot: bool,
    csv_name: str,
    plot_name: str,
) -> None:
    if not rows:
        return
    csv_path = out_dir / csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_DIT_LOSS_CSV_FIELDS), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out_row = {k: row[k] if k in row else "" for k in _DIT_LOSS_CSV_FIELDS}
            w.writerow(out_row)
    if not save_plot:
        return
    png_path = out_dir / plot_name
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [int(r["step"]) for r in rows]
        losses = [float(r["loss"]) for r in rows]
        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        ax.plot(steps, losses, linewidth=0.9, color="#2563eb")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("DualFlow vae_only DiT training loss")
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)
    except Exception as e:
        logger.warning("dualflow_dit: could not save loss plot to %s: %s", png_path, e)


def _lambda_warmup(step: int, warmup: int, final: float) -> float:
    if warmup <= 0:
        return final
    if step >= warmup:
        return final
    return final * float(step) / float(max(1, warmup))


class DualFlowDiTTrainer:
    def __init__(self, cfg: DualFlowDiTTrainConfig) -> None:
        self.cfg = cfg
        self._accelerator = Accelerator(
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        )
        self.device = self._accelerator.device
        dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32
        self.dtype = dtype

        self.transformer = load_ltx_transformer_from_json_channel_override(
            cfg.transformer_config,
            in_channels=cfg.c_vae,
            out_channels=cfg.c_vae,
            torch_dtype=dtype,
        )
        if cfg.transformer_weights:
            _load_stage2_transformer_weights(self.transformer, cfg.transformer_weights)

        from dualflow.ltx.transformer_ltx_ig import (
            LTXVideoTransformer3DModelWithIG,
            bootstrap_ig_heads_from_checkpoint,
        )

        self._ig_train_blocks = resolve_ig_train_blocks(cfg)
        self._ig_sampling_block = resolve_ig_sampling_block(cfg)
        ig_head_blocks = sorted({*self._ig_train_blocks, self._ig_sampling_block})

        if isinstance(self.transformer, LTXVideoTransformer3DModelWithIG):
            _wpath = (cfg.transformer_weights or "").strip() or None
            _aux, _leg = bootstrap_ig_heads_from_checkpoint(
                self.transformer,
                _wpath,
                ig_head_blocks,
                legacy_block_idx=self._ig_sampling_block,
            )
            if self._accelerator.is_main_process:
                if _aux:
                    logger.info(
                        "dualflow_dit: loaded per-block IG heads from checkpoint blocks=%s",
                        ig_head_blocks,
                    )
                elif _leg:
                    logger.info(
                        "dualflow_dit: migrated legacy ig_head.* to block %s.",
                        self._ig_sampling_block,
                    )
                else:
                    logger.info(
                        "dualflow_dit: initialized IG heads from main (blocks=%s).",
                        ig_head_blocks,
                    )
            if cfg.lambda_internal_guidance <= 0.0:
                for head in self.transformer.ig_heads.values():
                    for p in head.parameters():
                        p.requires_grad_(False)
                if self._accelerator.is_main_process:
                    logger.info(
                        "dualflow_dit: lambda_internal_guidance=0 -> freeze all IG head params for DDP-safe training."
                    )

        t_lat = cfg.latent_frames
        h_lat = cfg.latent_height
        w_lat = cfg.latent_width

        self._vae = load_vae(cfg.vae_model_source, dtype=torch.bfloat16 if cfg.dtype == "bf16" else torch.float32)
        self._vae.to(self.device, dtype=self._vae.dtype).eval()

        params = [p for p in self.transformer.parameters() if p.requires_grad]
        self._optimizer = AdamW(params, lr=cfg.learning_rate)

        if cfg.use_dummy_data:
            ds: Any = DualLatentDummyDataset(
                batch_items=max(cfg.steps, 8) * cfg.batch_size,
                latent_frames=t_lat,
                latent_height=h_lat,
                latent_width=w_lat,
                c_vae=cfg.c_vae,
                fps=24.0,
            )
            collate_fn = _dual_collate_vae_only
        else:
            assert cfg.video_root
            ds = OnlineVideoFolderDataset(
                cfg.video_root,
                total_source_frames=cfg.total_source_frames,
                video_width=cfg.video_width,
                video_height=cfg.video_height,
            )
            collate_fn = _online_collate

        self._loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=cfg.num_dataloader_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        preped = list(self._accelerator.prepare(self.transformer, self._optimizer, self._loader))
        self.transformer = preped[0]
        self._optimizer = preped[1]
        self._loader = preped[2]

        self._val_loader: DataLoader | None = None
        vc = cfg.validation
        if not cfg.use_dummy_data and vc.interval > 0 and vc.video_root:
            vds = OnlineVideoFolderDataset(
                vc.video_root,
                total_source_frames=cfg.total_source_frames,
                video_width=cfg.video_width,
                video_height=cfg.video_height,
            )
            self._val_loader = DataLoader(
                vds,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=_online_collate,
                num_workers=min(cfg.num_dataloader_workers, 2),
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )

        self._clip_grad_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self._best_loss_metric = float("inf")
        self._best_loss_ema: float | None = None

        if self._accelerator.is_main_process and cfg.lambda_internal_guidance > 0.0:
            logger.info(
                "dualflow_dit: internal guidance — train_blocks=%s sampling_block=%s lambda=%s C_vel=%s",
                self._ig_train_blocks,
                self._ig_sampling_block,
                cfg.lambda_internal_guidance,
                cfg.c_vae,
            )

    def _dit_forward_velocity(
        self,
        z_in: torch.Tensor,
        enc: torch.Tensor,
        mask: torch.Tensor,
        t_seq: torch.Tensor,
        num_frames_dit: int,
        nh: int,
        nw: int,
        rope: Any,
        lam_ig: float,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor] | None]:
        """DiT forward; when ``lam_ig > 0``, returns ``(pred_final, {block: pred_ig})`` in one pass."""
        raw_tm = self._accelerator.unwrap_model(self.transformer)
        kw: dict[str, Any] = {
            "hidden_states": z_in,
            "encoder_hidden_states": enc,
            "encoder_attention_mask": mask,
            "timestep": t_seq,
            "num_frames": num_frames_dit,
            "height": nh,
            "width": nw,
            "rope_interpolation_scale": rope,
            "video_coords": None,
            "return_dict": False,
        }
        if lam_ig > 0.0:
            from dualflow.ltx.transformer_ltx_ig import is_ltx_transformer_with_native_ig

            if not is_ltx_transformer_with_native_ig(raw_tm):
                raise RuntimeError(
                    "lambda_internal_guidance > 0 requires LTXVideoTransformer3DModelWithIG (dualflow.model_utils)."
                )
            kw["return_intermediate_velocities"] = True
            kw["intermediate_block_indices"] = list(self._ig_train_blocks)
            out = self.transformer(**kw)
            return out[0], out[1]
        tup = self.transformer(**kw)
        return tup[0], None

    def train(self) -> Path:
        cfg = self.cfg
        set_seed(cfg.seed)
        out = Path(cfg.output_dir)
        if self._accelerator.is_main_process:
            out.mkdir(parents=True, exist_ok=True)
            if cfg.save_training_config_record:
                _write_dualflow_dit_training_config_record(out, cfg)
            print(
                f"[dualflow] distributed: world_size={self._accelerator.num_processes} "
                f"local_rank={self._accelerator.local_process_index} | "
                f"per-GPU batch={cfg.batch_size} -> global_batch≈{cfg.batch_size * self._accelerator.num_processes}",
                flush=True,
            )
            print(
                f"[dualflow] vae_only DiT latent_grid=({cfg.latent_frames},{cfg.latent_height},{cfg.latent_width}) "
                f"compress_t={cfg.temporal_compression} compress_spatial={cfg.spatial_compression} "
                f"sample_every_n_steps={cfg.sample_every_n_steps} "
                f"log_every_n_steps={cfg.log_every_n_steps} "
                f"checkpoint_every={cfg.checkpoints.interval} "
                f"validation_every={cfg.validation.interval} "
                f"save_loss_csv={cfg.save_loss_csv} "
                f"max_grad_norm={cfg.max_grad_norm if cfg.max_grad_norm > 0 else 'off'} "
                f"save_best_loss={cfg.checkpoints.save_best_loss}",
                flush=True,
            )

        self.transformer.train()

        epoch = 0
        if hasattr(self._loader, "set_epoch"):
            self._loader.set_epoch(epoch)
        data_iter = iter(self._loader)
        t0 = time.time()
        loss_rows: list[dict[str, int | float | str]] = []
        for step in range(cfg.steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if hasattr(self._loader, "set_epoch"):
                    self._loader.set_epoch(epoch)
                data_iter = iter(self._loader)
                batch = next(data_iter)

            loss, log_t = self._step(batch, step)
            self._accelerator.backward(loss)
            mgn = float(cfg.max_grad_norm)
            if mgn > 0.0 and self._clip_grad_params:
                self._accelerator.clip_grad_norm_(self._clip_grad_params, mgn)
            self._optimizer.step()
            self._optimizer.zero_grad()

            loss_log = loss.detach()
            if self._accelerator.num_processes > 1:
                red = getattr(self._accelerator, "reduce", None)
                if red is not None:
                    loss_log = red(loss_log, reduction="mean")
            if self._accelerator.is_main_process and getattr(cfg.checkpoints, "save_best_loss", False):
                self._maybe_save_best_loss_checkpoint(float(loss_log.item()), step + 1, out)
            if cfg.save_loss_csv:
                row: dict[str, int | float | str] = {"step": step + 1}
                for fn in _DIT_LOSS_CSV_FIELDS:
                    if fn == "step":
                        continue
                    if fn not in log_t:
                        row[fn] = ""
                    else:
                        row[fn] = _reduce_scalar_for_log(self._accelerator, log_t[fn])
                if self._accelerator.is_main_process:
                    loss_rows.append(row)
            if self._accelerator.is_main_process and (step + 1) % cfg.log_every_n_steps == 0:
                logger.info(f"dualflow_dit step {step + 1}/{cfg.steps} loss={loss_log.item():.4f}")
            if (
                cfg.save_loss_csv
                and self._accelerator.is_main_process
                and loss_rows
                and (step + 1) % cfg.log_every_n_steps == 0
            ):
                _write_dualflow_dit_loss_artifacts(
                    out,
                    rows=loss_rows,
                    save_plot=cfg.save_loss_plot,
                    csv_name=cfg.loss_csv_filename,
                    plot_name=cfg.loss_plot_filename,
                )

            if cfg.checkpoints.interval > 0 and (step + 1) % cfg.checkpoints.interval == 0:
                if self._accelerator.is_main_process:
                    ck_step = out / f"dualflow_dit_step_{step + 1:07d}.safetensors"
                    self._save_weights_to_path(ck_step)
                    logger.info("dualflow_dit checkpoint -> %s", ck_step)
                    self._prune_step_checkpoints(out)
                self._accelerator.wait_for_everyone()

            if cfg.validation.interval > 0 and (step + 1) % cfg.validation.interval == 0:
                self._accelerator.wait_for_everyone()
                if self._accelerator.is_main_process and self._val_loader is not None:
                    self._run_validation(step)
                self._accelerator.wait_for_everyone()

            if cfg.sample_every_n_steps > 0 and (step + 1) % cfg.sample_every_n_steps == 0:
                if self._accelerator.is_main_process:
                    self._maybe_training_sample(step, out)
                self._accelerator.wait_for_everyone()

        self._accelerator.wait_for_everyone()
        if self._accelerator.is_main_process and cfg.save_loss_csv and loss_rows:
            _write_dualflow_dit_loss_artifacts(
                out,
                rows=loss_rows,
                save_plot=cfg.save_loss_plot,
                csv_name=cfg.loss_csv_filename,
                plot_name=cfg.loss_plot_filename,
            )
            logger.info(
                "dualflow_dit: wrote %s / %s (%s rows)",
                cfg.loss_csv_filename,
                cfg.loss_plot_filename,
                len(loss_rows),
            )
        ckpt_path = out / "dualflow_dit.safetensors"
        if self._accelerator.is_main_process:
            self._save_weights_to_path(ckpt_path)
            logger.info(f"saved {ckpt_path} in {time.time() - t0:.1f}s")
        return ckpt_path

    def _internal_guidance_sampling_kwargs(self) -> dict[str, Any]:
        cfg = self.cfg
        if abs(float(cfg.internal_guidance_sampling_scale) - 1.0) < 1e-6:
            return {}
        from dualflow.ltx.transformer_ltx_ig import is_ltx_transformer_with_native_ig

        if not is_ltx_transformer_with_native_ig(self._accelerator.unwrap_model(self.transformer)):
            return {}
        return {
            "ig_head": None,
            "ig_block_idx": int(self._ig_sampling_block),
            "ig_scale": float(cfg.internal_guidance_sampling_scale),
            "ig_min_timestep": cfg.internal_guidance_sampling_min_timestep,
        }

    def _maybe_training_sample(self, step: int, out: Path) -> None:
        """Periodic MP4: vae_only Euler denoising with optional conditioning."""
        cfg = self.cfg
        dit = self._accelerator.unwrap_model(self.transformer)
        was = dit.training
        dit.eval()
        try:
            nf = cfg.latent_frames
            nh = cfg.latent_height
            nw = cfg.latent_width
            sample_dir = out / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)

            cond_paths = [str(p).strip() for p in cfg.sample_conditioning_video_paths if str(p).strip()]
            use_cond = len(cond_paths) > 0
            sample_fps = float(cfg.train_sample_fps)
            ig_kw = self._internal_guidance_sampling_kwargs()

            def _one_sample_vae(
                *,
                vid_path: Path,
                conditioning_mask: torch.Tensor | None,
                z_pix_a: torch.Tensor | None,
                sample_fps_local: float,
                seed_offset: int,
            ) -> None:
                run_vae_latent_sampling(
                    dit,
                    self._vae,
                    c_vae=cfg.c_vae,
                    nf=nf,
                    nh=nh,
                    nw=nw,
                    num_inference_steps=cfg.sample_num_inference_steps,
                    vae_model_source=cfg.vae_model_source,
                    fps=sample_fps_local,
                    caption_channels=int(dit.config.caption_channels),
                    null_text_seq_len=cfg.null_text_seq_len,
                    seed=cfg.seed + step + seed_offset,
                    device=self.device,
                    dtype=self.dtype,
                    output_video_path=vid_path,
                    conditioning_mask=conditioning_mask,
                    z_pix_anchor=z_pix_a,
                    **ig_kw,
                )

            if use_cond:
                lf = rgb_frames_to_latent_frames(cfg.condition_source_frames, cfg.temporal_compression)
                cond_mask_base = conditioning_mask_packed(1, nf * nh * nw, lf, nh, nw, self.device)
                for ci, ref_s in enumerate(cond_paths):
                    ref = Path(ref_s)
                    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in ref.stem)[:48]
                    vid_path = sample_dir / f"step_{step + 1:07d}_vae_cond_{ci:02d}_{stem}.mp4"
                    if not ref.is_file():
                        logger.warning("dualflow_dit: conditioning video not found: %s — skip.", ref)
                        continue
                    try:
                        z_pix_a, fps_rd = self._load_condition_anchor(ref)
                        _one_sample_vae(
                            vid_path=vid_path,
                            conditioning_mask=cond_mask_base,
                            z_pix_a=z_pix_a,
                            sample_fps_local=float(fps_rd) if fps_rd > 0 else float(cfg.train_sample_fps),
                            seed_offset=ci * 17_389,
                        )
                        logger.info("dualflow_dit: training sample (vae_only, conditional) -> %s", vid_path)
                    except Exception as e:
                        logger.warning("dualflow_dit: conditional sample failed for %s (%s).", ref, e)
            else:
                vid_path = sample_dir / f"step_{step + 1:07d}_vae.mp4"
                _one_sample_vae(
                    vid_path=vid_path,
                    conditioning_mask=None,
                    z_pix_a=None,
                    sample_fps_local=sample_fps,
                    seed_offset=0,
                )
                logger.info("dualflow_dit: training sample (vae_only) -> %s", vid_path)
        finally:
            if was:
                self.transformer.train()

    def _load_condition_anchor(self, path: Path) -> tuple[torch.Tensor, float]:
        """Encode leading ``condition_source_frames`` into a full latent canvas for sampling."""
        from dualflow.sample_vae_resources import (
            encode_condition_anchor_latents,
            load_condition_bcfhw,
        )

        cfg = self.cfg
        bcfhw, fps = load_condition_bcfhw(
            video_path=path,
            condition_source_frames=cfg.condition_source_frames,
            video_width=cfg.video_width,
            video_height=cfg.video_height,
            device=self.device,
            dtype=self.dtype,
        )
        z_anchor = encode_condition_anchor_latents(
            self._vae,
            bcfhw,
            nf=cfg.latent_frames,
            nh=cfg.latent_height,
            nw=cfg.latent_width,
            condition_source_frames=cfg.condition_source_frames,
            temporal_compression=cfg.temporal_compression,
            device=self.device,
            dtype=self.dtype,
        )
        return z_anchor, float(fps)

    def _encode_vae_latents_only(self, bcfhw: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        enc = encode_video(self._vae, bcfhw, device=self.device, dtype=self.dtype)
        z_pix = enc["latents"].to(dtype=self.dtype)
        return z_pix, int(enc["num_frames"]), int(enc["height"]), int(enc["width"])

    def _encode_online_vae_only(self, batch: dict[str, Any]) -> tuple[torch.Tensor, int, int, int, float]:
        pv = batch["pixel_values"].to(self.device, dtype=self.dtype)
        bcfhw = pv.permute(0, 2, 1, 3, 4).contiguous()
        z_pix, nf, nh, nw = self._encode_vae_latents_only(bcfhw)
        return z_pix, nf, nh, nw, float(batch["fps"])

    def _forward_and_loss(self, batch: dict[str, Any], global_step: int, *, inference: bool) -> torch.Tensor:
        ctx = torch.inference_mode() if inference else contextlib.nullcontext()
        with ctx:
            loss, _ = self._forward_vae_latents_only_inner(batch, global_step)
            return loss

    def _forward_vae_latents_only_inner(
        self, batch: dict[str, Any], global_step: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Flow matching on LTX-VAE latents with optional Internal Guidance."""
        cfg = self.cfg
        if cfg.use_dummy_data:
            z_pix = batch["pix_latents"].to(self.device, dtype=self.dtype)
            nf = int(batch["num_frames"])
            nh = int(batch["height"])
            nw = int(batch["width"])
            fps = float(batch["fps"])
        else:
            z_pix, nf, nh, nw, fps = self._encode_online_vae_only(batch)

        b = z_pix.shape[0]
        rope = get_rope_scale_factors(fps)

        cond_mask = maybe_conditioning_mask(
            batch_size=b,
            seq_len=z_pix.shape[1],
            condition_source_rgb_frames=cfg.condition_source_frames,
            latent_height=nh,
            latent_width=nw,
            temporal_compression=cfg.temporal_compression,
            first_frame_conditioning_p=cfg.first_frame_conditioning_p,
            device=self.device,
        )

        t = sample_t_base_continuous(
            b,
            self.device,
            use_lognorm=cfg.transport_use_lognorm,
            train_eps=cfg.transport_train_eps,
            lognorm_mu=cfg.transport_lognorm_mu,
            lognorm_sigma=cfg.transport_lognorm_sigma,
            shift_lg=cfg.transport_shift_lg,
            shifted_mu=cfg.transport_shifted_mu,
        )
        sigma = continuous_t_to_sigma(t).view(b, 1, 1).to(dtype=self.dtype)
        noise_z = torch.randn_like(z_pix)
        ones = torch.ones(1, 1, 1, device=self.device, dtype=self.dtype)
        pix_n = (ones - sigma) * z_pix + sigma * noise_z
        cond_exp = cond_mask.unsqueeze(-1)
        pix_n = torch.where(cond_exp, z_pix, pix_n)
        z_in = pix_n

        t_id = continuous_t_to_timestep_id(t)
        t_seq = expand_timesteps_with_conditioning(cond_mask, t_id)

        cap_c = int(self._accelerator.unwrap_model(self.transformer).config.caption_channels)
        enc, mask = null_text_embeddings(
            b,
            cfg.null_text_seq_len,
            cap_c,
            self.device,
            self.dtype,
        )

        lam_ig = _lambda_warmup(
            global_step, cfg.lambda_internal_guidance_warmup_steps, cfg.lambda_internal_guidance
        )

        pred, pred_ig = self._dit_forward_velocity(
            z_in, enc, mask, t_seq, nf, nh, nw, rope, lam_ig
        )

        target = noise_z - z_pix
        loss_fm = _flow_mse_masked(pred, target, cond_mask)
        loss = loss_fm
        lam_ig_t = torch.tensor(lam_ig, device=loss.device, dtype=loss.dtype)
        log_t: dict[str, torch.Tensor] = {
            "loss_fm": loss_fm,
            "lambda_ig_eff": lam_ig_t,
        }
        if lam_ig > 0.0 and pred_ig:
            ig_losses = [_flow_mse_masked(p, target, cond_mask) for p in pred_ig.values()]
            loss_ig = torch.stack(ig_losses).mean()
            loss = loss + lam_ig * loss_ig
            log_t["loss_ig"] = loss_ig
            log_t["loss_ig_weighted"] = lam_ig * loss_ig
        log_t["loss"] = loss
        return loss, log_t

    def _step(self, batch: dict[str, Any], global_step: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._forward_vae_latents_only_inner(batch, global_step)

    def _maybe_save_best_loss_checkpoint(self, loss_value: float, step_one_based: int, out: Path) -> None:
        if not math.isfinite(loss_value):
            return
        cfg_ck = self.cfg.checkpoints
        decay = float(getattr(cfg_ck, "best_loss_ema_decay", 0.0))
        metric = loss_value
        if decay > 0.0:
            if self._best_loss_ema is None:
                self._best_loss_ema = loss_value
            else:
                self._best_loss_ema = decay * self._best_loss_ema + (1.0 - decay) * loss_value
            metric = self._best_loss_ema
        if metric >= self._best_loss_metric:
            return
        self._best_loss_metric = metric
        best_path = out / "dualflow_dit_best_loss.safetensors"
        self._save_weights_to_path(best_path)
        meta_path = out / "dualflow_dit_best_loss.json"
        payload: dict[str, float | int] = {
            "step": int(step_one_based),
            "metric": float(metric),
            "raw_loss_at_save": float(loss_value),
            "best_loss_ema_decay": float(decay),
        }
        if self._best_loss_ema is not None:
            payload["loss_ema"] = float(self._best_loss_ema)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(
            "dualflow_dit: new best loss metric=%.6g at step %s -> %s",
            metric,
            step_one_based,
            best_path.name,
        )

    def _save_weights_to_path(self, ckpt_path: Path) -> None:
        tensors: dict[str, torch.Tensor] = {}
        dit_sd = self._accelerator.unwrap_model(self.transformer).state_dict()
        for k, v in dit_sd.items():
            tensors[f"transformer.{k}"] = v
        save_file(tensors, str(ckpt_path))

    def _prune_step_checkpoints(self, out: Path) -> None:
        k = self.cfg.checkpoints.keep_last_n
        if k < 0:
            return
        paths = sorted(out.glob("dualflow_dit_step_*.safetensors"))
        if len(paths) <= k:
            return
        for p in paths[: -k]:
            try:
                p.unlink()
            except OSError:
                logger.warning("could not remove old checkpoint %s", p)

    def _run_validation(self, step: int) -> None:
        cfg = self.cfg
        vc = cfg.validation
        assert self._val_loader is not None
        was_dit = self.transformer.training
        self.transformer.eval()
        total = 0.0
        n = 0
        it = iter(self._val_loader)
        try:
            for _ in range(vc.num_batches):
                batch = next(it)
                loss = self._forward_and_loss(batch, step, inference=True)
                total += float(loss.detach().cpu())
                n += 1
        except StopIteration:
            pass
        if n > 0:
            logger.info(
                "dualflow_dit validation step %s loss=%.4f (mean over %s batches)",
                step + 1,
                total / n,
                n,
            )
        if was_dit:
            self.transformer.train()
