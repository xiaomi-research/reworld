"""Official navtest PDMS evaluation aligned with ``run_pdm_score_videodrive.py``."""

from __future__ import annotations

import json
import logging
import lzma
import os
import pickle
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from omegaconf import OmegaConf
from yaml import Loader, dump, load

from hydra.utils import instantiate

from navsim.agents.videodrive.pdms_eval_dist import (
    InferenceSampler,
    broadcast_object,
    gather_pickled_rows,
    rank as _rank,
)
from navsim.agents.videodrive.videodrive_agent import VideoDriveAgent
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _official_navtest_scene_filter_yaml() -> Path:
    return (
        _repo_root()
        / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )


def _official_scoring_parameters_yaml() -> Path:
    return _repo_root() / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"


def load_official_navtest_scene_filter() -> SceneFilter:
    """Same ``SceneFilter`` as Hydra ``train_test_split.scene_filter`` for navtest."""
    cfg = OmegaConf.load(str(_official_navtest_scene_filter_yaml()))
    return instantiate(cfg)


def load_official_pdm_simulator_scorer() -> tuple[PDMSimulator, PDMScorer]:
    """Same simulator/scorer as ``default_run_pdm_score.yaml`` scoring parameters."""
    cfg = OmegaConf.load(str(_official_scoring_parameters_yaml()))
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling
    return simulator, scorer


def resolve_official_navtest_paths() -> tuple[Path, Path, Path]:
    openscene = Path(os.environ["OPENSCENE_DATA_ROOT"])
    navsim_exp = Path(os.environ.get("NAVSIM_EXP_ROOT", str(openscene)))
    data_path = openscene / "navsim_logs/test"
    sensor_blobs = openscene / "sensor_blobs/test"
    metric_cache = navsim_exp / "metric_cache"
    return data_path, sensor_blobs, metric_cache


def build_official_tokens_to_evaluate(device: torch.device) -> list[str]:
    """Token list identical to ``run_pdm_score_videodrive.main`` rank-0 logic."""
    data_path, _, metric_cache_path = resolve_official_navtest_paths()
    scene_filter = load_official_navtest_scene_filter()
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    if _rank() == 0:
        metric_cache_loader = MetricCacheLoader(metric_cache_path)
        tokens = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
        missing = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
        unused = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
        if missing > 0:
            logger.warning("Missing metric cache for %d navtest tokens.", missing)
        if unused > 0:
            logger.warning("Unused metric cache for %d navtest tokens.", unused)
    else:
        tokens = []

    return broadcast_object(tokens, device=device, src=0)


def resolve_model_checkpoint_path(checkpoint_dir: str | Path) -> Path:
    ckpt_dir = Path(checkpoint_dir)
    if ckpt_dir.is_file():
        return ckpt_dir
    preferred = ckpt_dir / "diffusion_pytorch_model.safetensors"
    if preferred.is_file():
        return preferred
    candidates = sorted(ckpt_dir.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No .safetensors checkpoint under {ckpt_dir}")
    return candidates[0]


def write_videodrive_eval_config(
    train_config_file: str,
    out_path: Path,
    *,
    eval_config_file: str | None = None,
    use_train_config_for_eval: bool = True,
    model_checkpoint: str | Path | None = None,
) -> str:
    """Build agent yaml for PDMS eval (tau0, no rerank).

    Default (``use_train_config_for_eval=True``): copy **train yaml** and only patch
    the checkpoint path. The optional ``eval_config_file`` template is ignored unless
    ``use_train_config_for_eval=False``.

    Checkpoint priority: ``model_checkpoint`` arg > ``train_cfg.diffusion_model.model_path``.
    """
    import copy

    with open(train_config_file, "r", encoding="utf-8") as f:
        train_cfg = load(f, Loader=Loader)

    if not use_train_config_for_eval and eval_config_file:
        with open(eval_config_file, "r", encoding="utf-8") as f:
            eval_cfg = load(f, Loader=Loader)
        _merge_train_inference_fields_into_eval(eval_cfg, train_cfg)
    else:
        eval_cfg = copy.deepcopy(train_cfg)

    _apply_eval_runtime_overrides(eval_cfg, train_cfg, model_checkpoint=model_checkpoint)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        dump(eval_cfg, f)
    return str(out_path)


def _apply_eval_runtime_overrides(
    eval_cfg: dict,
    train_cfg: dict,
    *,
    model_checkpoint: str | Path | None,
) -> None:
    """Patch checkpoint path only; tau0 eval does not use LAF / IG / renoise pools."""
    train_diffusion = train_cfg.get("diffusion_model", {})
    eval_diffusion = eval_cfg.setdefault("diffusion_model", {})
    ckpt = model_checkpoint or train_diffusion.get("model_path")
    if ckpt:
        eval_diffusion["model_path"] = str(resolve_model_checkpoint_path(ckpt))


def _merge_train_inference_fields_into_eval(eval_cfg: dict, train_cfg: dict) -> None:
    """Legacy path: merge train inference fields into an eval template yaml."""
    shared_train_keys = (
        "view_mode",
        "seed",
        "noisy_video",
        "fast_wam_inference",
        "fast_wam_strict",
        "num_inference_step",
        "cross_modal_align_blocks",
        "lambda_cross_modal_align",
        "cross_modal_align_grad_video",
    )
    for key in shared_train_keys:
        if key in train_cfg:
            eval_cfg[key] = train_cfg[key]


def _device_str(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{idx}"
    return str(device)


def _eval_output_dir(output_dir: Path, step: int, eval_tag: str) -> Path:
    if eval_tag:
        return output_dir / eval_tag
    return output_dir / f"step_{step:06d}"


def _aggregate_metrics(
    df: pd.DataFrame,
    *,
    step: int,
    checkpoint: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    valid_df = df[df["valid"] == True]  # noqa: E712
    metrics: Dict[str, Any] = {
        "step": step,
        "num_tokens": int(len(df)),
        "num_valid": int(len(valid_df)),
        "num_failed": int(len(df) - len(valid_df)),
        "pdms": float(valid_df["score"].mean()) if len(valid_df) else 0.0,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": checkpoint,
        "scene_filter": str(_official_navtest_scene_filter_yaml()),
    }
    if extra:
        metrics.update(extra)
    for col in (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "driving_direction_compliance",
    ):
        if col in valid_df.columns:
            metrics[col] = float(valid_df[col].mean())
    return metrics


def run_official_navtest_pdms_eval(
    *,
    train_config_file: str,
    output_dir: str | Path,
    step: int,
    device: torch.device,
    eval_config_file: str | None = None,
    use_train_config_for_eval: bool | None = None,
    model_checkpoint: str | Path | None = None,
    eval_tag: str = "",
) -> Dict[str, Any]:
    """
    Navtest PDMS eval using the same token set, scene filter, and PDM scorer as
    ``run_pdm_score_videodrive.py`` (single trajectory / tau0 only).
    """
    output_dir = Path(output_dir)
    step_dir = _eval_output_dir(output_dir, step, eval_tag)

    train_cfg = load(open(train_config_file, "r"), Loader=Loader)
    if use_train_config_for_eval is None:
        use_train_config_for_eval = bool(train_cfg.get("use_train_config_for_eval", True))
    if not use_train_config_for_eval and eval_config_file is None:
        eval_config_file = train_cfg.get("eval_config_file")

    train_diffusion = train_cfg.get("diffusion_model", {})
    ckpt_path = resolve_model_checkpoint_path(
        model_checkpoint or train_diffusion.get("model_path", "")
    )

    eval_cfg_path = write_videodrive_eval_config(
        train_config_file,
        step_dir / "eval_agent_config.yaml",
        eval_config_file=eval_config_file,
        use_train_config_for_eval=use_train_config_for_eval,
        model_checkpoint=ckpt_path,
    )

    data_path, sensor_blobs, metric_cache_path = resolve_official_navtest_paths()
    if not metric_cache_path.is_dir():
        raise FileNotFoundError(f"navtest metric cache not found: {metric_cache_path}")

    scene_filter = load_official_navtest_scene_filter()
    tokens_to_evaluate = build_official_tokens_to_evaluate(device)
    logger.info(
        "[rank %d] official navtest PDMS (step=%d tag=%r) on %d tokens",
        _rank(),
        step,
        eval_tag,
        len(tokens_to_evaluate),
    )

    traj_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    agent = VideoDriveAgent(
        trajectory_sampling=traj_sampling,
        config_file=eval_cfg_path,
        weight_dtype="bf16",
        device=_device_str(device),
    )
    agent.initialize()
    agent.eval()

    simulator, scorer = load_official_pdm_simulator_scorer()
    metric_cache_loader = MetricCacheLoader(metric_cache_path)

    eval_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=False,
    )

    local_rows: List[Dict[str, Any]] = []
    sampler = InferenceSampler(len(tokens_to_evaluate))
    for local_idx, token_idx in enumerate(sampler):
        token = tokens_to_evaluate[token_idx]
        row: Dict[str, Any] = {"token": token, "valid": True, "rank": _rank(), "step": step}
        try:
            with lzma.open(metric_cache_loader.metric_cache_paths[token], "rb") as f:
                metric_cache: MetricCache = pickle.load(f)
            agent_input = eval_scene_loader.get_agent_input_from_token(token)
            scene = eval_scene_loader.get_scene_from_token(token)
            trajectory = agent.compute_trajectory(agent_input, scene)
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            row.update(asdict(pdm_result))
        except Exception:
            logger.warning("[rank %d] PDMS eval failed for token %s", _rank(), token)
            traceback.print_exc()
            row["valid"] = False

        local_rows.append(row)
        if (local_idx + 1) % 20 == 0:
            logger.info("[rank %d] PDMS eval progress %d/%d", _rank(), local_idx + 1, len(sampler))

    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged = gather_pickled_rows(local_rows, device=device)
    if _rank() != 0:
        return {}

    df = pd.DataFrame(merged)
    extra: Dict[str, Any] = {"eval_tag": eval_tag}
    metrics = _aggregate_metrics(df, step=step, checkpoint=str(ckpt_path), extra=extra)

    csv_path = step_dir / "navtest_pdms.csv"
    df.to_csv(csv_path, index=False)
    metrics_path = step_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(
        "Step %d official navtest PDMS=%.4f (%d/%d valid). Saved %s",
        step,
        metrics["pdms"],
        metrics["num_valid"],
        metrics["num_tokens"],
        csv_path,
    )
    return metrics
