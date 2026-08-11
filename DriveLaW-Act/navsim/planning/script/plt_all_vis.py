#!/usr/bin/env python
"""Visualize DriveLaW-Act / VideoDrive planning trajectories on NAVSIM scenes.

Saves PNGs: left = front camera + projected trajectories (agent red, human green);
right = BEV map + agent vs human trajectories.

Example (from DriveLaW-Act root)::

    export OPENSCENE_DATA_ROOT=...
    export NUPLAN_MAPS_ROOT=...
    python navsim/planning/script/plt_all_vis.py \\
        --config-file navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim.yaml \\
        --max-samples 20

Or via ``scripts/evaluation/run_videodrive_agent_vis.sh``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from tqdm import tqdm

from navsim.agents.videodrive.videodrive_agent import VideoDriveAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.visualization.plots import plot_bev_and_camera_with_agent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def init_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        print(f"Distributed init: rank {rank}/{world_size}, local_rank {local_rank}", flush=True)
        return rank, world_size, local_rank
    return 0, 1, 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DriveLaW-Act planning trajectory visualization")
    p.add_argument(
        "--config-file",
        type=str,
        default="navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim.yaml",
        help="VideoDrive YAML (diffusion_model.model_path = planning checkpoint)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["mini", "test", "trainval"],
        help="OpenScene split folder under navsim_logs / sensor_blobs",
    )
    p.add_argument(
        "--scene-filter",
        type=str,
        default="navtest",
        help="Hydra scene_filter name under train_test_split/scene_filter",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="traj_plot_vis_drivelaw_act",
        help="Directory for PNG outputs",
    )
    p.add_argument(
        "--score-csv",
        type=str,
        default="",
        help="Optional PDM score CSV; if set, visualize tokens with score==1 (or --score-value)",
    )
    p.add_argument(
        "--score-value",
        type=float,
        default=1.0,
        help="When --score-csv is set, keep rows with this score",
    )
    p.add_argument(
        "--tokens",
        type=str,
        default="",
        help="Comma-separated token list (overrides CSV / uses these only)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Cap number of scenes to visualize (after filtering). 0 = no cap",
    )
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def load_token_entries(
    *,
    score_csv: str,
    score_value: float,
    tokens_arg: str,
    all_tokens: list[str],
    max_samples: int,
) -> list[dict]:
    if tokens_arg.strip():
        entries = [{"token": t.strip()} for t in tokens_arg.split(",") if t.strip()]
    elif score_csv.strip():
        df = pd.read_csv(score_csv)
        if "token" not in df.columns or "score" not in df.columns:
            raise ValueError(f"{score_csv} must contain columns: token, score")
        sub = df[df["score"] == score_value]
        entries = [{"token": str(t)} for t in sub["token"].tolist()]
        print(f"Loaded {len(entries)} tokens from CSV with score=={score_value}", flush=True)
    else:
        entries = [{"token": t} for t in all_tokens]
        print(f"Using all scene_loader tokens: {len(entries)}", flush=True)

    if max_samples > 0:
        entries = entries[:max_samples]
    return entries


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    openscene_data_root = Path(os.environ.get("OPENSCENE_DATA_ROOT", "")).expanduser()
    if not openscene_data_root.is_dir():
        raise FileNotFoundError(
            "Set OPENSCENE_DATA_ROOT to the OpenScene/NAVSIM dataset root "
            f"(got: {openscene_data_root!s})"
        )

    config_file = Path(args.config_file)
    if not config_file.is_file():
        # Allow paths relative to DriveLaW-Act root
        alt = Path(__file__).resolve().parents[3] / args.config_file
        if alt.is_file():
            config_file = alt
        else:
            raise FileNotFoundError(f"config-file not found: {args.config_file}")

    # Scene filter via Hydra (same as other NAVSIM scripts)
    filter_cfg_dir = str(
        (Path(__file__).resolve().parent / "config" / "common" / "train_test_split" / "scene_filter").resolve()
    )
    with hydra.initialize_config_dir(config_dir=filter_cfg_dir, version_base=None):
        cfg = hydra.compose(config_name=args.scene_filter)
    scene_filter: SceneFilter = instantiate(cfg)

    agent = VideoDriveAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        config_file=str(config_file),
        weight_dtype="",
        device=str(device),
    )
    agent.device = str(device)
    agent.initialize()
    agent.eval()
    agent.to(device)

    logs_path = openscene_data_root / f"navsim_logs/{args.split}"
    sensors_path = openscene_data_root / f"sensor_blobs/{args.split}"
    sensor_config = agent.get_sensor_config()

    # Must load decoded RGB (not path-only): VideoDrive forward_test needs features['images'].
    scene_loader = SceneLoader(
        logs_path,
        sensors_path,
        scene_filter,
        sensor_config=sensor_config,
        load_image_path=False,
    )

    entries = load_token_entries(
        score_csv=args.score_csv,
        score_value=args.score_value,
        tokens_arg=args.tokens,
        all_tokens=list(scene_loader.tokens),
        max_samples=int(args.max_samples),
    )
    if not entries:
        if rank == 0:
            print("No tokens to visualize.", flush=True)
        return

    local_entries = entries[rank::world_size]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"[vis] config={config_file}", flush=True)
        print(f"[vis] output_dir={output_dir.resolve()}", flush=True)
        print(f"[vis] total={len(entries)} local_rank{rank}={len(local_entries)}", flush=True)

    n_ok = 0
    n_fail = 0
    for entry in tqdm(local_entries, desc=f"Rank {rank} planning vis", disable=(rank != 0)):
        token = entry["token"]
        try:
            scene = scene_loader.get_scene_from_token(token)
            frame_idx = scene.scene_metadata.num_history_frames - 1
            # Pass the same scene twice: BEV/camera + agent_input both need decoded images.
            fig, _, _ = plot_bev_and_camera_with_agent(scene, scene, frame_idx, agent)
            out_path = output_dir / f"{token}_vis_traj.png"
            fig.savefig(out_path, bbox_inches="tight", dpi=int(args.dpi))
            plt.close(fig)
            n_ok += 1
            if rank == 0:
                print(f"[vis] saved {out_path}", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"[vis] FAILED token={token}: {e}", flush=True)
            plt.close("all")

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
    print(f"[vis] rank={rank} ok={n_ok} fail={n_fail} dir={output_dir.resolve()}", flush=True)
    if rank == 0:
        print(f"[vis] done. PNGs under {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
