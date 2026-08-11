"""Distributed navtest PDMS evaluation for VideoDrive FM training (tau0 only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from navsim.agents.videodrive.official_pdms_eval import run_official_navtest_pdms_eval

__all__ = [
    "run_navtest_pdms_eval",
    "write_videodrive_eval_config",
    "resolve_model_checkpoint_path",
]


def write_videodrive_eval_config(*args, **kwargs):
    from navsim.agents.videodrive.official_pdms_eval import write_videodrive_eval_config as _write

    return _write(*args, **kwargs)


def resolve_model_checkpoint_path(checkpoint_dir: str | Path):
    from navsim.agents.videodrive.official_pdms_eval import resolve_model_checkpoint_path as _resolve

    return _resolve(checkpoint_dir)


def run_navtest_pdms_eval(
    *,
    train_config_file: str,
    model_checkpoint: str | Path,
    output_dir: str | Path,
    step: int,
    device: torch.device,
    eval_config_file: str | None = None,
    use_train_config_for_eval: bool | None = None,
    eval_tag: str = "",
) -> Dict[str, Any]:
    """Official navtest PDMS (single trajectory / tau0 only)."""
    return run_official_navtest_pdms_eval(
        train_config_file=train_config_file,
        output_dir=output_dir,
        step=step,
        device=device,
        eval_config_file=eval_config_file,
        use_train_config_for_eval=use_train_config_for_eval,
        model_checkpoint=model_checkpoint,
        eval_tag=eval_tag,
    )
