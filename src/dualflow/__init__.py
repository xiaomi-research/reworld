"""DualFlow: vae_only LTX DiT training and sampling with Internal Guidance."""

from __future__ import annotations

import logging
import os
from logging import getLogger

from rich.logging import RichHandler

__version__ = "0.1.0"

IS_MULTI_GPU = os.environ.get("LOCAL_RANK") is not None
RANK = int(os.environ.get("LOCAL_RANK", "0"))

logging.basicConfig(
    level="INFO",
    format=f"\\[rank {RANK}] %(message)s" if IS_MULTI_GPU else "%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_time=False,
            markup=True,
        )
    ],
)

logger = getLogger("dualflow")
logger.setLevel(logging.DEBUG)
logger.propagate = True
if RANK != 0:
    logger.setLevel(logging.WARNING)

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

from dualflow.act_interface import WorldLatentShapes, world_latent_shapes
from dualflow.sample_dualflow import sample_dualflow_async
from dualflow.timestep_async import (
    continuous_t_to_sigma,
    continuous_t_to_timestep_id,
    sample_t_base_continuous,
    sample_ts_tz_continuous,
)
from dualflow.train_configs import (
    DualFlowDiTTrainConfig,
    DualFlowSampleConfig,
)
from dualflow.trainer_dualflow_dit import DualFlowDiTTrainer

__all__ = [
    "DualFlowDiTTrainer",
    "DualFlowDiTTrainConfig",
    "DualFlowSampleConfig",
    "WorldLatentShapes",
    "continuous_t_to_sigma",
    "continuous_t_to_timestep_id",
    "sample_dualflow_async",
    "sample_t_base_continuous",
    "sample_ts_tz_continuous",
    "world_latent_shapes",
]
