#!/usr/bin/env python
"""Train vae_only LTX DiT with Internal Guidance. Run from repository root after `pip install -e .`.

Multi-GPU / multi-node::

    torchrun --nproc_per_node=$RESOURCE_GPU \\
        --nnodes=$WORLD_SIZE --node_rank=$RANK \\
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
        scripts/train_dualflow_dit.py configs/dualflow/dit_train.online.example.yaml

    # Optional: override grad clip without editing YAML
    python scripts/train_dualflow_dit.py configs/dualflow/dit_train.online.example.yaml --max-grad-norm 1.0
"""

from pathlib import Path

import typer
import yaml

from dualflow.trainer_dualflow_dit import DualFlowDiTTrainer
from dualflow.train_configs import DualFlowDiTTrainConfig

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


@app.command()
def main(
    config_path: str = typer.Argument(..., help="YAML config for DualFlowDiTTrainConfig"),
    max_grad_norm: float | None = typer.Option(
        None,
        "--max-grad-norm",
        help="Override YAML: clip gradient norm before optimizer step (0 = disable). Omit to use config.",
    ),
) -> None:
    p = Path(config_path)
    if not p.exists():
        raise typer.Exit(f"Missing config: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = DualFlowDiTTrainConfig(**data)
    if max_grad_norm is not None:
        cfg = cfg.model_copy(update={"max_grad_norm": max_grad_norm})
    DualFlowDiTTrainer(cfg).train()


if __name__ == "__main__":
    app()
