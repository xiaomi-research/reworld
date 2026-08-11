#!/usr/bin/env python
"""Sample vae_only DiT (+ optional IG) from a YAML DualFlowSampleConfig.

Example::

    python scripts/sample_dualflow_dit.py configs/dualflow/sample_reworld_ig_future_from_condition.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from dualflow.sample_dualflow import sample_dualflow_async
from dualflow.train_configs import DualFlowSampleConfig

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


@app.command()
def main(
    config_path: str = typer.Argument(..., help="YAML config for DualFlowSampleConfig"),
) -> None:
    p = Path(config_path)
    if not p.exists():
        raise typer.Exit(f"Missing config: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = DualFlowSampleConfig(**data)
    outs = sample_dualflow_async(cfg)
    for o in outs:
        print(f"[sample] wrote {o}", flush=True)
    print(f"[sample] done ({len(outs)} file(s))", flush=True)


if __name__ == "__main__":
    app()
