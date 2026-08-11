"""Load LTX transformers from local diffusers JSON (dual DiT)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def safetensors_torch_load_device(device: torch.device) -> str:
    """Device string for :func:`safetensors.torch.load_file`.

    Rust ``safe_open`` rejects ``cpu:0``; CPU loads must use the literal ``cpu``.
    """
    if device.type == "cpu":
        return "cpu"
    if device.index is not None:
        return f"{device.type}:{device.index}"
    return device.type


def load_ltx_transformer_from_json_channel_override(
    config_path: str | Path,
    *,
    in_channels: int,
    out_channels: int,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Build ``LTXVideoTransformer3DModel`` from JSON but force ``in_channels`` / ``out_channels``.

    Use for ``dit_latent_mode=vae_only`` (I/O width ``c_vae``) while reusing an LTX JSON that may list
    a different channel count.
    """
    from dualflow.ltx.transformer_ltx_ig import LTXVideoTransformer3DModelWithIG

    path = Path(config_path)
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for key in ("_class_name", "_diffusers_version", "_name_or_path"):
        raw.pop(key, None)
    raw["in_channels"] = int(in_channels)
    raw["out_channels"] = int(out_channels)
    model = LTXVideoTransformer3DModelWithIG(**raw)
    return model.to(torch_dtype)


def load_ltx_transformer_from_json(
    config_path: str | Path,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Build `LTXVideoTransformer3DModel` from a directory or `config.json` file path.

    UNCERTAINTY: Requires `diffusers` with `LTXVideoTransformer3DModel`.
    """
    from dualflow.ltx.transformer_ltx_ig import LTXVideoTransformer3DModelWithIG

    path = Path(config_path)
    try:
        config_obj = LTXVideoTransformer3DModelWithIG.load_config(str(path))
    except Exception:
        if not path.is_file():
            raise
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        for key in ("_class_name", "_diffusers_version", "_name_or_path"):
            raw.pop(key, None)
        model = LTXVideoTransformer3DModelWithIG.from_config(raw)
        return model.to(torch_dtype)
    model = LTXVideoTransformer3DModelWithIG.from_config(config_obj)
    return model.to(torch_dtype)


def load_ltx_transformer_from_json_fallback(
    config_path: str | Path,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Fallback when `load_config` is unavailable for a bare JSON file."""
    from dualflow.ltx.transformer_ltx_ig import LTXVideoTransformer3DModelWithIG

    raw: dict[str, Any] = json.loads(Path(config_path).read_text(encoding="utf-8"))
    for key in ("_class_name", "_diffusers_version", "_name_or_path"):
        raw.pop(key, None)
    model = LTXVideoTransformer3DModelWithIG(**raw)
    return model.to(torch_dtype)


def load_ltx_transformer_weights(model: torch.nn.Module, path: str | Path) -> None:
    """Load weights from a merged dual checkpoint or a single-transformer safetensors (strict=False)."""
    from safetensors.torch import load_file

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    state = load_file(str(p))
    model.load_state_dict(state, strict=False)


def load_ltx_transformer_weights_prefixed(
    model: torch.nn.Module,
    path: str | Path,
    prefix: str,
) -> None:
    from safetensors.torch import load_file

    state = load_file(str(path))
    own: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith(prefix):
            own[k[len(prefix) :]] = v
    if not own:
        raise ValueError(f"No keys with prefix {prefix!r} in {path}")
    model.load_state_dict(own, strict=False)


def load_ltx_transformer_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    """Load DiT weights from LTX / DriveLaW / DualFlow stage-2 checkpoints (strict=False).

    Supports:
    - DualFlow ``dualflow_dit*.safetensors`` with ``transformer.*`` (or legacy ``transformer_joint.*``)
    - Flat Diffusers / DriveLaW ``diffusion_pytorch_model.safetensors``
    - A directory containing ``diffusion_pytorch_model.safetensors`` (optionally under ``transformer/``)
    """
    from safetensors.torch import load_file

    p = Path(path)
    if p.is_dir():
        candidates = [
            p / "diffusion_pytorch_model.safetensors",
            p / "transformer" / "diffusion_pytorch_model.safetensors",
        ]
        found = next((c for c in candidates if c.is_file()), None)
        if found is None:
            raise FileNotFoundError(
                f"No diffusion_pytorch_model.safetensors under {p} (checked transformer/ too)"
            )
        p = found
    if not p.is_file():
        raise FileNotFoundError(p)

    if p.suffix in (".safetensors", ".sft"):
        blob = load_file(str(p))
        if any(k.startswith("transformer.") for k in blob):
            load_ltx_transformer_weights_prefixed(model, p, "transformer.")
        elif any(k.startswith("transformer_joint.") for k in blob):
            load_ltx_transformer_weights_prefixed(model, p, "transformer_joint.")
        else:
            load_ltx_transformer_weights(model, p)
        return
    load_ltx_transformer_weights(model, p)


def null_text_embeddings(
    batch_size: int,
    sequence_length: int,
    caption_channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zeros for LTX ``encoder_hidden_states`` **before** ``caption_projection`` (``in_features=caption_channels``).

    This is **not** ``cross_attention_dim`` (the KV dim after projection); using the wrong width breaks
    ``PixArtAlphaTextProjection`` (e.g. 2048 vs 4096).
    """
    emb = torch.zeros(batch_size, sequence_length, caption_channels, device=device, dtype=dtype)
    mask = torch.ones(batch_size, sequence_length, dtype=torch.bool, device=device)
    return emb, mask
