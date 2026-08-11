"""CLOVER-compatible ego status packing for LAF PDM scorer."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch

# pose(3) + vel(2) + acc(2) + driving_command(4)
CLOVER_EGO_STATUS_DIM = 11


def pack_ego_status_vector(
    pose: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    driving_command: torch.Tensor,
) -> torch.Tensor:
    """Single timestep ego vector, same layout as CLOVER DrivoRFeatureBuilder."""
    return torch.cat(
        [
            pose.reshape(-1)[:3],
            velocity.reshape(-1)[:2],
            acceleration.reshape(-1)[:2],
            driving_command.reshape(-1)[:4],
        ],
        dim=-1,
    ).to(dtype=torch.float32)


def build_ego_status_from_agent_input_stack(ego_statuses) -> torch.Tensor:
    """Build (T, 11) ego history from NavSim ``AgentInput.ego_statuses``."""
    rows = []
    for ego_status in ego_statuses:
        if ego_status is None:
            continue
        rows.append(
            pack_ego_status_vector(
                torch.tensor(ego_status.ego_pose, dtype=torch.float32),
                torch.tensor(ego_status.ego_velocity, dtype=torch.float32),
                torch.tensor(ego_status.ego_acceleration, dtype=torch.float32),
                torch.tensor(ego_status.driving_command, dtype=torch.float32),
            )
        )
    if not rows:
        raise ValueError("No ego statuses available to build ego_status tensor.")
    return torch.stack(rows, dim=0)


def build_ego_status_from_videodrive_features(features: Dict[str, Any]) -> torch.Tensor:
    """
    Build (T, 11) from VideoDrive cache features.

    Uses ``ego_status`` when present (new caches). Otherwise reconstructs from
    ``history_trajectory``, ``vel``, ``acc``, ``driving_command`` (legacy caches).
    """
    if "ego_status" in features and torch.is_tensor(features["ego_status"]):
        ego = features["ego_status"].float()
        if ego.ndim == 1:
            ego = ego.unsqueeze(0)
        return ego

    history = features.get("history_trajectory")
    vel = features["vel"].float()
    acc = features["acc"].float()
    cmd = features["driving_command"].float()

    if history is None:
        pose = torch.zeros(3, dtype=torch.float32, device=vel.device)
        return pack_ego_status_vector(pose, vel, acc, cmd).unsqueeze(0)

    history = history.float()
    if history.ndim == 1:
        history = history.unsqueeze(0)
    rows = []
    for ti in range(history.shape[0]):
        pose = history[ti]
        v = vel if ti == history.shape[0] - 1 else torch.zeros_like(vel)
        a = acc if ti == history.shape[0] - 1 else torch.zeros_like(acc)
        c = cmd if ti == history.shape[0] - 1 else torch.zeros_like(cmd)
        rows.append(pack_ego_status_vector(pose, v, a, c))
    return torch.stack(rows, dim=0)


def batch_clover_ego_status_last(features: Dict[str, Any]) -> torch.Tensor:
    """Return (B, 11) last-timestep ego vectors for scorer ``hist_encoding``."""
    if "ego_status" in features and torch.is_tensor(features["ego_status"]):
        ego = features["ego_status"].float()
        if ego.ndim == 3:
            return ego[:, -1]
        if ego.ndim == 2:
            return ego
        raise ValueError(f"Unexpected ego_status shape: {tuple(ego.shape)}")

    vel = features["vel"]
    if not torch.is_tensor(vel):
        raise ValueError("features must contain tensor vel or ego_status.")
    if vel.ndim == 1:
        return build_ego_status_from_videodrive_features(features)[-1].unsqueeze(0)

    rows = []
    for bi in range(int(vel.shape[0])):
        feat_i = {
            k: (v[bi] if torch.is_tensor(v) and v.ndim > 0 else v)
            for k, v in features.items()
        }
        rows.append(build_ego_status_from_videodrive_features(feat_i)[-1])
    return torch.stack(rows, dim=0)


def resolve_clover_ego_encoding_input(
    ego_status: Optional[torch.Tensor],
    features: Optional[Dict[str, Any]] = None,
    *,
    full_history_status: bool = False,
) -> torch.Tensor:
    """
    Return (B, D_in) ego vector for ``hist_encoding``.

    CLOVER default (``full_history_status=false``): last timestep only → (B, 11).
    """
    if ego_status is None:
        if features is None:
            raise ValueError("Either ego_status or features must be provided.")
        if isinstance(features, dict) and any(torch.is_tensor(v) and v.ndim >= 1 for v in features.values()):
            batch_vecs = []
            batch_size = None
            for key in ("ego_status", "vel"):
                if key in features and torch.is_tensor(features[key]):
                    t = features[key]
                    if t.ndim >= 2:
                        batch_size = t.shape[0]
                        break
            if batch_size is None:
                vec = build_ego_status_from_videodrive_features(features)
                ego_status = vec.unsqueeze(0)
            else:
                ego_rows = []
                for bi in range(batch_size):
                    feat_i = {
                        k: (v[bi] if torch.is_tensor(v) and v.ndim > 0 else v)
                        for k, v in features.items()
                    }
                    ego_rows.append(build_ego_status_from_videodrive_features(feat_i))
                ego_status = torch.stack(ego_rows, dim=0)
        else:
            vec = build_ego_status_from_videodrive_features(features)
            ego_status = vec.unsqueeze(0)

    ego_status = ego_status.float()
    if ego_status.ndim == 1:
        ego_status = ego_status.unsqueeze(0)
    if ego_status.ndim == 3:
        # (B, T, 11)
        if full_history_status:
            return ego_status.reshape(ego_status.shape[0], -1)
        return ego_status[:, -1]
    if ego_status.ndim == 2 and ego_status.shape[-1] != CLOVER_EGO_STATUS_DIM:
        if full_history_status:
            return ego_status.reshape(ego_status.shape[0], -1)
        raise ValueError(f"Expected ego_status last dim {CLOVER_EGO_STATUS_DIM}, got {ego_status.shape}")
    return ego_status


def ego_status_from_context_dict(context_dict: Optional[Dict[str, torch.Tensor]]) -> Optional[torch.Tensor]:
    """Build (B, 11) from pipeline ``context_dict`` when full ``ego_status`` is absent."""
    if context_dict is None:
        return None
    if "ego_status" in context_dict:
        return resolve_clover_ego_encoding_input(context_dict["ego_status"], full_history_status=False)
    required = ("vel", "acc", "cmd_onehot")
    if not all(k in context_dict for k in required):
        return None
    vel = context_dict["vel"].float()
    acc = context_dict["acc"].float()
    cmd = context_dict["cmd_onehot"].float()
    if vel.ndim == 1:
        vel = vel.unsqueeze(0)
        acc = acc.unsqueeze(0)
        cmd = cmd.unsqueeze(0)
    b = vel.shape[0]
    pose = torch.zeros(b, 3, device=vel.device, dtype=vel.dtype)
    rows = [pack_ego_status_vector(pose[i], vel[i], acc[i], cmd[i]) for i in range(b)]
    return torch.stack(rows, dim=0)
