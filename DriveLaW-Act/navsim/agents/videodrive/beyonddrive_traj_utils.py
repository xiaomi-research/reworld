"""Trajectory helpers ported from BeyondDrive-main ``transfuser/utils.py`` (LTFv7 delta path)."""

from __future__ import annotations

import torch

# Copied from navsim/agents/transfuser/utils.py (BeyondDrive-main).
x_diff_min = -1.2698211669921875
x_diff_max = 7.475563049316406
x_diff_mean = 2.950225591659546

y_diff_min = -5.012081146240234
y_diff_max = 4.8563690185546875
y_diff_mean = 0.0607292577624321


def diff_traj(traj: torch.Tensor) -> torch.Tensor:
    """
    First-order delta + sin/cos heading (official LTFv7 / MeanFuser BeyondDrive RDE space).

    Source: BeyondDrive-main/navsim/agents/transfuser/utils.py::diff_traj
    """
    _b, _l, _ = traj.shape
    sin = traj[..., -1:].sin()
    cos = traj[..., -1:].cos()
    zero_pad = torch.zeros((_b, 1, 1), dtype=traj.dtype, device=traj.device)
    x_diff = traj[..., 0:1].diff(n=1, dim=1, prepend=zero_pad)
    x_diff = x_diff - x_diff_mean
    x_diff_range = max(abs(x_diff_max - x_diff_mean), abs(x_diff_min - x_diff_mean))
    x_diff_norm = x_diff / x_diff_range

    zero_pad = torch.zeros((_b, 1, 1), dtype=traj.dtype, device=traj.device)
    y_diff = traj[..., 1:2].diff(n=1, dim=1, prepend=zero_pad)
    y_diff = y_diff - y_diff_mean
    y_diff_range = max(abs(y_diff_max - y_diff_mean), abs(y_diff_min - y_diff_mean))
    y_diff_norm = y_diff / y_diff_range

    return torch.cat([x_diff_norm, y_diff_norm, sin, cos], -1)
