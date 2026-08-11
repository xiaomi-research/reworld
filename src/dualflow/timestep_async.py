"""SemanticLead-style asynchronous continuous timesteps for training (formula 12).

Base time ``t`` sampling follows SFD ``Transport.sample`` (``transport.py``): uniform on
``[t0, t1]`` or logit-normal when ``use_lognorm`` (Gaussian + logistic / sigmoid), then the same
``semfirst_delta_t`` split as SFD ``training_losses``:

``t_sem = t * (1 + Δt)``, ``t_tex = t_sem - Δt``, ``t_sem = clamp(max=1)``, ``t_tex = clamp(min=0)``.

Here ``t_s`` / ``t_z`` correspond to semantic / pixel (VAE) branches in DualFlow.
"""

from __future__ import annotations

from typing import Literal

import torch


def _sfd_check_interval_velocity_linear(*, train_eps: float) -> tuple[float, float]:
    """Match SFD ``Transport.check_interval`` for LINEAR (``ICPlan``) + VELOCITY, training.

    In that case the ``VPCPlan`` / non-velocity branches are skipped, so ``t0=0``, ``t1=1``
    regardless of ``train_eps`` (same as SFD LightningDiT image training defaults).
    """
    _ = train_eps  # reserved for parity if path/model options expand
    return 0.0, 1.0


def _sample_logit_normal(
    batch_size: int,
    device: torch.device,
    *,
    mu: float,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Match SFD ``Transport.sample_logit_normal`` — logistic transform of a Gaussian, in (0, 1)."""
    g = torch.randn((batch_size,), device=device, generator=generator)
    return torch.sigmoid(g * float(sigma) + float(mu))


def sample_ts_tz_continuous(
    batch_size: int,
    device: torch.device,
    delta_t: float,
    *,
    generator: torch.Generator | None = None,
    use_lognorm: bool = True,
    train_eps: float = 0.0,
    lognorm_mu: float = 0.0,
    lognorm_sigma: float = 1.0,
    shift_lg: bool = False,
    shifted_mu: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample coupled ``(t_s, t_z)`` — same construction as SFD semantic-first transport.

    **Base** ``t`` matches ``Transport.sample`` on ``[t0,t1]`` (here ``[0,1]`` for velocity-linear):
    if ``use_lognorm``, ``t = sigmoid(N(mu, sigma^2)) * span + t0`` (or ``shifted_mu`` when
    ``shift_lg``); else ``t ~ Uniform[t0, t1]``.

    Then ``t_s_raw = t * (1 + Δt)``, ``t_z = clamp(t_s_raw - Δt, min=0)``, ``t_s = clamp(t_s_raw, max=1)``,
    matching ``semfirst_delta_t`` in SFD ``transport.py``.
    """
    if delta_t < 0:
        raise ValueError("delta_t must be non-negative")
    t0, t1 = _sfd_check_interval_velocity_linear(train_eps=train_eps)
    span = t1 - t0
    if use_lognorm:
        if shift_lg:
            # SFD ``shift_lg`` branch uses ``sample_logit_normal(shifted_mu, 1, ...)`` (sigma fixed to 1).
            base01 = _sample_logit_normal(
                batch_size,
                device,
                mu=shifted_mu,
                sigma=1.0,
                generator=generator,
            )
        else:
            base01 = _sample_logit_normal(
                batch_size,
                device,
                mu=lognorm_mu,
                sigma=lognorm_sigma,
                generator=generator,
            )
        t_base = base01 * span + t0
    else:
        u = torch.rand((batch_size,), device=device, generator=generator)
        t_base = u * span + t0

    t_s = t_base * (1.0 + delta_t)
    t_z = torch.clamp(t_s - delta_t, min=0.0)
    t_s = torch.clamp(t_s, max=1.0)
    return t_s, t_z


def sample_t_base_continuous(
    batch_size: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
    use_lognorm: bool = True,
    train_eps: float = 0.0,
    lognorm_mu: float = 0.0,
    lognorm_sigma: float = 1.0,
    shift_lg: bool = False,
    shifted_mu: float = 0.0,
) -> torch.Tensor:
    """Sample a **single** continuous time ``t in [0,1]`` per batch item (SFD-style **joint** DiT training).

    Same base distribution as the first component of ``sample_ts_tz_continuous`` before the semantic-first
    split — one noise level for both semantic and pixel latents.
    """
    t0, t1 = _sfd_check_interval_velocity_linear(train_eps=train_eps)
    span = t1 - t0
    if use_lognorm:
        if shift_lg:
            base01 = _sample_logit_normal(
                batch_size,
                device,
                mu=shifted_mu,
                sigma=1.0,
                generator=generator,
            )
        else:
            base01 = _sample_logit_normal(
                batch_size,
                device,
                mu=lognorm_mu,
                sigma=lognorm_sigma,
                generator=generator,
            )
        return base01 * span + t0
    u = torch.rand((batch_size,), device=device, generator=generator)
    return u * span + t0


def continuous_t_to_sigma(t: torch.Tensor) -> torch.Tensor:
    """Map continuous t in [0,1] to sigma for (1-sigma)*x + sigma*eps.

    UNCERTAINTY: Identity sigma=t; replace with scheduler lookup if you need exact LTX FM parity.
    """
    return t.clamp(0.0, 1.0)


def continuous_t_to_timestep_id(t: torch.Tensor, max_period: int = 1000) -> torch.Tensor:
    """Map continuous t in [0,1] to integer timestep ids for DiT embedding."""
    return torch.round(t * float(max_period)).long().clamp(0, max_period)


def fuse_joint_timestep_ids(
    ts_id: torch.Tensor,
    tz_id: torch.Tensor,
    mode: Literal["mean", "sem", "pix"],
    *,
    max_period: int = 1000,
) -> torch.Tensor:
    """Fuse semantic / texture timestep ids for **one** LTX ``timestep`` tensor (single embedder).

    SFD's LightningDiT uses two embedders and ``cat(emb_sem, emb_tex)``. LTX has one global embedding,
    so we approximate: **mean** (default), or use **sem** / **pix** only.
    """
    if mode == "sem":
        return ts_id
    if mode == "pix":
        return tz_id
    return ((ts_id.float() + tz_id.float()) * 0.5).round().long().clamp(0, max_period)
