"""Adjoint-mode physics loss for full ODE-solver back-propagation.

The hybrid loss in rectified.py evaluates the physics residual at a single
projected clean sample. For tighter conservation (especially for ERA5
divergence on long time horizons) we also support an adjoint-mode loss
that integrates the velocity field with ``torchdiffeq.odeint_adjoint`` and
applies the residual at every step.

This file is intentionally a thin wrapper; production training defaults to
the simpler instantaneous loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class AdjointPhysicsLoss(nn.Module):
    """Integrate the flow ODE with torchdiffeq adjoint and accumulate residuals."""

    def __init__(
        self,
        velocity_model: nn.Module,
        physics_residual: nn.Module,
        steps: int = 25,
        weight: float = 1.0,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.physics_residual = physics_residual
        self.steps = int(steps)
        self.weight = float(weight)

    def forward(self, x_0: Tensor, condition: dict) -> Tensor:
        try:
            from torchdiffeq import odeint_adjoint as odeint  # noqa: WPS433
        except ImportError as exc:
            raise RuntimeError(
                "Adjoint mode requires torchdiffeq. `pip install torchdiffeq`."
            ) from exc

        ts = torch.linspace(0.0, 1.0, self.steps + 1, device=x_0.device, dtype=x_0.dtype)

        def odefn(t, x):
            return self.velocity_model(x, t.expand(x.shape[0]), condition)

        traj = odeint(odefn, x_0, ts, method="euler")  # (steps+1, B, C, H, W)
        loss = traj.new_zeros(())
        for k in range(1, traj.shape[0]):
            loss = loss + self.physics_residual(traj[k], condition)
        return self.weight * loss / max(traj.shape[0] - 1, 1)
