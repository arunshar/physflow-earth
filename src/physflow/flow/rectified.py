"""Conditional rectified flow with a hybrid velocity + physics-residual loss.

Standard rectified-flow training matches the constant velocity ``x_1 - x_0``
along the linear interpolant ``x_t = (1 - t) x_0 + t x_1``. We add a
physics-residual term evaluated on the projected clean sample
``hat_x_1 = x_t + (1 - t) v_theta``. Gradients flow through ``hat_x_1``
back to the velocity head, so the model learns to make residuals small
without explicit projection / Lagrangian solvers.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class FlowStepOutput:
    velocity_loss: Tensor
    physics_loss: Tensor
    total_loss: Tensor
    hat_x_1: Tensor


class RectifiedFlow(nn.Module):
    """Wrap a velocity model with the rectified-flow training step.

    Args:
        velocity_model: an ``nn.Module`` taking (x_t, t, condition) and
            returning the predicted velocity.
        physics_residual: optional composite ``PhysicsResidual``. When None,
            we fall back to vanilla rectified-flow training.
        physics_weight: lambda multiplier on the physics term.
    """

    def __init__(
        self,
        velocity_model: nn.Module,
        physics_residual: nn.Module | None = None,
        physics_weight: float = 1.0,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.physics_residual = physics_residual
        self.physics_weight = float(physics_weight)

    # -- training step ----------------------------------------------------------

    def training_step(self, x_1: Tensor, condition: dict) -> FlowStepOutput:
        """One rectified-flow gradient step.

        Inputs:
            x_1: (B, C, H, W) clean target sample.
            condition: dict including ``x_lr`` and any auxiliary conditioning.
        """
        bsz = x_1.shape[0]
        x_0 = torch.randn_like(x_1)
        t = torch.rand(bsz, device=x_1.device, dtype=x_1.dtype)
        t_view = t.view(bsz, *([1] * (x_1.dim() - 1)))

        x_t = (1.0 - t_view) * x_0 + t_view * x_1
        v_target = x_1 - x_0
        v_pred = self.velocity_model(x_t, t, condition)
        velocity_loss = (v_pred - v_target).pow(2).mean()

        hat_x_1 = x_t + (1.0 - t_view) * v_pred

        if self.physics_residual is not None:
            physics_loss = self.physics_residual(hat_x_1, condition)
        else:
            physics_loss = x_1.new_zeros(())

        total = velocity_loss + self.physics_weight * physics_loss
        return FlowStepOutput(
            velocity_loss=velocity_loss,
            physics_loss=physics_loss,
            total_loss=total,
            hat_x_1=hat_x_1.detach(),
        )
