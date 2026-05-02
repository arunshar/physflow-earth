"""Sanity-check that RectifiedFlow.training_step actually drops the velocity loss."""
from __future__ import annotations

import torch
import torch.nn as nn

from physflow.flow import RectifiedFlow


class TinyVelocity(nn.Module):
    """A trivial velocity model: predicts a constant per channel."""

    def __init__(self, channels: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x_t, t, condition):
        return self.bias.expand_as(x_t)


def test_velocity_loss_decreases():
    torch.manual_seed(0)
    model = TinyVelocity(channels=3)
    flow = RectifiedFlow(model)
    opt = torch.optim.SGD(model.parameters(), lr=1e-1)
    x_1 = torch.full((4, 3, 8, 8), 0.5)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        out = flow.training_step(x_1, condition={"x_lr": torch.zeros(4, 3, 4, 4)})
        out.total_loss.backward()
        opt.step()
        losses.append(out.velocity_loss.item())
    assert losses[-1] < losses[0]


def test_physics_loss_is_nonnegative():
    from physflow.physics import MassConservationResidual, PhysicsResidual
    model = TinyVelocity(channels=1)
    residual = PhysicsResidual([MassConservationResidual(scale_factor=4)])
    flow = RectifiedFlow(model, physics_residual=residual, physics_weight=1.0)
    x_hr = torch.randn(2, 1, 16, 16)
    out = flow.training_step(x_hr, condition={"x_lr": torch.randn(2, 1, 4, 4)})
    assert out.physics_loss.item() >= 0.0
    assert torch.isfinite(out.total_loss)
