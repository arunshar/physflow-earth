"""Inference samplers: 25-step Euler and 4-step consistency-distilled."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


@torch.no_grad()
def euler_sample(
    velocity_model: nn.Module,
    shape: tuple[int, ...],
    condition: dict,
    *,
    steps: int = 25,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Standard Euler integration of dx/dt = v_theta(x, t, c)."""
    device = device or torch.device("cpu")
    x = torch.randn(shape, device=device, dtype=dtype)
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)
    for k in range(steps):
        t = ts[k].expand(shape[0])
        v = velocity_model(x, t, condition)
        x = x + (ts[k + 1] - ts[k]) * v
    return x


@torch.no_grad()
def consistency_sample(
    consistency_model: nn.Module,
    shape: tuple[int, ...],
    condition: dict,
    *,
    schedule: tuple = (1.0, 0.5, 0.25, 0.0),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Multi-step consistency sampling (default: 4 steps)."""
    device = device or torch.device("cpu")
    x = torch.randn(shape, device=device, dtype=dtype)
    for t_val in schedule:
        t = x.new_full((shape[0],), float(t_val))
        x = consistency_model(x, t, condition)
    return x
