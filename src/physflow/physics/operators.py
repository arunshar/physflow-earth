"""Differentiable physics operators used by the residual heads.

All operators are pure PyTorch and accept tensors with leading batch and
channel dimensions, e.g. (B, C, H, W). They are designed to be chained
inside a training loss so gradients flow back through them.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def average_pool(x: Tensor, factor: int) -> Tensor:
    """Spatial average pooling with stride = factor.

    Used to project a high-resolution prediction down to the low-resolution
    grid for mass-conservation residuals: pool(x_HR, s) should equal x_LR.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return x
    return F.avg_pool2d(x, kernel_size=factor, stride=factor, ceil_mode=False)


def central_diff_x(x: Tensor, dx: float = 1.0) -> Tensor:
    """Central-difference derivative along the spatial-x axis (last dim).

    Returns a tensor of the same shape with first/last columns zero-padded
    so the operator is shape-preserving.
    """
    pad = F.pad(x, (1, 1), mode="replicate")
    diff = (pad[..., 2:] - pad[..., :-2]) / (2.0 * dx)
    return diff


def central_diff_y(x: Tensor, dy: float = 1.0) -> Tensor:
    """Central-difference derivative along the spatial-y axis (second-to-last dim)."""
    pad = F.pad(x, (0, 0, 1, 1), mode="replicate")
    diff = (pad[..., 2:, :] - pad[..., :-2, :]) / (2.0 * dy)
    return diff


def horizontal_divergence(u: Tensor, v: Tensor, dx: float = 1.0, dy: float = 1.0) -> Tensor:
    """Horizontal divergence du/dx + dv/dy. Inputs (B, H, W) or (B, 1, H, W)."""
    if u.shape != v.shape:
        raise ValueError(f"u and v must match shape, got {tuple(u.shape)} vs {tuple(v.shape)}")
    return central_diff_x(u, dx) + central_diff_y(v, dy)


def upsample_nearest(x: Tensor, factor: int) -> Tensor:
    """Nearest-neighbor upsample, used when projecting LR conditions to HR grids."""
    if factor == 1:
        return x
    return F.interpolate(x, scale_factor=factor, mode="nearest")
