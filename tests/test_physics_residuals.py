"""Verify physics residual operators are correct.

These ensure that:

- Average pooling on an HR tensor matches the LR by construction when the
  HR is a nearest-neighbor upsample of the LR (mass conservation).
- The divergence operator is shape-preserving and zero on a constant field.
- The band-ratio residual on x_HR == nearest-upsample(x_LR) is exactly zero.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from physflow.physics import (
    BandRatioResidual,
    DivergenceResidual,
    MassConservationResidual,
    average_pool,
    horizontal_divergence,
)
from physflow.physics.residual import BandIndex


def test_average_pool_inverse_of_nearest_upsample():
    lr = torch.randn(2, 3, 32, 32)
    hr = F.interpolate(lr, scale_factor=4, mode="nearest")
    pooled = average_pool(hr, factor=4)
    assert torch.allclose(pooled, lr, atol=1e-6)


def test_horizontal_divergence_zero_on_constant_field():
    u = torch.full((2, 16, 16), 3.0)
    v = torch.full((2, 16, 16), -1.5)
    div = horizontal_divergence(u, v)
    assert div.abs().max() < 1e-5


def test_horizontal_divergence_recovers_linear_gradient():
    H = W = 64
    grid_x = torch.linspace(0, 1, W).expand(H, W)
    u = grid_x.clone().unsqueeze(0).expand(2, -1, -1)        # u(x) = x -> du/dx = 1/(W-1) per cell
    v = torch.zeros_like(u)
    div = horizontal_divergence(u, v, dx=1.0, dy=1.0)
    expected = 1.0 / (W - 1)
    assert (div - expected).abs().mean() < 1e-2


def test_mass_conservation_residual_zero_on_perfect_upsample():
    lr = torch.randn(2, 1, 16, 16)
    hr = F.interpolate(lr, scale_factor=5, mode="nearest")
    res = MassConservationResidual(scale_factor=5).residual(hr, {"x_lr": lr})
    assert res.abs().max() < 1e-5


def test_band_ratio_residual_zero_on_perfect_upsample():
    lr = torch.rand(2, 4, 16, 16) + 0.5    # avoid dividing by zero
    hr = F.interpolate(lr, scale_factor=4, mode="nearest")
    bands = BandIndex(red=0, green=1, nir=3, swir=2)
    res = BandRatioResidual(scale_factor=4, bands=bands).residual(hr, {"x_lr": lr})
    assert res.abs().max() < 1e-4


def test_divergence_residual_module_runs():
    u = torch.randn(2, 1, 16, 16)
    v = torch.randn(2, 1, 16, 16)
    field = torch.cat([u, v], dim=1)
    res = DivergenceResidual().residual(field, {})
    assert res.shape == (2, 1, 16, 16)
