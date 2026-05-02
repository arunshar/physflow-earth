"""Physics residual modules.

Each residual is a differentiable ``nn.Module``-style class with a
``residual(x_HR, condition)`` method returning a tensor whose squared L2
norm enters the training loss. Composed into a ``PhysicsResidual`` bag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from physflow.physics.operators import (
    average_pool,
    horizontal_divergence,
)


class _BaseResidual(nn.Module):
    """All residuals share the same interface: residual(x_HR, condition) -> Tensor."""

    weight: float = 1.0

    def residual(self, x_hr: Tensor, condition: dict) -> Tensor:  # noqa: D401
        """Return raw residual tensor (model is supposed to drive ||r|| -> 0)."""
        raise NotImplementedError


# -- Sentinel-2 band ratios --------------------------------------------------


@dataclass
class BandIndex:
    """Index of band positions in a Sentinel-2 stack used by the residuals."""

    red: int     # B4
    nir: int     # B8
    green: int   # B3
    swir: int    # B11


class BandRatioResidual(_BaseResidual):
    """Penalize NDVI / NDWI deviations between SR(LR) average-pooled and LR.

    NDVI = (NIR - RED) / (NIR + RED + eps)
    NDWI = (GREEN - SWIR) / (GREEN + SWIR + eps)
    """

    def __init__(self, scale_factor: int, bands: BandIndex, weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.bands = bands
        self.weight = float(weight)
        self.eps = eps

    @staticmethod
    def _ndvi(x: Tensor, b: BandIndex, eps: float) -> Tensor:
        red, nir = x[:, b.red : b.red + 1], x[:, b.nir : b.nir + 1]
        return (nir - red) / (nir + red + eps)

    @staticmethod
    def _ndwi(x: Tensor, b: BandIndex, eps: float) -> Tensor:
        g, sw = x[:, b.green : b.green + 1], x[:, b.swir : b.swir + 1]
        return (g - sw) / (g + sw + eps)

    def residual(self, x_hr: Tensor, condition: dict) -> Tensor:
        x_lr = condition["x_lr"]
        ndvi_hr_avg = average_pool(self._ndvi(x_hr, self.bands, self.eps), self.scale_factor)
        ndwi_hr_avg = average_pool(self._ndwi(x_hr, self.bands, self.eps), self.scale_factor)
        ndvi_lr = self._ndvi(x_lr, self.bands, self.eps)
        ndwi_lr = self._ndwi(x_lr, self.bands, self.eps)
        return torch.cat([ndvi_hr_avg - ndvi_lr, ndwi_hr_avg - ndwi_lr], dim=1)


# -- ERA5 divergence-free wind -----------------------------------------------


class DivergenceResidual(_BaseResidual):
    """Penalize horizontal divergence of a wind field (channels = (u, v))."""

    def __init__(self, dx: float = 1.0, dy: float = 1.0, weight: float = 1.0):
        super().__init__()
        self.dx = float(dx)
        self.dy = float(dy)
        self.weight = float(weight)

    def residual(self, x_hr: Tensor, condition: dict) -> Tensor:
        u = x_hr[:, 0]
        v = x_hr[:, 1]
        return horizontal_divergence(u, v, dx=self.dx, dy=self.dy).unsqueeze(1)


# -- CHIRPS precipitation mass conservation ----------------------------------


class MassConservationResidual(_BaseResidual):
    """Penalize average-pool(SR) - LR. Ensures SR preserves coarse means."""

    def __init__(self, scale_factor: int, weight: float = 1.0):
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.weight = float(weight)

    def residual(self, x_hr: Tensor, condition: dict) -> Tensor:
        return average_pool(x_hr, self.scale_factor) - condition["x_lr"]


# -- composite ---------------------------------------------------------------


class PhysicsResidual(nn.Module):
    """Compose multiple residual heads. ``forward`` returns the weighted L2 sum."""

    def __init__(self, residuals: Sequence[_BaseResidual]):
        super().__init__()
        self.residuals = nn.ModuleList(residuals)

    def forward(self, x_hr: Tensor, condition: dict) -> Tensor:
        total = x_hr.new_zeros(())
        for r in self.residuals:
            res = r.residual(x_hr, condition)
            total = total + r.weight * res.pow(2).mean()
        return total

    def per_residual(self, x_hr: Tensor, condition: dict) -> dict:
        """Return per-residual L2 magnitudes for logging."""
        out = {}
        for r in self.residuals:
            out[type(r).__name__] = r.residual(x_hr, condition).pow(2).mean().detach()
        return out
