"""Ordinary kriging as a precomputed linear operator.

For a fixed coarse->fine geometry the ordinary-kriging weights depend only on the
sample / target coordinates (not on the values), so the (n+1)x(n+1) kriging
system with the Lagrange multiplier is solved ONCE at construction and the
resulting (fine_pixels, coarse_pixels) weight matrix is reused as a matmul. This
makes kriging a cheap, batched, GPU-friendly `nn.Module`.

Exponential variogram: gamma(h) = sill + nugget - sill*exp(-h/length_scale),
with nugget added on the diagonal (h = 0).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import KiCDPMConfig


def _exp_semivariogram(h: np.ndarray, sill: float, length_scale: float, nugget: float) -> np.ndarray:
    cov = sill * np.exp(-h / length_scale)
    return sill + nugget - cov  # gamma(0) = nugget


def _solve_weights(coarse: int, fine: int, sill: float, length_scale: float, nugget: float) -> np.ndarray:
    """Return ordinary-kriging weight matrix W of shape (fine*fine, coarse*coarse)."""
    # coarse sample locations are placed at cell centres on the fine grid
    step = fine / coarse
    cc = (np.arange(coarse) + 0.5) * step
    sr, scol = np.meshgrid(cc, cc, indexing="ij")
    sample = np.stack([sr.ravel(), scol.ravel()], axis=1)               # (n, 2)
    fr, fcol = np.meshgrid(np.arange(fine), np.arange(fine), indexing="ij")
    pred = np.stack([fr.ravel(), fcol.ravel()], axis=1).astype(float)   # (m, 2)
    n = sample.shape[0]

    d_ss = np.sqrt(((sample[:, None] - sample[None]) ** 2).sum(-1))
    g_ss = _exp_semivariogram(d_ss, sill, length_scale, nugget)
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = g_ss
    K[:n, n] = 1.0
    K[n, :n] = 1.0

    d_ps = np.sqrt(((pred[:, None] - sample[None]) ** 2).sum(-1))       # (m, n)
    g_ps = _exp_semivariogram(d_ps, sill, length_scale, nugget)
    rhs = np.ones((n + 1, pred.shape[0]))
    rhs[:n] = g_ps.T
    weights = np.linalg.lstsq(K, rhs, rcond=None)[0][:n].T              # (m, n)
    return weights.astype(np.float32)


class OrdinaryKriging(nn.Module):
    """Maps a coarse field (B, C, c, c) to a kriged fine field (B, C, f, f)."""

    def __init__(self, cfg: KiCDPMConfig) -> None:
        super().__init__()
        w = _solve_weights(cfg.coarse, cfg.fine, cfg.sill, cfg.length_scale, cfg.nugget)
        self.coarse, self.fine = cfg.coarse, cfg.fine
        # (fine*fine, coarse*coarse) constant operator, moves with the module
        self.register_buffer("weights", torch.from_numpy(w))

    def forward(self, coarse: torch.Tensor) -> torch.Tensor:
        b, c, ch, cw = coarse.shape
        flat = coarse.reshape(b, c, ch * cw)                  # (B, C, n)
        kriged = torch.einsum("mn,bcn->bcm", self.weights, flat)
        return kriged.reshape(b, c, self.fine, self.fine)
