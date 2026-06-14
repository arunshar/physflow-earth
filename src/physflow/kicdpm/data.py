"""Synthetic fine fields for Ki-CDPM (sea-level-like smooth random fields).

Each fine field is a sum of a few random Gaussian bumps plus a smooth large-scale
tilt, normalised to roughly unit variance. The coarse observation is an
average-pool of the fine field (the downscaling inverse problem). Everything is
generated on the fly so train/eval run anywhere; to use real regional sea-level
data, replace FieldDataset with a loader that yields the same (coarse, fine)
tensor pairs at the configured scale factor.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import KiCDPMConfig


def _random_field(rng: np.random.Generator, fine: int, n_bumps: int = 5) -> np.ndarray:
    r, c = np.meshgrid(np.arange(fine), np.arange(fine), indexing="ij")
    field = np.zeros((fine, fine), dtype=np.float64)
    for _ in range(n_bumps):
        cr, cc = rng.uniform(0, fine, 2)
        amp = rng.uniform(-2, 2)
        width = rng.uniform(fine / 8, fine / 3)
        field += amp * np.exp(-((r - cr) ** 2 + (c - cc) ** 2) / (2 * width ** 2))
    # smooth large-scale tilt
    field += rng.uniform(-1, 1) * (r / fine) + rng.uniform(-1, 1) * (c / fine)
    # deterministic high-frequency detail whose amplitude is set by the local
    # field value (bathymetry-like fine structure). Its period (~3 px) is below
    # the coarsening factor, so average-pooling removes it and bilinear / kriging
    # cannot represent it, yet it is a fixed function of the field and so is
    # recoverable by a conditional model -- the regime where downscaling helps.
    detail = 0.5 * np.tanh(field) * np.cos(2.0 * np.pi * (r + c) / 3.0)
    return field + detail


def average_pool(fine: np.ndarray, factor: int) -> np.ndarray:
    n = fine.shape[0] // factor
    return fine.reshape(n, factor, n, factor).mean(axis=(1, 3))


class FieldDataset(Dataset):
    def __init__(self, cfg: KiCDPMConfig, n: int = 4096, seed: int = 0) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        fines, coarses = [], []
        for _ in range(n):
            f = _random_field(rng, cfg.fine)
            fines.append(f)
            coarses.append(average_pool(f, cfg.scale_factor))
        fa = np.stack(fines).astype(np.float32)
        self.mean = float(fa.mean())
        self.std = float(fa.std()) + 1e-6
        self.fine = (fa - self.mean) / self.std
        ca = np.stack(coarses).astype(np.float32)
        self.coarse = (ca - self.mean) / self.std

    def __len__(self) -> int:
        return self.fine.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        # (C, H, W) with C=1
        return torch.from_numpy(self.coarse[i])[None], torch.from_numpy(self.fine[i])[None]
