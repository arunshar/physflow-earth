"""Datasets for PhysFlow-Earth.

The real project trains on WorldStrat (Sentinel-2 super-resolution) and a paired
ERA5 / CHIRPS climate-downscaling set. Those archives are large and gated, so to
keep ``python -m physflow.training.train`` and the eval benchmark RUNNABLE
without a multi-hundred-GB download, this module ships SYNTHETIC stand-ins that
yield the same ``{"x_hr", "x_lr"}`` contract (smooth low-frequency multi-band
fields, with the low-resolution view an average-pool of the high-resolution one).

These are NOT the real datasets and produce no benchmark numbers. To train on
real data, replace these classes with loaders over the actual archives that
return the same dict of ``(C, H, W)`` / ``(C, H/s, W/s)`` tensors; nothing in the
training loop, the flow, or the physics residuals changes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

__all__ = ["SyntheticPairs", "WorldStratDataset", "ERA5CHIRPSDataset"]


def _synth_pair(seed: int, channels: int, hr: int, scale: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A smooth multi-band HR field and its average-pooled LR view."""
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(1, channels, max(hr // 8, 2), max(hr // 8, 2), generator=g)
    x_hr = F.interpolate(latent, size=(hr, hr), mode="bicubic", align_corners=False)[0]
    x_hr = (x_hr - x_hr.amin()) / (x_hr.amax() - x_hr.amin() + 1e-6)  # [0,1] reflectance-like
    x_lr = F.avg_pool2d(x_hr[None], kernel_size=scale)[0]
    return x_hr.contiguous(), x_lr.contiguous()


class SyntheticPairs(Dataset):
    """Base synthetic SR dataset yielding {"x_hr", "x_lr"} dicts."""

    def __init__(self, n: int = 256, channels: int = 4, hr: int = 256, scale_factor: int = 4, seed: int = 0) -> None:
        self.n, self.channels, self.hr, self.scale, self.seed = n, channels, hr, scale_factor, seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        x_hr, x_lr = _synth_pair(self.seed * 100_003 + i, self.channels, self.hr, self.scale)
        return {"x_hr": x_hr, "x_lr": x_lr}


class WorldStratDataset(SyntheticPairs):
    """SYNTHETIC stand-in for WorldStrat Sentinel-2 SR (4-band, x4)."""

    def __init__(self, root: str | None = None, split: str = "train", *, channels: int = 4,
                 hr: int = 256, scale_factor: int = 4, n: int = 256) -> None:
        super().__init__(n=n, channels=channels, hr=hr, scale_factor=scale_factor,
                         seed=0 if split == "train" else 1)
        self.root, self.split = root, split


class ERA5CHIRPSDataset(SyntheticPairs):
    """SYNTHETIC stand-in for ERA5 / CHIRPS climate downscaling."""

    def __init__(self, root: str | None = None, split: str = "train", scale_factor: int = 4, *,
                 channels: int = 4, hr: int = 128, n: int = 256) -> None:
        super().__init__(n=n, channels=channels, hr=hr, scale_factor=scale_factor,
                         seed=2 if split == "train" else 3)
        self.root, self.split = root, split
