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

__all__ = [
    "SyntheticPairs",
    "WorldStratDataset",
    "ERA5CHIRPSDataset",
    "HRRRDataset",
    "make_hrrr_datasets",
]


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


# ---------------------------------------------------------------------------
# REAL data: HRRR 8x super-resolution (the PC-RF HRRR field set)
# ---------------------------------------------------------------------------
# Reads the HRRR target field (``target.nc``: precip, u, v at 128x128 over T
# timesteps) and builds a well-posed SR task by deriving the LR view as an
# average-pool of the HR field, so the mass-conservation residual
# (avg_pool(SR) - LR) is exact at ground truth. Channels are ordered
# ``[u, v, precip]`` so DivergenceResidual (which reads channels 0 and 1) acts
# on the wind field. Per-channel z-score statistics are computed on the TRAIN
# split only and applied to both splits (the z-score is affine, so it commutes
# with average-pool and leaves mass conservation exact at ground truth).

HRRR_CHANNELS = ("u", "v", "precip")  # order matters: divergence reads ch 0,1


def _load_hrrr_stack(root: str) -> "torch.Tensor":
    """Load target.nc into a (T, 3, H, W) tensor ordered [u, v, precip]."""
    import os

    import xarray as xr

    path = os.path.join(root, "target.nc")
    ds = xr.open_dataset(path)
    # target.nc data_vars are precip, u, v at (time, lat, lon)
    u = torch.from_numpy(ds["u"].values).float()
    v = torch.from_numpy(ds["v"].values).float()
    precip = torch.from_numpy(ds["precip"].values).float()
    ds.close()
    stack = torch.stack([u, v, precip], dim=1)  # (T, 3, H, W)
    return stack.contiguous()


def _degrade(hr: torch.Tensor, scale: int, mode: str) -> torch.Tensor:
    """Forward observation operator A: HR -> LR.

    ``pool``    average-pool (mass-exact; bicubic-up nearly inverts it, so the
                bicubic baseline is near-optimal and hard to beat on RMSE).
    ``bicubic`` anti-aliased bicubic downsample (the standard SR degradation;
                bicubic-up does NOT invert it, so the task is a genuine SR
                problem where high-frequency recovery matters).
    """
    if mode == "pool":
        return F.avg_pool2d(hr, kernel_size=scale)
    if mode == "bicubic":
        lr_hw = (hr.shape[-2] // scale, hr.shape[-1] // scale)
        return F.interpolate(hr, size=lr_hw, mode="bicubic", align_corners=False, antialias=True)
    raise ValueError(f"unknown degradation {mode!r}")


def make_hrrr_datasets(
    root: str = "/scratch.global/arunshar/hrrr128",
    scale_factor: int = 8,
    train_frac: float = 0.8,
    degradation: str = "pool",
) -> tuple["HRRRDataset", "HRRRDataset", dict]:
    """Build paired (train, val) HRRR SR datasets with shared train statistics.

    Returns ``(train_ds, val_ds, stats)`` where ``stats`` holds the per-channel
    mean/std used for the z-score (so predictions can be de-normalized) and the
    degradation operator used to make the LR view.
    """
    stack = _load_hrrr_stack(root)  # (T, 3, H, W)
    t_total = stack.shape[0]
    n_train = int(round(t_total * train_frac))
    train_raw, val_raw = stack[:n_train], stack[n_train:]

    # per-channel z-score from the TRAIN split only (no val leakage)
    mean = train_raw.mean(dim=(0, 2, 3), keepdim=True)  # (1,3,1,1)
    std = train_raw.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    stats = {
        "mean": mean.squeeze().tolist(),
        "std": std.squeeze().tolist(),
        "channels": list(HRRR_CHANNELS),
        "n_train": int(n_train),
        "n_val": int(t_total - n_train),
        "scale_factor": int(scale_factor),
        "degradation": degradation,
    }
    train_ds = HRRRDataset((train_raw - mean) / std, scale_factor, degradation)
    val_ds = HRRRDataset((val_raw - mean) / std, scale_factor, degradation)
    return train_ds, val_ds, stats


class HRRRDataset(Dataset):
    """Real HRRR SR pairs. Holds a pre-normalized (T, 3, H, W) HR tensor and
    derives the LR view with the chosen forward operator, yielding the
    {"x_hr", "x_lr"} contract the training loop and physics residuals expect."""

    def __init__(self, hr_norm: torch.Tensor, scale_factor: int = 8,
                 degradation: str = "pool") -> None:
        self.hr = hr_norm.contiguous()
        self.scale = int(scale_factor)
        self.degradation = degradation
        self.lr = _degrade(self.hr, self.scale, degradation)  # (T,3,H/s,W/s)

    def __len__(self) -> int:
        return self.hr.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"x_hr": self.hr[i], "x_lr": self.lr[i]}
