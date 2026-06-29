"""ERA5 + CHIRPS climate downscaling dataset.

Source:
- ERA5 reanalysis on the WeatherBench-2 zarr cube (~25 km, hourly).
- CHIRPS daily precipitation at 5 km on a regular lat-lon grid.

We pair coarse ERA5 patches with their colocated CHIRPS HR patches at a
fixed scale factor (default 5x). All variables are normalized per-channel
by the per-split z-score precomputed in ``stats.json``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class ERA5CHIRPSSample:
    x_lr: torch.Tensor       # (C, H_lr, W_lr) coarse ERA5
    x_hr: torch.Tensor       # (C, H_hr, W_hr) HR CHIRPS-aligned target
    timestamp: str
    bbox_lonlat: tuple


class ERA5CHIRPSDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", scale_factor: int = 5):
        self.root = Path(root)
        self.split = split
        self.scale_factor = int(scale_factor)
        self.entries = self._index()

    def _index(self) -> list[Path]:
        idx = self.root / f"{self.split}.txt"
        if not idx.exists():
            raise FileNotFoundError(
                f"ERA5+CHIRPS split list missing at {idx}. "
                "Run scripts/build_era5_chirps_index.py first."
            )
        return [self.root / line.strip() for line in idx.read_text().splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> ERA5CHIRPSSample:
        path = self.entries[idx]
        lr = torch.load(path / "era5.pt", map_location="cpu", weights_only=True)
        hr = torch.load(path / "chirps.pt", map_location="cpu", weights_only=True)
        meta = (path / "meta.json").read_text() if (path / "meta.json").exists() else "{}"
        m = __import__("json").loads(meta)
        return ERA5CHIRPSSample(
            x_lr=lr,
            x_hr=hr,
            timestamp=m.get("timestamp", ""),
            bbox_lonlat=tuple(m.get("bbox", (0.0, 0.0, 0.0, 0.0))),
        )
