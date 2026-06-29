"""WorldStrat paired LR Sentinel-2 / HR SPOT-6/7 super-resolution dataset.

Cornebise et al., NeurIPS 2022. Available on Hugging Face Datasets:
``worldstrat/worldstrat``. We load 4-band stacks (R, G, B, NIR) at the LR
(10 m) and HR (1.5 m) scales and normalize to [-1, 1].
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class WorldStratSample:
    x_lr: torch.Tensor    # (4, 64, 64)
    x_hr: torch.Tensor    # (4, 256, 256)
    aoi: str
    band_names: tuple[str, ...]


class WorldStratDataset(Dataset):
    BANDS = ("B4_red", "B3_green", "B2_blue", "B8_nir")

    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        self.split = split
        self.entries = self._index()

    def _index(self) -> list[Path]:
        idx = self.root / f"{self.split}.txt"
        if not idx.exists():
            raise FileNotFoundError(
                f"WorldStrat split list missing at {idx}. "
                "Run scripts/download_worldstrat.sh."
            )
        return [self.root / line.strip() for line in idx.read_text().splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> WorldStratSample:
        path = self.entries[idx]
        lr = torch.load(path / "lr.pt", map_location="cpu", weights_only=True)
        hr = torch.load(path / "hr.pt", map_location="cpu", weights_only=True)
        return WorldStratSample(
            x_lr=lr.clamp(-1.0, 1.0),
            x_hr=hr.clamp(-1.0, 1.0),
            aoi=path.name,
            band_names=self.BANDS,
        )
