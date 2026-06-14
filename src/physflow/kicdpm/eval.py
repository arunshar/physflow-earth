"""Evaluate Ki-CDPM downscaling vs bilinear and kriging-only baselines.

Reports REAL RMSE measured this run on a held-out synthetic split:

    bilinear      F.interpolate(coarse, scale, bilinear)
    kriging-only  the kriging field y (Ki-CDPM's structural prior, no diffusion)
    Ki-CDPM       y + learned diffusion residual

    python -m physflow.kicdpm.eval --ckpt /tmp/kicdpm.pt --n 512
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .config import KiCDPMConfig
from .data import FieldDataset
from .model import KiCDPM


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean().sqrt())


def _highpass(x: torch.Tensor, factor: int) -> torch.Tensor:
    """High-frequency component: x minus its coarsened-then-upsampled low pass."""
    low = F.interpolate(F.avg_pool2d(x, factor), scale_factor=factor, mode="bilinear", align_corners=False)
    return x - low


def evaluate(model: KiCDPM, cfg: KiCDPMConfig, n: int = 512, device: str | None = None,
             n_samples: int = 8) -> dict[str, float]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev).eval()
    ds = FieldDataset(cfg, n=n, seed=cfg.seed + 5)
    bil, kri, kic = [], [], []
    bil_hf, kic_hf = [], []  # high-frequency band (the detail interpolation misses)
    sf = cfg.scale_factor
    for i in range(0, len(ds), cfg.batch_size):
        coarse = torch.from_numpy(ds.coarse[i : i + cfg.batch_size])[:, None].to(dev)
        fine = torch.from_numpy(ds.fine[i : i + cfg.batch_size])[:, None].to(dev)
        bilinear = F.interpolate(coarse, scale_factor=sf, mode="bilinear", align_corners=False)
        y = model.krige(coarse)
        pred = model.downscale(coarse, n_samples=n_samples)
        bil.append(_rmse(bilinear, fine))
        kri.append(_rmse(y, fine))
        kic.append(_rmse(pred, fine))
        fine_hf = _highpass(fine, sf)
        bil_hf.append(_rmse(_highpass(bilinear, sf), fine_hf))
        kic_hf.append(_rmse(_highpass(pred, sf), fine_hf))
    mean = lambda xs: sum(xs) / len(xs)
    return {"rmse_bilinear": mean(bil), "rmse_kriging_only": mean(kri), "rmse_kicdpm": mean(kic),
            "rmse_highfreq_bilinear": mean(bil_hf), "rmse_highfreq_kicdpm": mean(kic_hf),
            "n_test": float(len(ds))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args()
    if a.ckpt:
        model = KiCDPM.from_checkpoint(a.ckpt)
        cfg = model.cfg
    else:
        cfg = KiCDPMConfig()
        model = KiCDPM(cfg)
    m = evaluate(model, cfg, n=a.n, device=a.device)
    print("Ki-CDPM downscaling RMSE (measured this run, lower is better):")
    for k, v in m.items():
        print(f"  {k:20s} {v:.4f}")


if __name__ == "__main__":
    main()
