"""Reproduce WorldStrat leaderboard with image and physics metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from physflow.data import WorldStratDataset
from physflow.models import PhysFlowPipeline
from physflow.physics.residual import BandIndex, BandRatioResidual


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().clamp_min(1e-12)
    return float(10 * torch.log10(1.0 / mse))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", required=True)
    p.add_argument("--root", default="data/worldstrat")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="results/worldstrat.json")
    args = p.parse_args()

    pipeline = PhysFlowPipeline.from_pretrained(args.pipeline)
    ds = WorldStratDataset(root=args.root, split=args.split)
    bands = BandIndex(red=0, nir=3, green=1, swir=3)  # SWIR not in WS; reuse NIR for stub
    ndvi_residual = BandRatioResidual(scale_factor=4, bands=bands)

    rows = []
    for i in range(len(ds)):
        s = ds[i]
        x_lr = s.x_lr.unsqueeze(0)
        with torch.no_grad():
            sr = pipeline(x_lr)
        target = s.x_hr.unsqueeze(0)
        ndvi_res = ndvi_residual.residual(sr, condition={"x_lr": x_lr}).abs().mean().item()
        rows.append({
            "aoi": s.aoi,
            "psnr": psnr(sr, target),
            "ndvi_residual": ndvi_res,
        })

    summary = {
        "psnr_mean": sum(r["psnr"] for r in rows) / max(len(rows), 1),
        "ndvi_residual_mean": sum(r["ndvi_residual"] for r in rows) / max(len(rows), 1),
        "n": len(rows),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_aoi": rows, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
