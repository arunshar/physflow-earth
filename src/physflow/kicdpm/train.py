"""Train Ki-CDPM on synthetic coarse/fine field pairs.

    python -m physflow.kicdpm.train --epochs 30 --n 4096 --out /tmp/kicdpm.pt

CPU-runnable for a smoke test; uses CUDA + AMP automatically when available.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .config import KiCDPMConfig
from .data import FieldDataset
from .model import KiCDPM


def train(cfg: KiCDPMConfig, n: int = 4096, out: str | None = None, device: str | None = None) -> KiCDPM:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    ds = FieldDataset(cfg, n=n, seed=cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)
    model = KiCDPM(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    for epoch in range(cfg.epochs):
        running = 0.0
        for coarse, fine in dl:
            coarse, fine = coarse.to(dev), fine.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                loss = model.training_loss(coarse, fine)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += float(loss)
        print(f"epoch {epoch + 1:3d}/{cfg.epochs}  eps_mse={running / len(dl):.4f}")
    if out:
        model.save(out)
        print(f"saved checkpoint -> {out}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--coarse", type=int, default=None)
    ap.add_argument("--scale-factor", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args()
    base = KiCDPMConfig()
    cfg = KiCDPMConfig(
        epochs=a.epochs or base.epochs,
        coarse=a.coarse or base.coarse,
        scale_factor=a.scale_factor or base.scale_factor,
    )
    train(cfg, n=a.n, out=a.out, device=a.device)


if __name__ == "__main__":
    main()
