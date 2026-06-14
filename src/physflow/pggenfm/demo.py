"""Self-contained demo: physics violation drops after guidance.

A toy generator proposes a high-resolution field that does NOT respect the coarse
observation (its block means drift and it has spurious divergence). Physics
guidance projects it back onto the mass-conservation + low-divergence manifold,
and the measured violation drops. Reports REAL before/after numbers.

    python -m physflow.pggenfm.demo
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from physflow.physics.residual import DivergenceResidual, MassConservationResidual, PhysicsResidual
from physflow.pggenfm.guidance import physics_guided_refine


def _toy_generator(x_lr: torch.Tensor, scale: int, noise: float = 0.3) -> torch.Tensor:
    """Propose an HR field: bilinear upsample plus structured noise (violates physics)."""
    up = F.interpolate(x_lr, scale_factor=scale, mode="bilinear", align_corners=False)
    g = torch.Generator().manual_seed(0)
    return up + noise * torch.randn(up.shape, generator=g)


def run_demo(batch: int = 4, channels: int = 2, lr_hw: int = 16, scale: int = 4,
             steps: int = 150, device: str | None = None) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gen = torch.Generator().manual_seed(1)
    x_lr = torch.randn(batch, channels, lr_hw, lr_hw, generator=gen).to(dev)
    condition = {"x_lr": x_lr}

    energy = PhysicsResidual([
        MassConservationResidual(scale_factor=scale, weight=1.0),
        DivergenceResidual(dx=1.0, dy=1.0, weight=0.2),
    ]).to(dev)

    x0 = _toy_generator(x_lr.cpu(), scale).to(dev)
    res = physics_guided_refine(x0, energy, condition, steps=steps, lr=0.2, anchor=0.05)
    return {
        "violation_before": res.violation_before,
        "violation_after": res.violation_after,
        "reduction_pct": 100.0 * res.reduction,
    }


def main() -> None:
    m = run_demo()
    print("Physics-guided generation (measured this run):")
    print(f"  physics violation before guidance : {m['violation_before']:.5f}")
    print(f"  physics violation after guidance  : {m['violation_after']:.5f}")
    print(f"  reduction                         : {m['reduction_pct']:.1f}%")


if __name__ == "__main__":
    main()
