"""HRRR 8x super-resolution: physics vs. no-physics vs. bicubic, multi-seed.

The de-risking experiment for the spatial-intelligence program: on REAL HRRR
fields (precip, u, v), does the physics-residual term beat a no-physics flow and
a non-learned bicubic baseline, and is a split-conformal interval calibrated?

For each seed we train two identical rectified-flow / DiT models that differ
ONLY in the physics term (mass-conservation + divergence-free wind vs. none),
evaluate both plus a bicubic baseline on a held-out temporal split, and report
RMSE, the two physics-residual magnitudes, and PSNR. We then fit a marginal
split-conformal radius on half the val set and measure empirical coverage on the
other half. Nothing about the model or the residuals is HRRR-specific; only the
data loader is (physflow.data.make_hrrr_datasets).

Usage:
    python scripts/run_hrrr_experiment.py --out results/hrrr --steps 3000
    python scripts/run_hrrr_experiment.py --smoke            # fast sanity run
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from physflow.data import make_hrrr_datasets
from physflow.flow import RectifiedFlow
from physflow.flow.sampling import euler_sample
from physflow.models import DiTVelocity
from physflow.physics import DivergenceResidual, MassConservationResidual, PhysicsResidual
from physflow.physics.operators import average_pool, horizontal_divergence

SCALE = 8
HR = 128


def build_physics(div_w: float, mass_w: float) -> PhysicsResidual:
    """Mass conservation (exact at GT) + a softer divergence-free wind prior."""
    return PhysicsResidual([
        DivergenceResidual(dx=1.0, dy=1.0, weight=div_w),
        MassConservationResidual(scale_factor=SCALE, weight=mass_w),
    ])


def mass_residual(pred: torch.Tensor, x_lr: torch.Tensor) -> float:
    return float((average_pool(pred, SCALE) - x_lr).pow(2).mean())


def div_residual(pred: torch.Tensor) -> float:
    return float(horizontal_divergence(pred[:, 0], pred[:, 1]).pow(2).mean())


def grad_rmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """RMSE of spatial gradients: edge/detail fidelity (penalizes blur)."""
    def g(x):
        return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]
    pgx, pgy = g(pred)
    ggx, ggy = g(gt)
    return float(((pgx - ggx).pow(2).mean() + (pgy - ggy).pow(2).mean()).sqrt())


def hf_spectral_rmse(pred: torch.Tensor, gt: torch.Tensor, cutoff: float = 0.25) -> float:
    """RMSE of the high-frequency band of the 2D spectrum: the detail a blurry
    interpolant cannot recover, where a generative model is meant to win."""
    P, G = torch.fft.rfft2(pred), torch.fft.rfft2(gt)
    h, w = pred.shape[-2], pred.shape[-1]
    fy = torch.fft.fftfreq(h).abs()[:, None]
    fx = torch.fft.rfftfreq(w).abs()[None, :]
    mask = ((fy > cutoff) | (fx > cutoff)).to(P.real.dtype)
    return float(((P - G).abs() * mask).pow(2).mean().sqrt())


@torch.no_grad()
def evaluate(velocity, val_hr, val_lr, device, eval_seed: int = 1234, mb: int = 16) -> dict:
    """Sample HR from LR (25-step Euler) and score against the GT HR field.

    The sampler noise is reseeded identically for every model so RMSE
    differences reflect the model, not the draw."""
    velocity.eval()
    preds = []
    for s in range(0, val_lr.shape[0], mb):
        lr = val_lr[s:s + mb].to(device)
        torch.manual_seed(eval_seed + s)  # same noise across models
        pred = euler_sample(velocity, (lr.shape[0], 3, HR, HR), {"x_lr": lr},
                            steps=25, device=device)
        preds.append(pred.cpu())
    pred = torch.cat(preds, dim=0)
    rmse = float((pred - val_hr).pow(2).mean().sqrt())
    data_range = float(val_hr.max() - val_hr.min())
    psnr = float(20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10((pred - val_hr).pow(2).mean()))
    return {
        "rmse": rmse,
        "psnr": psnr,
        "grad_rmse": grad_rmse(pred, val_hr),
        "hf_rmse": hf_spectral_rmse(pred, val_hr),
        "mass_resid": mass_residual(pred, val_lr),
        "div_resid": div_residual(pred),
        "_pred": pred,  # kept for conformal; stripped before json
    }


def conformal_coverage(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 0.1) -> dict:
    """Marginal split conformal: fit a constant radius on half the val set,
    measure empirical coverage at nominal (1-alpha) on the other half."""
    n = pred.shape[0]
    half = n // 2
    err_cal = (pred[:half] - gt[:half]).abs().flatten()
    q = float(torch.quantile(err_cal, 1.0 - alpha))
    err_test = (pred[half:] - gt[half:]).abs()
    cov = float((err_test <= q).float().mean())
    return {"nominal": 1.0 - alpha, "empirical_coverage": cov, "radius": q}


def train_one(physics: bool, train_ds, val_hr, val_lr, args, device, seed: int) -> dict:
    torch.manual_seed(seed)
    velocity = DiTVelocity(in_channels=3, hidden=args.hidden, depth=args.depth,
                           heads=args.heads, patch=8).to(device)
    residual = build_physics(args.div_w, args.mass_w) if physics else None
    flow = RectifiedFlow(velocity_model=velocity, physics_residual=residual,
                         physics_weight=(args.phys_weight if physics else 0.0)).to(device)
    opt = torch.optim.AdamW(velocity.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)

    velocity.train()
    it = iter(loader)
    t0 = time.time()
    last = 0.0
    for _step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x_hr = batch["x_hr"].to(device)
        cond = {"x_lr": batch["x_lr"].to(device)}
        out = flow.training_step(x_hr, cond)
        opt.zero_grad(set_to_none=True)
        out.total_loss.backward()
        opt.step()
        last = float(out.total_loss)
    metrics = evaluate(velocity, val_hr, val_lr, device)
    pred = metrics.pop("_pred")
    res = {"physics": physics, "seed": seed, "final_train_loss": last,
           "train_secs": round(time.time() - t0, 1), **metrics}
    if physics:
        res["conformal"] = conformal_coverage(pred, val_hr)
    return res


def bicubic_baseline(val_hr, val_lr) -> dict:
    pred = F.interpolate(val_lr, size=(HR, HR), mode="bicubic", align_corners=False)
    rmse = float((pred - val_hr).pow(2).mean().sqrt())
    data_range = float(val_hr.max() - val_hr.min())
    psnr = float(20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10((pred - val_hr).pow(2).mean()))
    return {"method": "bicubic", "rmse": rmse, "psnr": psnr,
            "grad_rmse": grad_rmse(pred, val_hr), "hf_rmse": hf_spectral_rmse(pred, val_hr),
            "mass_resid": mass_residual(pred, val_lr), "div_resid": div_residual(pred)}


def agg(runs, key):
    xs = [r[key] for r in runs]
    return {"mean": statistics.mean(xs), "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch.global/arunshar/hrrr128")
    ap.add_argument("--out", default="results/hrrr")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--phys-weight", type=float, default=1.0, dest="phys_weight")
    ap.add_argument("--mass-w", type=float, default=1.0, dest="mass_w")
    ap.add_argument("--div-w", type=float, default=0.1, dest="div_w")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--degradation", default="bicubic", choices=["pool", "bicubic"],
                    help="LR forward operator; bicubic is the standard SR degradation")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run for validation")
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.steps, args.hidden, args.depth = [0], 30, 64, 2

    device = torch.device(args.device)
    train_ds, val_ds, stats = make_hrrr_datasets(args.root, SCALE, degradation=args.degradation)
    val_hr = val_ds.hr
    val_lr = val_ds.lr
    if args.smoke:
        val_hr, val_lr = val_hr[:8], val_lr[:8]
    print(f"[hrrr] device={device} train={len(train_ds)} val={val_hr.shape[0]} "
          f"stats={stats['channels']} mean={[round(m,3) for m in stats['mean']]}", flush=True)

    runs = {"physics": [], "nophysics": []}
    for seed in args.seeds:
        for physics in (False, True):
            r = train_one(physics, train_ds, val_hr, val_lr, args, device, seed)
            tag = "physics" if physics else "nophysics"
            runs[tag].append(r)
            extra = f" cov={r['conformal']['empirical_coverage']:.3f}" if physics else ""
            print(f"[hrrr] seed={seed} {tag:9s} rmse={r['rmse']:.4f} grad={r['grad_rmse']:.4f} "
                  f"hf={r['hf_rmse']:.4f} mass={r['mass_resid']:.5f} div={r['div_resid']:.5f}{extra} "
                  f"({r['train_secs']}s)", flush=True)

    bicubic = bicubic_baseline(val_hr, val_lr)
    print(f"[hrrr] bicubic    rmse={bicubic['rmse']:.4f} grad={bicubic['grad_rmse']:.4f} "
          f"hf={bicubic['hf_rmse']:.4f} mass={bicubic['mass_resid']:.5f} div={bicubic['div_resid']:.5f}", flush=True)

    keys = ("rmse", "psnr", "grad_rmse", "hf_rmse", "mass_resid", "div_resid")
    summary = {
        "config": vars(args), "stats": stats, "bicubic": bicubic,
        "nophysics": {k: agg(runs["nophysics"], k) for k in keys},
        "physics": {k: agg(runs["physics"], k) for k in keys},
        "physics_conformal": agg([r["conformal"] for r in runs["physics"]], "empirical_coverage")
        if runs["physics"] else None,
        "runs": runs,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    p, n, b = summary["physics"], summary["nophysics"], bicubic
    print(f"\n=== HRRR 8x SR ({args.degradation} degradation): physics vs no-physics vs bicubic (mean over seeds) ===", flush=True)
    print(f"  RMSE       bicubic={b['rmse']:.4f}  nophysics={n['rmse']['mean']:.4f}  physics={p['rmse']['mean']:.4f}", flush=True)
    print(f"  grad_rmse  bicubic={b['grad_rmse']:.4f}  nophysics={n['grad_rmse']['mean']:.4f}  physics={p['grad_rmse']['mean']:.4f}", flush=True)
    print(f"  hf_rmse    bicubic={b['hf_rmse']:.4f}  nophysics={n['hf_rmse']['mean']:.4f}  physics={p['hf_rmse']['mean']:.4f}", flush=True)
    print(f"  mass_resid bicubic={b['mass_resid']:.5f}  nophysics={n['mass_resid']['mean']:.5f}  physics={p['mass_resid']['mean']:.5f}", flush=True)
    print(f"  div_resid  bicubic={b['div_resid']:.5f}  nophysics={n['div_resid']['mean']:.5f}  physics={p['div_resid']['mean']:.5f}", flush=True)
    if summary["physics_conformal"]:
        print(f"  conformal coverage (nominal 0.90): {summary['physics_conformal']['mean']:.3f}", flush=True)
    print(f"\n[hrrr] wrote {out/'summary.json'}", flush=True)
    (out / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
