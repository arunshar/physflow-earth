"""Tests for Ki-CDPM (kriging-informed conditional diffusion downscaling)."""

from __future__ import annotations

import torch

from physflow.kicdpm.config import KiCDPMConfig
from physflow.kicdpm.data import FieldDataset
from physflow.kicdpm.kriging import OrdinaryKriging
from physflow.kicdpm.model import KiCDPM


def _tiny() -> KiCDPMConfig:
    return KiCDPMConfig(coarse=4, scale_factor=4, base_ch=16, timesteps=40, ddim_steps=4, batch_size=16, epochs=1)


def test_kriging_shapes():
    cfg = _tiny()
    k = OrdinaryKriging(cfg)
    out = k(torch.randn(3, 1, cfg.coarse, cfg.coarse))
    assert out.shape == (3, 1, cfg.fine, cfg.fine)


def test_kriging_reproduces_constant():
    # ordinary-kriging weights sum to 1 (Lagrange constraint), so a constant
    # coarse field maps to the same constant on the fine grid
    cfg = _tiny()
    k = OrdinaryKriging(cfg)
    out = k(torch.full((2, 1, cfg.coarse, cfg.coarse), 0.7))
    assert torch.allclose(out, torch.full_like(out, 0.7), atol=1e-3)


def test_q_sample_roundtrip():
    cfg = _tiny()
    m = KiCDPM(cfg)
    r = torch.randn(2, 1, cfg.fine, cfg.fine)
    t = torch.zeros(2, dtype=torch.long)
    noise = torch.randn_like(r)
    x_t = m.diffusion.q_sample(r, t, noise)
    r_hat = m.diffusion.predict_x0(x_t, t, noise)
    assert torch.allclose(r, r_hat, atol=1e-4)


def test_training_loss_backward():
    cfg = _tiny()
    m = KiCDPM(cfg)
    coarse = torch.randn(2, 1, cfg.coarse, cfg.coarse)
    fine = torch.randn(2, 1, cfg.fine, cfg.fine)
    loss = m.training_loss(coarse, fine)
    assert loss.ndim == 0 and loss.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_downscale_shape_and_stability():
    cfg = _tiny()
    m = KiCDPM(cfg).eval()
    coarse = torch.randn(2, 1, cfg.coarse, cfg.coarse)
    out = m.downscale(coarse, n_samples=2)
    assert out.shape == (2, 1, cfg.fine, cfg.fine)
    assert torch.isfinite(out).all()  # the x0 clamp keeps sampling finite


def test_dataset_pairs():
    cfg = _tiny()
    ds = FieldDataset(cfg, n=8, seed=0)
    coarse, fine = ds[0]
    assert coarse.shape == (1, cfg.coarse, cfg.coarse)
    assert fine.shape == (1, cfg.fine, cfg.fine)
