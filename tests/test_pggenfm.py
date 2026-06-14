"""Tests for physics-guided generation (pggenfm)."""

from __future__ import annotations

import torch

from physflow.physics.residual import MassConservationResidual, PhysicsResidual
from physflow.pggenfm.demo import run_demo
from physflow.pggenfm.guidance import physics_guided_refine


def test_guidance_reduces_violation():
    scale = 4
    x_lr = torch.randn(2, 1, 8, 8)
    energy = PhysicsResidual([MassConservationResidual(scale_factor=scale, weight=1.0)])
    x0 = torch.randn(2, 1, 32, 32)  # arbitrary proposal, violates avgpool == x_lr
    res = physics_guided_refine(x0, energy, {"x_lr": x_lr}, steps=100, lr=0.2, anchor=0.0)
    assert res.violation_after < res.violation_before
    assert res.reduction > 0.5  # mass conservation is convex, guidance drives it down hard


def test_anchor_keeps_sample_near_proposal():
    x_lr = torch.randn(1, 1, 8, 8)
    energy = PhysicsResidual([MassConservationResidual(scale_factor=4, weight=1.0)])
    x0 = torch.randn(1, 1, 32, 32)
    strong = physics_guided_refine(x0, energy, {"x_lr": x_lr}, steps=80, lr=0.2, anchor=2.0)
    weak = physics_guided_refine(x0, energy, {"x_lr": x_lr}, steps=80, lr=0.2, anchor=0.0)
    # a stronger anchor keeps the guided sample closer to the proposal
    assert (strong.sample - x0).pow(2).mean() < (weak.sample - x0).pow(2).mean()


def test_demo_runs():
    # channels=2 so the divergence residual has a (u, v) pair to act on
    m = run_demo(batch=2, channels=2, lr_hw=8, scale=4, steps=60)
    assert m["violation_after"] < m["violation_before"]
