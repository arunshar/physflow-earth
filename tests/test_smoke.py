"""Smoke tests for HF Space deployment of physflow-earth."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_APP = REPO_ROOT / "space" / "app.py"


def _load_app_module():
    spec = importlib.util.spec_from_file_location("physflow_space_app", SPACE_APP)
    module = importlib.util.module_from_spec(spec)
    sys.modules["physflow_space_app"] = module
    spec.loader.exec_module(module)
    return module


# -- package imports ---------------------------------------------------------


def test_top_level_imports():
    import physflow
    from physflow import RectifiedFlow, PhysicsResidual
    assert physflow.__version__


def test_physics_imports():
    from physflow.physics import (
        BandRatioResidual,
        DivergenceResidual,
        MassConservationResidual,
        PhysicsResidual,
        average_pool,
        central_diff_x,
        central_diff_y,
        horizontal_divergence,
    )


def test_flow_imports():
    from physflow.flow import RectifiedFlow, euler_sample, consistency_sample


def test_models_imports():
    from physflow.models import (
        DiTVelocity,
        LRTokenizer,
        PhysFlowPipeline,
        PhysicsCodebook,
    )


# -- end-to-end DiT forward pass on synthetic data --------------------------


def test_dit_forward_shape():
    from physflow.models import DiTVelocity
    model = DiTVelocity(in_channels=4, hidden=64, depth=2, heads=4, patch=4, codebook_size=8)
    x = torch.randn(2, 4, 32, 32)
    t = torch.rand(2)
    cond = {"x_lr": torch.randn(2, 4, 16, 16)}
    out = model(x, t, cond)
    assert out.shape == (2, 4, 32, 32)
    assert torch.isfinite(out).all()


def test_pipeline_inference_shape():
    from physflow.models import DiTVelocity
    from physflow.models.pipeline import PhysFlowConfig, PhysFlowPipeline

    cfg = PhysFlowConfig(
        in_channels=4, hidden=64, depth=2, heads=4, patch=4,
        image_hw=(32, 32), sampler_steps=2,
    )
    model = DiTVelocity(in_channels=4, hidden=64, depth=2, heads=4, patch=4, codebook_size=8)
    pipe = PhysFlowPipeline(model, cfg)
    x_lr = torch.randn(1, 4, 16, 16)
    out = pipe(x_lr)
    assert out.shape == (1, 4, 32, 32)


def test_rectified_flow_step_e2e():
    from physflow.flow import RectifiedFlow
    from physflow.models import DiTVelocity
    from physflow.physics import MassConservationResidual, PhysicsResidual

    model = DiTVelocity(in_channels=1, hidden=32, depth=1, heads=4, patch=4, codebook_size=4)
    res = PhysicsResidual([MassConservationResidual(scale_factor=4)])
    flow = RectifiedFlow(model, physics_residual=res, physics_weight=1.0)
    x_hr = torch.randn(2, 1, 16, 16)
    out = flow.training_step(x_hr, condition={"x_lr": torch.randn(2, 1, 4, 4)})
    assert torch.isfinite(out.total_loss)
    out.total_loss.backward()
    assert any(p.grad is not None for p in model.parameters())


# -- Gradio app smoke -------------------------------------------------------


def test_space_app_importable():
    module = _load_app_module()
    assert hasattr(module, "build_ui")
    assert hasattr(module, "downscale")


def test_space_ui_builds():
    gr = pytest.importorskip("gradio")
    module = _load_app_module()
    ui = module.build_ui()
    assert isinstance(ui, gr.Blocks)


def test_space_constants_defined():
    module = _load_app_module()
    assert isinstance(module.VARIABLES, list) and len(module.VARIABLES) >= 1
    assert isinstance(module.SCENARIOS, list) and len(module.SCENARIOS) >= 1


# -- requirements + readme --------------------------------------------------


def test_space_requirements_parseable():
    req = REPO_ROOT / "space" / "requirements.txt"
    assert req.exists()
    text = req.read_text().lower()
    assert "gradio" in text
    assert "torch" in text


def test_space_readme_has_hf_frontmatter():
    readme = REPO_ROOT / "space" / "README.md"
    assert readme.exists()
    body = readme.read_text()
    assert body.startswith("---\n")
    assert "sdk: gradio" in body
    assert "app_file:" in body
