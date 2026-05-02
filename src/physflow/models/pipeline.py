"""Diffusers-compatible pipeline wrapper for distribution + inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from physflow.flow.sampling import euler_sample
from physflow.models.dit import DiTVelocity


@dataclass
class PhysFlowConfig:
    in_channels: int = 4
    hidden: int = 768
    depth: int = 8
    heads: int = 12
    patch: int = 4
    image_hw: tuple[int, int] = (256, 256)
    sampler_steps: int = 25


class PhysFlowPipeline(nn.Module):
    """``from_pretrained`` loadable wrapper following the Diffusers convention."""

    def __init__(self, model: DiTVelocity, config: PhysFlowConfig):
        super().__init__()
        self.model = model
        self.config = config

    @torch.no_grad()
    def __call__(self, x_lr: Tensor, *, steps: int | None = None) -> Tensor:
        steps = steps or self.config.sampler_steps
        H, W = self.config.image_hw
        return euler_sample(
            self.model,
            shape=(x_lr.shape[0], self.config.in_channels, H, W),
            condition={"x_lr": x_lr},
            steps=steps,
            device=x_lr.device,
            dtype=x_lr.dtype,
        )

    @classmethod
    def from_pretrained(cls, repo_id_or_path: str | Path) -> "PhysFlowPipeline":
        from huggingface_hub import snapshot_download

        path = (
            Path(repo_id_or_path)
            if Path(repo_id_or_path).exists()
            else Path(snapshot_download(repo_id=str(repo_id_or_path)))
        )
        cfg = PhysFlowConfig(**__import__("json").loads((path / "config.json").read_text()))
        model = DiTVelocity(
            in_channels=cfg.in_channels,
            hidden=cfg.hidden,
            depth=cfg.depth,
            heads=cfg.heads,
            patch=cfg.patch,
        )
        sd = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        return cls(model, cfg)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        (path / "config.json").write_text(__import__("json").dumps(self.config.__dict__))
