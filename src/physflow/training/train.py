"""Training entry point.

    python -m physflow.training.train +experiment=sentinel2_x4
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def make_loader(cfg: DictConfig) -> DataLoader:
    if cfg.dataset.kind == "worldstrat":
        from physflow.data import WorldStratDataset
        ds = WorldStratDataset(root=cfg.dataset.root, split=cfg.dataset.split)
    elif cfg.dataset.kind == "era5_chirps":
        from physflow.data import ERA5CHIRPSDataset
        ds = ERA5CHIRPSDataset(root=cfg.dataset.root, split=cfg.dataset.split, scale_factor=cfg.dataset.scale_factor)
    else:
        raise ValueError(f"unknown dataset {cfg.dataset.kind!r}")
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.workers)


def make_residual(cfg: DictConfig):
    if not cfg.physics.enabled:
        return None
    from physflow.physics import (
        BandRatioResidual,
        DivergenceResidual,
        MassConservationResidual,
        PhysicsResidual,
    )
    from physflow.physics.residual import BandIndex
    items = []
    if cfg.physics.band_ratio.enabled:
        items.append(
            BandRatioResidual(
                scale_factor=cfg.physics.scale_factor,
                bands=BandIndex(**cfg.physics.band_ratio.bands),
                weight=cfg.physics.band_ratio.weight,
            )
        )
    if cfg.physics.divergence.enabled:
        items.append(
            DivergenceResidual(
                dx=cfg.physics.divergence.dx,
                dy=cfg.physics.divergence.dy,
                weight=cfg.physics.divergence.weight,
            )
        )
    if cfg.physics.mass_conservation.enabled:
        items.append(
            MassConservationResidual(
                scale_factor=cfg.physics.scale_factor,
                weight=cfg.physics.mass_conservation.weight,
            )
        )
    return PhysicsResidual(items)


@hydra.main(version_base=None, config_path=str(Path(__file__).parents[3] / "configs"), config_name="default")
def main(cfg: DictConfig) -> None:
    logger.info("config:\n%s", OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.device)

    from physflow.flow import RectifiedFlow
    from physflow.models import DiTVelocity, PhysFlowPipeline
    from physflow.models.pipeline import PhysFlowConfig

    velocity = DiTVelocity(
        in_channels=cfg.model.in_channels,
        hidden=cfg.model.hidden,
        depth=cfg.model.depth,
        heads=cfg.model.heads,
        patch=cfg.model.patch,
    ).to(device)
    flow = RectifiedFlow(
        velocity_model=velocity,
        physics_residual=make_residual(cfg),
        physics_weight=cfg.physics.weight,
    ).to(device)
    opt = torch.optim.AdamW(velocity.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)

    loader = make_loader(cfg)
    if cfg.logging.wandb:
        import wandb
        wandb.init(project="physflow-earth", config=OmegaConf.to_container(cfg, resolve=True))

    step = 0
    for epoch in range(cfg.train.epochs):
        for batch in loader:
            x_hr = batch.x_hr.to(device) if hasattr(batch, "x_hr") else batch["x_hr"].to(device)
            x_lr = batch.x_lr.to(device) if hasattr(batch, "x_lr") else batch["x_lr"].to(device)
            opt.zero_grad()
            out = flow.training_step(x_hr, condition={"x_lr": x_lr})
            out.total_loss.backward()
            opt.step()
            step += 1
            if step % cfg.logging.every == 0:
                logger.info(
                    "step=%d v=%.4f phys=%.4f total=%.4f",
                    step, out.velocity_loss.item(), out.physics_loss.item(), out.total_loss.item(),
                )
                if cfg.logging.wandb:
                    wandb.log({
                        "velocity_loss": out.velocity_loss.item(),
                        "physics_loss": out.physics_loss.item(),
                        "total_loss": out.total_loss.item(),
                        "step": step,
                    })

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = PhysFlowPipeline(velocity, PhysFlowConfig(
        in_channels=cfg.model.in_channels,
        hidden=cfg.model.hidden,
        depth=cfg.model.depth,
        heads=cfg.model.heads,
        patch=cfg.model.patch,
        image_hw=tuple(cfg.image_hw),
        sampler_steps=cfg.sampling.steps,
    ))
    pipeline.save_pretrained(out_dir)
    logger.info("saved to %s", out_dir)


if __name__ == "__main__":
    main()
