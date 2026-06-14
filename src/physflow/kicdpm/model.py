"""Ki-CDPM: kriging base + learned conditional-diffusion residual."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import KiCDPMConfig
from .diffusion import GridDiffusion
from .kriging import OrdinaryKriging
from .unet import ConditionalUNet2D


class KiCDPM(nn.Module):
    def __init__(self, cfg: KiCDPMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or KiCDPMConfig()
        self.kriging = OrdinaryKriging(self.cfg)
        self.unet = ConditionalUNet2D(self.cfg)
        self.diffusion = GridDiffusion(self.unet, self.cfg)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def krige(self, coarse: torch.Tensor) -> torch.Tensor:
        return self.kriging(coarse)

    def training_loss(self, coarse: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
        y = self.kriging(coarse)
        residual = fine - y
        return self.diffusion.loss(residual, y)

    @torch.no_grad()
    def downscale(self, coarse: torch.Tensor, steps: int | None = None, n_samples: int = 1) -> torch.Tensor:
        """Coarse field (B, C, c, c) -> fine field (B, C, f, f) = kriging + residual.

        n_samples > 1 averages the diffusion residual (a Monte-Carlo estimate of
        the conditional mean E[r | y]), which is the RMSE-optimal point estimate.
        """
        self.eval()
        y = self.kriging(coarse)
        residual = self.diffusion.sample(y, steps=steps)
        for _ in range(n_samples - 1):
            residual = residual + self.diffusion.sample(y, steps=steps)
        return y + residual / n_samples

    # ----------------------------------------------------------------- io
    @classmethod
    def from_checkpoint(cls, path: str, map_location: str | torch.device = "cpu") -> "KiCDPM":
        ckpt = torch.load(path, map_location=map_location)
        model = cls(ckpt.get("cfg") or KiCDPMConfig())
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model

    def save(self, path: str) -> None:
        torch.save({"model": self.state_dict(), "cfg": self.cfg}, path)
