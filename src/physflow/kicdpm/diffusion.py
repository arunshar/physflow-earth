"""Gaussian DDPM over the kriging residual, conditioned on the kriging field.

eps-prediction. The diffusion models the residual r = fine - y (y = kriging
field), so the network only has to learn the high-frequency structure kriging
smooths away. DDIM is used for sampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KiCDPMConfig
from .unet import ConditionalUNet2D


def _cosine_betas(t: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, t, t + 1)
    acp = torch.cos(((x / t) + s) / (1 + s) * torch.pi * 0.5) ** 2
    acp = acp / acp[0]
    return (1 - acp[1:] / acp[:-1]).clamp(1e-8, 0.999)


def _gather(a: torch.Tensor, idx: torch.Tensor, ndim: int) -> torch.Tensor:
    return a.gather(0, idx).reshape(idx.shape[0], *([1] * (ndim - 1)))


class GridDiffusion(nn.Module):
    def __init__(self, model: ConditionalUNet2D, cfg: KiCDPMConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.num_t = cfg.timesteps
        betas = _cosine_betas(cfg.timesteps) if cfg.beta_schedule == "cosine" else torch.linspace(1e-4, 0.02, cfg.timesteps)
        acp = torch.cumprod(1 - betas, dim=0)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("sqrt_acp", acp.sqrt())
        self.register_buffer("sqrt_one_minus_acp", (1 - acp).sqrt())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return _gather(self.sqrt_acp, t, x0.ndim) * x0 + _gather(self.sqrt_one_minus_acp, t, x0.ndim) * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (x_t - _gather(self.sqrt_one_minus_acp, t, x_t.ndim) * eps) / _gather(self.sqrt_acp, t, x_t.ndim)

    def loss(self, residual: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b = residual.shape[0]
        t = torch.randint(0, self.num_t, (b,), device=residual.device)
        noise = torch.randn_like(residual)
        x_t = self.q_sample(residual, t, noise)
        eps_hat = self.model(x_t, y, t)
        return F.mse_loss(eps_hat, noise)

    @torch.no_grad()
    def sample(self, y: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        """Sample a residual conditioned on y, via deterministic DDIM."""
        steps = steps or self.cfg.ddim_steps
        device = y.device
        ts = torch.linspace(self.num_t - 1, 0, steps + 1).round().long().to(device)
        x = torch.randn_like(y)
        for i in range(steps):
            t_cur = ts[i].expand(y.shape[0])
            t_nxt = ts[i + 1].expand(y.shape[0])
            eps = self.model(x, y, t_cur)
            x0 = self.predict_x0(x, t_cur, eps).clamp(-self.cfg.x0_clamp, self.cfg.x0_clamp)
            acp_nxt = _gather(self.alphas_cumprod, t_nxt, x.ndim)
            x = acp_nxt.sqrt() * x0 + (1 - acp_nxt).sqrt() * eps
        return x
