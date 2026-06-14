"""Small 2D conditional U-Net denoiser for Ki-CDPM.

Predicts the diffusion noise epsilon given the noisy residual x_t concatenated
with the kriging condition y, plus a sinusoidal timestep embedding injected into
every residual block (FiLM-style additive bias).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KiCDPMConfig


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet2D(nn.Module):
    """eps_theta(x_t, y, t): in = (residual C + condition C), out = residual C."""

    def __init__(self, cfg: KiCDPMConfig) -> None:
        super().__init__()
        c = cfg.channels
        base = cfg.base_ch
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_dim, cfg.time_dim), nn.SiLU(), nn.Linear(cfg.time_dim, cfg.time_dim)
        )
        self.time_dim = cfg.time_dim
        self.in_conv = nn.Conv2d(2 * c, base, 3, padding=1)
        chs = [base * m for m in cfg.ch_mult]
        # encoder
        self.down = nn.ModuleList()
        prev = base
        for ch in chs:
            self.down.append(ResBlock(prev, ch, cfg.time_dim))
            prev = ch
        self.pool = nn.AvgPool2d(2)
        # bottleneck
        self.mid = ResBlock(prev, prev, cfg.time_dim)
        # decoder (mirror)
        self.up = nn.ModuleList()
        for ch in reversed(chs):
            self.up.append(ResBlock(prev + ch, ch, cfg.time_dim))
            prev = ch
        self.out = nn.Sequential(nn.GroupNorm(min(8, prev), prev), nn.SiLU(), nn.Conv2d(prev, c, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.time_dim))
        h = self.in_conv(torch.cat([x_t, y], dim=1))
        skips = []
        for i, blk in enumerate(self.down):
            h = blk(h, temb)
            skips.append(h)
            if i < len(self.down) - 1:
                h = self.pool(h)
        h = self.mid(h, temb)
        for blk in self.up:
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = blk(torch.cat([h, skip], dim=1), temb)
        return self.out(h)
