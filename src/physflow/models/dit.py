"""Diffusion Transformer (DiT) backbone with cross-attention to a physics codebook.

The model is a standard DiT (Peebles & Xie, ICCV 2023) with two extensions:

1. LR-tokenized conditioning concatenated to the patch sequence.
2. A cross-attention layer in each block that attends to a learned
   physics codebook (64 embeddings).

Output is a velocity prediction in the same shape as the input image.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from physflow.models.conditioning import LRTokenizer, PhysicsCodebook


def sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """Standard sinusoidal time embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    arg = t.unsqueeze(-1).to(freqs) * freqs
    emb = torch.cat([arg.sin(), arg.cos()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class DiTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int = 16, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(hidden, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, int(hidden * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden * mlp_ratio), hidden),
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x: Tensor, c: Tensor, codebook: Tensor) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h)
        x = x + gate1.unsqueeze(1) * attn_out
        cross_out, _ = self.cross_attn(self.norm_cross(x), codebook, codebook)
        x = x + cross_out
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.mlp(h)
        return x


class DiTVelocity(nn.Module):
    """DiT-XL velocity model with physics codebook cross-attention.

    Inputs:
        x_t: (B, C, H, W)
        t:   (B,)         normalized to [0, 1]
        condition: dict with at least ``x_lr`` (B, C, H_lr, W_lr).
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 1152,
        depth: int = 8,
        heads: int = 16,
        patch: int = 4,
        codebook_size: int = 64,
    ):
        super().__init__()
        self.patch = patch
        self.in_channels = in_channels
        self.hidden = hidden

        self.patch_embed = nn.Conv2d(in_channels, hidden, kernel_size=patch, stride=patch)
        self.lr_embed = LRTokenizer(in_channels=in_channels, hidden=hidden, patch=patch)
        self.codebook = PhysicsCodebook(n_codes=codebook_size, hidden=hidden)

        self.t_proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

        self.blocks = nn.ModuleList(DiTBlock(hidden, heads=heads) for _ in range(depth))
        self.norm_out = nn.LayerNorm(hidden, elementwise_affine=False)
        self.head = nn.Linear(hidden, patch * patch * in_channels)

    def forward(self, x: Tensor, t: Tensor, condition: dict) -> Tensor:
        B, C, H, W = x.shape
        z = self.patch_embed(x).flatten(2).transpose(1, 2)             # (B, N, hidden)
        z_lr = self.lr_embed(condition["x_lr"])                         # (B, N_lr, hidden)
        z = torch.cat([z, z_lr], dim=1)
        codebook = self.codebook(B)
        c = self.t_proj(sinusoidal_embedding(t, self.hidden))
        for blk in self.blocks:
            z = blk(z, c, codebook)
        z = self.norm_out(z)
        n_patches = (H // self.patch) * (W // self.patch)
        z = z[:, :n_patches]                                            # drop LR tokens
        z = self.head(z)                                                # (B, N, p*p*C)
        z = z.transpose(1, 2).reshape(B, self.patch * self.patch * C, H // self.patch, W // self.patch)
        z = F.pixel_shuffle(z, self.patch)
        return z
