"""Conditioning blocks for PhysFlow-Earth.

We condition on (a) the low-res input via a frozen-VAE-tokenizer surrogate,
(b) a learned codebook of 64 physics-constraint embeddings cross-attended
in every DiT block. The codebook is the artifact that gives the model the
inductive bias to respect band ratios / divergence / mass conservation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LRTokenizer(nn.Module):
    """Patch-tokenize the LR conditioning input.

    For Sentinel-2 SR the LR is the same modality as HR. We patch-embed at
    a coarser stride and concatenate the resulting tokens to the DiT
    sequence. (In production we would use a frozen VAE encoder; the patch
    tokenizer is a simpler stand-in that compiles cleanly without external
    weights.)
    """

    def __init__(self, in_channels: int, hidden: int, patch: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden, kernel_size=patch, stride=patch)

    def forward(self, x_lr: Tensor) -> Tensor:
        z = self.proj(x_lr)            # (B, C, h, w)
        return z.flatten(2).transpose(1, 2)  # (B, h*w, C)


class PhysicsCodebook(nn.Module):
    """Learned codebook of physics-constraint embeddings (64 codes, dim=hidden).

    Cross-attended in every DiT block. The model is free to use these codes
    as queries / keys for "this region needs to preserve NDVI" type
    inductive biases.
    """

    def __init__(self, n_codes: int = 64, hidden: int = 512):
        super().__init__()
        self.codes = nn.Parameter(torch.randn(n_codes, hidden) * 0.02)

    def forward(self, batch_size: int) -> Tensor:
        return self.codes.unsqueeze(0).expand(batch_size, -1, -1)
