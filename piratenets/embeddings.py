"""Input embeddings.

PeriodicEmbedding: exact periodic boundary conditions (Expert's Guide Sec. 6.3,
after Dong & Ni 2021). A network that only sees (cos(w x), sin(w x)) with
w = 2 pi / P is P-periodic in x together with all of its x-derivatives.

FourierFeatures: random Fourier features gamma(v) = [cos(B v), sin(B v)] with
B_ij ~ N(0, scale^2) at initialization (Expert's Guide Sec. 4.2, eq. 4.3). B is
trained together with the network by default (docs/decisions.md, D5).
"""
import math

import torch
from torch import nn


class PeriodicEmbedding(nn.Module):
    """(t, x) -> [t, cos(w x), sin(w x)] with w = 2 pi / period."""

    out_dim = 3

    def __init__(self, period: float):
        super().__init__()
        self.omega = 2.0 * math.pi / period

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        wx = self.omega * x
        return torch.stack([t, torch.cos(wx), torch.sin(wx)], dim=-1)


class Identity2D(nn.Module):
    """(t, x) -> [t, x]: no boundary embedding."""

    out_dim = 2

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([t, x], dim=-1)


class FourierFeatures(nn.Module):
    """v (..., in_dim) -> [cos(v B^T), sin(v B^T)] (..., embed_dim)."""

    def __init__(self, in_dim: int, embed_dim: int, scale: float, trainable: bool = True):
        super().__init__()
        if embed_dim % 2:
            raise ValueError("embed_dim must be even (half cosines, half sines)")
        self.out_dim = embed_dim
        B = scale * torch.randn(embed_dim // 2, in_dim)
        if trainable:
            self.B = nn.Parameter(B)
        else:
            # A buffer: never trained, but saved with checkpoints.
            self.register_buffer("B", B)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        proj = v @ self.B.T
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
