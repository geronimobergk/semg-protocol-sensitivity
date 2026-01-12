from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn


def choose_gn_groups(num_channels: int, preferred: int = 8) -> int:
    """Choose a GroupNorm group count that divides num_channels."""
    if num_channels <= 0:
        raise ValueError("num_channels must be positive.")
    g = min(max(1, preferred), num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return g


class GNAct(nn.Module):
    """GroupNorm + channel-wise PReLU for stable, batch-independent normalization."""

    def __init__(self, num_channels: int, preferred_groups: int = 8):
        super().__init__()
        groups = choose_gn_groups(num_channels, preferred_groups)
        self.norm = nn.GroupNorm(groups, num_channels)
        self.act = nn.PReLU(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(x))


class ConvBlock(nn.Module):
    """
    SAME-padded conv block to preserve feature geometry across layers.

    Expected input: (B, C, H_electrodes, W_time)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        pool_size: Tuple[int, int] = (1, 4),
        dropout: float = 0.0,
        norm_groups: int = 8,
        attn: nn.Module | None = None,
    ):
        super().__init__()
        kh, kw = kernel_size
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError("kernel_size must be odd for SAME padding.")
        pad = (kh // 2, kw // 2)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=pad, bias=False
        )
        self.norm_act = GNAct(out_channels, preferred_groups=norm_groups)
        self.attn = attn if attn is not None else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.drop = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm_act(x)
        x = self.attn(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class MLPHead(nn.Module):
    """LayerNorm-based MLP head for batch-size independence."""

    def __init__(
        self,
        hidden: Sequence[int] = (256, 128),
        dropout: float = 0.3,
        num_classes: int = 17,
    ):
        super().__init__()
        if len(hidden) != 2:
            raise ValueError("hidden must have two entries, e.g. (256, 128).")
        h1, h2 = hidden

        self.net = nn.Sequential(
            nn.LazyLinear(h1),
            nn.LayerNorm(h1),
            nn.PReLU(h1),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.PReLU(h2),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
