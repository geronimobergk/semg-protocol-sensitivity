from __future__ import annotations

import torch
from torch import nn


class SpatioTemporalAttention(nn.Module):
    """
    Lightweight gating: pool -> conv -> sigmoid gates.

    Expects x: (B, C, H_electrodes, W_time)
    """

    def __init__(self, temporal_kernel: int = 7, spatial_kernel: int = 3):
        super().__init__()
        if temporal_kernel % 2 == 0 or spatial_kernel % 2 == 0:
            raise ValueError("temporal_kernel and spatial_kernel must be odd.")
        tpad = temporal_kernel // 2
        spad = spatial_kernel // 2

        self.temporal_gate = nn.Conv2d(
            2, 1, kernel_size=(1, temporal_kernel), padding=(0, tpad), bias=False
        )
        self.spatial_gate = nn.Conv2d(
            2, 1, kernel_size=(spatial_kernel, 1), padding=(spad, 0), bias=False
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatio-temporal attention to input tensor x.

        Args:
            x: Input tensor of shape (B, C, H_electrodes, W_time_samples).

        Returns:
            Output tensor of same shape.
        """
        # Temporal gate over W
        avg_t = x.mean(dim=(1, 2), keepdim=True)  # (B,1,1,W)
        max_t = x.amax(dim=(1, 2), keepdim=True)  # (B,1,1,W)
        t = self.act(self.temporal_gate(torch.cat([avg_t, max_t], dim=1)))
        x = x * t

        # Spatial gate over H
        avg_s = x.mean(dim=(1, 3), keepdim=True)  # (B,1,H,1)
        max_s = x.amax(dim=(1, 3), keepdim=True)  # (B,1,H,1)
        s = self.act(self.spatial_gate(torch.cat([avg_s, max_s], dim=1)))
        return x * s
