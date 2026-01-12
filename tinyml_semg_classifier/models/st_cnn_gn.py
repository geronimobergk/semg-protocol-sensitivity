from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn

from .attention import SpatioTemporalAttention
from .blocks import ConvBlock, MLPHead


class SpatioTemporalCNN_GN(nn.Module):
    """
    Controlled comparison backbone:
      - Conv blocks with GroupNorm (batch-independent)
      - Optional spatio-temporal gating
      - LN-based classifier head
    """

    def __init__(
        self,
        num_electrodes: int | None = None,
        num_samples: int | None = None,
        conv_channels: Sequence[int] = (64, 64, 128),
        kernel_size: Tuple[int, int] = (3, 3),
        pool_sizes: Sequence[Tuple[int, int]] = ((1, 4), (2, 2), (2, 2)),
        conv_dropout: float = 0.2,
        head_hidden: Sequence[int] = (256, 128),
        head_dropout: float = 0.3,
        num_classes: int = 17,
        use_attention: bool = False,
        attn_temporal_kernel: int = 7,
        attn_spatial_kernel: int = 3,
        gn_groups: int = 8,
    ):
        super().__init__()

        if num_electrodes is not None and num_electrodes <= 0:
            raise ValueError("num_electrodes must be positive if provided.")
        if num_samples is not None and num_samples <= 0:
            raise ValueError("num_samples must be positive if provided.")
        if len(conv_channels) != len(pool_sizes):
            raise ValueError("pool_sizes must match conv_channels length.")

        attn_mod: nn.Module | None = None
        if use_attention:
            attn_mod = SpatioTemporalAttention(
                temporal_kernel=attn_temporal_kernel,
                spatial_kernel=attn_spatial_kernel,
            )

        blocks = []
        in_ch = 1  # input expected as (B,1,H_electrodes,W_time_samples)
        for out_ch, pool in zip(conv_channels, pool_sizes):
            blocks.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    pool_size=pool,
                    dropout=conv_dropout,
                    norm_groups=gn_groups,
                    attn=attn_mod,  # applied consistently at same stage in each block
                )
            )
            in_ch = out_ch

        self._num_electrodes = num_electrodes
        self._num_samples = num_samples

        self.features = nn.Sequential(*blocks)
        self.head = MLPHead(
            hidden=head_hidden, dropout=head_dropout, num_classes=num_classes
        )

    def _validate_input(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either (B,H,W) or (B,1,H,W)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(
                f"Expected input (B,H,W) or (B,1,H,W), got {tuple(x.shape)}"
            )

        # Optional semantic checks (catch config/data mismatch early)
        if self._num_electrodes is not None and x.shape[2] != self._num_electrodes:
            raise ValueError(
                f"Input has H={x.shape[2]} electrodes, but config num_electrodes={self._num_electrodes}"
            )
        if self._num_samples is not None and x.shape[3] != self._num_samples:
            raise ValueError(
                f"Input has W={x.shape[3]} samples, but config num_samples={self._num_samples}"
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._validate_input(x)
        feats = self.features(x).flatten(1)
        return self.head(feats)


class ST_CNN_GN(SpatioTemporalCNN_GN):
    def __init__(self, **kwargs):
        super().__init__(use_attention=False, **kwargs)


class ST_Attn_CNN_GN(SpatioTemporalCNN_GN):
    def __init__(self, **kwargs):
        super().__init__(use_attention=True, **kwargs)
