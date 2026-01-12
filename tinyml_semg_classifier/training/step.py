from __future__ import annotations

from typing import Any

import torch
from torch import nn


def split_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, Any | None]:
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        xb, yb = batch[0], batch[1]
        meta = batch[2] if len(batch) > 2 else None
        return xb, yb, meta
    raise ValueError("Expected batch to be a tuple of (X, y[, meta]).")


def train_step(
    model: nn.Module,
    batch: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, int, torch.Tensor, torch.Tensor]:
    xb, yb, _ = split_batch(batch)
    xb = xb.to(device)
    yb = yb.to(device)
    optimizer.zero_grad()
    logits = model(xb)
    loss = criterion(logits, yb)
    loss.backward()
    optimizer.step()
    batch_size = xb.size(0)
    return float(loss.item()), batch_size, logits, yb
