from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def is_improvement(value: float | None, best: float | None, mode: str) -> bool:
    if value is None:
        return False
    if best is None:
        return True
    if mode == "max":
        return value > best
    if mode == "min":
        return value < best
    raise ValueError("mode must be 'min' or 'max'.")
