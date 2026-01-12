from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def _resolve_sample_ids(loader) -> list[str]:
    dataset = getattr(loader, "dataset", None)
    rows = getattr(dataset, "rows", None) if dataset is not None else None
    if rows is None:
        raise ValueError("Predict loader dataset must expose rows for sample_ids.")
    sample_ids = []
    for row in rows:
        value = row.get("sample_id")
        if value in (None, ""):
            raise ValueError("Predict loader row missing sample_id.")
        sample_ids.append(str(value))
    return sample_ids


def predict(
    model: nn.Module,
    loader,
    device: torch.device,
    run_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    include_probs: bool = True,
) -> Path:
    run_dir = Path(run_dir)
    if checkpoint_path is None:
        checkpoint_path = run_dir / "checkpoints" / "best.pt"
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for prediction: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    sample_ids = _resolve_sample_ids(loader)
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    prob_chunks: list[np.ndarray] = []

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                xb, yb = batch[0], batch[1]
            else:
                raise ValueError("Expected batch to be a tuple of (X, y[, meta]).")
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)

            y_true_chunks.append(yb.detach().cpu().numpy())
            y_pred_chunks.append(preds.detach().cpu().numpy())
            if include_probs:
                prob_chunks.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

    if was_training:
        model.train()

    y_true = np.concatenate(y_true_chunks) if y_true_chunks else np.array([])
    y_pred = np.concatenate(y_pred_chunks) if y_pred_chunks else np.array([])
    if include_probs:
        probs = np.concatenate(prob_chunks) if prob_chunks else np.array([])

    if len(sample_ids) != len(y_true):
        raise RuntimeError(
            "Mismatch between sample_ids and predictions: "
            f"{len(sample_ids)} vs {len(y_true)}"
        )

    output_path = run_dir / "preds.npz"
    payload = {
        "y_true": y_true,
        "y_pred": y_pred,
        "sample_ids": np.asarray(sample_ids, dtype=str),
    }
    if include_probs:
        payload["probs"] = probs
    np.savez(output_path, **payload)
    LOGGER.info("Wrote predictions to %s", output_path)
    return output_path
