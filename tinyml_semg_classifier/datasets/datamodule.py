from __future__ import annotations

from dataclasses import dataclass
import functools
from pathlib import Path
import random
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .manifest import resolve_sample_id, validate_manifest_rows


class ZScoreNormalizer:
    def __init__(self, mean, std, eps=1e-8):
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError("Z-score mean/std must be 1D arrays.")
        if mean.shape != std.shape:
            raise ValueError("Z-score mean/std must have matching shapes.")
        self.mean = mean
        self.std = std
        self.eps = float(eps)
        self._mean = mean.reshape(-1, 1)
        self._std = std.reshape(-1, 1)

    def __call__(self, window):
        return (window - self._mean) / (self._std + self.eps)

    def to_config(self):
        return {
            "type": "zscore",
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "eps": self.eps,
        }


@dataclass(frozen=True)
class LabelMapping:
    labels: list[int]
    mapping: dict[int, int]

    @property
    def num_classes(self) -> int:
        return len(self.labels)


def build_label_mapping(manifest_rows: list[dict]) -> LabelMapping:
    labels = sorted({int(row["gesture_id"]) for row in manifest_rows})
    mapping = {label: idx for idx, label in enumerate(labels)}
    return LabelMapping(labels=labels, mapping=mapping)


def normalize_manifest_rows(manifest_rows: list[dict]) -> list[dict]:
    validate_manifest_rows(manifest_rows)
    normalized = []
    for row in manifest_rows:
        sample_id = resolve_sample_id(row)
        normalized.append(
            {
                "sample_id": str(sample_id),
                "subject_id": int(row["subject_id"]),
                "rep_id": int(row["rep_id"]),
                "gesture_id": int(row["gesture_id"]),
                "exercise_id": int(row["exercise_id"]),
                "sample_start": int(row["sample_start"]),
                "sample_end": int(row["sample_end"]),
                "path": row["path"],
                "local_idx": int(row["local_idx"]),
            }
        )
    return normalized


def _resolve_path(path_str: str, manifest_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return manifest_dir / path


def _ensure_window_shape(
    window: np.ndarray, num_electrodes: int, num_samples: int
) -> np.ndarray:
    if window.ndim != 2:
        raise ValueError(f"Expected 2D window, got shape {window.shape}")
    if window.shape == (num_samples, num_electrodes):
        return window.T
    if window.shape == (num_electrodes, num_samples):
        return window
    raise ValueError(
        "Window shape must be (num_electrodes, num_samples) or (num_samples, num_electrodes), "
        f"got {window.shape}."
    )


def _ensure_batch_window_shape(
    windows: np.ndarray, num_electrodes: int, num_samples: int, path: Path
) -> np.ndarray:
    if windows.ndim != 3:
        raise ValueError(f"Expected 3D window array from {path}, got {windows.shape}.")
    shape = windows.shape[1:]
    if shape == (num_samples, num_electrodes):
        return np.transpose(windows, (0, 2, 1))
    if shape == (num_electrodes, num_samples):
        return windows
    raise ValueError(
        f"Unexpected window shape {windows.shape} in {path}; "
        f"expected (*, {num_samples}, {num_electrodes}) or "
        f"(*, {num_electrodes}, {num_samples})."
    )


def compute_zscore_stats_from_rows(
    rows: list[dict],
    num_electrodes: int,
    num_samples: int,
    manifest_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("Cannot compute z-score stats on an empty split.")
    sums = np.zeros(int(num_electrodes), dtype=np.float64)
    sumsq = np.zeros(int(num_electrodes), dtype=np.float64)
    total_count = 0
    grouped: dict[Path, list[int]] = {}
    for row in rows:
        path = _resolve_path(row["path"], manifest_dir)
        grouped.setdefault(path, []).append(int(row["local_idx"]))

    for path, local_indices in grouped.items():
        data = _load_windows_array(path)
        windows = data[np.asarray(local_indices, dtype=np.int64)]
        windows = np.asarray(windows, dtype=np.float32)
        windows = _ensure_batch_window_shape(windows, num_electrodes, num_samples, path)
        sums += windows.sum(axis=(0, 2), dtype=np.float64)
        sumsq += np.square(windows).sum(axis=(0, 2), dtype=np.float64)
        total_count += windows.shape[0] * windows.shape[2]

    if total_count == 0:
        raise ValueError("Cannot compute z-score stats with zero samples.")

    mean = sums / total_count
    var = sumsq / total_count - mean**2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def _load_windows_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if suffix == ".npz":
        with np.load(path) as data:
            if "X" not in data:
                raise KeyError(f"Missing X field in {path}.")
            return np.asarray(data["X"])
    raise ValueError(f"Unsupported window file format: {path.suffix}")


class WindowDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        label_map: LabelMapping,
        num_electrodes: int,
        num_samples: int,
        manifest_dir: Path,
        normalizer: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.rows = rows
        self.label_map = label_map
        self.num_electrodes = int(num_electrodes)
        self.num_samples = int(num_samples)
        self.manifest_dir = Path(manifest_dir)
        self.normalizer = normalizer
        self._cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = _resolve_path(row["path"], self.manifest_dir)
        local_idx = int(row["local_idx"])
        label = self.label_map.mapping[int(row["gesture_id"])]

        data = self._load_subject(path)
        window = data[local_idx]
        window = np.asarray(window, dtype=np.float32)
        window = _ensure_window_shape(window, self.num_electrodes, self.num_samples)
        if not window.flags.writeable:
            window = window.copy()
        if self.normalizer is not None:
            window = self.normalizer(window)
        meta = {
            "sample_id": str(row["sample_id"]),
            "subject_id": int(row["subject_id"]),
            "rep_id": int(row["rep_id"]),
            "gesture_id": int(row["gesture_id"]),
            "exercise_id": int(row["exercise_id"]),
            "sample_start": int(row["sample_start"]),
            "sample_end": int(row["sample_end"]),
        }
        return torch.from_numpy(window), label, meta

    def _load_subject(self, path: Path) -> np.ndarray:
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        if not path.exists():
            raise FileNotFoundError(f"Missing window file: {path}")
        data = _load_windows_array(path)
        self._cache[path] = data
        return data


def _split_rows(manifest_rows: list[dict], indices: Iterable[str]) -> list[dict]:
    lookup = {str(row["sample_id"]): row for row in manifest_rows}
    rows = []
    for idx in indices:
        key = str(idx)
        if key not in lookup:
            raise ValueError(f"Split index {key} not found in manifest.")
        rows.append(lookup[key])
    return rows


def _seed_worker(worker_id: int, base_seed: int) -> None:
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)


def build_dataloaders(
    manifest_rows: list[dict],
    split: dict,
    num_electrodes: int,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    manifest_dir: Path,
    normalization: dict | None = None,
) -> tuple[dict[str, DataLoader | None], LabelMapping, ZScoreNormalizer | None]:
    normalized = normalize_manifest_rows(manifest_rows)
    label_map = build_label_mapping(normalized)

    train_rows = _split_rows(normalized, split.get("train_idx", []))
    val_rows = _split_rows(normalized, split.get("val_idx", []))
    test_rows = _split_rows(normalized, split.get("test_idx", []))

    normalizer = None
    if normalization is not None:
        if not isinstance(normalization, dict):
            raise ValueError("normalization config must be a mapping.")
        norm_type = str(normalization.get("type", "zscore")).strip().lower()
        if norm_type in ("none", "off", "disabled"):
            normalizer = None
        elif norm_type != "zscore":
            raise ValueError(f"Unsupported normalization type: {norm_type}")
        else:
            mean, std = compute_zscore_stats_from_rows(
                train_rows,
                num_electrodes,
                num_samples,
                manifest_dir,
            )
            normalizer = ZScoreNormalizer(
                mean,
                std,
                normalization.get("eps", 1e-8),
            )

    train_dataset = WindowDataset(
        train_rows,
        label_map,
        num_electrodes,
        num_samples,
        manifest_dir,
        normalizer=normalizer,
    )
    val_dataset = (
        WindowDataset(
            val_rows,
            label_map,
            num_electrodes,
            num_samples,
            manifest_dir,
            normalizer=normalizer,
        )
        if val_rows
        else None
    )
    test_dataset = WindowDataset(
        test_rows,
        label_map,
        num_electrodes,
        num_samples,
        manifest_dir,
        normalizer=normalizer,
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    worker_init = None
    if num_workers > 0:
        worker_init = functools.partial(_seed_worker, base_seed=int(seed))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        generator=generator,
        worker_init_fn=worker_init,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            worker_init_fn=worker_init,
        )
        if val_dataset is not None
        else None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        worker_init_fn=worker_init,
    )

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    return loaders, label_map, normalizer


class DataModule:
    def __init__(
        self,
        manifest_rows: list[dict],
        split: dict,
        num_electrodes: int,
        num_samples: int,
        batch_size: int,
        num_workers: int,
        seed: int,
        manifest_dir: Path,
        normalization: dict | None = None,
    ) -> None:
        self._manifest_rows = manifest_rows
        self._split = split
        self._num_electrodes = int(num_electrodes)
        self._num_samples = int(num_samples)
        self._batch_size = int(batch_size)
        self._num_workers = int(num_workers)
        self._seed = int(seed)
        self._manifest_dir = Path(manifest_dir)
        self._normalization = normalization
        self.label_map: LabelMapping | None = None
        self.loaders: dict[str, DataLoader | None] = {}
        self.normalizer: ZScoreNormalizer | None = None
        self.normalization_config: dict | None = None

    def setup(self) -> None:
        loaders, label_map, normalizer = build_dataloaders(
            self._manifest_rows,
            self._split,
            self._num_electrodes,
            self._num_samples,
            self._batch_size,
            self._num_workers,
            self._seed,
            self._manifest_dir,
            normalization=self._normalization,
        )
        self.label_map = label_map
        self.loaders = loaders
        self.normalizer = normalizer
        self.normalization_config = (
            None if normalizer is None else normalizer.to_config()
        )
