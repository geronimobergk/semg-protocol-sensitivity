from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def iter_windows(
    emg: np.ndarray,
    restimulus: np.ndarray,
    rerepetition: np.ndarray,
    window_samples: int,
    hop_samples: int,
) -> Iterable[Tuple[np.ndarray, int, int, int, int]]:
    """Yield (window, label, repetition, start, end) for one recording."""
    total = emg.shape[0]
    if restimulus.shape[0] != total or rerepetition.shape[0] != total:
        raise ValueError("Label lengths do not match EMG length.")

    for end in range(window_samples - 1, total, hop_samples):
        start = end - window_samples + 1
        window = emg[start : end + 1, :]
        label = int(restimulus[end])
        rep = int(rerepetition[end])
        yield window, label, rep, start, end


def window_signal(
    signal: np.ndarray, window_samples: int, hop_samples: int
) -> Iterable[np.ndarray]:
    """Yield raw windows from a 2D signal array."""
    total = signal.shape[0]
    for end in range(window_samples - 1, total, hop_samples):
        start = end - window_samples + 1
        yield signal[start : end + 1]
