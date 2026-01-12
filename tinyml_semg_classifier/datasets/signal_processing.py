import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, iirnotch


ArrayLike = NDArray[np.floating]


def butter_bandpass(
    emg: ArrayLike,
    fs_hz: float,
    low_hz: float = 8.0,
    high_hz: float = 500.0,
    order: int = 4,
) -> ArrayLike:
    """Zero-phase Butterworth band-pass filter for sEMG."""
    if high_hz >= fs_hz / 2:
        raise ValueError("high_hz must be < Nyquist frequency.")

    b, a = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs_hz)
    return filtfilt(b, a, emg, axis=0)


def notch_50hz(
    emg: ArrayLike,
    fs_hz: float,
    quality: float = 30.0,
) -> ArrayLike:
    """50 Hz notch filter to suppress powerline interference."""
    b, a = iirnotch(50.0, quality, fs=fs_hz)
    return filtfilt(b, a, emg, axis=0)


def zscore_normalize(
    emg: ArrayLike,
    eps: float = 1e-8,
) -> ArrayLike:
    """Channel-wise Z-score normalization."""
    mean = emg.mean(axis=0, keepdims=True)
    std = emg.std(axis=0, keepdims=True)
    return (emg - mean) / (std + eps)
