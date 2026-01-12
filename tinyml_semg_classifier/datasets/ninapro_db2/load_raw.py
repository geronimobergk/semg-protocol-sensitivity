from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

KNOWN_EXERCISES: tuple[str, ...] = ("E1", "E2", "E3")
EXERCISE_MAP: dict[str, str] = {"B": "E1", "C": "E2", "D": "E3"}
EXERCISE_ID: dict[str, int] = {"E1": 1, "E2": 2, "E3": 3}

SUBJECT_DIR_RE = re.compile(r"^(?:db2_)?s(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class RawRecording:
    exercise_code: str
    exercise_id: int
    emg: np.ndarray
    restimulus: np.ndarray
    rerepetition: np.ndarray
    path: Path


def parse_subject_id(name: str) -> int | None:
    match = SUBJECT_DIR_RE.match(name.strip())
    return int(match.group(1)) if match else None


def collect_subject_dirs(raw_dir: Path) -> dict[int, Path]:
    raw_dir = Path(raw_dir)
    sid = parse_subject_id(raw_dir.name)
    if sid is not None:
        return {sid: raw_dir}

    subject_dirs: dict[int, Path] = {}
    for subject_dir in sorted(raw_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        sid = parse_subject_id(subject_dir.name)
        if sid is not None:
            subject_dirs[sid] = subject_dir
    return subject_dirs


def normalize_exercise_token(token: Any) -> str | None:
    token = str(token).strip().upper()
    if token in ("ALL", "*"):
        return None
    if token in KNOWN_EXERCISES:
        return token
    if token in EXERCISE_MAP:
        return EXERCISE_MAP[token]
    raise ValueError(f"Unknown exercise token: {token!r}")


def resolve_exercises(exercises: Any) -> list[str]:
    if exercises is None:
        raise ValueError("dataset.exercises is required.")
    if isinstance(exercises, (list, tuple)):
        tokens = list(exercises)
    else:
        tokens = [t.strip() for t in str(exercises).split(",") if t.strip()]

    normalized: list[str] = []
    for token in tokens:
        code = normalize_exercise_token(token)
        if code is None:
            return list(KNOWN_EXERCISES)
        normalized.append(code)

    if not normalized:
        raise ValueError("No exercises specified in config.")
    return sorted(set(normalized))


def infer_exercise_code(path: Path) -> str | None:
    stem = path.stem
    for code in KNOWN_EXERCISES:
        if re.search(rf"(?:^|_){code}(?:_|$)", stem, re.IGNORECASE):
            return code
    return None


def load_db2_mat(
    path: Path, channels: int = 12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    emg = np.asarray(data["emg"])
    restimulus = np.asarray(data["restimulus"]).reshape(-1)

    if "rerepetition" in data:
        rerepetition = data["rerepetition"]
    elif "repetition" in data:
        rerepetition = data["repetition"]
    else:
        raise KeyError(
            f"{path}: missing repetition labels ('rerepetition'/'repetition')"
        )

    rerepetition = np.asarray(rerepetition).reshape(-1)

    if emg.ndim != 2:
        raise ValueError(f"{path}: EMG should be 2D, got {emg.shape}")

    if emg.shape[1] != channels and emg.shape[0] == channels:
        emg = emg.T

    length = min(emg.shape[0], restimulus.shape[0], rerepetition.shape[0])
    return emg[:length], restimulus[:length], rerepetition[:length]


def load_raw_subject(
    raw_dir: str | Path,
    subject_id: int,
    exercises: list[str],
    channels: int = 12,
) -> list[RawRecording]:
    raw_dir = Path(raw_dir)
    subject_dirs = collect_subject_dirs(raw_dir)
    subject_dir = subject_dirs.get(subject_id)
    if subject_dir is None:
        raise FileNotFoundError(f"Subject {subject_id} not found in {raw_dir}")

    mat_files = sorted(Path(subject_dir).rglob("*.mat"))
    filtered = []
    for mat_path in mat_files:
        code = infer_exercise_code(mat_path)
        if code is None or code not in exercises:
            continue
        filtered.append((mat_path, code))

    recordings: list[RawRecording] = []
    for mat_path, code in filtered:
        emg, restimulus, rerepetition = load_db2_mat(mat_path, channels=channels)
        recordings.append(
            RawRecording(
                exercise_code=code,
                exercise_id=EXERCISE_ID[code],
                emg=emg,
                restimulus=restimulus,
                rerepetition=rerepetition,
                path=mat_path,
            )
        )

    return recordings
