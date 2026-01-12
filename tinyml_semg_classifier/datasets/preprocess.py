from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import build_context
from ..utils.hashing import stable_hash
from ..utils.io import parse_subjects_spec, write_json
from ..utils.logging import get_logger
from .signal_processing import butter_bandpass, notch_50hz
from .manifest import compute_sample_id, write_manifest
from .ninapro_db2.load_raw import (
    collect_subject_dirs,
    load_raw_subject,
    resolve_exercises,
)
from .ninapro_db2.windowing import iter_windows

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class WindowRecord:
    window: np.ndarray
    gesture_id: int
    rep_id: int
    exercise_code: str
    exercise_id: int
    sample_start: int
    sample_end: int


def _validate_preprocess_cfg(cfg: dict) -> None:
    dataset_cfg = cfg.get("dataset")
    preprocess_cfg = cfg.get("preprocess")
    manifest_cfg = cfg.get("manifest")

    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset configuration is required.")
    if not isinstance(preprocess_cfg, dict):
        raise ValueError("preprocess configuration is required.")
    if not isinstance(manifest_cfg, dict):
        raise ValueError("manifest configuration is required.")

    source = str(dataset_cfg.get("source", dataset_cfg.get("name", ""))).lower()
    if source == "fixture":
        if "fixture_path" not in dataset_cfg:
            raise ValueError("dataset.fixture_path is required for fixture source.")
    else:
        if dataset_cfg.get("name") != "ninapro_db2":
            raise ValueError("dataset.name must be 'ninapro_db2'.")
        for key in ("raw_dir", "sampling_rate_hz", "channels", "keep_rest"):
            if key not in dataset_cfg:
                raise ValueError(f"dataset.{key} is required.")
        if "exercises" not in dataset_cfg and "exercise" not in dataset_cfg:
            raise ValueError("dataset.exercises (or dataset.exercise) is required.")

    for key in ("window_ms", "window_samples", "hop_ms", "hop_samples", "output"):
        if key not in preprocess_cfg:
            raise ValueError(f"preprocess.{key} is required.")

    output_cfg = preprocess_cfg.get("output")
    if not isinstance(output_cfg, dict):
        raise ValueError("preprocess.output must be a mapping.")
    for key in ("windows_path", "meta_path"):
        if key not in output_cfg:
            raise ValueError(f"preprocess.output.{key} is required.")

    manifest_out = manifest_cfg.get("output")
    if not isinstance(manifest_out, dict):
        raise ValueError("manifest.output must be a mapping.")
    if "manifest_csv" not in manifest_out:
        raise ValueError("manifest.output.manifest_csv is required.")


def preprocess_emg(emg: np.ndarray, fs_hz: float) -> np.ndarray:
    emg_bp = butter_bandpass(emg, fs_hz, low_hz=8.0, high_hz=500.0, order=4)
    emg_notch = notch_50hz(emg_bp, fs_hz, quality=30.0)
    return emg_notch


def extract_windows(
    emg_proc: np.ndarray,
    restimulus: np.ndarray,
    rerepetition: np.ndarray,
    window_samples: int,
    hop_samples: int,
    keep_rest: bool,
    exercise_code: str,
    exercise_id: int,
) -> list[WindowRecord]:
    windows: list[WindowRecord] = []
    for window, label, rep, start, end in iter_windows(
        emg_proc, restimulus, rerepetition, window_samples, hop_samples
    ):
        if not keep_rest and int(label) == 0:
            continue
        window = np.asarray(window, dtype=np.float32)
        if window.ndim != 2:
            raise ValueError(f"Expected window to be 2D, got {window.shape}")
        window = window.T
        windows.append(
            WindowRecord(
                window=window,
                gesture_id=int(label),
                rep_id=int(rep),
                exercise_code=exercise_code,
                exercise_id=exercise_id,
                sample_start=int(start),
                sample_end=int(end),
            )
        )
    return windows


def build_label_map(
    meta_rows: list[dict],
    disambiguate_exercises: bool,
) -> tuple[str, list[dict] | None, dict[tuple[int, int], int] | None]:
    if not disambiguate_exercises:
        label_map_id = stable_hash({"type": "raw"})
        return label_map_id, None, None

    pairs = sorted(
        {(int(row["exercise_id"]), int(row["gesture_id"])) for row in meta_rows}
    )
    mapping = {pair: idx + 1 for idx, pair in enumerate(pairs)}
    entries = [
        {"exercise_id": ex, "gesture_id": gid, "mapped_id": mapping[(ex, gid)]}
        for ex, gid in pairs
    ]
    label_map_id = stable_hash(entries)
    return label_map_id, entries, mapping


def _ensure_fixture_shape(
    windows: np.ndarray, num_electrodes: int, num_samples: int, path: Path
) -> None:
    if windows.ndim != 3:
        raise ValueError(f"Expected 3D windows from {path}, got {windows.shape}")
    shape = windows.shape[1:]
    if shape not in {(num_electrodes, num_samples), (num_samples, num_electrodes)}:
        raise ValueError(
            f"Fixture windows shape {windows.shape} does not match "
            f"expected (*, {num_electrodes}, {num_samples}) or "
            f"(*, {num_samples}, {num_electrodes})."
        )


def preprocess_fixture(cfg: dict) -> Path:
    dataset_cfg = cfg["dataset"]
    preprocess_cfg = cfg["preprocess"]
    manifest_cfg = cfg["manifest"]

    context = build_context(cfg)
    fixture_path = Path(dataset_cfg["fixture_path"]).expanduser()
    if not fixture_path.is_absolute():
        fixture_path = (Path.cwd() / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Missing fixture file: {fixture_path}")

    meta_path = Path(preprocess_cfg["output"]["meta_path"].format(**context))
    manifest_path = Path(manifest_cfg["output"]["manifest_csv"].format(**context))

    cache_enabled = bool(preprocess_cfg.get("cache", False))
    if cache_enabled and meta_path.exists() and manifest_path.exists():
        LOGGER.info("Preprocess cache hit: %s", meta_path)
        return manifest_path

    data = np.load(fixture_path)
    try:
        if "X" not in data:
            raise KeyError(f"Fixture missing X array: {fixture_path}")
        windows = np.asarray(data["X"])

        num_samples = int(preprocess_cfg["window_samples"])
        num_electrodes = int(dataset_cfg.get("channels", windows.shape[1]))
        _ensure_fixture_shape(windows, num_electrodes, num_samples, fixture_path)

        def _load_field(name: str, default: int | None = None) -> np.ndarray:
            if name in data:
                values = np.asarray(data[name])
            elif default is not None:
                values = np.full((windows.shape[0],), default)
            else:
                raise KeyError(f"Fixture missing {name} array: {fixture_path}")
            if values.shape[0] != windows.shape[0]:
                raise ValueError(f"Fixture {name} length mismatch in {fixture_path}")
            return values

        subject_id = _load_field("subject_id")
        rep_id = _load_field("rep_id")
        gesture_id = _load_field("gesture_id")
        exercise_id = _load_field("exercise_id", default=1)
        sample_start = _load_field("sample_start", default=0)
        sample_end = _load_field("sample_end", default=num_samples - 1)
    finally:
        data.close()

    meta_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for idx in range(windows.shape[0]):
        row_sample_id = compute_sample_id(
            {
                "subject_id": int(subject_id[idx]),
                "exercise_id": int(exercise_id[idx]),
                "rep_id": int(rep_id[idx]),
                "gesture_id": int(gesture_id[idx]),
                "sample_start": int(sample_start[idx]),
                "sample_end": int(sample_end[idx]),
            }
        )
        meta_rows.append(
            {
                "sample_id": row_sample_id,
                "subject_id": int(subject_id[idx]),
                "rep_id": int(rep_id[idx]),
                "gesture_id": int(gesture_id[idx]),
                "exercise": "fixture",
                "exercise_id": int(exercise_id[idx]),
                "sample_start": int(sample_start[idx]),
                "sample_end": int(sample_end[idx]),
                "sampling_rate_hz": float(dataset_cfg.get("sampling_rate_hz", 0.0)),
                "window_samples": int(preprocess_cfg["window_samples"]),
                "hop_samples": int(preprocess_cfg["hop_samples"]),
            }
        )
        manifest_rows.append(
            {
                "sample_id": row_sample_id,
                "subject_id": int(subject_id[idx]),
                "rep_id": int(rep_id[idx]),
                "gesture_id": int(gesture_id[idx]),
                "exercise_id": int(exercise_id[idx]),
                "sample_start": int(sample_start[idx]),
                "sample_end": int(sample_end[idx]),
                "path": str(fixture_path),
                "local_idx": idx,
            }
        )

    label_map_id, label_map_entries, mapping = build_label_map(
        meta_rows, bool(dataset_cfg.get("disambiguate_exercises", False))
    )
    if mapping is not None:
        for row in meta_rows:
            raw_gesture = int(row["gesture_id"])
            row["gesture_id_raw"] = raw_gesture
            row["gesture_id"] = mapping[(int(row["exercise_id"]), raw_gesture)]
            row["label_map_id"] = label_map_id
        for row in manifest_rows:
            raw_gesture = int(row["gesture_id"])
            row["gesture_id"] = mapping[(int(row["exercise_id"]), raw_gesture)]
    else:
        for row in meta_rows:
            row["label_map_id"] = label_map_id

    meta_payload: dict[str, Any] = {"rows": meta_rows, "label_map_id": label_map_id}
    if label_map_entries is not None:
        meta_payload["label_map"] = label_map_entries

    write_json(meta_path, meta_payload)
    write_manifest(manifest_path, manifest_rows)
    LOGGER.info("Fixture preprocess wrote %s and %s", meta_path, manifest_path)
    return manifest_path


def preprocess(cfg: dict) -> Path:
    _validate_preprocess_cfg(cfg)

    dataset_cfg = cfg["dataset"]
    preprocess_cfg = cfg["preprocess"]
    manifest_cfg = cfg["manifest"]

    source = str(dataset_cfg.get("source", dataset_cfg.get("name", ""))).lower()
    if source == "fixture":
        return preprocess_fixture(cfg)

    fs_hz = float(dataset_cfg["sampling_rate_hz"])
    channels = int(dataset_cfg["channels"])
    keep_rest = bool(dataset_cfg["keep_rest"])
    disambiguate = bool(dataset_cfg.get("disambiguate_exercises", False))

    window_ms = float(preprocess_cfg["window_ms"])
    hop_ms = float(preprocess_cfg["hop_ms"])
    window_samples = int(preprocess_cfg["window_samples"])
    hop_samples = int(preprocess_cfg["hop_samples"])

    expected_window = int(round(window_ms * fs_hz / 1000.0))
    expected_hop = int(round(hop_ms * fs_hz / 1000.0))
    if window_samples != expected_window:
        raise ValueError(
            "preprocess.window_samples does not match window_ms and sampling_rate_hz "
            f"({window_samples} != {expected_window})."
        )
    if hop_samples != expected_hop:
        raise ValueError(
            "preprocess.hop_samples does not match hop_ms and sampling_rate_hz "
            f"({hop_samples} != {expected_hop})."
        )

    context = build_context(cfg)
    raw_dir = Path(dataset_cfg["raw_dir"])
    exercises = resolve_exercises(
        dataset_cfg.get("exercises", dataset_cfg.get("exercise"))
    )

    subject_dirs = collect_subject_dirs(raw_dir)
    if not subject_dirs:
        raise FileNotFoundError(f"No DB2 subject folders found in {raw_dir}")

    subjects = parse_subjects_spec(dataset_cfg.get("subjects"))
    if not subjects:
        raise ValueError("No subjects specified for preprocessing.")

    selected = sorted(set(subjects))
    missing = [sid for sid in selected if sid not in subject_dirs]
    if missing:
        raise ValueError(f"Requested subjects not found in {raw_dir}: {missing}")

    windows_template = preprocess_cfg["output"]["windows_path"]
    meta_path = Path(preprocess_cfg["output"]["meta_path"].format(**context))
    manifest_path = Path(manifest_cfg["output"]["manifest_csv"].format(**context))

    cache_enabled = bool(preprocess_cfg.get("cache", False))
    if cache_enabled and meta_path.exists() and manifest_path.exists():
        LOGGER.info("Preprocess cache hit: %s", meta_path)
        return manifest_path

    meta_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for subject_id in selected:
        LOGGER.info(f"Preprocessing subject {subject_id}...")
        recordings = load_raw_subject(raw_dir, subject_id, exercises, channels=channels)
        if not recordings:
            LOGGER.warning("No recordings for subject %s", subject_id)
            continue

        subject_windows: list[np.ndarray] = []
        local_idx = 0
        for recording in recordings:
            emg_proc = preprocess_emg(recording.emg, fs_hz).astype(
                np.float32, copy=False
            )
            windows = extract_windows(
                emg_proc,
                recording.restimulus,
                recording.rerepetition,
                window_samples,
                hop_samples,
                keep_rest,
                recording.exercise_code,
                recording.exercise_id,
            )
            if not windows:
                continue

            for window_record in windows:
                sample_id = compute_sample_id(
                    {
                        "subject_id": subject_id,
                        "exercise_id": window_record.exercise_id,
                        "rep_id": window_record.rep_id,
                        "gesture_id": window_record.gesture_id,
                        "sample_start": window_record.sample_start,
                        "sample_end": window_record.sample_end,
                    }
                )
                subject_windows.append(window_record.window)
                meta_rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "rep_id": window_record.rep_id,
                        "gesture_id": window_record.gesture_id,
                        "exercise": window_record.exercise_code,
                        "exercise_id": window_record.exercise_id,
                        "sample_start": window_record.sample_start,
                        "sample_end": window_record.sample_end,
                        "sampling_rate_hz": fs_hz,
                        "window_samples": window_samples,
                        "hop_samples": hop_samples,
                    }
                )
                manifest_rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "rep_id": window_record.rep_id,
                        "gesture_id": window_record.gesture_id,
                        "exercise_id": window_record.exercise_id,
                        "sample_start": window_record.sample_start,
                        "sample_end": window_record.sample_end,
                        "path": windows_template.format(
                            **context, subject_id=subject_id
                        ),
                        "local_idx": local_idx,
                    }
                )
                local_idx += 1

        if subject_windows:
            windows_path = Path(
                windows_template.format(**context, subject_id=subject_id)
            )
            windows_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(windows_path, np.stack(subject_windows, axis=0))
            LOGGER.info("Wrote windows for subject %s -> %s", subject_id, windows_path)

    if not meta_rows:
        raise RuntimeError("No windows extracted. Check raw data and filters.")

    label_map_id, label_map_entries, mapping = build_label_map(meta_rows, disambiguate)
    if mapping is not None:
        for row in meta_rows:
            raw_gesture = int(row["gesture_id"])
            row["gesture_id_raw"] = raw_gesture
            row["gesture_id"] = mapping[(int(row["exercise_id"]), raw_gesture)]
            row["label_map_id"] = label_map_id
        for row in manifest_rows:
            raw_gesture = int(row["gesture_id"])
            row["gesture_id"] = mapping[(int(row["exercise_id"]), raw_gesture)]
    else:
        for row in meta_rows:
            row["label_map_id"] = label_map_id

    meta_payload: dict[str, Any] = {"rows": meta_rows, "label_map_id": label_map_id}
    if label_map_entries is not None:
        meta_payload["label_map"] = label_map_entries

    write_json(meta_path, meta_payload)
    write_manifest(manifest_path, manifest_rows)

    LOGGER.info("Preprocess wrote %s and %s", meta_path, manifest_path)
    return manifest_path
