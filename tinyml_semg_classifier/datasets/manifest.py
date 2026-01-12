from __future__ import annotations

import csv
from pathlib import Path

from ..utils.hashing import stable_hash
from ..utils.io import ensure_dir


MANIFEST_FIELDS = [
    "sample_id",
    "subject_id",
    "exercise_id",
    "rep_id",
    "gesture_id",
    "sample_start",
    "sample_end",
    "path",
    "local_idx",
]


def compute_sample_id(payload: dict) -> str:
    required = (
        "subject_id",
        "exercise_id",
        "rep_id",
        "gesture_id",
        "sample_start",
        "sample_end",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing payload keys for sample_id: {missing}")
    canonical = {
        "subject_id": int(payload["subject_id"]),
        "exercise_id": int(payload["exercise_id"]),
        "rep_id": int(payload["rep_id"]),
        "gesture_id": int(payload["gesture_id"]),
        "sample_start": int(payload["sample_start"]),
        "sample_end": int(payload["sample_end"]),
    }
    return stable_hash(canonical)


def resolve_sample_id(row: dict) -> str:
    sample_id = row.get("sample_id")
    if sample_id in (None, ""):
        raise ValueError("Manifest row missing sample_id.")
    return str(sample_id)


def validate_manifest_rows(rows: list[dict]) -> None:
    required = [
        "sample_id",
        "subject_id",
        "exercise_id",
        "rep_id",
        "gesture_id",
        "sample_start",
        "sample_end",
        "path",
        "local_idx",
    ]
    missing_fields = [field for field in required if not rows or field not in rows[0]]
    if missing_fields:
        raise ValueError(f"Manifest missing required fields: {missing_fields}")
    sample_ids = []
    for row in rows:
        for field in required:
            if row.get(field) in (None, ""):
                raise ValueError(f"Manifest row missing {field}.")
        sample_ids.append(str(row["sample_id"]))
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Manifest contains duplicate sample_id values.")


def write_manifest(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else MANIFEST_FIELDS
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_manifest(path: Path) -> list[dict]:
    path = Path(path)
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]
    validate_manifest_rows(rows)
    return rows
