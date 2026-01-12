from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_yaml(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def parse_subjects_spec(spec: Any, fallback: Any | None = None) -> list[int]:
    if spec is None:
        spec = fallback
    if isinstance(spec, list):
        return [int(value) for value in spec]
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            return []
        if text.lower() == "all" and fallback is not None and fallback != spec:
            return parse_subjects_spec(fallback)
        if "-" in text:
            start, end = text.split("-", 1)
            return list(range(int(start), int(end) + 1))
        if "," in text:
            return [int(value.strip()) for value in text.split(",") if value.strip()]
        return [int(text)]
    return []
