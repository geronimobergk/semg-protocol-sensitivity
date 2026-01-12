from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

from ...utils.hashing import stable_hash
from ...utils.logging import get_logger
from ..manifest import resolve_sample_id
from ...utils.io import parse_subjects_spec


PROTOCOL_REGISTRY: dict[str, type["ProtocolBase"]] = {}
LOGGER = get_logger(__name__)


def register_protocol(name: str):
    def decorator(cls: type["ProtocolBase"]):
        PROTOCOL_REGISTRY[name] = cls
        return cls

    return decorator


def get_protocol(name: str) -> type["ProtocolBase"]:
    if name not in PROTOCOL_REGISTRY:
        raise KeyError(f"Unknown protocol type: {name}")
    return PROTOCOL_REGISTRY[name]


def split_hash(
    manifest_hash: str, protocol: str, protocol_cfg: dict, instance: dict
) -> str:
    payload = {
        "manifest_hash": manifest_hash,
        "protocol": protocol,
        "protocol_cfg": _clean_protocol_cfg(protocol_cfg),
        "instance": instance,
    }
    return stable_hash(payload)


def instance_to_tag(instance: dict) -> str:
    if not instance:
        return "all"
    parts = [f"{key}={value}" for key, value in sorted(instance.items())]
    return "_".join(parts).replace(" ", "")


def split_path(output_dir: Path, instance: dict) -> Path:
    tag = instance_to_tag(instance)
    return Path(output_dir) / f"instance={tag}" / "split.json"


def list_subjects(manifest_rows: list[dict], spec: Any | None) -> list[int]:
    subjects = parse_subjects_spec(spec)
    values = {int(row["subject_id"]) for row in manifest_rows if "subject_id" in row}
    if subjects:
        missing = sorted(set(subjects) - values)
        if missing:
            raise ValueError(f"Subjects missing from manifest: {missing}")
        return sorted(subjects)
    return sorted(values)


def _clean_protocol_cfg(protocol_cfg: dict) -> dict:
    if not isinstance(protocol_cfg, dict):
        return {"value": protocol_cfg}
    return {key: value for key, value in protocol_cfg.items() if key != "output_dir"}


def _parse_reps_list(value: Any, label: str) -> list[int]:
    if value is None:
        raise ValueError(f"protocol.reps.{label} is required.")
    if isinstance(value, (list, tuple, set)):
        reps = [int(item) for item in value]
    elif isinstance(value, int):
        reps = [int(value)]
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            reps = []
        elif "-" in text:
            start, end = text.split("-", 1)
            reps = list(range(int(start), int(end) + 1))
        elif "," in text:
            reps = [int(part.strip()) for part in text.split(",") if part.strip()]
        else:
            reps = [int(text)]
    else:
        raise ValueError(f"protocol.reps.{label} must be a list or string.")
    reps = sorted(set(reps))
    if not reps:
        raise ValueError(f"protocol.reps.{label} cannot be empty.")
    return reps


def resolve_reps_cfg(cfg: dict) -> dict[str, list[int]]:
    if not isinstance(cfg, dict):
        raise ValueError("protocol config must be a mapping.")
    reps_cfg = cfg.get("reps")
    if not isinstance(reps_cfg, dict):
        raise ValueError("protocol.reps must be a mapping.")
    all_reps = _parse_reps_list(reps_cfg.get("all"), "all")
    test_reps = _parse_reps_list(reps_cfg.get("test"), "test")
    val_reps = _parse_reps_list(reps_cfg.get("val"), "val")

    all_set = set(all_reps)
    test_set = set(test_reps)
    val_set = set(val_reps)
    if not test_set.issubset(all_set):
        raise ValueError("protocol.reps.test must be a subset of reps.all.")
    if not val_set.issubset(all_set):
        raise ValueError("protocol.reps.val must be a subset of reps.all.")
    overlap = sorted(test_set & val_set)
    if overlap:
        raise ValueError(
            "protocol.reps.test and protocol.reps.val must be disjoint "
            f"(overlap: {overlap})."
        )

    train_reps = sorted(all_set - test_set - val_set)
    if not train_reps:
        raise ValueError("protocol.reps leaves no repetitions for training.")
    return {
        "all": all_reps,
        "train": train_reps,
        "val": val_reps,
        "test": test_reps,
    }


def filter_indices(
    manifest_rows: list[dict],
    subject_ids: set[int] | None = None,
    rep_ids: set[int] | None = None,
    exclude_subjects: set[int] | None = None,
) -> list[str]:
    indices: list[str] = []
    for row in manifest_rows:
        subject_id = int(row["subject_id"])
        rep_id = int(row["rep_id"])
        if subject_ids is not None and subject_id not in subject_ids:
            continue
        if exclude_subjects is not None and subject_id in exclude_subjects:
            continue
        if rep_ids is not None and rep_id not in rep_ids:
            continue
        indices.append(str(row["sample_id"]))
    return sorted(indices)


def split_counts(total: int) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 0, 1
    val = max(1, int(total * 0.2))
    test = max(1, int(total * 0.2))
    train = total - val - test
    if train < 1:
        train = 1
        if val > 1:
            val -= 1
        elif test > 1:
            test -= 1
        else:
            val = 0
    return train, val, test


def dummy_split_indices(
    indices: list[str], seed: int
) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    n_train, n_val, n_test = split_counts(len(shuffled))
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


@dataclass
class Split:
    protocol: str
    instance: dict
    hash: str
    train_idx: list[int]
    val_idx: list[int]
    test_idx: list[int]
    notes: dict | None = None

    def to_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "instance": self.instance,
            "hash": self.hash,
            "num_train_windows": len(self.train_idx),
            "num_val_windows": len(self.val_idx),
            "num_test_windows": len(self.test_idx),
            "train_idx": self.train_idx,
            "val_idx": self.val_idx,
            "test_idx": self.test_idx,
            "notes": self.notes or {},
        }


class ProtocolBase:
    name = ""

    def enumerate_instances(self, cfg: dict, manifest_rows: list[dict]) -> list[dict]:
        raise NotImplementedError

    def make_split(
        self,
        manifest_rows: list[dict],
        cfg: dict,
        instance: dict,
        manifest_hash: str,
    ) -> Split:
        raise NotImplementedError


def assert_split_invariants(
    split: dict,
    manifest_rows: list[dict],
    protocol_name: str,
    protocol_cfg: dict,
    allow_missing_classes: bool = False,
) -> None:
    lookup = {str(row["sample_id"]): row for row in manifest_rows}

    def rows_for(indices: list[str]) -> list[dict]:
        rows = []
        for idx in indices:
            key = str(idx)
            if key not in lookup:
                raise ValueError(f"Split index {key} not found in manifest.")
            rows.append(lookup[key])
        return rows

    train_rows = rows_for(split.get("train_idx", []))
    val_rows = rows_for(split.get("val_idx", []))
    test_rows = rows_for(split.get("test_idx", []))

    train_ids = {str(resolve_sample_id(row)) for row in train_rows}
    val_ids = {str(resolve_sample_id(row)) for row in val_rows}
    test_ids = {str(resolve_sample_id(row)) for row in test_rows}

    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap:
        raise ValueError("Split sample_ids overlap between train/val/test.")

    if protocol_cfg.get("reps") is not None:

        def rep_ids(rows: list[dict]) -> set[int]:
            return {int(row["rep_id"]) for row in rows}

        train_reps = rep_ids(train_rows)
        val_reps = rep_ids(val_rows)
        test_reps = rep_ids(test_rows)
        if train_reps & test_reps:
            raise ValueError("Split repetitions overlap between train and test.")
        if val_reps & test_reps:
            raise ValueError("Split repetitions overlap between val and test.")

    protocol_type = str(protocol_cfg.get("type", protocol_name)).lower()
    if protocol_type == "loso":
        train_subjects = {int(row["subject_id"]) for row in train_rows}
        val_subjects = {int(row["subject_id"]) for row in val_rows}
        test_subjects = {int(row["subject_id"]) for row in test_rows}
        if (train_subjects | val_subjects) & test_subjects:
            raise ValueError("LOSO split mixes test subject into train/val.")

    all_classes = {
        int(row["gesture_id"])
        for row in manifest_rows
        if row.get("gesture_id") not in (None, "")
    }
    for name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        if not rows:
            continue
        classes = {
            int(row["gesture_id"])
            for row in rows
            if row.get("gesture_id") not in (None, "")
        }
        missing = sorted(all_classes - classes)
        if missing:
            if allow_missing_classes:
                LOGGER.warning("Split %s missing classes: %s", name, missing)
            else:
                raise ValueError(f"Split {name} missing classes: {missing}")
