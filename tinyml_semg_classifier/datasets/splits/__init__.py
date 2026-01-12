from __future__ import annotations

from pathlib import Path

from . import loso, pooled_repdisjoint, single_subject_repdisjoint  # noqa: F401
from .base import assert_split_invariants, get_protocol, split_hash, split_path
from ..manifest import load_manifest
from ...utils.hashing import hash_file
from ...utils.io import read_json, write_json
from ...utils.logging import get_logger

LOGGER = get_logger(__name__)


def generate_splits(cfg: dict) -> list[Path]:
    manifest_path = Path(cfg["manifest"]["output"]["manifest_csv"])
    manifest_rows = load_manifest(manifest_path)
    manifest_hash = hash_file(manifest_path)
    splits_cfg = cfg.get("splits", {})
    cache_enabled = bool(splits_cfg.get("cache", False))
    allow_missing = bool(splits_cfg.get("allow_missing_classes", False))

    outputs: list[Path] = []
    protocols_cfg = cfg.get("protocols", {})
    plan_protocols = cfg.get("plan", {}).get("protocols")
    if plan_protocols:
        protocol_items = []
        for name in plan_protocols:
            if name not in protocols_cfg:
                raise KeyError(f"Protocol '{name}' not found in config.")
            protocol_items.append((name, protocols_cfg[name]))
    else:
        protocol_items = list(protocols_cfg.items())

    for protocol_name, protocol_cfg in protocol_items:
        protocol_type = protocol_cfg.get("type", protocol_name)
        protocol = get_protocol(protocol_type)()
        instances = protocol.enumerate_instances(protocol_cfg, manifest_rows)
        output_dir = Path(protocol_cfg["output_dir"])
        for instance in instances:
            split_file = split_path(output_dir, instance)
            if cache_enabled and split_file.exists():
                expected_hash = split_hash(
                    manifest_hash, protocol.name, protocol_cfg, instance
                )
                try:
                    cached = read_json(split_file)
                except Exception as exc:
                    LOGGER.warning(
                        "Failed to read cached split %s: %s", split_file, exc
                    )
                else:
                    if cached.get("hash") == expected_hash:
                        outputs.append(split_file)
                        LOGGER.info("Split cache hit: %s", split_file)
                        continue
            split = protocol.make_split(
                manifest_rows,
                protocol_cfg,
                instance,
                manifest_hash,
            )
            assert_split_invariants(
                split.to_dict(),
                manifest_rows,
                protocol_name,
                protocol_cfg,
                allow_missing_classes=allow_missing,
            )
            write_json(split_file, split.to_dict())
            outputs.append(split_file)
            LOGGER.info("Wrote split %s", split_file)

    return outputs
