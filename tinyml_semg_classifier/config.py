from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .utils.io import read_yaml
from .utils.logging import get_logger


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def build_context(cfg: dict) -> dict:
    experiment = cfg.get("experiment", {})
    preprocess = cfg.get("preprocess", {})
    manifest = cfg.get("manifest", {})
    return {
        "experiment_id": experiment.get("id"),
        "artifacts_root": experiment.get("artifacts_root"),
        "runs_root": experiment.get("runs_root"),
        "reports_root": experiment.get("reports_root"),
        "preprocess_id": preprocess.get("id"),
        "manifest_id": manifest.get("id"),
        "profile": cfg.get("profile"),
    }


def resolve_templates(obj: Any, context: dict) -> Any:
    if isinstance(obj, str):
        try:
            return obj.format_map(_SafeDict(context))
        except Exception:
            return obj
    if isinstance(obj, list):
        return [resolve_templates(item, context) for item in obj]
    if isinstance(obj, dict):
        return {key: resolve_templates(value, context) for key, value in obj.items()}
    return obj


def resolve_config(raw: dict) -> dict:
    cfg = deepcopy(raw)
    context = build_context(cfg)
    cfg = resolve_templates(cfg, context)
    context = build_context(cfg)
    cfg = resolve_templates(cfg, context)
    return cfg


def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_profile_path(profile: str, base_path: str | Path) -> Path:
    profile_path = Path(profile)
    if profile_path.suffix in (".yml", ".yaml"):
        if profile_path.is_absolute():
            return profile_path
        candidate = Path.cwd() / profile_path
        if candidate.exists():
            return candidate
        return Path(base_path).resolve().parent / profile_path
    if profile_path.exists():
        return profile_path.resolve()
    candidates = [
        Path.cwd() / "configs" / "profiles" / f"{profile}.yaml",
        Path.cwd() / "configs" / "profiles" / f"{profile}.yml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Profile not found: {profile}")


def _normalize_overrides(overrides: dict | Iterable[dict] | None) -> list[dict]:
    if overrides is None:
        return []
    if isinstance(overrides, dict):
        return [overrides]
    return [value for value in overrides]


def validate_and_freeze_profile(cfg: dict) -> None:
    logger = get_logger(__name__)
    profile = str(cfg.get("profile") or "full").strip().lower()
    if not profile:
        profile = "full"
    cfg["profile"] = profile

    full_like_profiles = {"full", "full_part1", "full_part2"}

    if profile not in full_like_profiles:
        logger.info("RUN PROFILE: %s (NOT FOR REPORTING RESULTS)", profile.upper())
    else:
        logger.info("RUN PROFILE: %s", profile.upper())

    allow_caps = bool(cfg.get("profile_allow_caps", False))
    train_cfg = cfg.setdefault("train", {})
    eval_cfg = cfg.setdefault("eval", {})
    plan_cfg = cfg.setdefault("plan", {})
    preprocess_cfg = cfg.setdefault("preprocess", {})
    dataset_cfg = cfg.setdefault("dataset", {})
    splits_cfg = cfg.setdefault("splits", {})

    if profile in {"smoke", "dev", "dev_mini"}:
        plan_cfg["max_jobs"] = 1
        train_cfg["num_workers"] = 0
        train_cfg["device"] = "cpu"
        max_steps_cap = 10 if profile == "smoke" else 50
        max_epochs_cap = 1 if profile == "smoke" else 3
        max_batches_cap = 5 if profile == "smoke" else 20
        max_steps = train_cfg.get("max_steps")
        if max_steps is None or int(max_steps) > max_steps_cap:
            train_cfg["max_steps"] = max_steps_cap
        max_epochs = train_cfg.get("max_epochs")
        if max_epochs is None or int(max_epochs) > max_epochs_cap:
            train_cfg["max_epochs"] = max_epochs_cap
        max_batches = eval_cfg.get("max_batches")
        if max_batches is None or int(max_batches) > max_batches_cap:
            eval_cfg["max_batches"] = max_batches_cap
        latency_cfg = eval_cfg.setdefault("latency", {})
        latency_cfg["enabled"] = False
        preprocess_cfg["cache"] = False
        splits_cfg.setdefault("allow_missing_classes", True)
    elif profile == "bench":
        splits_cfg.setdefault("allow_missing_classes", False)
    elif profile == "dry_run":
        if str(dataset_cfg.get("source", "")).lower() == "fixture":
            raise ValueError("dataset.source=fixture is not permitted for dry_run.")
        splits_cfg.setdefault("allow_missing_classes", False)
    elif profile in full_like_profiles:
        if not allow_caps:
            if train_cfg.get("max_steps") is not None:
                raise ValueError(
                    "train.max_steps is only permitted in smoke/dev/dev_mini profiles."
                )
            if eval_cfg.get("max_batches") is not None:
                raise ValueError(
                    "eval.max_batches is only permitted in smoke/dev/dev_mini profiles."
                )
            if str(dataset_cfg.get("source", "")).lower() == "fixture":
                raise ValueError(
                    "dataset.source=fixture is only permitted in smoke/dev/dev_mini profiles."
                )
        splits_cfg.setdefault("allow_missing_classes", False)
    else:
        logger.warning("Unknown profile '%s'; guardrails not applied.", profile)


def load_config(
    path: str,
    profile: str | None = None,
    overrides: dict | Iterable[dict] | None = None,
) -> dict:
    raw = read_yaml(path) or {}
    profile_name = None
    if profile:
        profile_path = _resolve_profile_path(profile, path)
        profile_cfg = read_yaml(profile_path) or {}
        raw = _deep_merge(raw, profile_cfg)
        profile_name = Path(profile_path).stem
    if profile_name is None:
        profile_name = str(raw.get("profile") or "full")
    raw["profile"] = profile_name

    for override in _normalize_overrides(overrides):
        raw = _deep_merge(raw, override)

    cfg = resolve_config(raw)
    validate_and_freeze_profile(cfg)
    return cfg
