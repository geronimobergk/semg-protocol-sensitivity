from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import shutil
from pathlib import Path
import statistics
import time
from typing import Any

import torch
from torch import nn

from .datasets.datamodule import DataModule
from .datasets.manifest import load_manifest
from .datasets.splits.base import assert_split_invariants
from .models.registry import create_model
from .training.loop import resolve_device
from .training.optim import build_optimizer
from .training.seed import seed_everything
from .training.step import split_batch, train_step
from .utils.io import read_json, write_json
from .utils.logging import configure_logging, get_logger

LOGGER = get_logger(__name__)

VRAM_HEADROOM = 1.30
RAM_HEADROOM = 1.50

DEFAULT_TRAIN_STEPS = 500
DEFAULT_VAL_STEPS = 100
DEFAULT_WARMUP_STEPS = 50

DEFAULT_RUN_CKPT_MB = 20.0
DEFAULT_RUN_LOGS_MB = 5.0
DEFAULT_CKPTS_PER_RUN = 2


def _steps_for_windows(num_windows: int, batch_size: int) -> int:
    if num_windows <= 0:
        return 0
    return int(math.ceil(num_windows / batch_size))


def _cycle_loader(loader) -> Any:
    while True:
        for batch in loader:
            yield batch


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _get_vram_peak_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    peak_reserved = torch.cuda.max_memory_reserved(device)
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_bytes = max(peak_reserved, peak_alloc)
    return float(peak_bytes) / float(1024**3)


def _get_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            return str(torch.cuda.get_device_name(device))
        except Exception:
            return "cuda"
    return "cpu"


def _detect_hardware(artifacts_root: Path, device: torch.device) -> dict[str, Any]:
    gpu_name = None
    vram_total_gb = None
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else 0
        try:
            props = torch.cuda.get_device_properties(index)
            gpu_name = str(props.name)
            vram_total_gb = float(props.total_memory) / float(1024**3)
        except Exception:
            gpu_name = "cuda"

    cpu_cores = os.cpu_count() or 1
    ram_total_gb = _measure_ram_total_gb()
    disk_free_gb = _measure_disk_free_gb(artifacts_root)

    return {
        "gpu_name": gpu_name,
        "vram_total_gb": vram_total_gb,
        "cpu_cores": int(cpu_cores),
        "ram_total_gb": ram_total_gb,
        "disk_free_gb": disk_free_gb,
    }


def _measure_rss_gb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        rss_bytes = psutil.Process().memory_info().rss
    except Exception:
        return None
    return float(rss_bytes) / float(1024**3)


def _measure_ram_total_gb() -> float | None:
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            return float(psutil.virtual_memory().total) / float(1024**3)
        except Exception:
            pass
    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            return float(page_size * phys_pages) / float(1024**3)
        except Exception:
            pass
    return None


def _measure_disk_free_gb(path: Path) -> float | None:
    try:
        usage = shutil.disk_usage(path)
    except Exception:
        return None
    return float(usage.free) / float(1024**3)


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += int((Path(root) / name).stat().st_size)
            except FileNotFoundError:
                continue
    return total


def _bytes_to_gb(num_bytes: int) -> float:
    return float(num_bytes) / float(1024**3)


def _estimate_runs_artifacts_gb(total_runs: int) -> float:
    total_mb = float(total_runs) * (
        DEFAULT_CKPTS_PER_RUN * DEFAULT_RUN_CKPT_MB + DEFAULT_RUN_LOGS_MB
    )
    return total_mb / 1024.0


def _resolve_models(cfg: dict) -> list[str]:
    plan_cfg = cfg.get("plan", {}) or {}
    models = plan_cfg.get("models") or list(cfg.get("models", {}).keys())
    if not models:
        raise ValueError("No models found in plan or config.")
    return [str(model_id) for model_id in models]


def _resolve_num_seeds(cfg: dict, seeds: int | None) -> int:
    if seeds is not None:
        num_seeds = int(seeds)
    else:
        seed_list = cfg.get("train", {}).get("seeds", [0])
        num_seeds = len(seed_list) if seed_list else 1
    if num_seeds < 1:
        raise ValueError("seeds must be >= 1.")
    return int(num_seeds)


def _resolve_max_epochs(cfg: dict, epochs: int | None) -> int:
    train_cfg = cfg.get("train", {})
    cfg_max_epochs = int(train_cfg.get("max_epochs", train_cfg.get("epochs", 1)))
    max_epochs = int(epochs) if epochs is not None else cfg_max_epochs
    if max_epochs < 1:
        raise ValueError("epochs must be >= 1.")
    return int(max_epochs)


def _resolve_max_steps(cfg: dict) -> int | None:
    train_cfg = cfg.get("train", {})
    max_steps = train_cfg.get("max_steps")
    return int(max_steps) if max_steps is not None else None


def _resolve_max_jobs(cfg: dict, max_jobs: int | None, default: int) -> int:
    if max_jobs is not None:
        value = int(max_jobs)
    else:
        plan_cfg = cfg.get("plan", {}) or {}
        value = int(plan_cfg.get("max_jobs", default) or default)
    return max(1, value)


def _resolve_protocol_name(split: dict, split_path: Path) -> str:
    protocol_name = str(split.get("protocol", "")).strip()
    if not protocol_name:
        raise ValueError(f"Split missing protocol name: {split_path}")
    return protocol_name


def _validate_split_basic(split: dict) -> None:
    train_idx = set(split.get("train_idx", []))
    val_idx = set(split.get("val_idx", []))
    test_idx = set(split.get("test_idx", []))
    if not train_idx:
        raise ValueError("Split has no training indices.")
    if not test_idx:
        raise ValueError("Split has no test indices.")
    if train_idx & val_idx or train_idx & test_idx or val_idx & test_idx:
        raise ValueError("Split indices overlap between train/val/test.")


def _select_bench_splits(cfg: dict, split_path: str | Path | None) -> list[Path]:
    if split_path is not None:
        path = Path(split_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        return [path]

    protocols = cfg.get("plan", {}).get("protocols") or list(cfg.get("protocols", {}))
    preferred = []
    for name in ("loso", "pooled_repdisjoint"):
        if name in protocols:
            preferred.append(name)

    selected: list[Path] = []
    for name in preferred:
        protocol_cfg = cfg.get("protocols", {}).get(name)
        if not isinstance(protocol_cfg, dict):
            continue
        output_dir = protocol_cfg.get("output_dir")
        if not output_dir:
            continue
        candidates = sorted(Path(output_dir).rglob("split.json"))
        if candidates:
            selected.append(candidates[0])

    if selected:
        return selected

    artifacts_root = Path(cfg["experiment"]["artifacts_root"])
    candidates = sorted((artifacts_root / "splits").rglob("split.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No split.json files found under {artifacts_root / 'splits'}."
        )
    return [candidates[0]]


def _eval_step(
    model: nn.Module,
    batch: Any,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    xb, yb, _ = split_batch(batch)
    xb = xb.to(device)
    yb = yb.to(device)
    with torch.no_grad():
        logits = model(xb)
        _ = criterion(logits, yb)


def _measure_throughput(
    *,
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
    batch_size: int,
    mode: str,
) -> dict[str, float | None]:
    if mode == "train":
        model.train()
    else:
        model.eval()

    iterator = _cycle_loader(loader)
    for _ in range(warmup_steps):
        batch = next(iterator)
        if mode == "train":
            train_step(model, batch, criterion, optimizer, device)
        else:
            _eval_step(model, batch, criterion, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    _sync_if_cuda(device)
    start = time.perf_counter()
    rss_peak = None
    for _ in range(measure_steps):
        batch = next(iterator)
        if mode == "train":
            train_step(model, batch, criterion, optimizer, device)
        else:
            _eval_step(model, batch, criterion, device)
        rss_now = _measure_rss_gb()
        if rss_now is not None:
            rss_peak = rss_now if rss_peak is None else max(rss_peak, rss_now)
    _sync_if_cuda(device)
    duration = time.perf_counter() - start

    steps_per_sec = float(measure_steps) / duration if duration > 0 else 0.0
    samples_per_sec = steps_per_sec * float(batch_size)
    return {
        "steps": float(measure_steps),
        "duration_sec": float(duration),
        "steps_per_sec": float(steps_per_sec),
        "samples_per_sec": float(samples_per_sec),
        "vram_peak_gb": _get_vram_peak_gb(device),
        "rss_peak_gb": rss_peak,
    }


def _bench_model_on_split(
    cfg: dict,
    split_path: Path,
    model_id: str,
    device: torch.device,
    warmup_steps: int,
    train_steps: int,
    val_steps: int,
    manifest_rows: list[dict],
    manifest_path: Path,
) -> dict[str, float | None]:
    split = read_json(split_path)
    protocol_name = _resolve_protocol_name(split, split_path)
    if protocol_name not in cfg.get("protocols", {}):
        raise KeyError(f"Protocol '{protocol_name}' not found in config.")
    assert_split_invariants(
        split,
        manifest_rows,
        protocol_name,
        cfg["protocols"][protocol_name],
        allow_missing_classes=bool(
            cfg.get("splits", {}).get("allow_missing_classes", False)
        ),
    )
    _validate_split_basic(split)

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 0))
    normalization_cfg = (
        train_cfg["normalization"]
        if "normalization" in train_cfg
        else {"type": "zscore"}
    )

    model_cfg = cfg["models"][model_id]
    architecture = model_cfg["architecture"]
    model_params = model_cfg.get("params", {})
    num_classes = int(model_params.get("num_classes", 0))
    if num_classes <= 0:
        raise ValueError("Model params must define num_classes.")
    num_electrodes = int(model_params.get("num_electrodes", 0))
    num_samples = int(model_params.get("num_samples", 0))
    if num_electrodes <= 0 or num_samples <= 0:
        raise ValueError("Model params must define num_electrodes/num_samples.")

    seed_list = train_cfg.get("seeds", [0])
    seed = int(seed_list[0]) if seed_list else 0
    seed_everything(seed)

    datamodule = DataModule(
        manifest_rows=manifest_rows,
        split=split,
        num_electrodes=num_electrodes,
        num_samples=num_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        manifest_dir=manifest_path.parent,
        normalization=normalization_cfg,
    )
    datamodule.setup()
    if datamodule.label_map is None:
        raise RuntimeError("DataModule failed to build label mapping.")
    if datamodule.label_map.num_classes != num_classes:
        raise ValueError(
            "num_classes mismatch: model expects "
            f"{num_classes}, manifest has {datamodule.label_map.num_classes}"
        )

    model = create_model(architecture, model_params).to(device)
    optimizer = build_optimizer(model.parameters(), train_cfg.get("optimizer", {}))
    criterion = nn.CrossEntropyLoss()

    train_loader = datamodule.loaders["train"]
    val_loader = datamodule.loaders.get("val")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_metrics = _measure_throughput(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        warmup_steps=warmup_steps,
        measure_steps=train_steps,
        batch_size=batch_size,
        mode="train",
    )

    val_metrics: dict[str, float | None] = {
        "steps": float(val_steps),
        "duration_sec": 0.0,
        "steps_per_sec": None,
        "samples_per_sec": None,
        "vram_peak_gb": 0.0,
        "rss_peak_gb": None,
    }
    if val_loader is not None and val_steps > 0:
        val_metrics = _measure_throughput(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            warmup_steps=warmup_steps,
            measure_steps=val_steps,
            batch_size=batch_size,
            mode="eval",
        )

    vram_peak = max(
        float(train_metrics.get("vram_peak_gb") or 0.0),
        float(val_metrics.get("vram_peak_gb") or 0.0),
    )
    rss_candidates = [
        value
        for value in (train_metrics.get("rss_peak_gb"), val_metrics.get("rss_peak_gb"))
        if value is not None
    ]
    rss_peak = max(rss_candidates) if rss_candidates else None

    result = {
        "train_steps": int(train_steps),
        "train_duration_sec": float(train_metrics["duration_sec"]),
        "train_steps_per_sec": float(train_metrics["steps_per_sec"]),
        "train_samples_per_sec": float(train_metrics["samples_per_sec"]),
        "val_steps": int(val_steps),
        "val_duration_sec": float(val_metrics["duration_sec"]),
        "val_steps_per_sec": val_metrics["steps_per_sec"],
        "val_samples_per_sec": val_metrics["samples_per_sec"],
        "vram_peak_gb": float(vram_peak),
        "ram_rss_peak_gb": rss_peak,
    }

    del model, optimizer, datamodule
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def _aggregate_bench_results(
    per_split_results: list[dict[str, float | None]],
    train_steps: int,
    val_steps: int,
) -> dict[str, float | None]:
    if not per_split_results:
        raise ValueError("No benchmark results collected.")

    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(statistics.median(values))

    train_steps_per_sec = _median(
        [float(item["train_steps_per_sec"]) for item in per_split_results]
    )
    train_samples_per_sec = _median(
        [float(item["train_samples_per_sec"]) for item in per_split_results]
    )

    val_steps_vals = [
        float(item["val_steps_per_sec"])
        for item in per_split_results
        if item.get("val_steps_per_sec") is not None
    ]
    val_samples_vals = [
        float(item["val_samples_per_sec"])
        for item in per_split_results
        if item.get("val_samples_per_sec") is not None
    ]
    val_steps_per_sec = _median(val_steps_vals) if val_steps_vals else None
    val_samples_per_sec = _median(val_samples_vals) if val_samples_vals else None

    vram_peak_gb = max(
        float(item.get("vram_peak_gb") or 0.0) for item in per_split_results
    )
    rss_candidates = [
        float(item["ram_rss_peak_gb"])
        for item in per_split_results
        if item.get("ram_rss_peak_gb") is not None
    ]
    rss_peak_gb = max(rss_candidates) if rss_candidates else None

    return {
        "train_steps": int(train_steps),
        "train_steps_per_sec": float(train_steps_per_sec),
        "train_samples_per_sec": float(train_samples_per_sec),
        "val_steps": int(val_steps),
        "val_steps_per_sec": val_steps_per_sec,
        "val_samples_per_sec": val_samples_per_sec,
        "vram_peak_gb": float(vram_peak_gb),
        "ram_rss_peak_gb": rss_peak_gb,
    }


def run_benchmark(
    cfg: dict,
    split_path: str | Path | None,
    warmup_steps: int,
    train_steps: int,
    val_steps: int,
    device_override: str | None,
) -> dict[str, Any]:
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0.")
    if train_steps < 1:
        raise ValueError("train_steps must be >= 1.")
    if val_steps < 0:
        raise ValueError("val_steps must be >= 0.")

    manifest_path = Path(cfg["manifest"]["output"]["manifest_csv"])
    manifest_rows = load_manifest(manifest_path)
    device_setting = (
        device_override
        if device_override is not None
        else cfg.get("train", {}).get("device")
    )
    device = resolve_device(device_setting)

    bench_splits = _select_bench_splits(cfg, split_path)
    split_meta: list[dict[str, str]] = []
    for path in bench_splits:
        split = read_json(path)
        split_meta.append(
            {
                "path": str(path),
                "protocol": str(split.get("protocol", "")),
            }
        )

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 0))

    bench_results: dict[str, dict[str, float | None]] = {}
    for model_id in _resolve_models(cfg):
        per_split_results: list[dict[str, float | None]] = []
        for split_path_item in bench_splits:
            per_split_results.append(
                _bench_model_on_split(
                    cfg=cfg,
                    split_path=split_path_item,
                    model_id=model_id,
                    device=device,
                    warmup_steps=warmup_steps,
                    train_steps=train_steps,
                    val_steps=val_steps,
                    manifest_rows=manifest_rows,
                    manifest_path=manifest_path,
                )
            )
        bench_results[model_id] = _aggregate_bench_results(
            per_split_results, train_steps=train_steps, val_steps=val_steps
        )
        LOGGER.info(
            "Bench %s: train_steps_per_sec=%.3f vram_peak=%.3f GB",
            model_id,
            bench_results[model_id]["train_steps_per_sec"],
            bench_results[model_id]["vram_peak_gb"],
        )

    return {
        "device": str(device),
        "device_name": _get_device_name(device),
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "warmup_steps": int(warmup_steps),
        "train_steps": int(train_steps),
        "val_steps": int(val_steps),
        "splits": split_meta,
        "models": bench_results,
    }


def _probe_worker(
    cfg: dict,
    split_path: str,
    model_id: str,
    warmup_steps: int,
    train_steps: int,
    result_queue: mp.Queue,
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    configure_logging()

    try:
        torch.set_num_threads(1)
        device = torch.device("cuda:0")

        manifest_path = Path(cfg["manifest"]["output"]["manifest_csv"])
        manifest_rows = load_manifest(manifest_path)
        split = read_json(split_path)

        train_cfg = cfg.get("train", {})
        batch_size = int(train_cfg.get("batch_size", 32))
        num_workers = int(train_cfg.get("num_workers", 0))
        normalization_cfg = (
            train_cfg["normalization"]
            if "normalization" in train_cfg
            else {"type": "zscore"}
        )

        model_cfg = cfg["models"][model_id]
        architecture = model_cfg["architecture"]
        model_params = model_cfg.get("params", {})
        num_classes = int(model_params.get("num_classes", 0))
        if num_classes <= 0:
            raise ValueError("Model params must define num_classes.")
        num_electrodes = int(model_params.get("num_electrodes", 0))
        num_samples = int(model_params.get("num_samples", 0))
        if num_electrodes <= 0 or num_samples <= 0:
            raise ValueError("Model params must define num_electrodes/num_samples.")

        seed_list = train_cfg.get("seeds", [0])
        seed = int(seed_list[0]) if seed_list else 0
        seed_everything(seed)

        datamodule = DataModule(
            manifest_rows=manifest_rows,
            split=split,
            num_electrodes=num_electrodes,
            num_samples=num_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            manifest_dir=manifest_path.parent,
            normalization=normalization_cfg,
        )
        datamodule.setup()
        if datamodule.label_map is None:
            raise RuntimeError("DataModule failed to build label mapping.")
        if datamodule.label_map.num_classes != num_classes:
            raise ValueError(
                "num_classes mismatch: model expects "
                f"{num_classes}, manifest has {datamodule.label_map.num_classes}"
            )

        model = create_model(architecture, model_params).to(device)
        optimizer = build_optimizer(model.parameters(), train_cfg.get("optimizer", {}))
        criterion = nn.CrossEntropyLoss()
        train_loader = datamodule.loaders["train"]

        if device.type == "cuda":
            torch.cuda.empty_cache()

        metrics = _measure_throughput(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            warmup_steps=warmup_steps,
            measure_steps=train_steps,
            batch_size=batch_size,
            mode="train",
        )
        steps_per_sec = float(metrics["steps_per_sec"])
        t_step_sec = 0.0 if steps_per_sec <= 0 else 1.0 / steps_per_sec

        result_queue.put(
            {
                "steps_per_sec": steps_per_sec,
                "samples_per_sec": float(metrics["samples_per_sec"]),
                "t_step_sec": t_step_sec,
                "vram_peak_gb": float(metrics["vram_peak_gb"] or 0.0),
                "ram_rss_peak_gb": metrics.get("rss_peak_gb"),
            }
        )
    except Exception as exc:
        result_queue.put({"error": str(exc)})


def _run_probe_for_k(
    cfg: dict,
    split_path: str,
    model_id: str,
    warmup_steps: int,
    train_steps: int,
    k: int,
) -> list[dict[str, Any]]:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    workers: list[mp.Process] = []
    for idx in range(k):
        proc = ctx.Process(
            name=f"probe-worker-{k}-{idx}",
            target=_probe_worker,
            args=(
                cfg,
                split_path,
                model_id,
                warmup_steps,
                train_steps,
                result_queue,
            ),
        )
        proc.start()
        workers.append(proc)

    results: list[dict[str, Any]] = []
    max_wait = max(120.0, float(warmup_steps + train_steps) * 2.0)
    deadline = time.monotonic() + max_wait
    while len(results) < len(workers):
        timeout = max(0.5, min(5.0, deadline - time.monotonic()))
        if timeout <= 0:
            break
        try:
            results.append(result_queue.get(timeout=timeout))
        except queue.Empty:
            if all(not proc.is_alive() for proc in workers):
                break
            continue

    for proc in workers:
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)

    if len(results) < len(workers):
        raise RuntimeError("Probe workers exited without reporting results.")

    for result in results:
        if "error" in result:
            raise RuntimeError(result["error"])
    return results


def probe_jobs_per_gpu(
    cfg: dict,
    bench: dict[str, Any],
    max_k: int,
    warmup_steps: int,
    train_steps: int,
    device_override: str | None,
) -> dict[str, Any]:
    if max_k < 1:
        raise ValueError("max_k must be >= 1.")

    device_setting = (
        device_override
        if device_override is not None
        else cfg.get("train", {}).get("device")
    )
    device = resolve_device(device_setting)
    if device.type != "cuda" or not torch.cuda.is_available():
        LOGGER.info("Probe skipped (CUDA unavailable).")
        return {
            "status": "skipped",
            "reason": "cuda_unavailable",
            "selected_jobs_per_gpu": 1,
            "table": [],
        }

    bench_models = bench.get("models") or {}
    if not bench_models:
        raise ValueError("bench results missing models data.")

    splits = bench.get("splits") or []
    if not splits:
        raise ValueError("bench results missing split metadata.")
    split_path = str(splits[0].get("path", "")).strip()
    if not split_path:
        raise ValueError("bench results missing split path.")
    if not Path(split_path).exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    model_id = None
    best_vram = -1.0
    worst_speed = None
    for candidate_id, entry in bench_models.items():
        vram_peak = float(entry.get("vram_peak_gb") or 0.0)
        train_steps_per_sec = float(entry.get("train_steps_per_sec") or 0.0)
        if vram_peak > best_vram:
            model_id = str(candidate_id)
            best_vram = vram_peak
            worst_speed = train_steps_per_sec
        elif vram_peak == best_vram and worst_speed is not None:
            if train_steps_per_sec < worst_speed:
                model_id = str(candidate_id)
                worst_speed = train_steps_per_sec
    if model_id is None:
        model_id = str(next(iter(bench_models.keys())))

    try:
        props = torch.cuda.get_device_properties(0)
        vram_total_gb = float(props.total_memory) / float(1024**3)
        device_name = str(props.name)
    except Exception:
        vram_total_gb = 0.0
        device_name = "cuda:0"

    LOGGER.info(
        "Probing jobs/GPU using model=%s on %s (max_k=%s).",
        model_id,
        device_name,
        max_k,
    )

    table: list[dict[str, Any]] = []
    baseline_t_step = None
    status = "ok"
    error = None

    for k in range(1, max_k + 1):
        try:
            results = _run_probe_for_k(
                cfg,
                split_path,
                model_id,
                warmup_steps,
                train_steps,
                k,
            )
        except Exception as exc:
            error = str(exc)
            status = "partial" if table else "error"
            LOGGER.warning("Probe failed for k=%s: %s", k, exc)
            break

        samples_per_sec = [float(res["samples_per_sec"]) for res in results]
        t_steps = [float(res["t_step_sec"]) for res in results]
        vram_peaks = [float(res["vram_peak_gb"]) for res in results]

        t_step_med = statistics.median(t_steps) if t_steps else 0.0
        if baseline_t_step is None:
            baseline_t_step = t_step_med

        agg_samples_sec = float(sum(samples_per_sec))
        per_job_samples_sec_p50 = (
            float(statistics.median(samples_per_sec)) if samples_per_sec else 0.0
        )
        vram_sum_gb = float(sum(vram_peaks))
        slowdown = None
        if baseline_t_step and baseline_t_step > 0:
            slowdown = float(t_step_med) / float(baseline_t_step)

        safe = True
        if vram_total_gb > 0:
            safe = vram_sum_gb * VRAM_HEADROOM <= vram_total_gb

        table.append(
            {
                "k": int(k),
                "agg_samples_sec": agg_samples_sec,
                "per_job_samples_sec_p50": per_job_samples_sec_p50,
                "vram_sum_gb": vram_sum_gb,
                "slowdown": slowdown,
                "safe": bool(safe),
            }
        )

    safe_rows = [row for row in table if row.get("safe")] if table else []
    if safe_rows:
        best = max(safe_rows, key=lambda row: (row["agg_samples_sec"], -row["k"]))
        selected_jobs_per_gpu = int(best["k"])
    else:
        selected_jobs_per_gpu = 1
        if table:
            LOGGER.warning(
                "No safe probe result found; falling back to jobs_per_gpu=1."
            )

    for row in table:
        row["chosen"] = bool(row.get("k") == selected_jobs_per_gpu)

    return {
        "status": status,
        "error": error,
        "device": "cuda:0",
        "device_name": device_name,
        "model_id": model_id,
        "max_k": int(max_k),
        "vram_total_gb": float(vram_total_gb),
        "vram_headroom": float(VRAM_HEADROOM),
        "table": table,
        "selected_jobs_per_gpu": int(selected_jobs_per_gpu),
    }


def compute_workload(
    cfg: dict,
    splits_root: str | Path | None = None,
    seeds: int | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    artifacts_root = Path(cfg["experiment"]["artifacts_root"])
    splits_root = (
        Path(splits_root) if splits_root is not None else artifacts_root / "splits"
    )
    split_paths = sorted(splits_root.rglob("split.json"))
    if not split_paths:
        raise FileNotFoundError(f"No split.json files found under {splits_root}.")

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 32))
    max_epochs = _resolve_max_epochs(cfg, epochs)
    max_steps = _resolve_max_steps(cfg)
    num_seeds = _resolve_num_seeds(cfg, seeds)

    total_train_windows = 0
    total_val_windows = 0
    total_test_windows = 0
    total_train_steps = 0
    total_val_steps = 0
    total_test_steps = 0

    for split_path in split_paths:
        split = read_json(split_path)
        num_train = int(split.get("num_train_windows", 0))
        num_val = int(split.get("num_val_windows", 0))
        num_test = int(split.get("num_test_windows", 0))
        total_train_windows += num_train
        total_val_windows += num_val
        total_test_windows += num_test

        train_steps_per_epoch = _steps_for_windows(num_train, batch_size)
        val_steps_per_epoch = _steps_for_windows(num_val, batch_size)
        test_steps = _steps_for_windows(num_test, batch_size)

        if train_steps_per_epoch == 0:
            effective_epochs = 0
        elif max_steps is None:
            effective_epochs = max_epochs
        else:
            effective_epochs = min(
                max_epochs, int(math.ceil(max_steps / train_steps_per_epoch))
            )

        train_steps = train_steps_per_epoch * effective_epochs
        if max_steps is not None:
            train_steps = min(train_steps, max_steps)

        total_train_steps += train_steps
        total_val_steps += val_steps_per_epoch * effective_epochs
        total_test_steps += test_steps

    total_train_steps_all = total_train_steps * num_seeds
    total_val_steps_all = total_val_steps * num_seeds
    total_test_steps_all = total_test_steps * num_seeds

    models = _resolve_models(cfg)
    total_runs = len(split_paths) * len(models) * num_seeds

    return {
        "splits_root": str(splits_root),
        "splits": {
            "count": len(split_paths),
            "num_train_windows": total_train_windows,
            "num_val_windows": total_val_windows,
            "num_test_windows": total_test_windows,
        },
        "train": {
            "batch_size": int(batch_size),
            "max_epochs": int(max_epochs),
            "max_steps": int(max_steps) if max_steps is not None else None,
            "num_seeds": int(num_seeds),
            "overrides": {
                "num_seeds": seeds is not None,
                "max_epochs": epochs is not None,
            },
        },
        "runs": {
            "models": models,
            "num_models": len(models),
            "num_seeds": int(num_seeds),
            "num_splits": len(split_paths),
            "total_runs": int(total_runs),
        },
        "steps": {
            "train": int(total_train_steps_all),
            "val": int(total_val_steps_all),
            "test": int(total_test_steps_all),
            "total": int(
                total_train_steps_all + total_val_steps_all + total_test_steps_all
            ),
        },
    }


def compute_time_breakdown(
    workload: dict[str, Any], bench: dict[str, Any]
) -> tuple[dict[str, dict[str, float | None]], float]:
    bench_models = bench.get("models") or {}
    if not isinstance(bench_models, dict) or not bench_models:
        raise ValueError("bench results missing models data.")

    runs = workload.get("runs") or {}
    models = runs.get("models") or []
    if not models:
        raise ValueError("Workload missing runs.models.")

    steps = workload.get("steps") or {}
    total_train_steps_all = int(steps.get("train", 0))
    total_val_steps_all = int(steps.get("val", 0))
    total_test_steps_all = int(steps.get("test", 0))

    per_model: dict[str, dict[str, float | None]] = {}
    total_time_sec = 0.0

    for model_id in models:
        if model_id not in bench_models:
            raise KeyError(f"Bench results missing model '{model_id}'.")
        entry = bench_models[model_id] or {}
        train_steps_per_sec = float(entry.get("train_steps_per_sec") or 0.0)
        val_steps_per_sec = entry.get("val_steps_per_sec")
        val_steps_per_sec = (
            float(val_steps_per_sec)
            if val_steps_per_sec is not None
            else train_steps_per_sec
        )
        if train_steps_per_sec <= 0:
            raise ValueError(
                f"Bench results missing throughput for model '{model_id}'."
            )
        if val_steps_per_sec <= 0:
            val_steps_per_sec = train_steps_per_sec

        train_time = (
            float(total_train_steps_all) / float(train_steps_per_sec)
            if total_train_steps_all > 0
            else 0.0
        )
        val_time = (
            float(total_val_steps_all) / float(val_steps_per_sec)
            if total_val_steps_all > 0
            else 0.0
        )
        test_time = (
            float(total_test_steps_all) / float(val_steps_per_sec)
            if total_test_steps_all > 0
            else 0.0
        )
        total_model_time = train_time + val_time + test_time
        total_time_sec += total_model_time

        per_model[model_id] = {
            "train_steps_per_sec": float(train_steps_per_sec),
            "val_steps_per_sec": float(val_steps_per_sec),
            "train_time_sec": float(train_time),
            "val_time_sec": float(val_time),
            "test_time_sec": float(test_time),
            "total_time_sec": float(total_model_time),
        }

    return per_model, total_time_sec


def predict_walltime(
    workload: dict[str, Any],
    bench: dict[str, Any],
    jobs_per_gpu: int,
    max_gpus: int,
    alpha: float,
    max_jobs: int,
) -> dict[str, Any]:
    if max_gpus < 1:
        raise ValueError("max_gpus must be >= 1.")
    if jobs_per_gpu < 1:
        raise ValueError("jobs_per_gpu must be >= 1.")
    if alpha < 1.0:
        LOGGER.warning("alpha=%.3f < 1.0; clamping to 1.0.", alpha)
        alpha = 1.0

    per_model, total_time_sec = compute_time_breakdown(workload, bench)
    total_runs = int((workload.get("runs") or {}).get("total_runs") or 0)
    if total_runs <= 0:
        total_runs = 1

    table: list[dict[str, Any]] = []
    for g in range(1, max_gpus + 1):
        concurrency = min(total_runs, g * jobs_per_gpu, max_jobs)
        concurrency = max(1, concurrency)
        wall_sec = (total_time_sec * float(alpha)) / float(concurrency)
        table.append(
            {
                "gpus": int(g),
                "effective_concurrency": int(concurrency),
                "wall_sec": float(wall_sec),
                "wall_hours": float(wall_sec / 3600.0),
            }
        )

    return {
        "alpha": float(alpha),
        "jobs_per_gpu": int(jobs_per_gpu),
        "max_jobs": int(max_jobs),
        "total_time_sec": float(total_time_sec),
        "total_time_hours": float(total_time_sec / 3600.0),
        "wall_time_by_gpus": table,
        "per_model": per_model,
    }


def recommend_resources(
    cfg: dict,
    bench: dict[str, Any],
    workload: dict[str, Any],
    jobs_per_gpu: int,
    max_gpus: int,
    max_jobs: int,
) -> dict[str, Any]:
    bench_models = bench.get("models") or {}
    if not bench_models:
        raise ValueError("bench results missing models data.")

    vram_per_job = 0.0
    ram_rss_peak_gb = 0.0
    for entry in bench_models.values():
        vram_peak = entry.get("vram_peak_gb")
        vram_per_job = max(
            vram_per_job, float(vram_peak) if vram_peak is not None else 0.0
        )
        rss_peak = entry.get("ram_rss_peak_gb")
        if rss_peak is not None:
            ram_rss_peak_gb = max(ram_rss_peak_gb, float(rss_peak))

    vram_per_gpu_required = vram_per_job * jobs_per_gpu * VRAM_HEADROOM

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 32))
    if ram_rss_peak_gb > 0:
        ram_per_job_gb = float(ram_rss_peak_gb)
        ram_source = "bench_rss"
    else:
        ram_per_job_gb = max(4.0, 1.0 + 0.5 * (batch_size / 256.0))
        ram_source = "heuristic"

    total_runs = int((workload.get("runs") or {}).get("total_runs") or 0)
    total_runs = max(1, total_runs)
    total_concurrency = min(total_runs, max_gpus * jobs_per_gpu, max_jobs)
    total_concurrency = max(1, total_concurrency)

    ram_total_required = ram_per_job_gb * float(total_concurrency) * RAM_HEADROOM

    num_workers = int(train_cfg.get("num_workers", 0))
    cores_per_job = max(1, num_workers + 1)
    cpu_cores = int(total_concurrency * cores_per_job)

    artifacts_root = Path(cfg["experiment"]["artifacts_root"])
    runs_root = Path(cfg["experiment"]["runs_root"])
    artifacts_gb = _bytes_to_gb(_path_size_bytes(artifacts_root))
    runs_existing_gb = _bytes_to_gb(_path_size_bytes(runs_root))
    runs_estimated_gb = _estimate_runs_artifacts_gb(total_runs)
    disk_required_gb = artifacts_gb + runs_estimated_gb

    return {
        "jobs_per_gpu": int(jobs_per_gpu),
        "vram_per_job_gb": float(vram_per_job),
        "vram_per_gpu_required_gb": float(vram_per_gpu_required),
        "ram_per_job_gb": float(ram_per_job_gb),
        "ram_total_required_gb": float(ram_total_required),
        "ram_headroom": float(RAM_HEADROOM),
        "ram_source": ram_source,
        "cpu_cores": int(cpu_cores),
        "cores_per_job": int(cores_per_job),
        "disk_required_gb": float(disk_required_gb),
        "disk": {
            "artifacts_gb": float(artifacts_gb),
            "runs_existing_gb": float(runs_existing_gb),
            "runs_estimated_gb": float(runs_estimated_gb),
        },
    }


def run_size(
    cfg: dict,
    split_path: str | Path | None = None,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    train_steps: int = DEFAULT_TRAIN_STEPS,
    val_steps: int = DEFAULT_VAL_STEPS,
    device_override: str | None = None,
    max_k: int = 4,
    max_gpus: int = 8,
    alpha: float = 1.2,
    seeds: int | None = None,
    epochs: int | None = None,
    max_jobs: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    bench = run_benchmark(
        cfg,
        split_path=split_path,
        warmup_steps=warmup_steps,
        train_steps=train_steps,
        val_steps=val_steps,
        device_override=device_override,
    )

    probe = probe_jobs_per_gpu(
        cfg,
        bench,
        max_k=max_k,
        warmup_steps=warmup_steps,
        train_steps=train_steps,
        device_override=device_override,
    )
    jobs_per_gpu = int(probe.get("selected_jobs_per_gpu") or 1)

    workload = compute_workload(cfg, seeds=seeds, epochs=epochs)
    max_jobs_effective = _resolve_max_jobs(cfg, max_jobs, max_gpus * jobs_per_gpu)

    predictions = predict_walltime(
        workload,
        bench,
        jobs_per_gpu=jobs_per_gpu,
        max_gpus=max_gpus,
        alpha=alpha,
        max_jobs=max_jobs_effective,
    )
    resources = recommend_resources(
        cfg,
        bench,
        workload,
        jobs_per_gpu=jobs_per_gpu,
        max_gpus=max_gpus,
        max_jobs=max_jobs_effective,
    )

    baseline_per_model: dict[str, dict[str, float | None]] = {}
    for model_id, entry in (bench.get("models") or {}).items():
        payload: dict[str, float | None] = {
            "samples_sec": float(entry.get("train_samples_per_sec") or 0.0),
            "vram_peak_gb": float(entry.get("vram_peak_gb") or 0.0),
        }
        rss_peak = entry.get("ram_rss_peak_gb")
        if rss_peak is not None:
            payload["rss_peak_gb"] = float(rss_peak)
        baseline_per_model[str(model_id)] = payload

    probe_rows = probe.get("table") or []
    walltime_rows = [
        {
            "gpus": int(row["gpus"]),
            "concurrency": int(row["effective_concurrency"]),
            "wall_hours": float(row["wall_hours"]),
        }
        for row in (predictions.get("wall_time_by_gpus") or [])
    ]

    recommendation = {
        "jobs_per_gpu": int(jobs_per_gpu),
        "vram_per_gpu_gb": float(resources.get("vram_per_gpu_required_gb") or 0.0),
        "ram_total_gb": float(resources.get("ram_total_required_gb") or 0.0),
        "cpu_cores": int(resources.get("cpu_cores") or 0),
        "ssd_gb": float(resources.get("disk_required_gb") or 0.0),
    }

    artifacts_root = Path(cfg["experiment"]["artifacts_root"])
    device_setting = (
        device_override
        if device_override is not None
        else cfg.get("train", {}).get("device")
    )
    output = {
        "hardware_detected": _detect_hardware(
            artifacts_root, resolve_device(device_setting)
        ),
        "baseline_per_model": baseline_per_model,
        "probe_jobs_per_gpu": probe_rows,
        "recommendation": recommendation,
        "walltime_by_gpus": walltime_rows,
        "workload": workload,
    }

    output_path = (
        Path(output_path).expanduser()
        if output_path is not None
        else artifacts_root / "sizing" / "sizing.json"
    )
    write_json(output_path, output)

    wall_by_gpus = predictions.get("wall_time_by_gpus") or []
    if wall_by_gpus:
        LOGGER.info(
            "Sizing: jobs_per_gpu=%s vram>=%.1fGB ram>=%.1fGB disk>=%.1fGB (wrote %s)",
            resources.get("jobs_per_gpu"),
            resources.get("vram_per_gpu_required_gb"),
            resources.get("ram_total_required_gb"),
            resources.get("disk_required_gb"),
            output_path,
        )
    else:
        LOGGER.info("Wrote sizing results to %s", output_path)

    return output
