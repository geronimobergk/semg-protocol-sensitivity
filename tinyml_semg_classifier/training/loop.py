from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil
import socket
import time
from typing import Any

import torch
from torch import nn

from ..datasets.datamodule import DataModule
from ..datasets.manifest import load_manifest
from ..datasets.splits.base import (
    assert_split_invariants,
    get_protocol,
    instance_to_tag,
    split_path,
)
from ..models.registry import create_model
from ..eval.efficiency import count_parameters, estimate_macs_flops, measure_latency
from ..eval.predict import predict
from ..utils.io import read_json, write_json, write_yaml
from ..utils.logging import get_logger
from .checkpointing import is_improvement, save_checkpoint
from .metrics import ClassificationStats
from .optim import build_optimizer, build_scheduler
from .seed import seed_everything
from .step import split_batch, train_step

LOGGER = get_logger(__name__)


def resolve_device(device_setting: str | None) -> torch.device:
    if (
        device_setting is None
        or str(device_setting).strip() == ""
        or device_setting == "auto"
    ):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_setting)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def _force_cuda_device(cfg: dict) -> dict:
    resolved = deepcopy(cfg)
    train_cfg = resolved.setdefault("train", {})
    device_setting = train_cfg.get("device")
    if device_setting is None:
        return resolved
    text = str(device_setting).strip()
    if text.startswith("cuda:"):
        train_cfg["device"] = "cuda"
    return resolved


def _lock_path_for_run_dir(run_dir: Path) -> Path:
    return run_dir.with_name(f"{run_dir.name}.lock")


def _try_acquire_lock(lock_path: Path, payload: dict) -> bool:
    lock_path = Path(lock_path)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(lock_path, flags)
    except FileExistsError:
        return False
    except OSError as exc:
        raise RuntimeError(f"Failed to create lock file: {lock_path}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            import json

            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return True


def _write_run_env(run_dir: Path, job: dict) -> None:
    env = {
        "pid": int(os.getpid()),
        "hostname": socket.gethostname(),
        "time_unix": time.time(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "model": job.get("model_id"),
        "protocol": job.get("protocol_name"),
        "seed": job.get("seed"),
        "instance": job.get("instance"),
    }
    env["torch"] = {
        "version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count())
        if torch.cuda.is_available()
        else 0,
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            env["torch"]["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    write_json(run_dir / "run_env.json", env)


def _execute_job(
    cfg: dict,
    job: dict,
    manifest_rows: list[dict],
    manifest_path: Path,
) -> tuple[str, str | None]:
    train_cache = bool(cfg.get("train", {}).get("cache", True))
    model_id = str(job["model_id"])
    protocol_name = str(job["protocol_name"])
    instance = job["instance"]
    seed = int(job["seed"])
    run_dir = Path(job["run_dir"])
    source_split = Path(job["source_split"])

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path_for_run_dir(run_dir)
    lock_payload = {
        "pid": int(os.getpid()),
        "hostname": socket.gethostname(),
        "time_unix": time.time(),
        "run_dir": str(run_dir),
        "model": model_id,
        "protocol": protocol_name,
        "seed": seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    if not _try_acquire_lock(lock_path, lock_payload):
        LOGGER.info("Skipping locked run %s (lock: %s)", run_dir, lock_path)
        return "skipped_locked", str(lock_path)

    try:
        if run_dir.exists():
            if train_cache:
                LOGGER.info("Skipping existing run %s", run_dir)
                return "skipped_existing", None
            shutil.rmtree(run_dir)

        if not source_split.exists():
            raise FileNotFoundError(
                f"Missing split file: {source_split}. Run splits first."
            )

        run_dir.mkdir(parents=True, exist_ok=False)

        resolved_cfg = deepcopy(cfg)
        run_meta = {
            "model": model_id,
            "protocol": protocol_name,
            "instance": instance,
            "seed": seed,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        resolved_cfg["run"] = run_meta
        write_yaml(run_dir / "resolved_config.yaml", resolved_cfg)

        target_split = run_dir / "split.json"
        target_split.write_text(
            source_split.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        _write_run_env(run_dir, job)

        _run_single(
            cfg=cfg,
            run_dir=run_dir,
            model_id=model_id,
            protocol_name=protocol_name,
            instance=instance,
            seed=seed,
            manifest_rows=manifest_rows,
            manifest_path=manifest_path,
        )
        return "completed", None
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            LOGGER.warning("Failed to remove lock file %s", lock_path)


def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    log_every: int | None = None,
    logger=None,
    max_steps: int | None = None,
    start_step: int = 0,
) -> tuple[dict, int]:
    model.train()
    stats = ClassificationStats.create(num_classes)
    total_loss = 0.0
    total_samples = 0
    steps_run = 0
    for step, batch in enumerate(loader, start=1):
        global_step = start_step + step
        if max_steps is not None and global_step > max_steps:
            break
        loss_value, batch_size, logits, yb = train_step(
            model, batch, criterion, optimizer, device
        )
        total_loss += loss_value * batch_size
        total_samples += batch_size
        stats.update(logits, yb)
        steps_run += 1
        if log_every and logger and step % log_every == 0:
            logger.info("Step %s: loss=%.4f", step, loss_value)
    metrics = stats.summarize()
    metrics["loss"] = total_loss / max(1, total_samples)
    return metrics, steps_run


def eval_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    include_confusion: bool = False,
    include_per_class: bool = False,
    label_names: list[str] | None = None,
    max_batches: int | None = None,
) -> dict:
    model.eval()
    stats = ClassificationStats.create(num_classes)
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            xb, yb, _ = split_batch(batch)
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            stats.update(logits, yb)
            if max_batches is not None and step >= max_batches:
                break
    metrics = stats.summarize(
        include_confusion=include_confusion,
        include_per_class=include_per_class,
        label_names=label_names,
    )
    metrics["loss"] = total_loss / max(1, total_samples)
    return metrics


def _flatten_metrics(prefix: str, metrics: dict) -> dict:
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
        if key != "confusion_matrix"
    }


def _validate_split(split: dict) -> None:
    train_idx = set(split.get("train_idx", []))
    val_idx = set(split.get("val_idx", []))
    test_idx = set(split.get("test_idx", []))
    if not train_idx:
        raise ValueError("Split has no training indices.")
    if not test_idx:
        raise ValueError("Split has no test indices.")
    if train_idx & val_idx or train_idx & test_idx or val_idx & test_idx:
        raise ValueError("Split indices overlap between train/val/test.")


def _monitor_value(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def _label_names(label_map) -> list[str]:
    return [str(label) for label in label_map.labels]


def _summary_split(metrics: dict, prefix: str) -> dict:
    out: dict[str, float] = {}
    for key in ("balanced_accuracy", "macro_f1", "accuracy", "loss"):
        value = metrics.get(f"{prefix}_{key}")
        if value is not None:
            out[key] = float(value)
    return out


def _run_single(
    cfg: dict,
    run_dir: Path,
    model_id: str,
    protocol_name: str,
    instance: dict,
    seed: int,
    manifest_rows: list[dict],
    manifest_path: Path,
) -> None:
    train_cfg = cfg.get("train", {})
    device = resolve_device(train_cfg.get("device"))
    seed_everything(seed)
    max_steps = train_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None

    split_path_local = run_dir / "split.json"
    if not split_path_local.exists():
        raise FileNotFoundError(
            f"Missing split.json for run: {split_path_local}. Run splits first."
        )
    split = read_json(split_path_local)
    assert_split_invariants(
        split,
        manifest_rows,
        protocol_name,
        cfg["protocols"][protocol_name],
        allow_missing_classes=bool(
            cfg.get("splits", {}).get("allow_missing_classes", False)
        ),
    )
    _validate_split(split)

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
    normalization_cfg = (
        train_cfg["normalization"]
        if "normalization" in train_cfg
        else {"type": "zscore"}
    )

    datamodule = DataModule(
        manifest_rows=manifest_rows,
        split=split,
        num_electrodes=num_electrodes,
        num_samples=num_samples,
        batch_size=int(train_cfg.get("batch_size", 32)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        seed=int(seed),
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
    label_names = _label_names(datamodule.label_map)
    if datamodule.normalization_config is not None:
        write_json(run_dir / "normalization.json", datamodule.normalization_config)

    train_loader = datamodule.loaders["train"]
    val_loader = datamodule.loaders["val"]
    test_loader = datamodule.loaders["test"]

    model = create_model(architecture, model_params).to(device)
    optimizer = build_optimizer(model.parameters(), train_cfg.get("optimizer", {}))
    max_epochs = int(train_cfg.get("max_epochs", train_cfg.get("epochs", 1)))
    if max_epochs < 1:
        raise ValueError("train.max_epochs must be >= 1.")
    scheduler_spec = build_scheduler(
        optimizer, train_cfg.get("scheduler"), total_epochs=max_epochs
    )
    if scheduler_spec and hasattr(scheduler_spec.scheduler, "set_initial_lr"):
        scheduler_spec.scheduler.set_initial_lr()
    criterion = nn.CrossEntropyLoss()

    log_every = train_cfg.get("log_every")
    log_every = int(log_every) if log_every is not None else None

    early_cfg = train_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    monitor = str(early_cfg.get("monitor", "val_loss"))
    mode = str(early_cfg.get("mode", "min")).lower()
    patience = int(early_cfg.get("patience", 10))
    min_epochs = int(early_cfg.get("min_epochs", 0))
    if early_enabled and (val_loader is None) and monitor.startswith("val_"):
        raise ValueError("Early stopping requires validation data.")

    checkpoint_cfg = train_cfg.get("checkpoint", {})
    save_best = bool(checkpoint_cfg.get("save_best", True))
    save_last = bool(checkpoint_cfg.get("save_last", True))
    primary_ckpt = str(checkpoint_cfg.get("primary", "last")).lower()

    train_log_path = run_dir / "train.log"
    train_log_path.write_text("", encoding="utf-8")
    metrics_history: list[dict[str, Any]] = []
    best_metric = None
    best_epoch = None
    bad_epochs = 0

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints_dir / "best.pt"
    last_path = checkpoints_dir / "last.pt"

    eval_cfg = cfg.get("eval", {})
    max_batches = eval_cfg.get("max_batches")
    max_batches = int(max_batches) if max_batches is not None else None

    global_step = 0
    for epoch in range(1, max_epochs + 1):
        train_metrics, steps_run = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes,
            log_every=log_every,
            logger=LOGGER,
            max_steps=max_steps,
            start_step=global_step,
        )
        global_step += steps_run
        val_metrics = {}
        if val_loader is not None:
            val_metrics = eval_epoch(
                model,
                val_loader,
                criterion,
                device,
                num_classes,
                max_batches=max_batches,
            )

        lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
        epoch_metrics = {"epoch": epoch, "lr": float(lr)}
        epoch_metrics.update(_flatten_metrics("train", train_metrics))
        if val_metrics:
            epoch_metrics.update(_flatten_metrics("val", val_metrics))
        metrics_history.append(epoch_metrics)

        with train_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"epoch={epoch} metrics={epoch_metrics}\n")

        monitor_value = _monitor_value(epoch_metrics, monitor)
        if is_improvement(monitor_value, best_metric, mode):
            best_metric = monitor_value
            best_epoch = epoch
            bad_epochs = 0
            if save_best:
                save_checkpoint(
                    best_path,
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "metrics": epoch_metrics,
                        "config": {
                            "model_id": model_id,
                            "protocol": protocol_name,
                            "instance": instance,
                            "seed": seed,
                        },
                    },
                )
        else:
            bad_epochs += 1

        if save_last:
            save_checkpoint(
                last_path,
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": epoch_metrics,
                    "config": {
                        "model_id": model_id,
                        "protocol": protocol_name,
                        "instance": instance,
                        "seed": seed,
                    },
                },
            )

        if scheduler_spec:
            if scheduler_spec.step_on_metric and monitor_value is None:
                raise ValueError(
                    f"Scheduler requires monitor metric '{monitor}' to be present."
                )
            if scheduler_spec.step_on_metric:
                scheduler_spec.scheduler.step(monitor_value)
            else:
                scheduler_spec.scheduler.step()

        if early_enabled and epoch >= min_epochs and bad_epochs >= patience:
            LOGGER.info("Early stopping at epoch %s", epoch)
            break
        if max_steps is not None and global_step >= max_steps:
            LOGGER.info("Reached max_steps=%s at epoch %s", max_steps, epoch)
            break

    write_json(run_dir / "metrics_history.json", metrics_history)

    eval_checkpoint = best_path if best_path.exists() else last_path
    eval_checkpoint_name = "best" if eval_checkpoint == best_path else "last"
    if eval_checkpoint.exists():
        checkpoint = torch.load(eval_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        selected_epoch = checkpoint.get("epoch", best_epoch)
    else:
        selected_epoch = metrics_history[-1]["epoch"] if metrics_history else None

    test_metrics = eval_epoch(
        model,
        test_loader,
        criterion,
        device,
        num_classes,
        include_confusion=True,
        include_per_class=True,
        label_names=label_names,
        max_batches=max_batches,
    )
    write_json(run_dir / "confusion_matrix.json", test_metrics["confusion_matrix"])

    eval_cfg = cfg.get("eval", {})
    include_probs = bool(eval_cfg.get("save_predictions", False))
    pred_checkpoint = best_path if best_path.exists() else eval_checkpoint
    try:
        predict(
            model,
            test_loader,
            device,
            run_dir,
            checkpoint_path=pred_checkpoint,
            include_probs=include_probs,
        )
    except FileNotFoundError:
        LOGGER.warning("Skipping predictions; no checkpoint found in %s", run_dir)

    summary_epoch = (
        best_epoch
        if (best_epoch is not None and best_path.exists())
        else selected_epoch
    )
    summary_metrics = {}
    if summary_epoch is not None:
        for entry in metrics_history:
            if entry["epoch"] == summary_epoch:
                summary_metrics = entry
                break

    train_summary = _summary_split(summary_metrics, "train")
    val_summary = _summary_split(summary_metrics, "val")
    num_epochs = metrics_history[-1]["epoch"] if metrics_history else 0

    input_tensor = torch.zeros(
        (1, num_electrodes, num_samples), dtype=torch.float32, device=device
    )
    num_parameters = count_parameters(model)
    macs, flops = estimate_macs_flops(model, input_tensor)
    latency = None
    latency_cfg = eval_cfg.get("latency", {})
    if bool(latency_cfg.get("enabled", False)):
        latency_device_setting = latency_cfg.get("device")
        try:
            latency_device = (
                resolve_device(latency_device_setting)
                if latency_device_setting is not None
                else device
            )
        except RuntimeError as exc:
            LOGGER.warning("Skipping latency measurement: %s", exc)
        else:
            if latency_device != device:
                model.to(latency_device)
                input_tensor = input_tensor.to(latency_device)
            latency = measure_latency(
                model,
                input_tensor,
                warmup=int(latency_cfg.get("warmup", 50)),
                iters=int(latency_cfg.get("iters", 200)),
            )

    metrics = {
        "model": model_id,
        "seed": int(seed),
        "protocol": protocol_name,
        "protocol_instance": instance,
        "train": train_summary,
        "val": val_summary,
        "test": {
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
        },
        "training": {
            "best_epoch": best_epoch,
            "num_epochs": int(num_epochs),
            "primary_checkpoint": primary_ckpt,
        },
        "model_stats": {
            "num_parameters": int(num_parameters),
            "macs_per_window": int(macs),
            "flops_per_window": int(flops),
            "latency": None if latency is None else float(latency),
        },
    }
    if "accuracy" in test_metrics:
        metrics["test"]["accuracy"] = float(test_metrics["accuracy"])
    if "loss" in test_metrics:
        metrics["test"]["loss"] = float(test_metrics["loss"])
    if "per_class" in test_metrics:
        metrics["test"]["per_class"] = test_metrics["per_class"]
    if "confusion_matrix" in test_metrics:
        metrics["test"]["confusion_matrix"] = test_metrics["confusion_matrix"]
    if eval_checkpoint_name:
        metrics["training"]["checkpoint"] = eval_checkpoint_name
    write_json(run_dir / "metrics.json", metrics)

    LOGGER.info("Completed run %s", run_dir)


def run(cfg: dict, max_jobs: int | None = None) -> None:
    plan_cfg = cfg.get("plan", {})
    models = plan_cfg.get("models") or list(cfg.get("models", {}).keys())
    protocols = plan_cfg.get("protocols") or list(cfg.get("protocols", {}).keys())
    seeds = cfg.get("train", {}).get("seeds", [0])

    runs_root = Path(cfg["experiment"]["runs_root"])
    manifest_path = Path(cfg["manifest"]["output"]["manifest_csv"])
    manifest_rows = load_manifest(manifest_path)

    jobs: list[dict[str, Any]] = []
    for model_id in models:
        for protocol_name in protocols:
            protocol_cfg = cfg["protocols"][protocol_name]
            protocol_type = protocol_cfg.get("type", protocol_name)
            protocol = get_protocol(protocol_type)()
            instances = protocol.enumerate_instances(protocol_cfg, manifest_rows)

            for instance in instances:
                instance_tag = instance_to_tag(instance)
                for seed in seeds:
                    run_dir = (
                        runs_root
                        / model_id
                        / protocol_name
                        / instance_tag
                        / f"seed{seed}"
                    )
                    source_split = split_path(
                        Path(protocol_cfg["output_dir"]), instance
                    )
                    jobs.append(
                        {
                            "model_id": model_id,
                            "protocol_name": protocol_name,
                            "instance": instance,
                            "seed": int(seed),
                            "run_dir": str(run_dir),
                            "source_split": str(source_split),
                        }
                    )

    max_jobs_effective = (
        int(max_jobs) if max_jobs is not None else int(plan_cfg.get("max_jobs", 1) or 1)
    )
    if max_jobs_effective < 1:
        raise ValueError("max_jobs must be >= 1.")

    if max_jobs_effective == 1:
        for job in jobs:
            _execute_job(
                cfg=cfg,
                job=job,
                manifest_rows=manifest_rows,
                manifest_path=manifest_path,
            )
        return

    worker_devices: list[str | None] = [None] * max_jobs_effective
    train_device = resolve_device(cfg.get("train", {}).get("device"))
    if train_device.type == "cuda" and torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            tokens = [tok.strip() for tok in visible.split(",") if tok.strip()]
        else:
            tokens = [str(idx) for idx in range(int(torch.cuda.device_count()))]
        if tokens:
            if max_jobs_effective > len(tokens):
                jobs_per_gpu = (max_jobs_effective + len(tokens) - 1) // len(tokens)
                LOGGER.info(
                    "max_jobs=%s exceeds available GPUs=%s; scheduling up to %s jobs per GPU.",
                    max_jobs_effective,
                    len(tokens),
                    jobs_per_gpu,
                )
            worker_devices = [
                tokens[idx % len(tokens)] for idx in range(max_jobs_effective)
            ]

    from .parallel import run_jobs_parallel

    run_jobs_parallel(
        cfg=cfg,
        jobs=jobs,
        max_jobs=max_jobs_effective,
        worker_devices=worker_devices,
    )
