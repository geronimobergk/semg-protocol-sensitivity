from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback
from typing import Any

from ..utils.logging import configure_logging, get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ParallelRunResult:
    status: str
    run_dir: str
    detail: str | None = None
    traceback: str | None = None


def _resolve_per_process_memory_fraction(
    cfg: dict, jobs_per_device: int
) -> float | None:
    parallel_cfg = cfg.get("parallel", {})
    if isinstance(parallel_cfg, dict) and "per_process_memory_fraction" in parallel_cfg:
        value = parallel_cfg.get("per_process_memory_fraction")
        if value in (None, 0, "0", "0.0"):
            return None
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid parallel.per_process_memory_fraction=%r; ignoring.", value
            )
            return None
        if fraction <= 0 or fraction > 1:
            LOGGER.warning(
                "parallel.per_process_memory_fraction must be in (0, 1]; ignoring."
            )
            return None
        return fraction
    if jobs_per_device == 2:
        return 0.45
    return None


def _build_worker_memory_fractions(
    cfg: dict, worker_devices: list[str | None]
) -> list[float | None]:
    counts = Counter(str(device) for device in worker_devices if device is not None)
    fractions: list[float | None] = []
    for device in worker_devices:
        if device is None:
            fractions.append(None)
            continue
        jobs_per_device = counts[str(device)]
        fractions.append(_resolve_per_process_memory_fraction(cfg, jobs_per_device))
    return fractions


def _kill_process(proc: mp.Process) -> None:
    try:
        proc.kill()
    except AttributeError:
        proc.terminate()
    except OSError:
        pass


def _shutdown_workers(
    workers: list[mp.Process], timeout: float = 5.0, force: bool = False
) -> None:
    if force:
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
        for proc in workers:
            proc.join(timeout=timeout)
        for proc in workers:
            if proc.is_alive():
                _kill_process(proc)
        for proc in workers:
            proc.join(timeout=timeout)
        return

    for proc in workers:
        proc.join(timeout=timeout)
    for proc in workers:
        if proc.is_alive():
            proc.terminate()
    for proc in workers:
        proc.join(timeout=timeout)
    for proc in workers:
        if proc.is_alive():
            _kill_process(proc)
    for proc in workers:
        proc.join(timeout=timeout)


def _worker_main(
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    cfg: dict,
    cuda_visible_devices: str | None,
    memory_fraction: float | None,
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    configure_logging()

    import torch

    torch.set_num_threads(1)
    if cuda_visible_devices is not None and torch.cuda.is_available():
        if memory_fraction is not None:
            try:
                torch.cuda.set_per_process_memory_fraction(
                    float(memory_fraction), device=0
                )
            except Exception:
                LOGGER.warning(
                    "Failed to set per-process memory fraction=%s", memory_fraction
                )

    from . import loop as loop_mod
    from ..datasets.manifest import load_manifest

    manifest_path = Path(cfg["manifest"]["output"]["manifest_csv"])
    manifest_rows = load_manifest(manifest_path)

    worker_cfg = cfg
    if cuda_visible_devices is not None:
        train_cfg = worker_cfg.setdefault("train", {})
        train_cfg["device"] = "cuda:0"

    while True:
        try:
            job = job_queue.get()
        except (EOFError, OSError):
            return
        if job is None:
            return
        run_dir = str(job.get("run_dir", ""))
        try:
            status, detail = loop_mod._execute_job(
                cfg=worker_cfg,
                job=job,
                manifest_rows=manifest_rows,
                manifest_path=manifest_path,
            )
        except Exception as exc:
            result_queue.put(
                ParallelRunResult(
                    status="error",
                    run_dir=run_dir,
                    detail=str(exc),
                    traceback=traceback.format_exc(),
                )
            )
        else:
            result_queue.put(
                ParallelRunResult(status=status, run_dir=run_dir, detail=detail)
            )


def _spawn_workers(
    ctx: mp.context.BaseContext,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    cfg: dict,
    worker_devices: list[str | None],
    worker_memory_fractions: list[float | None],
) -> list[mp.Process]:
    workers: list[mp.Process] = []
    for idx, (device, memory_fraction) in enumerate(
        zip(worker_devices, worker_memory_fractions)
    ):
        proc = ctx.Process(
            name=f"traineval-worker-{idx}",
            target=_worker_main,
            args=(job_queue, result_queue, cfg, device, memory_fraction),
        )
        proc.start()
        workers.append(proc)
    return workers


def run_jobs_parallel(
    cfg: dict,
    jobs: list[dict[str, Any]],
    max_jobs: int,
    worker_devices: list[str | None],
) -> None:
    if max_jobs < 1:
        raise ValueError("max_jobs must be >= 1.")
    if not jobs:
        LOGGER.info("No runs to schedule.")
        return

    if not worker_devices:
        worker_devices = [None] * max_jobs
    concurrency = min(max_jobs, len(worker_devices))
    worker_devices = list(worker_devices[:concurrency])
    if concurrency < 1:
        concurrency = 1
        worker_devices = [None]

    worker_memory_fractions = _build_worker_memory_fractions(cfg, worker_devices)

    ctx = mp.get_context("spawn")
    job_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()

    workers = _spawn_workers(
        ctx,
        job_queue,
        result_queue,
        cfg,
        worker_devices,
        worker_memory_fractions,
    )
    shutdown = False
    try:
        for job in jobs:
            job_queue.put(job)
        for _ in workers:
            job_queue.put(None)

        completed = 0
        skipped = 0
        start = time.perf_counter()
        while completed < len(jobs):
            try:
                result: ParallelRunResult = result_queue.get(timeout=5.0)
            except queue.Empty:
                alive = [proc for proc in workers if proc.is_alive()]
                if not alive:
                    raise RuntimeError("All workers exited unexpectedly.")
                continue

            if result.status == "error":
                _shutdown_workers(workers, force=True)
                shutdown = True
                raise RuntimeError(
                    f"Run failed in {result.run_dir}: {result.detail}\n{result.traceback}"
                )
            if result.status.startswith("skipped"):
                skipped += 1

            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                elapsed = time.perf_counter() - start
                LOGGER.info(
                    "Completed %s/%s runs (skipped=%s) in %.1fs",
                    completed,
                    len(jobs),
                    skipped,
                    elapsed,
                )
    finally:
        if not shutdown:
            _shutdown_workers(workers, force=False)
