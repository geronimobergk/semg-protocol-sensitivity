"""Download NinaPro DB2 preprocessed files into the raw data directory."""

from __future__ import annotations

import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from typing import Any

from .load_raw import collect_subject_dirs
from ...utils.io import parse_subjects_spec
from ...utils.logging import get_logger

LOGGER = get_logger(__name__)

DB2_BASE_URL = "https://ninapro.hevs.ch/files/DB2_Preproc/"
DB2_FILE_TEMPLATE = "DB2_s{subject_id}.zip"


def _remote_size(url: str, timeout: int) -> int | None:
    try:
        response = requests.head(url, timeout=timeout)
        if response.status_code == 200 and "Content-Length" in response.headers:
            return int(response.headers["Content-Length"])
    except Exception:
        return None
    return None


def download_file(
    url: str,
    out_path: Path,
    timeout: int = 30,
    retries: int = 3,
    chunk_size: int = 1024 * 64,
) -> bool:
    out_path = Path(out_path)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            total = _remote_size(url, timeout)
            if (
                out_path.exists()
                and total is not None
                and out_path.stat().st_size == total
            ):
                LOGGER.info("Skipping existing %s (size matches)", out_path.name)
                return True

            if tmp_path.exists():
                tmp_path.unlink()

            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                total_bytes = int(response.headers.get("Content-Length") or 0) or None
                with tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    desc=out_path.name,
                    leave=False,
                ) as pbar:
                    with open(tmp_path, "wb") as handle:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            pbar.update(len(chunk))

            tmp_path.replace(out_path)

            if total is not None and out_path.stat().st_size != total:
                raise IOError("Downloaded file size does not match Content-Length")

            return True
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            LOGGER.warning("Attempt %s failed for %s: %s", attempt, url, exc)
            time.sleep(2**attempt)

    LOGGER.error("Failed to download %s after %s attempts", url, retries)
    return False


def extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(dest_dir)
        return True
    except zipfile.BadZipFile:
        LOGGER.error("Bad zip file: %s", zip_path)
        return False


def download_subjects(
    dest_dir: Path,
    subject_ids: list[int],
    base_url: str,
    workers: int,
    keep_zip: bool,
    timeout: int,
    retries: int,
    chunk_size: int,
) -> None:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for subject_id in subject_ids:
        name = DB2_FILE_TEMPLATE.format(subject_id=subject_id)
        url = base_url.rstrip("/") + "/" + name
        out_file = dest_dir / name
        tasks.append((subject_id, url, out_file))

    results = []
    max_workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_file,
                url,
                out_path,
                timeout=timeout,
                retries=retries,
                chunk_size=chunk_size,
            ): (subject_id, url, out_path)
            for subject_id, url, out_path in tasks
        }
        for future in as_completed(futures):
            subject_id, url, out_path = futures[future]
            try:
                ok = future.result()
            except Exception as exc:
                LOGGER.error("Download error for %s: %s", url, exc)
                ok = False
            results.append((subject_id, out_path, ok))

    for subject_id, out_path, ok in results:
        if not ok:
            LOGGER.warning("Skipping extraction for subject %s", subject_id)
            continue
        LOGGER.info("Extracting %s", out_path.name)
        extracted = extract_zip(out_path, dest_dir)
        if extracted and not keep_zip:
            out_path.unlink(missing_ok=True)

    LOGGER.info("DB2 download complete for %s subjects", len(subject_ids))


def _require_setting(cfg: dict, key: str) -> Any:
    value = cfg.get(key)
    if value is None:
        raise ValueError(f"download configuration must include '{key}'")
    return value


def download_db2(cfg: dict) -> None:
    dataset_cfg = cfg.get("dataset")
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset configuration is required for DB2 download.")

    raw_dir_value = dataset_cfg.get("raw_dir")
    if raw_dir_value is None:
        raise ValueError("dataset.raw_dir is required for DB2 download.")
    raw_dir = Path(raw_dir_value)

    download_cfg = dataset_cfg.get("download")
    if not isinstance(download_cfg, dict):
        raise ValueError("dataset.download configuration is required.")

    workers = int(_require_setting(download_cfg, "workers"))
    keep_zip = bool(_require_setting(download_cfg, "keep_zip"))
    timeout = int(_require_setting(download_cfg, "timeout"))
    retries = int(_require_setting(download_cfg, "retries"))
    chunk_size = int(_require_setting(download_cfg, "chunk_size"))

    subjects = parse_subjects_spec(dataset_cfg.get("subjects"))
    if not subjects:
        raise ValueError("No subjects specified for DB2 download.")

    skip_existing = bool(download_cfg.get("skip_existing"))
    if skip_existing:
        existing_subjects: set[int] = set()
        if raw_dir.exists():
            existing_subjects = set(collect_subject_dirs(raw_dir).keys())
        remaining = sorted(set(subjects) - existing_subjects)
        if not remaining:
            LOGGER.info(
                "All requested subjects already exist in %s; skipping download.",
                raw_dir,
            )
            return
        subjects = remaining

    invalid = [subject for subject in subjects if subject < 1 or subject > 40]
    if invalid:
        raise ValueError(f"DB2 subjects must be in 1-40, got {invalid}")

    download_subjects(
        raw_dir,
        sorted(set(subjects)),
        base_url=DB2_BASE_URL,
        workers=workers,
        keep_zip=keep_zip,
        timeout=timeout,
        retries=retries,
        chunk_size=chunk_size,
    )
