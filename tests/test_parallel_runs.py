import json
import os
from pathlib import Path

import pytest

from tinyml_semg_classifier import cli
from tinyml_semg_classifier.training import loop as loop_mod
from tinyml_semg_classifier.utils.io import write_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs/experiments/protocol_sensitivity_semg_cnn.yml"


@pytest.mark.devmini
def test_dev_mini_parallel_max_jobs_writes_run_env(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    runs_root = tmp_path / "runs"
    overrides = {
        "experiment": {
            "artifacts_root": str(artifacts_root),
            "runs_root": str(runs_root),
            "reports_root": str(tmp_path / "reports"),
        }
    }
    overrides_path = tmp_path / "overrides.yml"
    write_yaml(overrides_path, overrides)

    cli.main(
        [
            "run",
            "-c",
            str(BASE_CONFIG),
            "--profile",
            "dev_mini",
            "--overrides",
            str(overrides_path),
            "--max-jobs",
            "2",
        ]
    )

    run_dir = runs_root / "tiny_cnn" / "pooled_repdisjoint" / "all" / "seed0"
    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists()

    env_path = run_dir / "run_env.json"
    assert env_path.exists()
    env = json.loads(env_path.read_text(encoding="utf-8"))
    assert "cuda_visible_devices" in env
    assert int(env["pid"]) != os.getpid()

    lock_path = run_dir.with_name(f"{run_dir.name}.lock")
    assert not lock_path.exists()


def test_run_lock_is_atomic(tmp_path: Path) -> None:
    lock_path = tmp_path / "seed0.lock"
    payload = {"pid": 123}
    assert loop_mod._try_acquire_lock(lock_path, payload)
    assert not loop_mod._try_acquire_lock(lock_path, payload)
    lock_path.unlink()
    assert loop_mod._try_acquire_lock(lock_path, payload)
    lock_path.unlink()
