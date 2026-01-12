import json
from pathlib import Path

import pytest

from tinyml_semg_classifier import cli
from tinyml_semg_classifier.utils.io import write_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs/experiments/protocol_sensitivity_semg_cnn.yml"


def _write_overrides(tmp_path: Path, overrides: dict) -> Path:
    path = tmp_path / "overrides.yml"
    write_yaml(path, overrides)
    return path


@pytest.mark.smoke
def test_smoke_run_pooled_repdisjoint_cpu(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    runs_root = tmp_path / "runs"
    overrides = {
        "experiment": {
            "artifacts_root": str(artifacts_root),
            "runs_root": str(runs_root),
            "reports_root": str(tmp_path / "reports"),
        },
        "plan": {"protocols": ["pooled_repdisjoint"]},
    }
    overrides_path = _write_overrides(tmp_path, overrides)

    cli.main(
        [
            "run",
            "-c",
            str(BASE_CONFIG),
            "--profile",
            "smoke",
            "--overrides",
            str(overrides_path),
        ]
    )

    run_dir = runs_root / "tiny_cnn" / "pooled_repdisjoint" / "all" / "seed0"
    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "test" in metrics
    assert "balanced_accuracy" in metrics["test"]
    assert "macro_f1" in metrics["test"]
    assert (run_dir / "split.json").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "preds.npz").exists()
    assert (
        artifacts_root
        / "splits"
        / "protocol=pooled_repdisjoint"
        / "instance=all"
        / "split.json"
    ).exists()


@pytest.mark.smoke
def test_smoke_run_loso_single_subject(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    runs_root = tmp_path / "runs"
    overrides = {
        "experiment": {
            "artifacts_root": str(artifacts_root),
            "runs_root": str(runs_root),
            "reports_root": str(tmp_path / "reports"),
        },
        "plan": {"protocols": ["loso"]},
        "protocols": {"loso": {"subjects": [1]}},
    }
    overrides_path = _write_overrides(tmp_path, overrides)

    cli.main(
        [
            "run",
            "-c",
            str(BASE_CONFIG),
            "--profile",
            "smoke",
            "--overrides",
            str(overrides_path),
        ]
    )

    run_dir = runs_root / "tiny_cnn" / "loso" / "test_subject=1" / "seed0"
    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "test" in metrics
    assert "balanced_accuracy" in metrics["test"]
    assert "macro_f1" in metrics["test"]
    assert (run_dir / "split.json").exists()
