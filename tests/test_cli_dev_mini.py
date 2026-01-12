import json
from pathlib import Path

import pytest

from tinyml_semg_classifier import cli
from tinyml_semg_classifier.utils.io import write_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs/experiments/protocol_sensitivity_semg_cnn.yml"


@pytest.mark.devmini
def test_dev_mini_run(tmp_path: Path) -> None:
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
        ]
    )

    pooled_dir = runs_root / "tiny_cnn" / "pooled_repdisjoint" / "all" / "seed0"
    metrics_path = pooled_dir / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "test" in metrics
    assert "macro_f1" in metrics["test"]

    loso_dir = runs_root / "tiny_cnn" / "loso" / "test_subject=1" / "seed0"
    assert (loso_dir / "metrics.json").exists()
