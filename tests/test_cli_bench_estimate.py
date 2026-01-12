import json
import math
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
def test_smoke_size(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    runs_root = tmp_path / "runs"
    overrides = {
        "experiment": {
            "artifacts_root": str(artifacts_root),
            "runs_root": str(runs_root),
            "reports_root": str(tmp_path / "reports"),
        }
    }
    overrides_path = _write_overrides(tmp_path, overrides)

    cli.main(
        [
            "prepare",
            "-c",
            str(BASE_CONFIG),
            "--profile",
            "smoke",
            "--overrides",
            str(overrides_path),
        ]
    )
    cli.main(
        [
            "splits",
            "-c",
            str(BASE_CONFIG),
            "--profile",
            "smoke",
            "--overrides",
            str(overrides_path),
        ]
    )
    cli.main(
        [
            "size",
            "-c",
            str(BASE_CONFIG),
            "--profile",
            "smoke",
            "--overrides",
            str(overrides_path),
            "--warmup-steps",
            "1",
            "--bench-train-steps",
            "2",
            "--bench-val-steps",
            "1",
            "--device",
            "cpu",
            "--max-k",
            "1",
            "--max-gpus",
            "1",
            "--alpha",
            "1.0",
        ]
    )

    sizing_path = artifacts_root / "sizing" / "sizing.json"
    assert sizing_path.exists()
    sizing = json.loads(sizing_path.read_text(encoding="utf-8"))

    assert "hardware_detected" in sizing
    assert "baseline_per_model" in sizing
    assert "probe_jobs_per_gpu" in sizing
    assert "recommendation" in sizing
    assert "walltime_by_gpus" in sizing
    assert "workload" in sizing

    model_entry = next(iter(sizing["baseline_per_model"].values()))
    assert "samples_sec" in model_entry
    assert "vram_peak_gb" in model_entry
    assert sizing["walltime_by_gpus"]


def test_size_overrides_seeds_and_epochs(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    runs_root = tmp_path / "runs"
    cfg = {
        "profile": "bench",
        "experiment": {
            "id": "test_size_overrides",
            "artifacts_root": str(artifacts_root),
            "runs_root": str(runs_root),
            "reports_root": str(tmp_path / "reports"),
        },
        "dataset": {
            "source": "fixture",
            "fixture_path": "tests/fixtures/tiny_windows_subject1_2.npz",
            "sampling_rate_hz": 2000,
            "channels": 2,
            "subjects": [1, 2],
        },
        "preprocess": {
            "id": "fixture_overrides",
            "window_ms": 4,
            "hop_ms": 2,
            "window_samples": 8,
            "hop_samples": 4,
            "cache": False,
            "output": {
                "windows_path": "{artifacts_root}/data/{preprocess_id}/windows_s{subject_id}.npy",
                "meta_path": "{artifacts_root}/data/{preprocess_id}/meta.json",
            },
        },
        "manifest": {
            "id": "fixture_overrides",
            "output": {
                "manifest_csv": "{artifacts_root}/manifests/{manifest_id}/manifest.csv"
            },
        },
        "protocols": {
            "loso": {
                "type": "loso",
                "output_dir": "{artifacts_root}/splits/protocol=loso",
                "subjects": [1, 2],
                "reps": {"all": [1, 2, 3], "test": [2], "val": [3]},
            }
        },
        "models": {
            "tiny_cnn": {
                "architecture": "ST_CNN_GN",
                "params": {
                    "num_electrodes": 2,
                    "num_samples": 8,
                    "conv_channels": [4],
                    "kernel_size": [3, 3],
                    "pool_sizes": [[1, 1]],
                    "conv_dropout": 0.0,
                    "gn_groups": 1,
                    "head_hidden": [8, 4],
                    "head_dropout": 0.0,
                    "num_classes": 2,
                },
            }
        },
        "train": {
            "device": "cpu",
            "seeds": [0],
            "max_epochs": 1,
            "batch_size": 4,
            "num_workers": 0,
            "log_every": 1,
            "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0},
            "checkpoint": {"primary": "last", "save_best": False, "save_last": False},
            "early_stopping": {"enabled": False},
        },
        "eval": {"latency": {"enabled": False}},
        "plan": {"models": ["tiny_cnn"], "protocols": ["loso"], "max_jobs": 1},
    }
    config_path = tmp_path / "config.yml"
    write_yaml(config_path, cfg)

    cli.main(["prepare", "-c", str(config_path)])
    cli.main(["splits", "-c", str(config_path)])
    cli.main(
        [
            "size",
            "-c",
            str(config_path),
            "--warmup-steps",
            "1",
            "--bench-train-steps",
            "2",
            "--bench-val-steps",
            "1",
            "--device",
            "cpu",
            "--max-k",
            "1",
            "--max-gpus",
            "1",
            "--alpha",
            "1.0",
            "--seeds",
            "3",
            "--epochs",
            "5",
        ]
    )

    sizing_path = artifacts_root / "sizing" / "sizing.json"
    assert sizing_path.exists()
    sizing = json.loads(sizing_path.read_text(encoding="utf-8"))
    workload = sizing["workload"]

    assert workload["train"]["num_seeds"] == 3
    assert workload["train"]["max_epochs"] == 5
    assert workload["train"]["overrides"]["num_seeds"] is True
    assert workload["train"]["overrides"]["max_epochs"] is True

    split_paths = sorted((artifacts_root / "splits").rglob("split.json"))
    assert split_paths
    batch_size = 4
    epochs = 5
    requested_seeds = 3
    expected_train = 0
    expected_val = 0
    expected_test = 0
    for split_path in split_paths:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_steps_per_epoch = int(
            math.ceil(int(split.get("num_train_windows", 0)) / batch_size)
        )
        val_steps_per_epoch = int(
            math.ceil(int(split.get("num_val_windows", 0)) / batch_size)
        )
        test_steps = int(math.ceil(int(split.get("num_test_windows", 0)) / batch_size))
        effective_epochs = epochs if train_steps_per_epoch > 0 else 0
        expected_train += train_steps_per_epoch * effective_epochs
        expected_val += val_steps_per_epoch * effective_epochs
        expected_test += test_steps

    assert workload["steps"]["train"] == expected_train * requested_seeds
    assert workload["steps"]["val"] == expected_val * requested_seeds
    assert workload["steps"]["test"] == expected_test * requested_seeds
