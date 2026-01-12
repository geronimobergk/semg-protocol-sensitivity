import numpy as np
import torch

from tinyml_semg_classifier.datasets.datamodule import DataModule
from tinyml_semg_classifier.datasets.manifest import compute_sample_id


def _build_manifest_rows(path: str) -> list[dict]:
    rows = []
    for idx in range(4):
        sample_start = idx * 3
        sample_end = sample_start + 2
        sample_id = compute_sample_id(
            {
                "subject_id": 1,
                "exercise_id": 1,
                "rep_id": 1,
                "gesture_id": 1,
                "sample_start": sample_start,
                "sample_end": sample_end,
            }
        )
        rows.append(
            {
                "sample_id": sample_id,
                "subject_id": 1,
                "exercise_id": 1,
                "rep_id": 1,
                "gesture_id": 1,
                "sample_start": sample_start,
                "sample_end": sample_end,
                "path": path,
                "local_idx": idx,
            }
        )
    return rows


def test_zscore_normalization_uses_train_split(tmp_path):
    windows = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]],
            [[11.0, 11.0, 11.0], [21.0, 21.0, 21.0]],
        ],
        dtype=np.float32,
    )
    data_path = tmp_path / "windows.npy"
    np.save(data_path, windows)

    manifest_rows = _build_manifest_rows(str(data_path))
    split = {
        "train_idx": [manifest_rows[0]["sample_id"], manifest_rows[1]["sample_id"]],
        "val_idx": [manifest_rows[2]["sample_id"]],
        "test_idx": [manifest_rows[3]["sample_id"]],
    }

    datamodule = DataModule(
        manifest_rows=manifest_rows,
        split=split,
        num_electrodes=2,
        num_samples=3,
        batch_size=1,
        num_workers=0,
        seed=0,
        manifest_dir=tmp_path,
        normalization={"type": "zscore", "eps": 0.0},
    )
    datamodule.setup()

    assert datamodule.normalizer is not None
    np.testing.assert_allclose(datamodule.normalizer.mean, [1.0, 2.0])
    np.testing.assert_allclose(datamodule.normalizer.std, [1.0, 1.0])

    val_loader = datamodule.loaders["val"]
    assert val_loader is not None
    xb, _, _ = next(iter(val_loader))
    expected = torch.tensor([[[9.0, 9.0, 9.0], [18.0, 18.0, 18.0]]], dtype=xb.dtype)
    assert torch.allclose(xb, expected)
