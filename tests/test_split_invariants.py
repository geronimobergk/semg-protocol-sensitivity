import pytest

from tinyml_semg_classifier.datasets.manifest import compute_sample_id
from tinyml_semg_classifier.datasets.splits.base import assert_split_invariants


def test_split_invariants_accepts_disjoint_split():
    sample_id_0 = compute_sample_id(
        {
            "subject_id": 1,
            "exercise_id": 1,
            "rep_id": 1,
            "gesture_id": 1,
            "sample_start": 0,
            "sample_end": 1,
        }
    )
    sample_id_1 = compute_sample_id(
        {
            "subject_id": 1,
            "exercise_id": 1,
            "rep_id": 2,
            "gesture_id": 2,
            "sample_start": 2,
            "sample_end": 3,
        }
    )
    sample_id_2 = compute_sample_id(
        {
            "subject_id": 2,
            "exercise_id": 1,
            "rep_id": 3,
            "gesture_id": 1,
            "sample_start": 4,
            "sample_end": 5,
        }
    )
    manifest_rows = [
        {
            "sample_id": sample_id_0,
            "subject_id": 1,
            "exercise_id": 1,
            "rep_id": 1,
            "gesture_id": 1,
            "sample_start": 0,
            "sample_end": 1,
            "path": "fixture.npy",
            "local_idx": 0,
        },
        {
            "sample_id": sample_id_1,
            "subject_id": 1,
            "exercise_id": 1,
            "rep_id": 2,
            "gesture_id": 2,
            "sample_start": 2,
            "sample_end": 3,
            "path": "fixture.npy",
            "local_idx": 1,
        },
        {
            "sample_id": sample_id_2,
            "subject_id": 2,
            "exercise_id": 1,
            "rep_id": 3,
            "gesture_id": 1,
            "sample_start": 4,
            "sample_end": 5,
            "path": "fixture.npy",
            "local_idx": 2,
        },
    ]
    split = {
        "train_idx": [sample_id_0],
        "val_idx": [sample_id_2],
        "test_idx": [sample_id_1],
    }
    assert_split_invariants(
        split,
        manifest_rows,
        "pooled_repdisjoint",
        {"reps": {"all": [1, 2, 3], "test": [2], "val": [3]}},
        allow_missing_classes=True,
    )


def test_split_invariants_detect_overlap():
    sample_id_0 = compute_sample_id(
        {
            "subject_id": 1,
            "exercise_id": 1,
            "rep_id": 1,
            "gesture_id": 1,
            "sample_start": 0,
            "sample_end": 1,
        }
    )
    sample_id_1 = compute_sample_id(
        {
            "subject_id": 2,
            "exercise_id": 1,
            "rep_id": 2,
            "gesture_id": 2,
            "sample_start": 2,
            "sample_end": 3,
        }
    )
    manifest_rows = [
        {
            "sample_id": sample_id_0,
            "subject_id": 1,
            "exercise_id": 1,
            "rep_id": 1,
            "gesture_id": 1,
            "sample_start": 0,
            "sample_end": 1,
            "path": "fixture.npy",
            "local_idx": 0,
        },
        {
            "sample_id": sample_id_1,
            "subject_id": 2,
            "exercise_id": 1,
            "rep_id": 2,
            "gesture_id": 2,
            "sample_start": 2,
            "sample_end": 3,
            "path": "fixture.npy",
            "local_idx": 1,
        },
    ]
    split = {"train_idx": [sample_id_0], "val_idx": [], "test_idx": [sample_id_0]}
    with pytest.raises(ValueError, match="overlap"):
        assert_split_invariants(
            split,
            manifest_rows,
            "pooled_repdisjoint",
            {"reps": {"all": [1, 2], "test": [2], "val": [1]}},
            allow_missing_classes=True,
        )


def test_split_invariants_detects_loso_leakage():
    sample_id_0 = compute_sample_id(
        {
            "subject_id": 1,
            "exercise_id": 1,
            "rep_id": 1,
            "gesture_id": 1,
            "sample_start": 0,
            "sample_end": 1,
        }
    )
    sample_id_1 = compute_sample_id(
        {
            "subject_id": 1,
            "exercise_id": 1,
            "rep_id": 2,
            "gesture_id": 2,
            "sample_start": 2,
            "sample_end": 3,
        }
    )
    manifest_rows = [
        {
            "sample_id": sample_id_0,
            "subject_id": 1,
            "exercise_id": 1,
            "rep_id": 1,
            "gesture_id": 1,
            "sample_start": 0,
            "sample_end": 1,
            "path": "fixture.npy",
            "local_idx": 0,
        },
        {
            "sample_id": sample_id_1,
            "subject_id": 1,
            "exercise_id": 1,
            "rep_id": 2,
            "gesture_id": 2,
            "sample_start": 2,
            "sample_end": 3,
            "path": "fixture.npy",
            "local_idx": 1,
        },
    ]
    split = {"train_idx": [sample_id_0], "val_idx": [], "test_idx": [sample_id_1]}
    with pytest.raises(ValueError, match="LOSO split mixes"):
        assert_split_invariants(
            split,
            manifest_rows,
            "loso",
            {"type": "loso"},
            allow_missing_classes=True,
        )
