import pytest

from tinyml_semg_classifier.datasets.manifest import compute_sample_id
from tinyml_semg_classifier.datasets.splits.base import list_subjects, resolve_reps_cfg
from tinyml_semg_classifier.datasets.splits.loso import LOSO
from tinyml_semg_classifier.datasets.splits.pooled_repdisjoint import PooledRepDisjoint
from tinyml_semg_classifier.datasets.splits.single_subject_repdisjoint import (
    SingleSubjectRepDisjoint,
)


def _build_manifest(subjects, reps, samples_per_rep=2):
    rows = []
    idx = 0
    for subject_id in subjects:
        for rep_id in reps:
            for _ in range(samples_per_rep):
                sample_start = idx * 2
                sample_end = sample_start + 1
                sample_id = compute_sample_id(
                    {
                        "subject_id": subject_id,
                        "exercise_id": 1,
                        "rep_id": rep_id,
                        "gesture_id": 1,
                        "sample_start": sample_start,
                        "sample_end": sample_end,
                    }
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "exercise_id": 1,
                        "rep_id": rep_id,
                        "gesture_id": 1,
                        "sample_start": sample_start,
                        "sample_end": sample_end,
                        "path": "fixture.npy",
                        "local_idx": idx,
                    }
                )
                idx += 1
    return rows


def _collect_indices(
    manifest_rows, subject_ids=None, rep_ids=None, exclude_subjects=None
):
    indices = set()
    for row in manifest_rows:
        subject_id = row["subject_id"]
        rep_id = row["rep_id"]
        if subject_ids is not None and subject_id not in subject_ids:
            continue
        if exclude_subjects is not None and subject_id in exclude_subjects:
            continue
        if rep_ids is not None and rep_id not in rep_ids:
            continue
        indices.add(row["sample_id"])
    return indices


def test_list_subjects_missing_raises():
    manifest = _build_manifest([1], [1])
    with pytest.raises(ValueError, match="Subjects missing"):
        list_subjects(manifest, "1-2")


def test_resolve_reps_cfg_overlap_raises():
    with pytest.raises(ValueError, match="disjoint"):
        resolve_reps_cfg({"reps": {"all": [1, 2], "test": [1], "val": [1]}})


def test_pooled_repdisjoint_split():
    manifest = _build_manifest([1, 2], [1, 2, 3])
    cfg = {"reps": {"all": [1, 2, 3], "test": [2], "val": [3]}}
    split = PooledRepDisjoint().make_split(manifest, cfg, {}, "hash")

    assert set(split.train_idx) == _collect_indices(manifest, rep_ids={1})
    assert set(split.val_idx) == _collect_indices(manifest, rep_ids={3})
    assert set(split.test_idx) == _collect_indices(manifest, rep_ids={2})


def test_loso_split_excludes_test_subject():
    manifest = _build_manifest([1, 2], [1, 2, 3])
    cfg = {"reps": {"all": [1, 2, 3], "test": [2], "val": [3]}}
    split = LOSO().make_split(manifest, cfg, {"test_subject": 2}, "hash")

    assert set(split.train_idx) == _collect_indices(
        manifest, rep_ids={1}, exclude_subjects={2}
    )
    assert set(split.val_idx) == _collect_indices(
        manifest, rep_ids={3}, exclude_subjects={2}
    )
    assert set(split.test_idx) == _collect_indices(
        manifest, rep_ids={2}, subject_ids={2}
    )


def test_single_subject_repdisjoint_split():
    manifest = _build_manifest([1, 2], [1, 2, 3])
    cfg = {"reps": {"all": [1, 2, 3], "test": [2], "val": [3]}}
    split = SingleSubjectRepDisjoint().make_split(manifest, cfg, {"subject": 1}, "hash")

    assert set(split.train_idx) == _collect_indices(
        manifest, subject_ids={1}, rep_ids={1}
    )
    assert set(split.val_idx) == _collect_indices(
        manifest, subject_ids={1}, rep_ids={3}
    )
    assert set(split.test_idx) == _collect_indices(
        manifest, subject_ids={1}, rep_ids={2}
    )
