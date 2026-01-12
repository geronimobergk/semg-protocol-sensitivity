import numpy as np

from tinyml_semg_classifier.datasets.ninapro_db2.windowing import iter_windows


def test_iter_windows_emits_expected_segments():
    emg = np.arange(12, dtype=np.float32).reshape(6, 2)
    restimulus = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    rerepetition = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)

    windows = list(iter_windows(emg, restimulus, rerepetition, 3, 2))
    assert len(windows) == 2

    window, label, rep, start, end = windows[0]
    np.testing.assert_array_equal(window, emg[0:3])
    assert label == restimulus[end]
    assert rep == rerepetition[end]
    assert (start, end) == (0, 2)
