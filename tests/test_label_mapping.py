from tinyml_semg_classifier.datasets.datamodule import build_label_mapping


def test_build_label_mapping_sorted():
    rows = [
        {"gesture_id": 5},
        {"gesture_id": 2},
        {"gesture_id": 5},
        {"gesture_id": 3},
    ]
    mapping = build_label_mapping(rows)
    assert mapping.labels == [2, 3, 5]
    assert mapping.mapping == {2: 0, 3: 1, 5: 2}
