import torch

from tinyml_semg_classifier.training.metrics import ClassificationStats


def test_classification_stats_metrics():
    stats = ClassificationStats.create(2)
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [3.0, -1.0]])
    targets = torch.tensor([0, 1, 0])
    stats.update(logits, targets)

    metrics = stats.summarize(include_confusion=True, include_per_class=True)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["confusion_matrix"]["matrix"] == [[2, 0], [0, 1]]
