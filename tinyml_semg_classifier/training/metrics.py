from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ClassificationStats:
    num_classes: int
    per_class_count: torch.Tensor
    per_class_pred: torch.Tensor
    per_class_correct: torch.Tensor
    confusion: torch.Tensor

    @classmethod
    def create(cls, num_classes: int) -> "ClassificationStats":
        return cls(
            num_classes=int(num_classes),
            per_class_count=torch.zeros(int(num_classes), dtype=torch.long),
            per_class_pred=torch.zeros(int(num_classes), dtype=torch.long),
            per_class_correct=torch.zeros(int(num_classes), dtype=torch.long),
            confusion=torch.zeros(
                (int(num_classes), int(num_classes)), dtype=torch.long
            ),
        )

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = logits.argmax(dim=1)
        targets = targets.detach().view(-1).to("cpu")
        preds = preds.detach().view(-1).to("cpu")
        self.per_class_count += torch.bincount(targets, minlength=self.num_classes)
        self.per_class_pred += torch.bincount(preds, minlength=self.num_classes)
        correct_mask = preds == targets
        if correct_mask.any():
            self.per_class_correct += torch.bincount(
                targets[correct_mask], minlength=self.num_classes
            )
        pair_index = targets * self.num_classes + preds
        self.confusion += torch.bincount(
            pair_index, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)

    def summarize(
        self,
        include_confusion: bool = False,
        include_per_class: bool = False,
        label_names: list[str] | None = None,
    ) -> dict:
        counts = self.per_class_count.to(torch.float32)
        preds = self.per_class_pred.to(torch.float32)
        correct = self.per_class_correct.to(torch.float32)
        total = counts.sum().item()
        accuracy = (correct.sum().item() / total) if total > 0 else 0.0

        per_class_acc = torch.where(
            counts > 0, correct / counts, torch.zeros_like(counts)
        )
        mask = counts > 0
        if mask.any():
            balanced_accuracy = per_class_acc[mask].mean().item()
        else:
            balanced_accuracy = 0.0

        precision = torch.where(preds > 0, correct / preds, torch.zeros_like(preds))
        recall = torch.where(counts > 0, correct / counts, torch.zeros_like(counts))
        denom = precision + recall
        per_class_f1 = torch.where(
            denom > 0, 2 * precision * recall / denom, torch.zeros_like(denom)
        )
        if mask.any():
            macro_f1 = per_class_f1[mask].mean().item()
        else:
            macro_f1 = 0.0

        if label_names is None:
            label_names = [str(idx) for idx in range(self.num_classes)]
        else:
            label_names = [str(name) for name in label_names]
            if len(label_names) != self.num_classes:
                label_names = [str(idx) for idx in range(self.num_classes)]

        metrics = {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "macro_f1": float(macro_f1),
        }
        if include_per_class:
            metrics["per_class"] = {
                "accuracy": {
                    label_names[idx]: float(per_class_acc[idx].item())
                    for idx in range(self.num_classes)
                },
                "f1": {
                    label_names[idx]: float(per_class_f1[idx].item())
                    for idx in range(self.num_classes)
                },
            }
        if include_confusion:
            metrics["confusion_matrix"] = {
                "labels": label_names,
                "matrix": self.confusion.tolist(),
            }
        return metrics
