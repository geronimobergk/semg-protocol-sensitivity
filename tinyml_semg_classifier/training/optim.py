from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SchedulerSpec:
    scheduler: object
    step_on_metric: bool


class WarmupCosineScheduler:
    def __init__(self, optimizer, total_epochs, warmup_epochs=0, min_lr=0.0):
        if total_epochs is None or int(total_epochs) < 1:
            raise ValueError("total_epochs must be a positive integer.")
        self.optimizer = optimizer
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.min_lr = float(min_lr)
        self.base_lrs = []
        for group in optimizer.param_groups:
            base_lr = float(group.get("lr", 0.0))
            group.setdefault("initial_lr", base_lr)
            self.base_lrs.append(base_lr)
        self.step_count = 0
        remaining = max(self.total_epochs - self.warmup_epochs, 1)
        if self.total_epochs > self.warmup_epochs:
            self._cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=remaining,
                eta_min=self.min_lr,
                last_epoch=0,
            )
        else:
            self._cosine = None

    def _apply_warmup(self, step: int) -> None:
        if self.warmup_epochs <= 0:
            return
        progress = min(step / max(1, self.warmup_epochs), 1.0)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * progress

    def set_initial_lr(self) -> None:
        if self.warmup_epochs <= 0:
            return
        self.step_count = 1
        self._apply_warmup(self.step_count)

    def step(self) -> None:
        self.step_count += 1
        if self.step_count <= self.warmup_epochs:
            self._apply_warmup(self.step_count)
            return
        if self._cosine is None:
            return
        if self.step_count == self.warmup_epochs + 1:
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
                group["lr"] = base_lr
        self._cosine.step()


def build_optimizer(params, cfg: dict) -> torch.optim.Optimizer:
    if not isinstance(cfg, dict):
        raise ValueError("optimizer config must be a mapping.")
    name = str(cfg.get("name", "adamw")).strip().lower()
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))

    if name in {"adamw", "adam_w"}:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(cfg.get("momentum", 0.0))
        return torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    raise ValueError(f"Unsupported optimizer: {cfg.get('name')}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict | None, total_epochs: int
) -> SchedulerSpec | None:
    if not cfg:
        return None
    if not isinstance(cfg, dict):
        raise ValueError("scheduler config must be a mapping.")
    name = str(cfg.get("name", "")).strip().lower()
    if name in {"cosine", "cosineannealing", "cosine_annealing"}:
        warmup_epochs = int(cfg.get("warmup_epochs", 0))
        min_lr = float(cfg.get("min_lr", 0.0))
        scheduler = WarmupCosineScheduler(
            optimizer,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr,
        )
        return SchedulerSpec(scheduler=scheduler, step_on_metric=False)
    if name in {"plateau", "reduceonplateau", "reduce_on_plateau"}:
        mode = cfg.get("mode", "min")
        patience = int(cfg.get("patience", 10))
        factor = float(cfg.get("factor", 0.1))
        threshold = float(cfg.get("threshold", 0.0))
        threshold_mode = cfg.get("threshold_mode", "rel")
        cooldown = int(cfg.get("cooldown", 0))
        eps = float(cfg.get("eps", 1e-8))
        min_lr = float(cfg.get("min_lr", 0.0))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            patience=patience,
            factor=factor,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            eps=eps,
            min_lr=min_lr,
        )
        return SchedulerSpec(scheduler=scheduler, step_on_metric=True)
    raise ValueError(f"Unsupported scheduler: {cfg.get('name')}")
