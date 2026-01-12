from __future__ import annotations

from .base import (
    ProtocolBase,
    Split,
    filter_indices,
    register_protocol,
    resolve_reps_cfg,
    split_hash,
)


@register_protocol("pooled_repdisjoint")
class PooledRepDisjoint(ProtocolBase):
    name = "pooled_repdisjoint"

    def enumerate_instances(self, cfg: dict, manifest_rows: list[dict]) -> list[dict]:
        return [{}]

    def make_split(
        self,
        manifest_rows: list[dict],
        cfg: dict,
        instance: dict,
        manifest_hash: str,
    ) -> Split:
        reps_cfg = resolve_reps_cfg(cfg)
        train = filter_indices(manifest_rows, rep_ids=set(reps_cfg["train"]))
        val = filter_indices(manifest_rows, rep_ids=set(reps_cfg["val"]))
        test = filter_indices(manifest_rows, rep_ids=set(reps_cfg["test"]))
        split_id = split_hash(manifest_hash, self.name, cfg, instance)
        return Split(
            protocol=self.name,
            instance=instance,
            hash=split_id,
            train_idx=train,
            val_idx=val,
            test_idx=test,
            notes={
                "reps_train": reps_cfg["train"],
                "reps_val": reps_cfg["val"],
                "reps_test": reps_cfg["test"],
            },
        )
