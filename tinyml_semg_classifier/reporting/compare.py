from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..utils.io import read_json
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

META_KEYS = {"model", "protocol", "instance", "seed"}

EN_DASH = "\u2013"
PLUS_MINUS = "\u00b1"

MODEL_LABELS = {
    "st_cnn_gn": "ST-CNN",
    "st_cnn": "ST-CNN",
    "tiny_cnn": "ST-CNN",
    "st_attn_cnn_gn": "ST-Attn-CNN",
    "st_attn_cnn": "ST-Attn-CNN",
    "tiny_attn_cnn": "ST-Attn-CNN",
}
MODEL_ORDER = ["ST-CNN", "ST-Attn-CNN"]

PROTOCOL_ORDER = [
    "single_subject_repdisjoint",
    "pooled_repdisjoint",
    "loso",
]
PROTOCOL_LABELS_TABLE1 = {
    "single_subject_repdisjoint": "Single-subject, rep-disjoint",
    "pooled_repdisjoint": "Pooled, rep-disjoint",
    "loso": "Cross-subject (LOSO)",
}
PROTOCOL_LABELS_TABLE2 = {
    "single_subject_repdisjoint": "Single-subject, repetition-disjoint",
    "pooled_repdisjoint": "Pooled, repetition-disjoint",
    "loso": "Cross-subject (LOSO)",
}
PROTOCOL_GENERALIZATION_AXIS = {
    "single_subject_repdisjoint": "Repetitions (within-user)",
    "pooled_repdisjoint": "Repetitions (seen users)",
    "loso": "Subjects (unseen users)",
}


def _parse_run_meta(metrics_path: Path, runs_root: Path) -> dict | None:
    rel = metrics_path.relative_to(runs_root)
    if len(rel.parts) < 5:
        return None
    model_id, protocol, instance, seed_part = rel.parts[:4]
    seed = seed_part
    if seed_part.startswith("seed"):
        seed = seed_part.replace("seed", "", 1)
        try:
            seed = int(seed)
        except ValueError:
            seed = seed_part.replace("seed", "", 1)
    return {
        "model": model_id,
        "protocol": protocol,
        "instance": instance,
        "seed": seed,
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _flatten_metrics(metrics: dict) -> dict:
    flat: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_metrics = metrics.get(split)
        if isinstance(split_metrics, dict):
            for key, value in split_metrics.items():
                if _is_number(value):
                    flat[f"{split}_{key}"] = value
    training = metrics.get("training")
    if isinstance(training, dict):
        for key, value in training.items():
            if _is_number(value):
                flat[f"training_{key}"] = value
    model_stats = metrics.get("model_stats")
    if isinstance(model_stats, dict):
        for key, value in model_stats.items():
            if _is_number(value):
                flat[f"model_stats_{key}"] = value
    for key, value in metrics.items():
        if key in ("train", "val", "test", "training", "model_stats"):
            continue
        if _is_number(value):
            flat[key] = value
    return flat


def _aggregate_group(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot aggregate empty rows.")
    first = rows[0]
    instances = sorted({row["instance"] for row in rows})
    out: dict[str, Any] = {
        "model": first["model"],
        "protocol": first["protocol"],
        "seed": first["seed"],
        "instance": instances[0] if len(instances) == 1 else "aggregate",
        "instance_count": len(instances),
        "instances": ";".join(instances),
    }

    metric_keys = set()
    for row in rows:
        metric_keys.update({key for key in row.keys() if key not in META_KEYS})

    for key in sorted(metric_keys):
        values = [row.get(key) for row in rows if _is_number(row.get(key))]
        if not values:
            continue
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        std_val = statistics.pstdev(values) if len(values) > 1 else 0.0
        out[key] = mean_val
        out[f"{key}_median"] = median_val
        out[f"{key}_std"] = std_val
    return out


def _collect_metric_rows(runs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(runs_root.rglob("metrics.json")):
        meta = _parse_run_meta(metrics_path, runs_root) or {}
        metrics = read_json(metrics_path)
        for key in ("model", "protocol", "seed"):
            if key in metrics:
                meta[key] = metrics[key]
        row = {**meta, **_flatten_metrics(metrics)}
        rows.append(row)
    return rows


def _write_compare_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        fieldnames = ["model", "protocol", "instance", "seed", "instance_count"]
    else:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _ordered_model_labels(rows: list[dict]) -> list[str]:
    labels = {
        _model_label(str(row.get("model", "")))
        for row in rows
        if row.get("model") is not None
    }
    ordered = [label for label in MODEL_ORDER if label in labels]
    for label in sorted(labels):
        if label not in ordered:
            ordered.append(label)
    return ordered or MODEL_ORDER


def _collect_values_by_instance(
    rows: list[dict], protocol: str, model_label: str, metric: str
) -> dict[str, list[float]]:
    values_by_instance: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("protocol") != protocol:
            continue
        label = _model_label(str(row.get("model", "")))
        if label != model_label:
            continue
        value = row.get(metric)
        if _is_number(value):
            instance = str(row.get("instance", "unknown"))
            values_by_instance[instance].append(float(value))
    return values_by_instance


def _mean_std_from_instances(
    values_by_instance: dict[str, list[float]],
) -> tuple[float | None, float | None]:
    if not values_by_instance:
        return None, None
    if len(values_by_instance) == 1:
        values = next(iter(values_by_instance.values()))
        if not values:
            return None, None
        mean_val = statistics.mean(values)
        std_val = statistics.pstdev(values) if len(values) > 1 else 0.0
        return mean_val, std_val
    instance_means = [
        statistics.mean(values) for values in values_by_instance.values() if values
    ]
    if not instance_means:
        return None, None
    mean_val = statistics.mean(instance_means)
    std_val = statistics.pstdev(instance_means) if len(instance_means) > 1 else 0.0
    return mean_val, std_val


def _format_mean_std(mean_val: float | None, std_val: float | None) -> str:
    if mean_val is None or std_val is None:
        return "n/a"
    return f"{mean_val * 100:.1f} {PLUS_MINUS} {std_val * 100:.1f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}"


def _format_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1f}"


def _median_value(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def _format_count(value: float | None) -> str:
    if value is None:
        return "n/a"
    abs_value = abs(value)
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs_value >= threshold:
            scaled = value / threshold
            precision = 2 if abs(scaled) < 10 else 1
            text = f"{scaled:.{precision}f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return f"{value:.0f}"


def _format_latency(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _collect_model_stat_values(
    rows: list[dict], model_label: str, key: str
) -> list[float]:
    values: list[float] = []
    for row in rows:
        label = _model_label(str(row.get("model", "")))
        if label != model_label:
            continue
        value = row.get(key)
        if _is_number(value):
            values.append(float(value))
    return values


def _metric_stats(
    rows: list[dict], protocols: list[str], model_labels: list[str], metric: str
) -> dict[tuple[str, str], tuple[float | None, float | None]]:
    stats: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    for protocol in protocols:
        for model_label in model_labels:
            values_by_instance = _collect_values_by_instance(
                rows, protocol, model_label, metric
            )
            stats[(protocol, model_label)] = _mean_std_from_instances(
                values_by_instance
            )
    return stats


def _rank_models(
    stats: dict[tuple[str, str], tuple[float | None, float | None]],
    protocol: str,
    model_labels: list[str],
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for label in model_labels:
        mean_val, _ = stats.get((protocol, label), (None, None))
        if mean_val is None:
            continue
        ranked.append((label, mean_val))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _append_efficiency_table(lines: list[str], rows: list[dict]) -> None:
    lines.extend(
        [
            "",
            f"#### Table 4 {EN_DASH} Model efficiency summary",
            "",
            "| Model       | Parameters | MACs / Inference Window | FLOPs / Inference Window | Inference Latency [ms] |",
            "| ----------- | ---------- | ----------------------- | ------------------------ | ---------------------- |",
        ]
    )
    for model_label in MODEL_ORDER:
        params = _median_value(
            _collect_model_stat_values(rows, model_label, "model_stats_num_parameters")
        )
        macs = _median_value(
            _collect_model_stat_values(rows, model_label, "model_stats_macs_per_window")
        )
        flops = _median_value(
            _collect_model_stat_values(
                rows, model_label, "model_stats_flops_per_window"
            )
        )
        if flops is None and macs is not None:
            flops = macs * 2
        latency = _median_value(
            _collect_model_stat_values(rows, model_label, "model_stats_latency")
        )
        lines.append(
            "| "
            f"{model_label} | "
            f"{_format_count(params)} | "
            f"{_format_count(macs)} | "
            f"{_format_count(flops)} | "
            f"{_format_latency(latency)} |"
        )


def _write_protocol_tables(
    path: Path, rows: list[dict], efficiency_rows: list[dict]
) -> None:
    protocols = PROTOCOL_ORDER
    model_labels = _ordered_model_labels(rows)
    ba_stats = _metric_stats(rows, protocols, model_labels, "test_balanced_accuracy")
    f1_stats = _metric_stats(rows, protocols, model_labels, "test_macro_f1")

    lines = [
        f"#### Table 1 {EN_DASH} Performance across evaluation protocols (mean {PLUS_MINUS} std)",
        "",
        "| Protocol                     | Model       | Balanced Acc. [%] | Macro-F1 [%] |",
        "| ---------------------------- | ----------- | ----------------- | ------------ |",
    ]
    for protocol in protocols:
        protocol_label = PROTOCOL_LABELS_TABLE1.get(protocol, protocol)
        first = True
        for model_label in model_labels:
            ba_mean, ba_std = ba_stats.get((protocol, model_label), (None, None))
            f1_mean, f1_std = f1_stats.get((protocol, model_label), (None, None))
            ba_text = _format_mean_std(ba_mean, ba_std)
            if ba_text != "n/a":
                ba_text = f"**{ba_text}**"
            f1_text = _format_mean_std(f1_mean, f1_std)
            lines.append(
                f"| {protocol_label if first else ''} | {model_label} | {ba_text} | {f1_text} |"
            )
            first = False

    lines.extend(
        [
            "",
            f"### Table 2 {EN_DASH} Protocol-Dependent Model Ranking",
            "",
            "| Evaluation Protocol                 | Primary Generalization Axis | Better Model | Δ Balanced Accuracy (pp) | Ranking Stability |",
            "| ----------------------------------- | --------------------------- | ------------ | ------------------------ | ----------------- |",
        ]
    )

    baseline_protocol = (
        "pooled_repdisjoint" if "pooled_repdisjoint" in protocols else protocols[0]
    )
    baseline_ranked = _rank_models(ba_stats, baseline_protocol, model_labels)
    baseline_top = baseline_ranked[0][0] if baseline_ranked else None

    for protocol in protocols:
        ranked = _rank_models(ba_stats, protocol, model_labels)
        better_model = ranked[0][0] if ranked else "n/a"
        delta = None
        if len(ranked) > 1:
            delta = (ranked[0][1] - ranked[1][1]) * 100
        stability = "n/a"
        if baseline_top is not None and ranked:
            stability = "Stable" if ranked[0][0] == baseline_top else "**Inverted**"
        lines.append(
            "| "
            f"{PROTOCOL_LABELS_TABLE2.get(protocol, protocol)} | "
            f"{PROTOCOL_GENERALIZATION_AXIS.get(protocol, 'n/a')} | "
            f"{better_model} | "
            f"{_format_delta(delta)} | "
            f"{stability} |"
        )

    lines.extend(
        [
            "",
            f"#### Table 3 {EN_DASH} Protocol sensitivity (Balanced Accuracy)",
            "",
            "| Model       | Pooled BA [%] | LOSO BA [%] | Relative Drop [%] |",
            "| ----------- | ------------- | ----------- | ----------------- |",
        ]
    )
    for model_label in model_labels:
        pooled_mean, _ = ba_stats.get(("pooled_repdisjoint", model_label), (None, None))
        loso_mean, _ = ba_stats.get(("loso", model_label), (None, None))
        drop = None
        if pooled_mean is not None and pooled_mean > 0 and loso_mean is not None:
            drop = (pooled_mean - loso_mean) / pooled_mean * 100
        drop_text = _format_percent(drop)
        if drop_text != "n/a":
            drop_text = f"**{drop_text}**"
        lines.append(
            "| "
            f"{model_label} | "
            f"{_format_percent(pooled_mean)} | "
            f"{_format_percent(loso_mean)} | "
            f"{drop_text} |"
        )

    _append_efficiency_table(lines, efficiency_rows)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def compare(cfg: dict) -> Path:
    runs_root = Path(cfg["experiment"]["runs_root"])
    reports_root = Path(cfg["experiment"]["reports_root"])
    reports_root.mkdir(parents=True, exist_ok=True)

    run_rows = _collect_metric_rows(runs_root)
    grouped: dict[tuple[str, str, Any], list[dict]] = defaultdict(list)
    for row in run_rows:
        key = (row["model"], row["protocol"], row["seed"])
        grouped[key].append(row)

    summary_rows = [_aggregate_group(rows) for rows in grouped.values()]
    summary_rows.sort(key=lambda r: (r["protocol"], r["model"], str(r["seed"])))

    compare_path = reports_root / "compare.csv"
    _write_compare_csv(compare_path, summary_rows)

    tables_path = reports_root / "protocol_tables.md"
    _write_protocol_tables(tables_path, run_rows, summary_rows)

    LOGGER.info("Wrote compare outputs to %s", reports_root)
    return compare_path
