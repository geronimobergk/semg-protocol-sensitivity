# Developer guide (contributor notes)

This repo is a CLI-driven, config-first experiment harness for evaluating **protocol sensitivity** in compact sEMG CNNs.

## Setup

```bash
uv sync
```

## Quality gates (run before pushing)

```bash
uv run ruff format .
uv run ruff check .
uv run pytest -q
```

Pytest markers:

- `-m smoke`: fast end-to-end (fixture dataset)
- `-m devmini`: longer fixture run (opt-in)

## How to run experiments

- Main experiment config: `configs/experiments/protocol_sensitivity_semg_cnn.yml`
- Prefer variants via `--overrides <yaml>` rather than editing the base config.
- Use profiles for guardrails:
  - `full` for reportable runs
  - `smoke`/`dev_mini` for fixture validation
  - `dry_run` for a small real-data end-to-end check

Example (fixture smoke):

```bash
uv run tinyml-semg run -c configs/experiments/protocol_sensitivity_semg_cnn.yml --profile smoke
```

## Adding a new protocol

1) Implement a `ProtocolBase` in `tinyml_semg_classifier/datasets/splits/` and register it via `@register_protocol`.
2) Add a `protocols.<name>` entry in the experiment YAML (define `type`, `reps`, `subjects` if needed, and `output_dir`).
3) Add/extend tests to enforce leakage invariants (see `tests/test_split_invariants.py`).

## Adding a new model

1) Add the module under `tinyml_semg_classifier/models/`.
2) Register the architecture string in `tinyml_semg_classifier/models/registry.py`.
3) Add a `models.<id>` entry in the experiment YAML (ensure `num_electrodes`, `num_samples`, `num_classes` match the manifest).
4) Validate with `uv run pytest -q` (and optionally `tinyml-semg bench/estimate` if you changed efficiency-critical code).

## Reporting changes

`tinyml-semg report` aggregates `runs/**/metrics.json` into `reports/<experiment_id>/`.
If you change metric keys or model labels, update `tinyml_semg_classifier/reporting/compare.py` accordingly.
