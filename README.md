# Evaluation Protocols Matter — A Controlled Study on Protocol Sensitivity in sEMG CNNs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/geronimobergk/semg-protocol-sensitivity/blob/main/notebooks/00_colab_quickstart.ipynb)

Config-driven PyTorch experiments to quantify how **evaluation protocol choice** changes
absolute performance and **model ranking** for CNN-based sEMG classifiers on
NinaPro DB2 (Exercise B).

- Dataset: NinaPro DB2, sEMG-only, Exercise B (default: no rest class)
- Models: `ST_CNN_GN` vs `ST_Attn_CNN_GN` (same backbone; attention toggle)
- Protocols: `pooled_repdisjoint`, `loso` (LOSO cross-subject), `single_subject_repdisjoint`
- Outputs: split files, per-run metrics/predictions, runtime/VRAM estimates, paper-ready tables

## Setup

This repo uses `uv`:

```bash
uv sync
```

## Quick sanity run (no raw data)

Runs the full pipeline on a tiny fixture dataset (CPU) to validate the install:

```bash
uv run tinyml-semg run -c configs/experiments/protocol_sensitivity_semg_cnn.yml --profile smoke
```

Profiles other than `full` are guarded and **not for reporting results**.

## Run the experiment (NinaPro DB2)

1) Provide raw data (see `data/README.md`) or use the downloader:

```bash
uv run tinyml-semg download -c configs/experiments/protocol_sensitivity_semg_cnn.yml
```

2) One-shot pipeline:

```bash
uv run tinyml-semg run -c configs/experiments/protocol_sensitivity_semg_cnn.yml
```

Or stage-by-stage:

```bash
uv run tinyml-semg prepare  -c configs/experiments/protocol_sensitivity_semg_cnn.yml
uv run tinyml-semg splits   -c configs/experiments/protocol_sensitivity_semg_cnn.yml
uv run tinyml-semg traineval -c configs/experiments/protocol_sensitivity_semg_cnn.yml
uv run tinyml-semg report   -c configs/experiments/protocol_sensitivity_semg_cnn.yml
```

## Configs, profiles, overrides

- Base experiment: `configs/experiments/protocol_sensitivity_semg_cnn.yml`
- Profiles (budget/guardrails): `configs/profiles/*.yaml` (`smoke`, `dev_mini`, `dry_run`, `bench`)
- Overrides: `--overrides path/to/override.yaml` (repeatable YAML deep-merge; applied after profile)

Typical override (redirect outputs, restrict protocols/models):

```yaml
experiment:
  artifacts_root: /abs/path/artifacts
  runs_root: /abs/path/runs
  reports_root: /abs/path/reports
plan:
  protocols: [loso]
  models: [st_cnn_gn]
train:
  seeds: [1, 3, 7]
```

See `configs/README.md` for the config schema and layering rules.

## Outputs

- `artifacts/` (gitignored): window cache, manifests, protocol splits, sizing (`sizing.json`)
- `runs/` (gitignored): per-run `resolved_config.yaml`, `split.json`, `metrics.json`, `preds.npz`, checkpoints, env metadata
- `reports/`: aggregated `compare.csv` + `protocol_tables.md` (used by `REPORT.md`)

## Sizing and scheduling

```bash
uv run tinyml-semg size -c configs/experiments/protocol_sensitivity_semg_cnn.yml --profile bench --max-k 4 --max-gpus 8
```

## Protocol correctness and reproducibility

- Splits are validated for leakage (sample ID overlap; repetition overlap; LOSO subject isolation).
- Split hashes include the manifest file hash; each run stores the resolved config + environment snapshot.
- Repro checklist: `docs/REPRODUCIBILITY.md`

## Repository layout

- `configs/`: experiment config + run profiles
- `data/`: raw data expectations (not distributed)
- `docs/`: reproducibility notes
- `reports/`: generated tables/CSVs (paper-facing)
- `runs/`: run outputs (gitignored)
- `tests/`: smoke tests and invariants (splits/leakage)
- `tinyml_semg_classifier/`: datasets, splits, models, training, eval, reporting
