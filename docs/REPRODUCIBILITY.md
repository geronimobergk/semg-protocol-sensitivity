# Reproducibility and protocol correctness

This repo is built around a single principle: **protocol choice is part of the model**.
Results are only meaningful if splits, model selection, and reporting are traceable and leak-free.

## What is recorded per run

Each run directory under `experiment.runs_root` contains (at minimum):

- `resolved_config.yaml`: full resolved config + `{model, protocol, instance, seed}`
- `split.json`: the exact split used for the run (copied from `artifacts/`)
- `run_env.json`: torch version, CUDA availability, device metadata, host info
- `normalization.json`: train-fit normalization statistics (when enabled)
- `checkpoints/{best,last}.pt`: checkpoint(s) used for evaluation
- `metrics_history.json`: per-epoch metrics
- `metrics.json`: final train/val/test metrics (+ efficiency stats when enabled)
- `confusion_matrix.json`
- `preds.npz`: test predictions (and optional probabilities)

## Split integrity guarantees

Splits are generated from the manifest and validated before training:

- **No overlap**: sample IDs are disjoint across train/val/test.
- **Repetition disjointness**: repetitions do not overlap between train/val/test when `protocol.reps` is defined.
- **LOSO isolation**: the held-out test subject never appears in train/val for `loso`.
- **Traceability**: `split.hash` includes the manifest file hash + protocol config + instance tag.

Validation hooks:

- `uv run pytest -q` (includes leakage/invariant tests in `tests/`)
- `uv run tinyml-semg splits -c <config>` regenerates `split.json` files under `experiment.artifacts_root`

## Reproducing report tables

1) Run the experiment (prefer `profile=full` for reportable results):

    ```bash
    uv run tinyml-semg run -c configs/experiments/protocol_sensitivity_semg_cnn.yml
    ```

2) Generate aggregated tables:

    ```bash
    uv run tinyml-semg report -c configs/experiments/protocol_sensitivity_semg_cnn.yml
    ```

    Outputs are written to `experiment.reports_root` (e.g. `reports/<experiment_id>/`).

## Making changes without breaking comparability

- Use a new `experiment.id` when you change the protocol set, model set, or reporting logic.
- Use a new `preprocess.id` / `manifest.id` when you change windowing or preprocessing.
- Apply all local path/protocol/model tweaks via `--overrides` YAML (and keep it with the results).
- Never tune on test subjects; LOSO validation must use training subjects only.
