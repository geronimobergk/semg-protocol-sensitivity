# Configs

Entry point: `configs/experiments/protocol_sensitivity_semg_cnn.yml`.

## How configs are applied

The CLI loads and resolves configuration in this order:

1. Base experiment YAML (`-c/--config`)
2. Optional profile overlay (`--profile <name|path>`) from `configs/profiles/` (deep-merged)
3. Optional override YAML files (`--overrides path.yaml`, repeatable; deep-merged in order)
4. Template expansion for strings like `{artifacts_root}` / `{experiment_id}`

This is a lightweight, Hydra-like workflow (YAML + overlays), but it does not use Hydra.

## Profiles

Profiles exist to cap work and/or enforce guardrails:

- `full` (default): real dataset only; rejects capped budgets unless `profile_allow_caps: true`
- `full_part1`, `full_part2`: full-guardrail runs that pin the model/seed plan for split execution
- `smoke`, `dev_mini`: fixture dataset + strict caps (fast validation; not for reporting)
- `dry_run`: real data, small subject subset, capped budgets (end-to-end pipeline check)
- `bench`: minimal training for micro-benchmarking

## Common override patterns

Redirect outputs (recommended if you don’t want generated files under the repo):

```yaml
experiment:
  artifacts_root: /abs/path/artifacts
  runs_root: /abs/path/runs
  reports_root: /abs/path/reports
```

Restrict the cartesian product (protocols/models/seeds):

```yaml
plan:
  protocols: [loso]
  models: [st_cnn_gn]
  max_jobs: 2
train:
  seeds: [1, 3, 7]
```

Restrict subjects for a real-data dry run:

```yaml
dataset:
  subjects: "1-4"
protocols:
  loso:
    subjects: "1-4"
  single_subject_repdisjoint:
    subjects: "1-4"
```

Run with overrides:

```bash
uv run tinyml-semg run -c configs/experiments/protocol_sensitivity_semg_cnn.yml \
  --overrides path/to/overrides.yaml
```

## Template variables

These keys are available for string templates inside the resolved config:

- `{experiment_id}`, `{artifacts_root}`, `{runs_root}`, `{reports_root}`
- `{preprocess_id}`, `{manifest_id}`, `{profile}`
