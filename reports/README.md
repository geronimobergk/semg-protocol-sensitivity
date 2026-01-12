# Reports

`reports/` contains aggregated tables/CSVs derived from `runs/` (not raw checkpoints).
Reports are deterministic given a fixed config, seeds, and split definition.

By default, each experiment writes into `experiment.reports_root` (e.g.
`reports/protocol_sensitivity_semg_cnn/`).

## Generate

After runs exist:

```bash
uv run tinyml-semg report -c configs/experiments/protocol_sensitivity_semg_cnn.yml
```

Or run the full pipeline:

```bash
uv run tinyml-semg run -c configs/experiments/protocol_sensitivity_semg_cnn.yml
```

## Outputs (per experiment)

- `compare.csv`: aggregated metrics across instances (e.g., LOSO subjects) per seed
- `protocol_tables.md`: markdown tables (performance, ranking stability, protocol sensitivity, efficiency)

If you redirect `runs_root` via `--overrides`, redirect `reports_root` as well so
`report` reads the correct run tree and writes alongside it.
