# NinaPro DB2 raw data (not distributed)

This repo does not ship NinaPro data. Obtain DB2 from the official source and
place the subject folders under `data/raw/` (or set `dataset.raw_dir` in your
experiment config).

- Reference: `https://ninapro.hevs.ch/instructions/DB2.html`

## Expected layout

```text
data/raw/
  DB2_s1/
    S1_E1_A1.mat
    S1_E2_A1.mat
    S1_E3_A1.mat
  DB2_s2/
    S2_E1_A1.mat
    S2_E2_A1.mat
    S2_E3_A1.mat
  ...
  DB2_s40/
    S40_E1_A1.mat
    S40_E2_A1.mat
    S40_E3_A1.mat
```

Notes:

- Subject folder names may be `DB2_s<N>`, `db2_s<N>`, or `s<N>` (case-insensitive).
- Files are discovered recursively; exercises are selected by `dataset.exercises`
  (this repo maps `B→E1`, `C→E2`, `D→E3`).
- The pipeline reads sEMG only; other modalities in the `.mat` are ignored.

## Optional download helper

`tinyml-semg download` downloads the Ninapro-hosted `DB2_Preproc/DB2_s<N>.zip`
archives for the subjects specified in your config and extracts them into
`dataset.raw_dir`:

```bash
uv run tinyml-semg download -c configs/experiments/protocol_sensitivity_semg_cnn.yml
```

## Loader expectations

Each `.mat` must contain:

- `emg` (samples×channels or channels×samples)
- `restimulus`
- `rerepetition` (or `repetition` as fallback)
