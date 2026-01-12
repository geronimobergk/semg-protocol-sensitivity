# Artifacts (generated, gitignored)

`artifacts/` stores derived, reproducible intermediate outputs (window caches, manifests,
protocol splits, sizing JSON). It is treated as a build cache:

- Safe to delete and regenerate.
- Do not edit artifacts by hand.
- Not tracked by git (see `.gitignore`).

By default, each experiment writes into `experiment.artifacts_root` (e.g.
`artifacts/protocol_sensitivity_semg_cnn/`).

## Typical layout (per experiment)

- `data/<preprocess_id>/`
  - `windows_s<subject_id>.npy`: windowed sEMG (shape matches model input assumptions)
  - `meta.json`: preprocessing metadata (IDs, hashes, etc.)
- `manifests/<manifest_id>/manifest.csv`: sample index for training/splitting
- `splits/protocol=<protocol_name>/instance=<tag>/split.json`: train/val/test indices + split hash
- `sizing/`
  - `bench.json`: micro-benchmark timings + VRAM peak
  - `estimate.json`: full cartesian-product runtime/resource estimate
  - `pod.json`: recommended GPU/concurrency plan derived from `estimate.json`
