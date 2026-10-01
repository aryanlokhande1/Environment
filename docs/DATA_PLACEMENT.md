# Data Placement

- Historical input: one combined Parquet or multiple March-April Parquets under `data/input/gold_history/`
- Frozen models: `artifacts/gold_events_v1/`
- Run-isolated outputs and checkpoints: `data/output/runs/<run_id>/`
- Immutable fitting/validation references: `data/reference/` (not read by normal runtime)

Local sources may be passed repeatedly with `--historical`, supplied as a
directory, or configured in `config/environment.yaml`. S3 source URIs use the
standard AWS credential chain and are cached inside the run directory. Large
historical files remain outside Git and are never modified.

The mandatory combined export is
`data/output/runs/<run_id>/combined_gold_events.parquet`. Daily May partitions
retain simulation row keys for restart/deduplication; the combined export has
exactly the canonical Gold columns. Fitting/validation input filenames and
schemas are separately declared in `config/reference_inputs.json`.
