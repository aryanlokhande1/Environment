# Data Placement

- Starting cumulative history: `data/input/gold_history/gold_events_start.parquet`
- Frozen models: `artifacts/gold_events_v1/`
- Local outputs and checkpoints: `data/output/`
- Optional non-model runtime references: `data/reference/`

For S3, set `ENVIRONMENT_INPUT_URI`, `ENVIRONMENT_OUTPUT_URI`, and `ENVIRONMENT_ARTIFACT_URI`. Credentials and bucket names are never committed. Normal inference needs starting cumulative history plus the frozen bundle; it does not need raw March-April training data.
