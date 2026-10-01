# Historical reference inputs

Runtime inference does not read this directory. Reproducible fitting and
matched-cohort validation require the files declared in
`config/reference_inputs.json`:

- `events_gold_1_bucket_000_20260331.parquet`
- `events_gold_2_bucket_000_20260430.parquet`
- `campaign_decision_opportunities_20260301_20260430.parquet`
- `reference_manifest.json`

The two Gold files contain immutable March-April telemetry. The decision log
must contain every evaluated opportunity, including NO_ACTION, ineligible,
scheduled, realized, terminal-suppressed, and delivery-suppressed records. One
row is one decision request and `decision_id` must be unique. All timestamps
are timezone-naive in the timezone named by the manifest. `application_id`
must be the lifecycle owner at the opportunity timestamp; ambiguous ownership
must be marked ineligible, never reassigned by `context_id` alone.

Decision-log values are constrained as follows:

- `eligibility_status`: `ELIGIBLE` or `INELIGIBLE`;
- `action`: `SEND` or `NO_ACTION` for eligible rows, null for ineligible rows;
- `outcome_status`: `NO_ACTION`, `SCHEDULED`, `REALIZED`,
  `SUPPRESSED_TERMINAL`, `SUPPRESSED_DELIVERY`, or `CANCELED`;
- `scheduled_send_datetime` is required for scheduled/realized/suppressed
  sends and cannot precede `opportunity_datetime`;
- `realized_send_datetime` is required only for `REALIZED` and cannot precede
  its scheduled timestamp;
- `suppression_reason` is required for suppressed/canceled rows;
- application creation, expiry, and terminal time are joined from the immutable
  Gold lifecycle using `application_id`, never transferred across applications.

The denominator for opportunity intensity is all logged decision requests in
the eligible application risk set. Timing is fitted from opportunity timestamps,
not from realized delivery timestamps. Send realization and suppression are
separate conditional models. Competing sends censor response follow-up at the
next realized exposure.

`reference_manifest.json` must declare `schema_version`, `created_at`,
`timezone`, `source_period`, `extraction_query_version`, and a `files` array.
Each file entry contains `filename`, `sha256`, `row_count`, source system, and
an immutable extraction identifier. Run:

```powershell
python -m environment.cli validate-reference-inputs
```

The raw Gold files currently exist in the adjacent research checkout but are
not copied into this standalone repository. The campaign decision log is not
present there or in this bundle. Realized `campaign_sent` rows are not an
acceptable substitute because they omit NO_ACTION and suppressed decisions.
