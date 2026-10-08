# Environment

## Purpose

Environment is the stochastic Personal Loan world model used after an external Transformer and RL policy choose an action. It consumes frozen empirical probabilities derived only from immutable March-April 2026 Gold Events. It does not train, refit, or learn from simulated data.

## Architecture

```text
S3 / starting Gold history
        ↓
Transformer → embedding → RL → Action
                              ↓
                         Environment
                              ↓
                    realized Gold events
                              ↓
                 updated cumulative history
                              ↓
                           repeat
```

The Transformer and RL policy are deliberately outside this repository.

## Environment behavior

The runtime reconstructs application-scoped state from visible cumulative history, applies business guards, and samples natural transitions, FIRST_PASS/REVISIT behavior, stage continuation or temporary inactivity, campaign response motifs, event timing, PTP hazard, and 30-day expiry. `NO_ACTION` still permits organic behavior. PTP is terminal for an application and is sampled only from the eligible post-AIP hazard. Pending events and deterministic counters are checkpointable. Competing sends censor unresolved response motifs, and sends pre-empted by PTP/expiry are recorded as suppressed rather than realized. Every decision and model draw produces separate audit metadata; audit rows never enter Gold history.

Application identity is enforced by `(context_id, simulated_time)`: a customer reuses the same `application_id` throughout its 30-day validity window. A modeled `Application Created` revisit during that window is retained as no-event probability mass and is never emitted or renormalized into another transition.

## Ground Truth

`artifacts/gold_events_v2/` is the current frozen probability bundle. It adds
the March-April-only weekday/residual arrival bootstrap and conditional Gold
`event_name` model. `gold_events_v1/` remains immutable for baseline
reproduction. `manifest.json` is the only runtime artifact inventory.

## Folder Structure

```text
config/                    runtime configuration
src/environment/           public API, models, persistence, runtime
artifacts/gold_events_v2/  authoritative corrected bundle
data/input/gold_history/   starting cumulative Gold history
data/reference/            immutable fitting/validation references (not runtime)
data/output/               local outputs/checkpoints
sql/                       optional PostgreSQL schema
tests/                     deterministic migration/parity tests
docs/                      integration and data documentation
```

## Data Placement

LOCAL INPUT: place one combined March-April Parquet or the two source Parquets
in `data/input/gold_history/`. Multiple `--historical` arguments, a directory,
and configured S3 URIs are also supported. Inputs are opened read-only and are
recorded by SHA-256, size, and modification time in the run manifest.

FROZEN ARTIFACTS: committed under `artifacts/gold_events_v2/`; do not replace individual files without issuing a new manifest/version.

OUTPUT: May runs are isolated under `data/output/runs/<run_id>/`:

```text
may/YYYY-MM-DD/gold_events.parquet
audit/YYYY-MM-DD/decisions.parquet
audit/YYYY-MM-DD/explanations.jsonl.gz
checkpoints/YYYY-MM-DD.json.gz
checkpoints/YYYY-MM-DD.commit.json
combined_gold_events.parquet
simulation_summary.json
validation_report.json
```

S3: set URI values through configuration or environment variables and use `S3IO`. No bucket or credential is embedded.

Normal runtime does not need the full March-April fitting data.

Refitting intra-day opportunities additionally requires the decision log in
`config/reference_inputs.json`. Realized Gold sends do not provide the missing
NO_ACTION or suppressed-opportunity denominators.

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Install optional adapters with `pip install -e ".[postgres,s3]"`.

## Test

```powershell
pytest
python -m environment.cli validate-bundle
python -m environment.cli validate-reference-inputs  # requires the declared reference files
```

## Run May 1-31

The default driver submits explicit WAIT/NO_ACTION intervals. It does not use
the historical campaign policy and never creates unsolicited sends. Organic
progression, inactivity, PTP and expiry still occur.

```powershell
python -m environment.cli validate-input --run-id may-demo `
  --historical data/input/gold_history/events_gold_1_bucket_000_20260331.parquet `
  --historical data/input/gold_history/events_gold_2_bucket_000_20260430.parquet

python -m environment.cli initialize-may --run-id may-demo `
  --historical data/input/gold_history/events_gold_1_bucket_000_20260331.parquet `
  --historical data/input/gold_history/events_gold_2_bucket_000_20260430.parquet

python -m environment.cli run-may --run-id may-demo `
  --historical data/input/gold_history/events_gold_1_bucket_000_20260331.parquet `
  --historical data/input/gold_history/events_gold_2_bucket_000_20260430.parquet
```

Run or retry one day, resume, inspect, export and validate:

```powershell
python -m environment.cli run-day --run-id may-demo --day 2026-05-01
python -m environment.cli resume-may --run-id may-demo
python -m environment.cli inspect-run --run-id may-demo
python -m environment.cli export-combined --run-id may-demo
python -m environment.cli validate-run --run-id may-demo
```

Committed days are idempotent. Resume discovers the latest hash-verified daily
checkpoint and does not replay prior partitions. The final canonical dataset is
`data/output/runs/may-demo/combined_gold_events.parquet`.

For one deterministic campaign demonstration, initialize with
`--send-example`. For an exact scripted action, repeat:

```powershell
--scripted-send "APPLICATION_ID,2026-05-03T09:00:00,WA,MORNING"
```

Scripted actions are test-driver inputs, not empirical Environment dynamics or
a trained policy.

## Time-advance API

RL chooses both the action and interval boundary. `NO_ACTION` means WAIT:

```python
result = env.advance(
    state,
    EnvironmentAction.no_action(),
    "2026-05-03T09:00:00",
    "2026-05-03T12:00:00",
)
```

The Environment advances the complete three hours. Organic or pending events
may occur, but no event is forced and an intermediate event does not interrupt
the requested wait. A campaign passed to `advance()` is sent at `start_time`;
response and journey effects use the frozen observational motif/timing model
and may remain pending beyond `end_time`.

## Single-application integration example

```python
import pandas as pd
from environment import Environment, EnvironmentAction, reconstruct_state_from_gold

history = pd.read_parquet("data/input/gold_history/gold_events_start.parquet")
state = reconstruct_state_from_gold(history, "APPLICATION_ID", "2026-05-03T09:00:00")
env = Environment("artifacts/gold_events_v2", seed=20260502, run_id="example")
result = env.step(
    state=state,
    action=EnvironmentAction.campaign("SMS", "AFTERNOON"),
    decision_time="2026-05-03T09:00:00",
)
updated_history = pd.concat([history, pd.DataFrame(result.events)], ignore_index=True)
```

For a persisted live Transformer/RL boundary, use `LocalClosedLoop`. It
validates the external 192-value embedding and action, atomically replaces the
cumulative Parquet file, and writes a restart checkpoint:

```python
from environment import Environment, LocalClosedLoop

env = Environment("artifacts/gold_events_v2", seed=20260502, run_id="run-001")
loop = LocalClosedLoop(
    env,
    input_history="data/input/gold_history/gold_events_start.parquet",
    cumulative_history="data/output/run-001/gold_events.parquet",
    checkpoint="data/output/run-001/checkpoint.json",
)
step = loop.run_step("APPLICATION_ID", "2026-05-03T09:00:00", transformer, rl_policy)
```

## Action Contract

Use `EnvironmentAction.no_action()` or `EnvironmentAction.campaign(channel_id, time_bucket, theme="THEME_PLACEHOLDER")`. Supported channels are `SMS`, `WA`, and `GRCS`; buckets are `MORNING`, `AFTERNOON`, `EVENING`, and `NIGHT`. No historical theme-specific effect is available, so all themes explicitly back off to the placeholder.

## Output Contract

`EnvironmentResult` contains realized Gold-shaped `events`, updated application `state`, `terminal`, serializable `pending_events`, `decisions`, and separate `explanations`. Decision status distinguishes `NO_ACTION`, `SCHEDULED`, `REALIZED`, `UNSCHEDULABLE_BEFORE_EXPIRY`, and `SUPPRESSED_BY_TERMINAL`. Only `events` should be appended to cumulative Gold history.

## PostgreSQL

PostgreSQL is optional. The runtime-state tables in `sql/environment_schema.sql`
remain separate from the read-only demonstration surface. The cumulative
March-April + May export can be loaded into the `goldfish` database under the
`environment_demo` schema. `context_id` is the primary customer identity;
`application_id` remains the nullable lifecycle identity. `event_row_id` is a
generated SQL key and is never treated as a Gold identifier.

Install the adapter and configure either `DATABASE_URL` or standard libpq
`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, and `PGPASSWORD` variables. Then:

```powershell
pip install -e ".[postgres]"
$env:PGHOST = "localhost"
$env:PGPORT = "5432"
$env:PGDATABASE = "goldfish"
$env:PGUSER = "goldfish"
$env:PGPASSWORD = "<local password>"

python -m environment.cli postgres-demo-validate-input
python -m environment.cli postgres-demo-init
python -m environment.cli postgres-demo-load
python -m environment.cli postgres-demo-validate `
  --expected-historical 9856993 `
  --expected-simulated 356243 `
  --expected-total 10213236
```

The load is one transaction and refuses to overwrite a nonempty table. Use
`postgres-demo-load --replace` only for an intentional atomic reload. The
schema creates:

```text
goldfish
└── environment_demo
    ├── combined_gold_events
    ├── v_monthly_summary
    ├── v_may_daily_summary
    ├── v_simulation_overview
    ├── v_may_journey_summary
    └── v_context_journey
```

`v_context_journey` is the event-level timeline across every application owned
by one customer. It includes stable chronological sequence numbers, the prior
event timestamp, and elapsed time. Query it without conflating customer and
application identity:

```sql
SELECT *
FROM environment_demo.v_context_journey
WHERE context_id = 'MAYCTX_4b5262410a8cda744f9617676952'
ORDER BY context_event_sequence;
```

For application checkpoint persistence, use `initialize_schema()` and
`PostgresStateStore`; that operational store is independent of the demo views.

## S3

Install the `s3` extra and configure the normal AWS credential chain. `S3IO` supports starting-history download, output upload, checkpoints, and a manifest-led bundle download boundary. Unit tests do not access AWS.

## Transformer/RL Integration

The Transformer encodes only committed cumulative history and produces the
external 192-D embedding. RL selects SEND or WAIT and the next simulation time.
Environment advances to that boundary, samples organic behavior and supported
action consequences, then returns only realized Gold rows. The scripted May
driver is replaceable by this live loop without changing Environment dynamics.
See `docs/RL_TRANSFORMER_CONTRACT.md`.

## Explainability

Audit metadata records state, action, model/hash, conditioning/backoff level, support, candidates or hazard, selected outcome, deterministic draw reference, timing source, pathway, and guards. Example:

```text
ACTION: SMS / THEME_PLACEHOLDER / AFTERNOON
STATE: Application Resume
ENVIRONMENT: campaign motif, STATE_CHANNEL, support=N
RESULT: campaign_clicked; later journey response sampled after an empirical delay
```

This is observational association, not a causal claim.

## Determinism

The global seed plus application identity, decision sequence, and draw purpose determine RNG streams. Persist both state and pending events after each committed step; replay from an identical checkpoint and action produces the same outcome.

## Known Limitation

Historical matched data contains approximately 1.52 raw sends per send-day.
That prevents statistical calibration of an autonomous historical opportunity
process without the decision log in `config/reference_inputs.json`. It does not
block live RL scheduling or the scripted driver: those callers explicitly choose
when to WAIT or SEND. Campaign associations remain observational, and the
NO_ACTION demonstration is not a campaign-volume fidelity claim.

## Migration Status

This repository was extracted from the current V3 research runtime with PTP provenance separation, censoring-aware PTP hazard, continuation/inactivity, campaign motifs, and deterministic daily closed-loop semantics. Production hardening adds strict action/history boundaries, terminal suppression telemetry, competing-send censoring, manifest-safe S3 download, atomic local append, and a fake-model integration test. The campaign opportunity limitation remains explicit; no unsupported recalibration was performed.

## Safety / Data Rules

- Historical source data is immutable.
- Simulated data never derives frozen ground-truth probabilities.
- Synthetic balancing data never enters environment dynamics.
- `application_id` is the lifecycle boundary; `context_id` is customer identity.
- Future events are never exposed before their scheduled time.
- Model updates require a new versioned manifest and hashes.

## Runtime semantics and recovery

Runtime code semantics are independent of the frozen empirical bundle. New
`Environment` and `MaySimulationRunner` instances default to `corrected-v1`,
configured by `environment.runtime_version` or `--runtime-version`. Explicit
`legacy-v2r2` retains the accepted V2r2 hazard and expiry behavior for reproduction:

```bash
environment run-may --run-id may-no-action-v2r2-20260502 \
  --runtime-version legacy-v2r2 --output-root data/output/phase1-legacy-validation
```

Use a separate output root; never overwrite the accepted AWS run. Runtime
version is saved in run manifests, summaries, checkpoints and stepped application
state and checked on resume. Older metadata without this field means legacy;
opening an old run preserves that choice. New corrected checkpoints cannot
resume as legacy or vice versa. Explicit CLI resume overrides must match saved
metadata. The frozen bundle and historical Parquets are unchanged.

The hazard assets contain **conditional interval probabilities** (events divided
by the risk set at the bin start), plus empirical conditional timing donors.
Corrected mode allocates each bin's hazard mass across its donors and conditions
on survival to the current elapsed time. It samples the remaining event time
once per continuation/AIP episode through the lifecycle deadline. Future events
and no-event outcomes are persisted; dividing a WAIT does not redraw exposure or
consume extra model RNG draws. New journey observations, a new eligible AIP or
campaign interruption/release can start a new applicable episode. This removes
the legacy combination of fractional-bin hazard and empty-window donor
suppression without fitting or scaling probabilities. Corrected output counts
are not required to equal legacy counts.

Corrected mode processes expiry even when WAIT ends exactly at creation plus
30 days. Ordinary Gold events retain half-open interval execution. Existing
transaction-before-expiry priority is retained for an already queued terminal
PTP exactly at the deadline; it wins that tie and clears the queue. Expiry emits
no Gold row. Later pending events are suppressed, and further advances of a
terminal application are rejected. Frozen V2 hazard draws and campaign motifs
schedule their realized outcomes strictly before expiry and the May endpoint.

The corrected local loop follows the day-wise runner's **manifest-last commit**
pattern: stage immutable history and checkpoint snapshots, flush their files and
directory entries, then atomically publish the checkpoint commit marker. The
public cumulative history is a projection of that committed snapshot. A crash
before the marker leaves staging uncommitted and ignored; a crash after it is
recovered by publishing the validated snapshot. Same-boundary retries return
without calling the controller or appending Gold again. Recovery validates both
snapshot hashes, checkpoint state, seed, runtime version, artifact hashes and
input history identity. Arbitrary history/checkpoint disagreement fails closed;
the immediately previous history projection is recognized as incomplete
publication and repaired. Pending events, RNG counters and simulated time are
part of the committed state. This single-application adapter uses a filesystem
lock for cooperating writers sharing its checkpoint; each loop owns its history
and checkpoint paths. Snapshot directories must remain beside their checkpoint.
No PostgreSQL transaction is required. Legacy mode retains its old local-loop
commit semantics solely for replay.


### Corrected-v2 training Environment

The new explicit `corrected-v2` runtime uses `artifacts/gold_events_v3/`.
Existing corrected-v1 defaults and frozen v1/v2 bundles remain available.
See [the external controller and empirical contract](docs/corrected-v2-contract.md)
for caller-controlled SEND/WAIT decisions, generic calendar bounds, scoped
observations, recovery, and stochastic namespaces. `config/corrected-v2.yaml`
is the explicit CLI configuration; it does not replace the older default.

Historical-only construction is in `tools/build_final_bundle.py`. Validation
uses `tools/validate_final_history.py`, `tools/validate_final_subpopulations.py`,
`tools/validate_final_run.py`, and `tools/evaluate_final_run.py`. Raw Phase 2
comparisons remain separate from the supplementary prerequisite-observability
audit. A full run requires an accepted medium gate in
`tools/run_final_validation.py`; rejected development candidates are retained
under `data/output/finalization-validation/`.

The final materiality-based freeze decision is documented in
[corrected-v2 certification](docs/corrected-v2-certification.md), with the
versioned [certification evidence](docs/corrected-v2-certification.json).
`tools/certify_frozen_candidate.py` evaluates existing full-run evidence;
it never simulates or refits. The earlier strict gate remains a diagnostic
record of statistically nonzero differences. Both raw historical comparisons
and supplementary business-guard comparisons remain disclosed.
