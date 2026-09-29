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

The runtime reconstructs application-scoped state from visible cumulative history, applies business guards, and samples natural transitions, FIRST_PASS/REVISIT behavior, stage continuation or temporary inactivity, campaign response motifs, event timing, PTP hazard, and 30-day expiry. `NO_ACTION` still permits organic behavior. PTP is terminal for an application and is sampled only from the eligible post-AIP hazard. Pending events and RNG state are checkpointable. Every decision and model draw produces separate audit metadata; audit rows never enter Gold history.

## Ground Truth

`artifacts/gold_events_v1/` is the frozen probability bundle. `manifest.json` is the only artifact inventory and pins every runtime input by SHA-256. Raw support tables, spreadsheets, and files not declared by the manifest are not runtime probabilities.

## Folder Structure

```text
config/                    runtime configuration
src/environment/           public API, models, persistence, runtime
artifacts/gold_events_v1/  authoritative frozen bundle
data/input/gold_history/   starting cumulative Gold history
data/reference/            optional non-model references
data/output/               local outputs/checkpoints
sql/                       optional PostgreSQL schema
tests/                     deterministic migration/parity tests
docs/                      integration and data documentation
```

## Data Placement

LOCAL INPUT: place `gold_events_start.parquet` in `data/input/gold_history/`.

FROZEN ARTIFACTS: committed under `artifacts/gold_events_v1/`; do not replace individual files without issuing a new manifest/version.

OUTPUT: local cumulative history and checkpoints belong in `data/output/`.

S3: set URI values through configuration or environment variables and use `S3IO`. No bucket or credential is embedded.

Normal runtime does not need the full March-April fitting data.

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
```

## Run / Example

```python
import pandas as pd
from environment import Environment, EnvironmentAction, reconstruct_state_from_gold

history = pd.read_parquet("data/input/gold_history/gold_events_start.parquet")
state = reconstruct_state_from_gold(history, "APPLICATION_ID", "2026-05-03T09:00:00")
env = Environment("artifacts/gold_events_v1", seed=20260502, run_id="example")
result = env.step(
    state=state,
    action=EnvironmentAction.campaign("SMS", "AFTERNOON"),
    decision_time="2026-05-03T09:00:00",
)
updated_history = pd.concat([history, pd.DataFrame(result.events)], ignore_index=True)
```

## Action Contract

Use `EnvironmentAction.no_action()` or `EnvironmentAction.campaign(channel_id, time_bucket, theme="THEME_PLACEHOLDER")`. Supported channels are `SMS`, `WA`, and `GRCS`; buckets are `MORNING`, `AFTERNOON`, `EVENING`, and `NIGHT`. No historical theme-specific effect is available, so all themes explicitly back off to the placeholder.

## Output Contract

`EnvironmentResult` contains realized Gold-shaped `events`, updated application `state`, `terminal`, serializable `pending_events`, `decisions`, and separate `explanations`. Only `events` should be appended to cumulative Gold history.

## PostgreSQL

PostgreSQL is optional. Install the `postgres` extra, set standard `PG*` variables, run `initialize_schema()`, and use `PostgresStateStore` to atomically upsert application checkpoints. The focused schema stores runs, decisions, Gold events, explanations, checkpoints, and pending events.

## S3

Install the `s3` extra and configure the normal AWS credential chain. `S3IO` supports starting-history download, output upload, checkpoints, and a manifest-led bundle download boundary. Unit tests do not access AWS.

## Transformer/RL Integration

The Transformer encodes only the current cumulative history. RL chooses one supported action. Environment samples the world response and returns realized events. Append those events, persist atomically, and encode again. See `docs/RL_TRANSFORMER_CONTRACT.md`.

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

Historical data contains approximately 1.52 raw sends per send-day, while the current validated runtime primarily exposes one daily decision opportunity. Campaign cadence is not claimed to be perfectly calibrated. `get_decision_opportunities()` is the compatibility boundary for a future list of intra-day opportunities. See `TODO_INTRA_DAY_CAMPAIGN_OPPORTUNITY_MODEL`.

## Migration Status

This repository was extracted from the current V3 research runtime with PTP provenance separation, censoring-aware PTP hazard, continuation/inactivity, campaign motifs, and deterministic daily closed-loop semantics. The campaign opportunity limitation remains explicit; no recalibration was performed during packaging.

## Safety / Data Rules

- Historical source data is immutable.
- Simulated data never derives frozen ground-truth probabilities.
- Synthetic balancing data never enters environment dynamics.
- `application_id` is the lifecycle boundary; `context_id` is customer identity.
- Future events are never exposed before their scheduled time.
- Model updates require a new versioned manifest and hashes.
