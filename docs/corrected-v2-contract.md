# Corrected-v2 candidate contract

The frozen `legacy-v2r2` and `corrected-v1` runtimes retain their v1/v2 assets
and replay behavior. `corrected-v2` requires `artifacts/gold_events_v3` and a
persisted stochastic namespace. The existing default remains corrected-v1
until the corrected-v2 fidelity gate and final validation are accepted.

## External controller

```python
from environment import Environment, EnvironmentAction, ExternalAgentSession

env = Environment(
    "artifacts/gold_events_v3", runtime_version="corrected-v2",
    seed=20260502, run_id="experiment-storage-id",
    stochastic_namespace="experiment-world-id",
    simulation_start="2026-06-01", simulation_end="2026-07-01",
)
session = ExternalAgentSession(
    env, application_id="application-id", start_time="2026-06-01T09:00:00",
    input_history="input.parquet", cumulative_history="cumulative.parquet",
    checkpoint="checkpoint.json",
)
observation = session.observe()
result = session.decide(EnvironmentAction.send("SMS", "MORNING"),
                        "2026-06-01T12:00:00")
result = session.decide(EnvironmentAction.wait(), "2026-06-01T18:00:00")
result = session.decide(EnvironmentAction.wait(), "2026-06-02T09:00:00")
```

SEND executes at the committed decision timestamp; the time bucket remains
an action attribute. WAIT is an explicit decision. The caller chooses every
next decision boundary. Organic events and pending effects advance internally;
midnight does not trigger an additional controller decision. Core calendar
bounds are optional and generic. The May validation runner retains May bounds.

`AgentObservation` contains `simulated_now`, application/customer identifiers,
and customer-scoped Gold-column history with `event_datetime <= simulated_now`,
including previous customer applications. No hidden state, probabilities,
future queues, or future outcomes belong to observations. `ExternalStep` holds
the next observation, realized Gold events, terminal status, and separate
`operator_audit`. Do not feed operator audit/checkpoints to an agent.

`Environment.advance(state, action, start_time, end_time)` is the lower-level
in-memory API. Its default inclusive end exposes events exactly at a caller's
next decision boundary. The reporting runner uses half-open daily Gold
partitions. `step` and automatic daily decision-opportunity APIs are rejected
for corrected-v2. Older runtimes retain their daily interface.

Recovery validates seed, run identity, stochastic namespace, input hash,
artifact hashes, calendar bounds, and application identity before use.
Committed retries at the same next decision time produce no duplicate events;
a retry with a different action is rejected. Local persistence uses immutable
history/state snapshots and a manifest-last commit marker with file locking.

## Empirical contract

All new models derive from immutable March-April Gold, never simulated May.
Finite distinct business packets preserve supported timestamp ties; duplicate
instrumentation and unsupported co-occurrences are excluded. PD-before-AIP and
terminal guards remain authoritative.

The rejected first candidate censored PTP when fitting natural continuation
and sampled PTP independently. The replacement samples mutually exclusive next
journey/eligible-terminal outcomes, preserving their empirical delay and
incidence. Missing-prerequisite terminal mass becomes inactivity, without
renormalization into extra journey activity. The frozen PTP files are retained
byte-for-byte for replay; corrected-v2 does not invoke that independent clock.
Eligibility, source age, and observed returning-customer status have separate
backoff pools. Sparse source/regime pools use source, stage, then empirical
fallback within those risk strata.

An application's first carried-risk episode uses complete historical lifecycle
landmarks matched by weekday, source, age, silence, and prior depth with
at least 30 unique applications, followed by deterministic backoff. Residual
survival censors each historical donor at its exact expiry and competes the
sampled event with the target expiry. Fine near-expiry age bins avoid mixing
hours with nine remaining days. Inactivity and exact positive delays remain. One draw per
risk episode, cached pending/no-event outcomes, prevents polling-dependent
resampling. Historical temporal limitations and medium-run gate outcomes must
be reviewed before calling this candidate training-ready.

Returning arrivals sample April weekday-specific returner rates and joint
prior-gap/outcome/history-depth donors. Contexts must have no overlapping
historical lifecycle, be at least 30 days past their latest creation, and not
have been assigned to another May arrival. Gap matching first preserves the
sampled prior outcome, then uses an exact-day or three-day neighborhood with
30 eligible histories; sparse cells use the nearest supported neighborhood.
It never falls back to all eligible contexts. Late-May immutable history
availability still shifts gaps above raw April quantiles; this is reported
alongside target errors and the closest available eligible histories.

`run_id` identifies storage/audit. `stochastic_namespace` controls corrected-v2
RNG and event keys, including event-name sampling. Same seed/namespace and
different run IDs yield the same logical world; changing namespace yields a
different realization. Namespace is persisted in state, checkpoints, manifest,
summary, and recovery identity. Older event-key semantics remain unchanged.
