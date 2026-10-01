# Architecture

The public `Environment.advance` boundary accepts application state, an
external action, and an explicit `[start_time, end_time)` interval. `NO_ACTION`
is WAIT. The core delegates to frozen runtime modules for guarded progression,
continuation, campaign motifs, PTP hazard, and timing. An event inside the
interval is recorded without shortening the caller-selected wait. `step`
remains the compatible daily convenience method.

Realized events are converted to the Gold contract; decisions and explanations
remain separate. Persistence adapters own local, S3, checkpoint, and optional
PostgreSQL boundaries. Artifact loading is manifest-only, schema-checked and
hash-verified before runtime initialization.

State is application-scoped. A context may later own a different application, but lifecycle prerequisites never transfer between applications. Pending future events are retained in the checkpoint and remain invisible to upstream models until realized. A competing send removes unresolved response events from the earlier exposure as censored. Terminal processing clears all later events and records any scheduled campaign send as suppressed.

`LocalClosedLoop` is the minimal local coordinator. It loads only history visible
at the decision timestamp, calls caller-supplied Transformer and RL interfaces,
validates the 192-D/action contracts, steps the Environment, atomically appends
Gold-shaped rows, and saves state plus pending events for restart. It is not a
Transformer or policy implementation.

`MaySimulationRunner` is a separate scripted driver over the same public
Environment API. It seeds carried-forward state from the frozen May-1 snapshot,
activates arrivals from the frozen calendar-matched arrival artifact, submits
explicit WAIT/SEND actions, and commits one run-scoped day at a time. The commit
marker is written only after Gold, audit and compressed state/pending files are
durable. Combined export streams immutable historical row groups followed by
May partitions into the canonical 24-column Gold schema.
