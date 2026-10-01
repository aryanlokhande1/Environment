# RL / Transformer Contract

1. Read only committed cumulative Gold rows visible at time `t`.
2. Transformer produces an embedding; it is not implemented here.
3. RL returns `NO_ACTION`/WAIT or `{campaign_sent: true, channel_id, theme, time_bucket}` and selects the next simulation boundary.
4. `Environment.advance(state, action, start_time, end_time)` applies the action at `start_time` and samples the world through `end_time` from the frozen bundle.
5. Append only realized Gold-shaped rows and commit state/pending/audit atomically.
6. Transformer reads the updated committed history for the next decision.

Theme-specific effects are unsupported by the historical bundle. The only accepted value is `THEME_PLACEHOLDER`, an explicit backoff rather than an invented effect. Observed campaign relationships are associative, not causal.

WAIT never disables organic behavior. The Environment may emit zero or more
events during the interval and does not request another decision before the
caller-selected boundary. Campaign engagement/progression can remain pending
after the boundary. SEND is never generated autonomously when an external
policy or scripted driver is supplied.

The opportunity API returns one opportunity for each active application-day in
the requested interval, bounded by activation, terminal time, and expiry. It
returns a list so a future evidence-backed intra-day model can return
`[t1, t2, ...]` without changing the `step` contract. A state may be stepped at
most once per day under the current bundle.

Policy output may be an `EnvironmentAction` or a mapping. Unknown fields,
unsupported channels/buckets/themes, non-boolean `campaign_sent`, and embeddings
whose shape is not exactly `(192,)` are rejected at the integration boundary.
