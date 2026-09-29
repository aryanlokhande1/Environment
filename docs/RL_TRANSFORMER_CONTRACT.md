# RL / Transformer Contract

1. Read only committed cumulative Gold rows visible at time `t`.
2. Transformer produces an embedding; it is not implemented here.
3. RL returns `NO_ACTION` or `{campaign_sent: true, channel_id, theme, time_bucket}`.
4. Environment samples realized world events and timing from the frozen bundle.
5. Append only realized Gold-shaped rows and commit state/pending/audit atomically.
6. Transformer reads the updated committed history for the next decision.

Theme-specific effects are unsupported by the historical bundle. The only accepted value is `THEME_PLACEHOLDER`, an explicit backoff rather than an invented effect. Observed campaign relationships are associative, not causal.

The opportunity API currently returns at most one active-day opportunity. It intentionally returns a list so a future `TODO_INTRA_DAY_CAMPAIGN_OPPORTUNITY_MODEL` can return `[t1, t2, ...]` without changing the `step` contract.
