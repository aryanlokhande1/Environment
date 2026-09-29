# Starting Gold history

For local runs put the starting Gold Events or cumulative history here, preferably as:

`data/input/gold_history/gold_events_start.parquet`

Required reconstruction columns are `application_id`, `context_id`, `event_datetime`, `journey_stage`, and `journey_substage`. Preserve the complete Gold schema when available, including `event_name`, channel fields, provenance, `is_simulated`, and stable row keys. Rows must contain only information visible at the decision time. Normal runtime does not require the raw March-April fitting data.
