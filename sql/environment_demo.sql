-- Read-only analytics surface for the cumulative March-April historical and
-- May simulated Gold Events export.  Gold timestamps are deliberately stored
-- without a time zone because the source contract is local wall-clock time.

CREATE SCHEMA IF NOT EXISTS environment_demo;

CREATE TABLE IF NOT EXISTS environment_demo.combined_gold_events (
    event_row_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    context_id text NOT NULL,
    application_id text,
    event_datetime timestamp without time zone NOT NULL,
    event_name text,
    journey_stage text,
    journey_substage text,
    data_origin text NOT NULL CHECK (data_origin IN (
        'HISTORICAL_MARCH', 'HISTORICAL_APRIL', 'SIMULATED_MAY'
    )),
    is_simulated boolean NOT NULL,
    site_subsection text,
    channel_id text,
    platform_type text,
    product_id text,
    partner_id text,
    trigger_type text,
    time_bucket text,
    visitnum double precision,
    utm_source text,
    utm_medium text,
    utm_campaign text,
    source_type text,
    campaign_name text,
    session_id text,
    channel text,
    utm_channel text,
    day_of_week_num bigint,
    initiated_by text,
    CONSTRAINT combined_gold_events_origin_consistent CHECK (
        (is_simulated AND data_origin = 'SIMULATED_MAY') OR
        (NOT is_simulated AND data_origin IN ('HISTORICAL_MARCH', 'HISTORICAL_APRIL'))
    )
);

COMMENT ON COLUMN environment_demo.combined_gold_events.context_id IS
    'Primary customer identity. A context may own multiple applications.';
COMMENT ON COLUMN environment_demo.combined_gold_events.application_id IS
    'Application lifecycle identity when supplied by Gold; historical rows may be NULL.';
COMMENT ON COLUMN environment_demo.combined_gold_events.event_row_id IS
    'Technical SQL key only; it is not a Gold business identifier.';

CREATE INDEX IF NOT EXISTS combined_gold_events_context_time_idx
    ON environment_demo.combined_gold_events (context_id, event_datetime, event_row_id);
CREATE INDEX IF NOT EXISTS combined_gold_events_application_time_idx
    ON environment_demo.combined_gold_events (application_id, event_datetime, event_row_id)
    WHERE application_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS combined_gold_events_time_idx
    ON environment_demo.combined_gold_events (event_datetime);
CREATE INDEX IF NOT EXISTS combined_gold_events_simulated_time_idx
    ON environment_demo.combined_gold_events (event_datetime, context_id)
    WHERE is_simulated;

CREATE OR REPLACE VIEW environment_demo.v_monthly_summary AS
SELECT
    date_trunc('month', event_datetime)::date AS month_start,
    data_origin,
    is_simulated,
    count(*)::bigint AS event_count,
    count(DISTINCT context_id)::bigint AS context_count,
    count(DISTINCT application_id)::bigint AS application_count,
    count(*) FILTER (WHERE journey_substage = 'AIP Approved')::bigint AS aip_event_count,
    count(*) FILTER (WHERE journey_substage = 'Push to Partner')::bigint AS ptp_event_count
FROM environment_demo.combined_gold_events
GROUP BY date_trunc('month', event_datetime)::date, data_origin, is_simulated;

CREATE OR REPLACE VIEW environment_demo.v_may_daily_summary AS
SELECT
    event_datetime::date AS event_date,
    count(*)::bigint AS event_count,
    count(DISTINCT context_id)::bigint AS context_count,
    count(DISTINCT application_id)::bigint AS application_count,
    count(*) FILTER (WHERE journey_substage = 'AIP Approved')::bigint AS aip_event_count,
    count(*) FILTER (WHERE journey_substage = 'Push to Partner')::bigint AS ptp_event_count,
    count(*) FILTER (WHERE journey_substage = 'campaign_sent')::bigint AS campaign_send_count,
    count(*) FILTER (
        WHERE journey_substage IN ('campaign_open_read', 'campaign_clicked')
    )::bigint AS campaign_response_count
FROM environment_demo.combined_gold_events
WHERE is_simulated
  AND event_datetime >= timestamp '2026-05-01 00:00:00'
  AND event_datetime < timestamp '2026-06-01 00:00:00'
GROUP BY event_datetime::date;

CREATE OR REPLACE VIEW environment_demo.v_simulation_overview AS
SELECT
    count(*)::bigint AS combined_event_count,
    count(*) FILTER (WHERE NOT is_simulated)::bigint AS historical_event_count,
    count(*) FILTER (WHERE is_simulated)::bigint AS simulated_event_count,
    count(DISTINCT context_id)::bigint AS combined_context_count,
    count(DISTINCT context_id) FILTER (WHERE is_simulated)::bigint AS simulated_context_count,
    count(DISTINCT application_id)::bigint AS combined_application_count,
    count(DISTINCT application_id) FILTER (WHERE is_simulated)::bigint AS simulated_application_count,
    min(event_datetime) AS first_event_datetime,
    max(event_datetime) AS last_event_datetime,
    min(event_datetime) FILTER (WHERE is_simulated) AS first_simulated_event_datetime,
    max(event_datetime) FILTER (WHERE is_simulated) AS last_simulated_event_datetime,
    count(DISTINCT event_datetime::date) FILTER (WHERE is_simulated)::bigint AS simulated_active_day_count
FROM environment_demo.combined_gold_events;

CREATE OR REPLACE VIEW environment_demo.v_may_journey_summary AS
SELECT
    context_id,
    application_id,
    min(event_datetime) AS first_may_event_datetime,
    max(event_datetime) AS last_may_event_datetime,
    count(*)::bigint AS event_count,
    count(DISTINCT event_datetime::date)::bigint AS active_day_count,
    bool_or(journey_substage = 'AIP Approved') AS reached_aip,
    bool_or(journey_substage = 'Push to Partner') AS reached_push_to_partner,
    count(*) FILTER (WHERE journey_substage = 'campaign_sent')::bigint AS campaign_send_count,
    count(*) FILTER (
        WHERE journey_substage IN ('campaign_open_read', 'campaign_clicked')
    )::bigint AS campaign_response_count
FROM environment_demo.combined_gold_events
WHERE is_simulated
  AND event_datetime >= timestamp '2026-05-01 00:00:00'
  AND event_datetime < timestamp '2026-06-01 00:00:00'
GROUP BY context_id, application_id;

-- Event-level, chronological customer timeline. context_id is deliberately the
-- partition key: application_id identifies a lifecycle, not the customer.
CREATE OR REPLACE VIEW environment_demo.v_context_journey AS
SELECT
    event_row_id,
    context_id,
    application_id,
    event_datetime,
    row_number() OVER context_timeline AS context_event_sequence,
    CASE WHEN application_id IS NOT NULL THEN
        row_number() OVER application_timeline
    END AS application_event_sequence,
    lag(event_datetime) OVER context_timeline AS previous_event_datetime,
    event_datetime - lag(event_datetime) OVER context_timeline AS time_since_previous_event,
    event_name,
    journey_stage,
    journey_substage,
    data_origin,
    is_simulated,
    channel,
    trigger_type,
    time_bucket,
    source_type,
    initiated_by
FROM environment_demo.combined_gold_events
WINDOW
    context_timeline AS (
        PARTITION BY context_id ORDER BY event_datetime, event_row_id
    ),
    application_timeline AS (
        PARTITION BY context_id, application_id ORDER BY event_datetime, event_row_id
    );
