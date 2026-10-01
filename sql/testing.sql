SELECT *
FROM environment_demo.combined_gold_events
ORDER BY event_row_id
LIMIT 20;

(
    SELECT *
    FROM environment_demo.combined_gold_events
    WHERE is_simulated = FALSE
    ORDER BY event_row_id
    LIMIT 10
)
UNION ALL
(
    SELECT *
    FROM environment_demo.combined_gold_events
    WHERE is_simulated = TRUE
    ORDER BY event_row_id
    LIMIT 10
);

SELECT
    data_origin,
    is_simulated,
    COUNT(*) AS total_events,
    COUNT(DISTINCT context_id) AS unique_context_ids,
    COUNT(DISTINCT application_id) AS unique_application_ids
FROM environment_demo.combined_gold_events
GROUP BY data_origin, is_simulated
ORDER BY data_origin;

SELECT
    data_origin,
    journey_stage,
    journey_substage,
    COUNT(*) AS event_count,
    COUNT(DISTINCT context_id) AS unique_context_ids,
    COUNT(DISTINCT application_id) AS unique_applications
FROM environment_demo.combined_gold_events
GROUP BY
    data_origin,
    journey_stage,
    journey_substage
ORDER BY
    data_origin,
    event_count DESC;

SELECT
    journey_stage,
    journey_substage,
    COUNT(*) AS event_count,
    COUNT(DISTINCT context_id) AS unique_context_ids,
    COUNT(DISTINCT application_id) AS unique_applications
FROM environment_demo.combined_gold_events
WHERE is_simulated = TRUE
GROUP BY journey_stage, journey_substage
ORDER BY event_count DESC;

SELECT
    context_id,
    COUNT(*) AS total_events,
    COUNT(DISTINCT application_id) AS application_count,
    MIN(event_datetime) AS first_event,
    MAX(event_datetime) AS last_event
FROM environment_demo.combined_gold_events
WHERE is_simulated = TRUE
GROUP BY context_id
ORDER BY total_events DESC
LIMIT 30;

WITH selected_context AS (
    SELECT context_id
    FROM environment_demo.combined_gold_events
    WHERE is_simulated = TRUE
      AND context_id <> 'MAYCTX_4b5262410a8cda744f9617676952'
    GROUP BY context_id
    HAVING COUNT(*) >= 5
    ORDER BY COUNT(*) DESC
    LIMIT 1
)
SELECT
    j.context_event_sequence,
    j.event_datetime,
    j.context_id,
    j.application_id,
    j.event_name,
    j.journey_stage,
    j.journey_substage,
    j.data_origin,
    j.is_simulated
FROM environment_demo.v_context_journey j
JOIN selected_context s
    ON j.context_id = s.context_id
ORDER BY j.context_event_sequence;

WITH carried_context AS (
    SELECT context_id
    FROM environment_demo.combined_gold_events
    GROUP BY context_id
    HAVING
        BOOL_OR(is_simulated = FALSE)
        AND BOOL_OR(is_simulated = TRUE)
    ORDER BY COUNT(*) DESC
    LIMIT 1
)
SELECT
    j.context_event_sequence,
    j.event_datetime,
    j.context_id,
    j.application_id,
    j.event_name,
    j.journey_stage,
    j.journey_substage,
    j.data_origin,
    j.is_simulated
FROM environment_demo.v_context_journey j
JOIN carried_context c
    ON j.context_id = c.context_id
ORDER BY j.context_event_sequence;

WITH selected_context AS (
    SELECT context_id
    FROM environment_demo.combined_gold_events
    WHERE is_simulated = TRUE
      AND context_id <> 'MAYCTX_4b5262410a8cda744f9617676952'
    GROUP BY context_id
    HAVING COUNT(*) BETWEEN 5 AND 15
    ORDER BY COUNT(*) DESC
    LIMIT 1
)
SELECT
    event_datetime,
    application_id,
    journey_stage,
    journey_substage,
    data_origin
FROM environment_demo.v_context_journey
WHERE context_id = (
    SELECT context_id
    FROM selected_context
)
ORDER BY context_event_sequence;

