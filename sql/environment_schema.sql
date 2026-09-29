CREATE TABLE IF NOT EXISTS environment_runs (
    run_id text PRIMARY KEY, seed bigint NOT NULL, artifact_bundle_version text NOT NULL,
    artifact_manifest_sha256 text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
    status text NOT NULL
);
CREATE TABLE IF NOT EXISTS environment_decisions (
    run_id text NOT NULL REFERENCES environment_runs(run_id), decision_id text NOT NULL,
    application_id text NOT NULL, context_id text NOT NULL, decision_time timestamptz NOT NULL,
    action_json jsonb NOT NULL, status text NOT NULL, PRIMARY KEY (run_id, decision_id)
);
CREATE TABLE IF NOT EXISTS environment_events (
    run_id text NOT NULL REFERENCES environment_runs(run_id), event_key text NOT NULL,
    application_id text NOT NULL, context_id text NOT NULL, event_datetime timestamptz NOT NULL,
    gold_row jsonb NOT NULL, PRIMARY KEY (run_id, event_key)
);
CREATE TABLE IF NOT EXISTS environment_explanations (
    run_id text NOT NULL REFERENCES environment_runs(run_id), explanation_id text NOT NULL,
    application_id text NOT NULL, explanation_time timestamptz NOT NULL,
    explanation_json jsonb NOT NULL, PRIMARY KEY (run_id, explanation_id)
);
CREATE TABLE IF NOT EXISTS environment_checkpoints (
    run_id text NOT NULL REFERENCES environment_runs(run_id), application_id text NOT NULL,
    checkpoint_time timestamptz NOT NULL, state_json jsonb NOT NULL, pending_json jsonb NOT NULL,
    PRIMARY KEY (run_id, application_id)
);
CREATE TABLE IF NOT EXISTS environment_pending_events (
    run_id text NOT NULL REFERENCES environment_runs(run_id), application_id text NOT NULL,
    pending_key text NOT NULL, due_at timestamptz NOT NULL, payload_json jsonb NOT NULL,
    PRIMARY KEY (run_id, application_id, pending_key)
);
