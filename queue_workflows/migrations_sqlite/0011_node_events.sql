-- workflow_node_events: append-only log of per-node per-attempt lifecycle events.
-- Keeps durably the claim, model-load, stall, trip, requeue, reassign, and terminal
-- signals that would otherwise be lost when subsequent attempts overwrite the
-- mutable workflow_node_jobs row. 'attempt' is the cross-attempt key; 'detail'
-- carries free-form metrics (max_sm_pct, ram_anchor_mb, exit_code, etc.).
-- ON DELETE CASCADE from workflow_runs keeps purge/restart_from clean.

CREATE TABLE IF NOT EXISTS workflow_node_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES workflow_runs(id)
                                ON DELETE CASCADE,
    node_id     TEXT NOT NULL,
    job_id      TEXT,
    attempt     INTEGER NOT NULL DEFAULT 0,
    event_type  TEXT NOT NULL,
    host_label  TEXT,
    queue       TEXT,
    model       TEXT,
    elapsed_s   REAL,
    error       TEXT,
    detail      TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    CONSTRAINT workflow_node_events_type_check CHECK (event_type IN (
        'claimed', 'model_load_start', 'model_load_done', 'progress_beat',
        'stall_suspected', 'stall_trip', 'gpu_health_trip', 'budget_trip',
        'requeued', 'reassigned', 'lease_renew', 'completed', 'failed',
        'cancelled', 'error'
    ))
);

CREATE INDEX IF NOT EXISTS workflow_node_events_node_idx
    ON workflow_node_events (run_id, node_id, created_at);

CREATE INDEX IF NOT EXISTS workflow_node_events_created_idx
    ON workflow_node_events (created_at);
