-- queue_workflows 0002 — workflow_node_jobs (the node-per-job queue).
--
-- The engine dispatches one *node* at a time, not a whole pipeline step. Each
-- node-job lives on either the 'cpu' queue (short-lived subprocess workers) or
-- the 'gpu' queue (long-lived workers with a model cache).
--
-- CONSOLIDATED final shape (plan §7 step 6 — discrete column-add migrations
-- folded into the base table): this table is created in the same shape the
-- ai_leads chain reached after migrations 002/005/006/009/016/019/030/038.
-- Columns folded in:
--   * pipeline_name (old 005)        — parent pipeline ref
--   * celery_task_id (old 009)       — legacy, unused post-Phase-5; kept for
--                                      fidelity (a separate cleanup drops it)
--   * resolved_inputs (old 016)      — execute-time $from snapshot
--   * host_label (old 030)           — claiming host
--   * input_spec (old 038)           — per-job awaiting-input widget spec
-- CHECK constraints folded to final form:
--   * required_model (old 006): CPU rows MUST NOT set required_model; GPU rows
--     MAY leave it NULL.  -> name 'workflow_node_jobs_check'.
--   * status (old 019): includes 'skipped'.  -> name
--     'workflow_node_jobs_status_check'.
-- The 'mode' column on workflow_runs (old 002) lives in 0001.
--
-- Idempotent: 'CREATE TABLE IF NOT EXISTS' so re-running on an already-migrated
-- ai_leads DB is a no-op. The named CHECK constraints match what the ai_leads
-- chain auto-named, so the existing live table satisfies them identically.

CREATE TABLE IF NOT EXISTS workflow_node_jobs (
    id                 TEXT PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
    node_id            TEXT NOT NULL,
    node_module        TEXT NOT NULL,
    pipeline_name      TEXT,
    queue              TEXT NOT NULL CHECK (queue IN ('cpu', 'gpu')),
    required_model     TEXT,
    status             TEXT NOT NULL,
    priority           INTEGER NOT NULL DEFAULT 100,
    worker_lane        INTEGER,
    inputs             TEXT NOT NULL DEFAULT '{}',
    resolved_inputs    TEXT,
    input_spec         TEXT,
    context_delta      TEXT NOT NULL DEFAULT '{}',
    host_label         TEXT,
    celery_task_id     TEXT,
    error              TEXT,
    vm_rss_mb_peak     INTEGER,
    seconds            REAL,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    started_at         TEXT,
    finished_at        TEXT,
    UNIQUE (run_id, node_id),
    CONSTRAINT workflow_node_jobs_status_check CHECK (status IN (
        'queued', 'running', 'completed', 'failed', 'cancelled',
        'awaiting_input', 'skipped'
    )),
    CONSTRAINT workflow_node_jobs_check
        CHECK (queue = 'gpu' OR required_model IS NULL)
);

-- Hot path: dispatcher pulls (queue, status, priority, created_at) on claim.
CREATE INDEX IF NOT EXISTS workflow_node_jobs_claim_idx
    ON workflow_node_jobs (queue, priority, created_at)
    WHERE status = 'queued';

CREATE INDEX IF NOT EXISTS workflow_node_jobs_run_idx ON workflow_node_jobs (run_id);
CREATE INDEX IF NOT EXISTS workflow_node_jobs_status_idx ON workflow_node_jobs (status);
-- For GPU-worker "prefer current model" ordering:
CREATE INDEX IF NOT EXISTS workflow_node_jobs_model_idx ON workflow_node_jobs (required_model)
    WHERE queue = 'gpu' AND status = 'queued';
-- old 005:
CREATE INDEX IF NOT EXISTS workflow_node_jobs_pipeline_idx
    ON workflow_node_jobs (pipeline_name);
-- old 030 (partial, NULLs skipped):
CREATE INDEX IF NOT EXISTS workflow_node_jobs_host_label_idx
    ON workflow_node_jobs (host_label)
    WHERE host_label IS NOT NULL;
-- old 009 (partial, NULLs skipped):
CREATE INDEX IF NOT EXISTS workflow_node_jobs_celery_task_id_idx
    ON workflow_node_jobs (celery_task_id)
    WHERE celery_task_id IS NOT NULL;
