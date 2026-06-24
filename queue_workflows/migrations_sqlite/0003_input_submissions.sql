-- queue_workflows 0003 — workflow_input_submissions (durable user-input store).
--
-- Rails inserts a row (status='pending') when the user submits a value for an
-- awaiting_input node; the Python InputListener polls, claims, calls
-- dispatcher.resume_after_input, then marks processed. Replaces a transient
-- notification channel (which dropped submissions on listener restart).
--
-- CONSOLIDATED final shape (old 008 + 014 + 034):
--   * claimed_at (old 014)            — reclaim a row stuck in 'processing'
--   * partial-unique (old 034): the base 008 status-agnostic UNIQUE
--     (run_id, node_id) is REPLACED by a partial UNIQUE INDEX that only
--     enforces uniqueness for in-flight rows (status IN ('pending',
--     'processing')) so legitimate re-submissions across retries don't 409.
--
-- Idempotent: `CREATE TABLE IF NOT EXISTS` + named-constraint forms.

CREATE TABLE IF NOT EXISTS workflow_input_submissions (
    id             TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES workflow_runs(id)
                                   ON DELETE CASCADE,
    node_id        TEXT NOT NULL,
    value          TEXT,
    status         TEXT NOT NULL DEFAULT 'pending',
    error          TEXT,
    claimed_at     TEXT,
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    processed_at   TEXT,
    CONSTRAINT workflow_input_submissions_status_check
        CHECK (status IN ('pending', 'processing', 'processed', 'failed'))
);

CREATE INDEX IF NOT EXISTS workflow_input_submissions_pending_idx
    ON workflow_input_submissions (created_at)
    WHERE status = 'pending';

-- Partial UNIQUE only for in-flight rows.
CREATE UNIQUE INDEX IF NOT EXISTS workflow_input_submissions_inflight_unique
    ON workflow_input_submissions (run_id, node_id)
    WHERE status IN ('pending', 'processing');
