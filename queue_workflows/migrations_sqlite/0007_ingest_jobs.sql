-- queue_workflows 0007 — ingest_jobs (claim-able periodic-work rows)
--
-- A periodic ingest unit (the host's run_fetch_all / run_load_all / … ) is a
-- STANDALONE callable, not a DAG node: no parent workflow_runs row, no $from
-- inputs, no dispatch-event outbox. A dedicated table carries the SAME
-- claim/lease columns as workflow_node_jobs so the lease-renew / reclaim
-- machinery is reused at the SQL-shape level.
--
-- ENGINE-OWNED but TASK-NAME-AGNOSTIC: the engine relaxes queue + status checks only;
-- the host validates task_name against its registered dispatch map BEFORE enqueue.

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id                TEXT NOT NULL,
    task_name         TEXT NOT NULL,
    queue             TEXT NOT NULL,
    reason            TEXT NOT NULL DEFAULT 'tick',
    status            TEXT NOT NULL DEFAULT 'queued',
    priority          INTEGER NOT NULL DEFAULT 100,
    result            TEXT,
    error             TEXT,
    seconds           REAL,
    claimed_by        TEXT,
    lease_expires_at  TEXT,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    started_at        TEXT,
    finished_at       TEXT,

    PRIMARY KEY (id),
    -- NB: NO queue CHECK here; engine is task-agnostic and the host validates before enqueue.
    CONSTRAINT ingest_jobs_status_check
        CHECK (status IN ('queued','running','completed','failed','cancelled'))
);

CREATE INDEX IF NOT EXISTS ingest_jobs_claim_idx
    ON ingest_jobs (queue, priority, created_at)
    WHERE status = 'queued';

CREATE INDEX IF NOT EXISTS ingest_jobs_lease_idx
    ON ingest_jobs (lease_expires_at)
    WHERE status = 'running';
