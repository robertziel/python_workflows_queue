-- 0017_project_tenant (SQLite)
-- Add project tenant identity to queue records and update worker_heartbeats PK

-- ── tenant tag on the queue records ────────────────────────────────────────
ALTER TABLE workflow_runs
    ADD COLUMN project TEXT NOT NULL DEFAULT '';

ALTER TABLE workflow_node_jobs
    ADD COLUMN project TEXT NOT NULL DEFAULT '';

ALTER TABLE ingest_jobs
    ADD COLUMN project TEXT NOT NULL DEFAULT '';

ALTER TABLE worker_heartbeats
    ADD COLUMN project TEXT NOT NULL DEFAULT '';

-- ── worker_heartbeats identity now includes project ────────────────────────
-- Rebuild table to update PRIMARY KEY from (host_label, queue) to (host_label, queue, project)
-- SQLite requires table rebuild for PK changes (ALTER TABLE DROP CONSTRAINT not supported for PKs)
CREATE TABLE worker_heartbeats_new (
    host_label               TEXT NOT NULL,
    queue                    TEXT NOT NULL,
    concurrency              INTEGER NOT NULL,
    current_model            TEXT,
    known_models             TEXT NOT NULL DEFAULT '[]',
    last_seen                TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    last_flagged_dead_at     TEXT,
    llm_servers_available    TEXT NOT NULL DEFAULT '["ollama"]',
    vram_total_mb            INTEGER,
    fits_models              TEXT NOT NULL DEFAULT '[]',
    project                  TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (host_label, queue, project)
);

INSERT INTO worker_heartbeats_new
    SELECT host_label, queue, concurrency, current_model, known_models,
           last_seen, last_flagged_dead_at, llm_servers_available,
           vram_total_mb, fits_models, project
    FROM worker_heartbeats;

DROP TABLE worker_heartbeats;

ALTER TABLE worker_heartbeats_new RENAME TO worker_heartbeats;

-- Recreate indexes on worker_heartbeats (from migrations 0005 and 0009)
CREATE INDEX IF NOT EXISTS worker_heartbeats_last_seen_idx
    ON worker_heartbeats (last_seen);

CREATE INDEX IF NOT EXISTS worker_heartbeats_flagged_dead_idx
    ON worker_heartbeats (last_flagged_dead_at)
    WHERE last_flagged_dead_at IS NOT NULL;

-- ── project-aware claim indexes (hot path) ─────────────────────────────────
CREATE INDEX IF NOT EXISTS workflow_node_jobs_project_claim_idx
    ON workflow_node_jobs (queue, project, priority, created_at)
    WHERE status = 'queued';

CREATE INDEX IF NOT EXISTS ingest_jobs_project_claim_idx
    ON ingest_jobs (queue, project, priority, created_at)
    WHERE status = 'queued';

-- Per-project run history / snapshot filter.
CREATE INDEX IF NOT EXISTS workflow_runs_project_idx
    ON workflow_runs (project, status);
