-- Reverse 0017 — drop the project tenant tag + restore the 2-col heartbeat PK

DROP INDEX IF EXISTS workflow_runs_project_idx;
DROP INDEX IF EXISTS ingest_jobs_project_claim_idx;
DROP INDEX IF EXISTS workflow_node_jobs_project_claim_idx;

-- Rebuild worker_heartbeats to remove project column and restore 2-col PK
-- (SAFE ONLY ON SINGLE-TENANT DB where all project values are empty string)
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
    PRIMARY KEY (host_label, queue)
);

INSERT INTO worker_heartbeats_new
    SELECT host_label, queue, concurrency, current_model, known_models,
           last_seen, last_flagged_dead_at, llm_servers_available,
           vram_total_mb, fits_models
    FROM worker_heartbeats;

DROP TABLE worker_heartbeats;

ALTER TABLE worker_heartbeats_new RENAME TO worker_heartbeats;

-- Recreate indexes on worker_heartbeats (from migrations 0005 and 0009)
CREATE INDEX IF NOT EXISTS worker_heartbeats_last_seen_idx
    ON worker_heartbeats (last_seen);

CREATE INDEX IF NOT EXISTS worker_heartbeats_flagged_dead_idx
    ON worker_heartbeats (last_flagged_dead_at)
    WHERE last_flagged_dead_at IS NOT NULL;

-- Drop project columns from other tables
ALTER TABLE ingest_jobs DROP COLUMN project;
ALTER TABLE workflow_node_jobs DROP COLUMN project;
ALTER TABLE workflow_runs DROP COLUMN project;
