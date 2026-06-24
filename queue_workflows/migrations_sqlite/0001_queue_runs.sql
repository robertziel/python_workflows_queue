-- queue_workflows 0001 — workflow_runs + workflow_run_files.
--
-- Runs are the queue's substrate — workers claim from `workflow_runs` via
-- row-level locks on (status='queued', priority, queued_at).
--
-- ENGINE-OWNED + parcel-agnostic: parcel_id is a plain nullable TEXT so the queue
-- schema stands alone on a parcel-less DB.
--
-- Idempotent: every CREATE uses `IF NOT EXISTS` so re-running this migration on an
-- already-populated DB is a no-op.

CREATE TABLE IF NOT EXISTS workflow_runs (
    id                 TEXT PRIMARY KEY,
    parcel_id          TEXT,
    workflow_name      TEXT NOT NULL,
    status             TEXT NOT NULL,
    priority           INTEGER NOT NULL DEFAULT 100,
    current_step_id    TEXT,
    progress_pct       REAL NOT NULL DEFAULT 0.0,
    steps_done         TEXT NOT NULL DEFAULT '[]',
    context            TEXT NOT NULL DEFAULT '{}',
    input_spec         TEXT,
    error              TEXT,
    out_dir            TEXT,
    mode               TEXT NOT NULL DEFAULT 'step'
                            CHECK (mode IN ('step', 'node')),
    resume_count       INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    updated_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    queued_at          TEXT,
    started_at         TEXT,
    finished_at        TEXT
);

-- Hot path: claim_next_queued() — partial index keeps it tiny.
CREATE INDEX IF NOT EXISTS workflow_runs_claim_idx
    ON workflow_runs (priority, queued_at)
    WHERE status = 'queued';

-- Per-parcel history view.
CREATE INDEX IF NOT EXISTS workflow_runs_parcel_created_idx
    ON workflow_runs (parcel_id, created_at DESC);

CREATE INDEX IF NOT EXISTS workflow_runs_status_idx ON workflow_runs (status);


CREATE TABLE IF NOT EXISTS workflow_run_files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
    step_id     TEXT NOT NULL,
    rel_path    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    is_primary  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    UNIQUE (run_id, rel_path)
);

CREATE INDEX IF NOT EXISTS workflow_run_files_run_idx ON workflow_run_files (run_id);
CREATE INDEX IF NOT EXISTS workflow_run_files_kind_idx ON workflow_run_files (kind);
