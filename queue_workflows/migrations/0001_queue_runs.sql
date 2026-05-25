-- queue_workflows 0001 — workflow_runs + workflow_run_files.
--
-- Runs are the queue's substrate — workers claim from `workflow_runs` via
-- `FOR UPDATE SKIP LOCKED` on (status='queued', priority, queued_at).
--
-- ENGINE-OWNED + parcel-agnostic: the original ai_leads migration 001 also
-- created `parcels` and FK'd `workflow_runs.parcel_id` to it. The engine drops
-- that FK (plan §5b) — `parcel_id` is a plain nullable TEXT here so the queue
-- schema stands alone on a parcel-less DB. A host that wants the FK adds it in
-- a domain migration that runs after both `parcels` and `workflow_runs` exist.
--
-- Idempotent (plan §5b): every CREATE uses `IF NOT EXISTS` so re-running this
-- chain on an already-populated DB (the ai_leads cutover) is a no-op. The
-- vestigial `current_step_idx` column (dropped by old migration 007) is
-- omitted entirely — fresh engine DBs never had it.

CREATE TABLE IF NOT EXISTS workflow_runs (
    id                 TEXT PRIMARY KEY,
    parcel_id          TEXT,
    workflow_name      TEXT NOT NULL,
    status             TEXT NOT NULL,
    priority           SMALLINT NOT NULL DEFAULT 100,
    current_step_id    TEXT,
    progress_pct       REAL NOT NULL DEFAULT 0.0,
    steps_done         JSONB NOT NULL DEFAULT '[]'::jsonb,
    context            JSONB NOT NULL DEFAULT '{}'::jsonb,
    input_spec         JSONB,
    error              TEXT,
    out_dir            TEXT,
    mode               TEXT NOT NULL DEFAULT 'step'
                            CHECK (mode IN ('step', 'node')),
    resume_count       SMALLINT NOT NULL DEFAULT 0,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    queued_at          TIMESTAMPTZ,
    started_at         TIMESTAMPTZ,
    finished_at        TIMESTAMPTZ
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
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
    step_id     TEXT NOT NULL,
    rel_path    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    size_bytes  BIGINT NOT NULL DEFAULT 0,
    is_primary  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, rel_path)
);

CREATE INDEX IF NOT EXISTS workflow_run_files_run_idx ON workflow_run_files (run_id);
CREATE INDEX IF NOT EXISTS workflow_run_files_kind_idx ON workflow_run_files (kind);
