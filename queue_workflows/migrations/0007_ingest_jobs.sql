-- queue_workflows 0007 — ingest_jobs (claim-able periodic-work rows) (= old 043).
--
-- A periodic ingest unit (the host's run_fetch_all / run_load_all / … ) is a
-- STANDALONE callable, not a DAG node: no parent workflow_runs row, no $from
-- inputs, no dispatch-event outbox. A dedicated table carries the SAME
-- claim/lease columns as workflow_node_jobs so the lease-renew / reclaim
-- machinery is reused at the SQL-shape level, plus its own NOTIFY trigger that
-- wakes the claim worker exactly like node_job_ready does for cpu/gpu.
--
-- ENGINE-OWNED but TASK-NAME-AGNOSTIC (plan §1f): the original ai_leads
-- migration 043 had `CONSTRAINT ingest_jobs_task_check CHECK (task_name IN
-- ('run_fetch_all','run_load_all','audit_freshness'))` — those are ai_leads
-- domain names. The engine RELAXES that CHECK (queue + status CHECK only); the
-- host validates task_name against its registered dispatch map BEFORE enqueue
-- (node_queue.enqueue_ingest_job checks config.ingest_task_map). This keeps
-- ingest_jobs reusable by any project's periodic work.
--
--   * task_name   — the host ingest callable to run.
--   * queue       — 'fetch' | 'load'; the claim worker LISTENs per queue.
--   * reason      — provenance ('tick' | 'boot' | 'manual').
--   * claimed_by / lease_expires_at — identical lease bookkeeping to
--                   workflow_node_jobs so reclaim_expired_leases sweeps both.
--
-- Idempotent (`CREATE TABLE IF NOT EXISTS`, etc.). NB: on the live ai_leads DB
-- the table already exists WITH the task-name CHECK from old 043 — re-running
-- this engine migration is a no-op (IF NOT EXISTS), so it leaves the existing
-- (stricter) CHECK in place there. That is harmless: ai_leads only ever
-- enqueues its three known task names, all of which satisfy the old CHECK; the
-- host-side Python validation is the canonical gate going forward.

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id                TEXT NOT NULL,
    task_name         TEXT NOT NULL,
    queue             TEXT NOT NULL,
    reason            TEXT NOT NULL DEFAULT 'tick',
    status            TEXT NOT NULL DEFAULT 'queued',
    priority          SMALLINT NOT NULL DEFAULT 100,
    result            JSONB,
    error             TEXT,
    seconds           DOUBLE PRECISION,
    claimed_by        TEXT,
    lease_expires_at  TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at        TIMESTAMPTZ,
    finished_at       TIMESTAMPTZ,

    PRIMARY KEY (id),
    CONSTRAINT ingest_jobs_queue_check
        CHECK (queue IN ('fetch','load')),
    -- NB: NO task-name CHECK here (plan §1f) — the host validates task_name
    -- against its registered ingest dispatch map before enqueue.
    CONSTRAINT ingest_jobs_status_check
        CHECK (status IN ('queued','running','completed','failed','cancelled'))
);

CREATE INDEX IF NOT EXISTS ingest_jobs_claim_idx
    ON ingest_jobs (queue, priority, created_at)
    WHERE status = 'queued';

CREATE INDEX IF NOT EXISTS ingest_jobs_lease_idx
    ON ingest_jobs (lease_expires_at)
    WHERE status = 'running';

CREATE OR REPLACE FUNCTION notify_ingest_job_ready() RETURNS trigger AS $$
BEGIN
    IF NEW.status = 'queued' THEN
        PERFORM pg_notify('ingest_job_ready', NEW.queue);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ingest_job_ready_notify ON ingest_jobs;
CREATE TRIGGER ingest_job_ready_notify
    AFTER INSERT OR UPDATE OF status ON ingest_jobs
    FOR EACH ROW EXECUTE FUNCTION notify_ingest_job_ready();
