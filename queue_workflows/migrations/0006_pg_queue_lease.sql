-- queue_workflows 0006 — PG-as-queue lease + wake primitives (= old 042).
--
--   * claimed_by        — which worker host owns the in-flight lease.
--   * lease_expires_at  — when the lease lapses; a reclaim sweep re-queues
--                         `running` rows past this point.
--   * ..._lease_idx     — partial index over the reclaim predicate.
--   * notify_node_job_ready() + node_job_ready_notify trigger — fire
--                         pg_notify('node_job_ready', <queue>) whenever a row
--                         becomes `queued` (fresh INSERT or a status flip back
--                         to queued, e.g. a lease reclaim). Lets idle claim
--                         workers block on LISTEN instead of polling; the
--                         NOTIFY rides inside the writer's txn so there's no
--                         "row queued but no wake" window.
--
-- Idempotent (`ADD COLUMN IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`,
-- `CREATE OR REPLACE FUNCTION`, `DROP TRIGGER IF EXISTS` + `CREATE TRIGGER`) so
-- re-running on the live ai_leads DB is a no-op.

ALTER TABLE workflow_node_jobs
    ADD COLUMN IF NOT EXISTS claimed_by text,
    ADD COLUMN IF NOT EXISTS lease_expires_at timestamptz;

CREATE INDEX IF NOT EXISTS workflow_node_jobs_lease_idx
    ON workflow_node_jobs (lease_expires_at)
    WHERE status = 'running';

CREATE OR REPLACE FUNCTION notify_node_job_ready() RETURNS trigger AS $$
BEGIN
    IF NEW.status = 'queued' THEN
        PERFORM pg_notify('node_job_ready', NEW.queue);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS node_job_ready_notify ON workflow_node_jobs;
CREATE TRIGGER node_job_ready_notify
    AFTER INSERT OR UPDATE OF status ON workflow_node_jobs
    FOR EACH ROW EXECUTE FUNCTION notify_node_job_ready();
