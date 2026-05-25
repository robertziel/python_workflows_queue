-- Revert 0006: drop the wake trigger + function, the lease index, the columns.
DROP TRIGGER IF EXISTS node_job_ready_notify ON workflow_node_jobs;
DROP FUNCTION IF EXISTS notify_node_job_ready();
DROP INDEX IF EXISTS workflow_node_jobs_lease_idx;

ALTER TABLE workflow_node_jobs
    DROP COLUMN IF EXISTS lease_expires_at,
    DROP COLUMN IF EXISTS claimed_by;
