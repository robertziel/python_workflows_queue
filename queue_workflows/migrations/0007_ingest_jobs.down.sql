-- Revert 0007: drop the ingest-job wake trigger + function, indexes, table.
DROP TRIGGER IF EXISTS ingest_job_ready_notify ON ingest_jobs;
DROP FUNCTION IF EXISTS notify_ingest_job_ready();
DROP INDEX IF EXISTS ingest_jobs_lease_idx;
DROP INDEX IF EXISTS ingest_jobs_claim_idx;
DROP TABLE IF EXISTS ingest_jobs CASCADE;
