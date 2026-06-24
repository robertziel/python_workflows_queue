-- Reverse 0017 — drop the project tenant tag + restore the 2-col heartbeat PK.
--
-- SAFE ONLY ON A SINGLE-TENANT (all-'') DB. If multi-tenancy was actually used
-- (two projects sharing the same host_label+queue), dropping `project` collapses
-- those into duplicate (host_label, queue) rows and the 2-col PK re-add throws —
-- inherent to reversing multi-tenant data into a single-tenant shape. The forward
-- path and the empty-DB migration roundtrip are unaffected.
DROP INDEX IF EXISTS workflow_runs_project_idx;
DROP INDEX IF EXISTS ingest_jobs_project_claim_idx;
DROP INDEX IF EXISTS workflow_node_jobs_project_claim_idx;

ALTER TABLE worker_heartbeats DROP CONSTRAINT IF EXISTS worker_heartbeats_pkey;
ALTER TABLE worker_heartbeats DROP COLUMN IF EXISTS project;
ALTER TABLE worker_heartbeats
    ADD CONSTRAINT worker_heartbeats_pkey PRIMARY KEY (host_label, queue);

ALTER TABLE ingest_jobs DROP COLUMN IF EXISTS project;
ALTER TABLE workflow_node_jobs DROP COLUMN IF EXISTS project;
ALTER TABLE workflow_runs DROP COLUMN IF EXISTS project;
