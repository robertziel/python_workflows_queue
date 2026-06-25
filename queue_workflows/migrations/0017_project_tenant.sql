-- queue_workflows 0017 — project (tenant) identity on the queue records.
--
-- WHY. The engine already runs ONE cpu + ONE gpu queue (0002) partitioned by
-- *resource*, never by project — but it assumed a single Postgres per project
-- (per-project deployment topology). To run ONE shared broker DB across all
-- projects (operator decision: "shared DB + per-project clients"), every queue
-- record carries a ``project`` tenant tag: each project's client enqueues +
-- claims ONLY its own project's rows, while the broker holds all projects'
-- jobs on the shared cpu/gpu (+ ingest) queues. Distinct from ``db_namespace``
-- (config.py) — that ISOLATES tenants on a shared redis/mongo so they can't see
-- each other; this column POOLS them into one queue with a filter, the inverse.
--
-- DESIGN — exact-match-always (the claim filters ``project = <client project>``
-- unconditionally). ``DEFAULT ''`` makes a single-tenant deploy byte-compatible:
-- every row is ``''`` and the filter ``project=''`` matches them all, so today's
-- behaviour is unchanged with zero host wiring. A multi-tenant client sets its
-- ``config.project`` and only ever sees its own rows.
--
-- Additive + idempotent (``IF NOT EXISTS`` / drop-then-add the PK) so re-running
-- on an already-migrated DB is a safe no-op. ``NOT NULL DEFAULT ''`` backfills
-- existing rows to the single-tenant sentinel with no separate backfill step.

-- ── tenant tag on the queue records ────────────────────────────────────────
ALTER TABLE workflow_runs
    ADD COLUMN IF NOT EXISTS project TEXT NOT NULL DEFAULT '';

ALTER TABLE workflow_node_jobs
    ADD COLUMN IF NOT EXISTS project TEXT NOT NULL DEFAULT '';

ALTER TABLE ingest_jobs
    ADD COLUMN IF NOT EXISTS project TEXT NOT NULL DEFAULT '';

ALTER TABLE worker_heartbeats
    ADD COLUMN IF NOT EXISTS project TEXT NOT NULL DEFAULT '';

-- ── worker_heartbeats identity now includes project ────────────────────────
-- A shared broker can run two projects' workers on the same machine
-- (``host_label``) + queue (e.g. host-a runs both ai_leads' and alpha's gpu
-- client). The old PK ``(host_label, queue)`` would make the second client's
-- upsert CLOBBER the first's heartbeat. The tenant tag disambiguates them, so
-- the worker identity becomes ``(host_label, queue, project)``. Drop-then-add
-- keeps this idempotent (re-running drops the existing PK and re-adds the same
-- 3-col one).
--
-- BREAKING for raw-SQL heartbeat writers: the 2-col unique constraint is gone.
-- Any consumer that upserts heartbeats with its own
-- ``INSERT … ON CONFLICT (host_label, queue)`` must move to
-- ``node_queue.upsert_worker_heartbeat`` (3-col ON CONFLICT) or it errors with
-- "no unique or exclusion constraint matching the ON CONFLICT specification".
ALTER TABLE worker_heartbeats DROP CONSTRAINT IF EXISTS worker_heartbeats_pkey;
ALTER TABLE worker_heartbeats
    ADD CONSTRAINT worker_heartbeats_pkey PRIMARY KEY (host_label, queue, project);

-- ── project-aware claim indexes (hot path) ─────────────────────────────────
-- The claim subselect now filters ``queue = ? AND status='queued' AND
-- project = ?``; a (queue, project, priority, created_at) partial index keeps it
-- as tight as the original (queue, priority, created_at) one.
CREATE INDEX IF NOT EXISTS workflow_node_jobs_project_claim_idx
    ON workflow_node_jobs (queue, project, priority, created_at)
    WHERE status = 'queued';

CREATE INDEX IF NOT EXISTS ingest_jobs_project_claim_idx
    ON ingest_jobs (queue, project, priority, created_at)
    WHERE status = 'queued';

-- Per-project run history / snapshot filter.
CREATE INDEX IF NOT EXISTS workflow_runs_project_idx
    ON workflow_runs (project, status);
