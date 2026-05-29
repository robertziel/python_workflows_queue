-- queue_workflows 0008 — multi-tenant ingest: per-job args + host-defined queue
-- names.
--
-- All additive + backward-compatible: ai_leads' fetch/load queues and its
-- no-args enqueues keep working unchanged. The only behavioural change is that
-- the queue-name allow-list moves from a DB CHECK to host-side validation —
-- exactly the move migration 0007 already made for task_name (the host
-- validates against its registered set before enqueue). This lets a non-ingest
-- domain (a second consumer — a non-DAG forecast service) route its own queues
-- (ingest/hydro/hydraulic/gpu) and carry per-job arguments, without forking the
-- shared engine.
--
-- Idempotent (`IF NOT EXISTS` / `IF EXISTS`) so re-running on the live ai_leads
-- DB is a no-op.

-- (G2) Per-job arguments for parametrised ingest tasks — e.g. a host's
-- run_scenario(scenario_id). DEFAULT '{}' so every existing INSERT stays valid
-- and ai_leads' periodic sweeps (which carry no args) are unaffected.
ALTER TABLE ingest_jobs
    ADD COLUMN IF NOT EXISTS args JSONB NOT NULL DEFAULT '{}'::jsonb;

-- (G1) Queue name is host-defined. Drop the hardcoded fetch/load CHECK; the
-- host validates `queue` against its registered ingest-queue set before enqueue
-- (node_queue.enqueue_ingest_job), mirroring the 0007 task_name gate. The
-- claim index on (queue, priority, created_at) keeps per-queue claims fast for
-- any queue name.
ALTER TABLE ingest_jobs
    DROP CONSTRAINT IF EXISTS ingest_jobs_queue_check;

-- (G5) Allow heartbeat rows for ingest-family workers (not just cpu/gpu) so a
-- host's queue snapshot can read live-worker counts per queue. Drop the
-- cpu/gpu-only CHECK; the (host_label, queue) primary key still enforces one
-- row per worker-queue.
ALTER TABLE worker_heartbeats
    DROP CONSTRAINT IF EXISTS worker_heartbeats_queue_check;
