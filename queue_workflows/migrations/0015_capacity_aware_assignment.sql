-- queue_workflows 0015 — capacity-aware GPU model assignment + unassignable flag.
--
-- WHY. A GPU "model" job names a model whose weights need N GB of VRAM
-- (ModelSpec.est_vram_gb). Today ANY gpu worker can claim ANY gpu job — the
-- capability gate filters only by known_models (the id is registered), NOT by
-- whether the claiming machine can actually hold the model. A model larger than
-- a machine's VRAM is claimed anyway and then fails / OOMs at load. And when NO
-- machine in the fleet is big enough, the node sits queued forever with no
-- visible reason. This migration adds the two data shapes that fix both:
--
--   worker_heartbeats.vram_total_mb — the machine's TOTAL GPU VRAM (MB), sampled
--     by the worker each heartbeat. NULL on a worker that hasn't advertised /
--     a non-GPU queue. Telemetry + the human-readable side of the red-flag event.
--
--   worker_heartbeats.fits_models — text[] of the registered model ids whose
--     est_vram_gb FITS this machine's vram_total_mb (worker-computed from the
--     registry it holds; the orchestrator has no registry, so the fit decision
--     is pushed to the worker and advertised as plain data). The claim gate
--     passes a worker's fits_models as its known_models, and the fleet
--     "no machine can run this model" check is a pure-SQL test of whether ANY
--     fresh heartbeat lists the model here. DEFAULT '{}' so a pre-capacity row
--     advertises "fits nothing" until a capacity-aware worker rewrites it — but
--     the claim gate falls back to claim-any on an EMPTY advertised set (a cold
--     worker must not wedge the queue), so '{}' is safe.
--
--   workflow_node_jobs.unassignable_at / unassignable_reason — the RED FLAG. A
--     queued gpu model-job that no live machine can fit is stamped here (NOT
--     failed, NOT a new lifecycle status): the node stays 'queued' because the
--     condition is transient — a big-enough machine can come online and the
--     flag clears. Surfaced in the queue UI as a red flag + emitted as an
--     'unassignable' node event.
--
-- event_type CHECK gains 'unassignable' (the per-node event the fleet sweep
-- emits, once, when it first flags a node — joins the existing 0011 timeline).
--
-- Additive + idempotent (IF NOT EXISTS / drop-then-add the CHECK) so re-running
-- on the live ai_leads DB is a safe no-op.

ALTER TABLE worker_heartbeats
    ADD COLUMN IF NOT EXISTS vram_total_mb INTEGER;

ALTER TABLE worker_heartbeats
    ADD COLUMN IF NOT EXISTS fits_models text[] NOT NULL DEFAULT '{}';

ALTER TABLE workflow_node_jobs
    ADD COLUMN IF NOT EXISTS unassignable_at TIMESTAMPTZ;

ALTER TABLE workflow_node_jobs
    ADD COLUMN IF NOT EXISTS unassignable_reason TEXT;

-- Extend the per-node event vocabulary with 'unassignable'. Drop + recreate the
-- CHECK so the migration is idempotent and the full allowed set stays in one
-- place (mirrors the 0011 definition + the new value).
ALTER TABLE workflow_node_events
    DROP CONSTRAINT IF EXISTS workflow_node_events_type_check;

ALTER TABLE workflow_node_events
    ADD CONSTRAINT workflow_node_events_type_check CHECK (event_type IN (
        'claimed', 'model_load_start', 'model_load_done', 'progress_beat',
        'stall_suspected', 'stall_trip', 'gpu_health_trip', 'budget_trip',
        'requeued', 'reassigned', 'lease_renew', 'completed', 'failed',
        'cancelled', 'error', 'unassignable'
    ));

-- Hot read for the fleet sweep: queued gpu model-jobs not yet flagged.
CREATE INDEX IF NOT EXISTS workflow_node_jobs_unassignable_idx
    ON workflow_node_jobs (queue, status)
    WHERE required_model IS NOT NULL;
