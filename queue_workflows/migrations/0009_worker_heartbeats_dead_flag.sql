-- queue_workflows 0009 — worker_heartbeats.last_flagged_dead_at (stale-worker
-- recovery marker).
--
-- WHY. A GPU hardware-hang can leave the claim-worker PROCESS wedged even when
-- the JOB it held is already recovered: a torch/HIP call wedged in the dead GPU
-- context blocks the worker (PID 1), its in-process GpuHealthWatchdog daemon
-- thread can't make progress (or its trip signal never satisfies because the
-- subprocess RAM is static AND the box-level GPU probe still reads the sidecar),
-- and so its `worker_heartbeats.last_seen` FREEZES — it stops claiming overflow
-- work, silently. The DB lease-reclaim re-queues the JOB onto a healthy host
-- (good), but nothing flags the dead PROCESS so an operator/host-supervisor can
-- bounce it.
--
-- The orchestrator (NodePool) is a SEPARATE process — GIL-independent of the
-- wedged worker — so it CAN observe the stale heartbeat. ``last_flagged_dead_at``
-- is the durable, queryable marker it stamps when it detects a worker whose
-- ``last_seen`` is stale WHILE that worker still owns a ``running`` job. It makes
-- the flag idempotent (the 0.5 s orchestrator tick re-flags only after the row
-- recovers + goes stale again, instead of logging every tick) and gives a
-- host-supervisor a single column to poll:
--
--     SELECT host_label, queue FROM worker_heartbeats
--      WHERE last_flagged_dead_at IS NOT NULL
--        AND last_flagged_dead_at > now() - interval '2 minutes';
--
-- Nullable, no default: a healthy worker never carries a flag. A worker that
-- recovers (its heartbeat refreshes) has the flag CLEARED by the same heartbeat
-- upsert, so a future hang re-flags cleanly.
--
-- Idempotent (``IF NOT EXISTS``) so re-running on the live ai_leads DB is a
-- no-op.

ALTER TABLE worker_heartbeats
    ADD COLUMN IF NOT EXISTS last_flagged_dead_at TIMESTAMPTZ;

-- Supports the supervisor's "which workers are flagged dead recently" poll.
CREATE INDEX IF NOT EXISTS worker_heartbeats_flagged_dead_idx
    ON worker_heartbeats (last_flagged_dead_at)
    WHERE last_flagged_dead_at IS NOT NULL;
