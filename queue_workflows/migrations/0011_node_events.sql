-- queue_workflows 0011 — workflow_node_events (durable per-node, per-attempt
-- event history).
--
-- WHY. workflow_node_jobs is a single MUTABLE row per (run_id, node_id): a
-- watchdog re-queue (0010) overwrites started_at / claimed_by / host_label and
-- only bumps watchdog_retries, so attempt N-1's worker, timing, and trip reason
-- are lost the instant attempt N is claimed. The rich lifecycle signals (claim,
-- model-load, stall suspected/tripped, gpu-health / budget trip, requeue,
-- reassign, terminal) today hit ONLY the worker logs. This append-only table
-- keeps them durably so cross-attempt failure / stall history is queryable
-- after the fact (surfaced as a per-node timeline in the workflow graph).
--
-- DESIGN. Append-only — no UPDATE path, no new mutation invariant. The terminal
-- + requeue events are written in the SAME txn as the state change (the proven
-- outbox-atomicity pattern, mirroring workflow_dispatch_events, 0004); every
-- other event is best-effort (own connection, swallow-on-failure) so an
-- event-write blip can NEVER fail the load-bearing claim / terminal / watchdog
-- path. ``attempt`` (= workflow_node_jobs.watchdog_retries at emit time) is the
-- cross-attempt key tying the N tries of one node together. ``detail`` carries
-- the free-form trip metrics the watchdogs already compute (max_sm_pct,
-- ram_anchor_mb, ram_now_mb, budget_s, exit_code, model_load_s, …).
--
-- ON DELETE CASCADE from workflow_runs keeps purge / restart_from clean (the
-- same cascade workflow_dispatch_events / workflow_node_jobs already get). Old
-- rows are pruned by prune_node_events(retention_days) on a NodePool sweep.
--
-- Additive + idempotent (IF NOT EXISTS) so re-running on the live ai_leads DB
-- is a safe no-op.

CREATE TABLE IF NOT EXISTS workflow_node_events (
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES workflow_runs(id)
                                ON DELETE CASCADE,
    node_id     TEXT NOT NULL,            -- logical id inside the workflow JSON
    job_id      TEXT,                     -- workflow_node_jobs.id at emit (nullable: survives row churn)
    attempt     SMALLINT NOT NULL DEFAULT 0,  -- = watchdog_retries at emit (cross-attempt key)
    event_type  TEXT NOT NULL,
    host_label  TEXT,                     -- emitting / claiming worker
    queue       TEXT,                     -- cpu / gpu / fetch / load
    model       TEXT,                     -- required_model, when relevant
    elapsed_s   DOUBLE PRECISION,         -- seconds in this attempt (trips / terminal)
    error       TEXT,                     -- trip reason / failure text (truncated)
    detail      JSONB NOT NULL DEFAULT '{}'::jsonb,  -- free-form metrics
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT workflow_node_events_type_check CHECK (event_type IN (
        'claimed', 'model_load_start', 'model_load_done', 'progress_beat',
        'stall_suspected', 'stall_trip', 'gpu_health_trip', 'budget_trip',
        'requeued', 'reassigned', 'lease_renew', 'completed', 'failed',
        'cancelled', 'error'
    ))
);

-- The hot read: one node's timeline, oldest→newest.
CREATE INDEX IF NOT EXISTS workflow_node_events_node_idx
    ON workflow_node_events (run_id, node_id, created_at);

-- Retention-sweep predicate (prune_node_events scans by age).
CREATE INDEX IF NOT EXISTS workflow_node_events_created_idx
    ON workflow_node_events (created_at);
