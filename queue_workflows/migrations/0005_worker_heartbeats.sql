-- queue_workflows 0005 — worker_heartbeats (per-host fleet capacity ledger).
--
-- Each claim worker upserts its (host_label, queue) row at startup and
-- refreshes last_seen every ~10s. Rails sums concurrency across rows whose
-- last_seen is recent enough — the denominator behind `CPU 1/N` / `GPU 0/N`
-- in the queue pill. A stopped worker stops refreshing and falls out of the
-- SUM filter within 30s (no DELETE on shutdown needed).
--
-- CONSOLIDATED final shape (old 031 + 032 + 037):
--   * current_model (old 032)  — GPU model-affinity hint for sticky routing.
--   * known_models (old 037)   — capability advertisement (model_registry
--                                .known_ids() snapshot) for pre-flight
--                                validation + affinity routing.
--
-- Idempotent (`IF NOT EXISTS`) so re-running on the live ai_leads DB is a
-- no-op.

CREATE TABLE IF NOT EXISTS worker_heartbeats (
    host_label    TEXT NOT NULL,
    queue         TEXT NOT NULL,
    concurrency   INTEGER NOT NULL,
    current_model TEXT,                                   -- old 032
    known_models  text[] NOT NULL DEFAULT '{}',           -- old 037
    last_seen     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (host_label, queue),
    CONSTRAINT worker_heartbeats_queue_check CHECK (queue IN ('cpu', 'gpu'))
);

-- Supports Rails' staleness filter (`last_seen > now() - interval '30s'`).
CREATE INDEX IF NOT EXISTS worker_heartbeats_last_seen_idx
    ON worker_heartbeats (last_seen);

-- old 037 — GIN over the array so `known_models @> ARRAY['x']` is O(log n).
CREATE INDEX IF NOT EXISTS worker_heartbeats_known_models_gin
    ON worker_heartbeats USING gin (known_models);
