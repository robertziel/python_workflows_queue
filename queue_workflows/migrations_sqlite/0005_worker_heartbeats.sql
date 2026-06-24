-- queue_workflows 0005 — worker_heartbeats (per-host fleet capacity ledger).
--
-- Each claim worker upserts its (host_label, queue) row at startup and
-- refreshes last_seen every ~10s. Sums concurrency across rows whose
-- last_seen is recent enough — the denominator behind capacity metrics
-- in the queue pill. A stopped worker stops refreshing and falls out of the
-- SUM filter within 30s (no DELETE on shutdown needed).
--
-- CONSOLIDATED final shape (old 031 + 032 + 037):
--   * current_model (old 032)  — GPU model-affinity hint for sticky routing.
--   * known_models (old 037)   — capability advertisement (model_registry
--                                .known_ids() snapshot) for pre-flight
--                                validation + affinity routing.
--
-- Idempotent (`IF NOT EXISTS`) so re-running on the live database is a
-- no-op.

CREATE TABLE IF NOT EXISTS worker_heartbeats (
    host_label    TEXT NOT NULL,
    queue         TEXT NOT NULL,
    concurrency   INTEGER NOT NULL,
    current_model TEXT,
    known_models  TEXT NOT NULL DEFAULT '[]',
    last_seen     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    PRIMARY KEY (host_label, queue)
);

-- Supports staleness filter (`last_seen > datetime('now', '-30 seconds')`).
CREATE INDEX IF NOT EXISTS worker_heartbeats_last_seen_idx
    ON worker_heartbeats (last_seen);
