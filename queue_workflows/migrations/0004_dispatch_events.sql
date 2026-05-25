-- queue_workflows 0004 — workflow_dispatch_events (durable dispatcher outbox).
--
-- A worker writes a `workflow_dispatch_events` row in the SAME txn as its
-- terminal mark_completed / mark_failed / mark_awaiting_input. NodePool._tick
-- drains unprocessed events on every dispatch cycle and invokes the dispatcher
-- callback (on_node_completed / on_node_failed / on_node_awaiting_input). On
-- callback failure the row stays processed_at IS NULL with attempts++; the
-- next tick retries. Exhausted retries flip the run to failed so the user sees
-- something instead of a stall.
--
-- = old migration 015. Idempotent (`IF NOT EXISTS`) so re-running on the live
-- ai_leads DB is a no-op.

CREATE TABLE IF NOT EXISTS workflow_dispatch_events (
    id           BIGSERIAL PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES workflow_runs(id)
                                 ON DELETE CASCADE,
    node_id      TEXT NOT NULL,
    kind         TEXT NOT NULL,
    processed_at TIMESTAMPTZ,
    error        TEXT,
    attempts     SMALLINT NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT workflow_dispatch_events_kind_check
        CHECK (kind IN ('completed', 'failed', 'awaiting_input'))
);

CREATE INDEX IF NOT EXISTS workflow_dispatch_events_unprocessed_idx
    ON workflow_dispatch_events (created_at)
    WHERE processed_at IS NULL;
