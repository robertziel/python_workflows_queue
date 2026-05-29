-- queue_workflows 0012 — worker_controls (operator ON/OFF desired-state per worker).
--
-- WHY a NEW table, not a column on worker_heartbeats. ``worker_heartbeats`` is
-- *observed* state: a live worker upserts its row every ~10 s and a stopped one
-- simply ages out of the freshness window (no DELETE on shutdown). But an OFF
-- control state must PERSIST precisely while the worker is NOT beating — exactly
-- when its heartbeat row would be aging out — so it cannot live there. This table
-- is *desired* state, written by an operator / Rails (which shares the DB) and
-- read by the worker; keeping the two apart also avoids the heartbeat upsert
-- clobbering an operator's control write (and vice-versa).
--
-- Keyed ``(host_label, queue)`` — the SAME identity worker_heartbeats and the
-- claim's ``claimed_by``/``queue`` use. A host runs several workers under one
-- ``host_label`` (host-c runs a cpu AND a gpu worker), so control MUST be
-- per-queue: turning off "host-a gpu" must not touch "host-a cpu".
--
--   * desired_state — 'on' | 'off'. A stable two-value enum ⇒ a DB CHECK.
--   * stop_policy   — how to transition on→off. 'hard' (kill in-flight + free
--                     RAM now) is the only one implemented today; 'drain' /
--                     'pause' will slot in later. Deliberately FREE-FORM TEXT
--                     (no CHECK) so a future policy needs NO migration — the
--                     worker validates it against its in-code STOP_POLICIES
--                     registry (queue_workflows.worker_control).
--   * requested_by  — provenance (operator / service name); informational.
--
-- The trigger NOTIFYs ``worker_control`` (payload ``host_label:queue``) on every
-- write so the worker's WorkerControlWatcher wakes immediately — mirroring
-- migration 0006/0007's node_job_ready / ingest_job_ready triggers, so a plain
-- INSERT/UPDATE from Rails wakes the worker with no app-side NOTIFY code and the
-- wake rides the writer's txn (no "row written but no wake" window).
--
-- Idempotent (``CREATE TABLE IF NOT EXISTS`` / ``CREATE OR REPLACE FUNCTION`` /
-- ``DROP TRIGGER IF EXISTS``) so re-running on the live ai_leads DB is a no-op.

CREATE TABLE IF NOT EXISTS worker_controls (
    host_label    TEXT NOT NULL,
    queue         TEXT NOT NULL,
    desired_state TEXT NOT NULL DEFAULT 'on',
    stop_policy   TEXT NOT NULL DEFAULT 'hard',
    requested_by  TEXT,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (host_label, queue),
    CONSTRAINT worker_controls_desired_state_check
        CHECK (desired_state IN ('on', 'off'))
);

CREATE OR REPLACE FUNCTION notify_worker_control() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('worker_control', NEW.host_label || ':' || NEW.queue);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS worker_control_notify ON worker_controls;
CREATE TRIGGER worker_control_notify
    AFTER INSERT OR UPDATE ON worker_controls
    FOR EACH ROW EXECUTE FUNCTION notify_worker_control();
