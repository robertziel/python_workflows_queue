-- Create worker_controls table with operator-controlled desired state per worker.
-- TIMESTAMPTZ converted to TEXT; now() converted to datetime('now').
-- Notify function and trigger omitted in SQLite (polling replaces LISTEN/NOTIFY).

CREATE TABLE IF NOT EXISTS worker_controls (
    host_label    TEXT NOT NULL,
    queue         TEXT NOT NULL,
    desired_state TEXT NOT NULL DEFAULT 'on',
    stop_policy   TEXT NOT NULL DEFAULT 'hard',
    requested_by  TEXT,
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),

    PRIMARY KEY (host_label, queue),
    CHECK (desired_state IN ('on', 'off'))
);
