-- Down for 0011 — drop the per-node event history table.
-- Append-only log with no dependents, so a plain DROP is safe.
DROP TABLE IF EXISTS workflow_node_events;
