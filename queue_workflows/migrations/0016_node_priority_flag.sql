-- queue_workflows 0016 — per-node "run next" priority flag.
--
-- A boolean an operator sets on a QUEUED node so the next worker asking for a
-- node in that queue/capability claims it BEFORE older + default-priority peers.
-- Distinct from the integer ``priority`` (which orders the band, DEFAULT 100):
-- ``is_priority`` is a binary "jump the queue" flag, sorted FIRST in the claim
-- ORDER BY — ahead of the priority band and, on GPU, ahead of the warm-model
-- affinity tiebreak (so a flagged cold-model node preempts a warm one; the model
-- reload is the accepted cost of "run this next"). DEFAULT FALSE so every
-- existing + freshly-written row keeps today's behaviour with no backfill.
ALTER TABLE workflow_node_jobs
    ADD COLUMN is_priority BOOLEAN NOT NULL DEFAULT FALSE;
