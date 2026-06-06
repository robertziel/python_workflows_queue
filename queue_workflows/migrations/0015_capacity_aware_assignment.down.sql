-- Revert 0015. Drop the index + capacity/flag columns, and restore the 0011
-- event_type CHECK (without 'unassignable').
DROP INDEX IF EXISTS workflow_node_jobs_unassignable_idx;

ALTER TABLE workflow_node_jobs DROP COLUMN IF EXISTS unassignable_reason;
ALTER TABLE workflow_node_jobs DROP COLUMN IF EXISTS unassignable_at;

ALTER TABLE worker_heartbeats DROP COLUMN IF EXISTS fits_models;
ALTER TABLE worker_heartbeats DROP COLUMN IF EXISTS vram_total_mb;

ALTER TABLE workflow_node_events
    DROP CONSTRAINT IF EXISTS workflow_node_events_type_check;

ALTER TABLE workflow_node_events
    ADD CONSTRAINT workflow_node_events_type_check CHECK (event_type IN (
        'claimed', 'model_load_start', 'model_load_done', 'progress_beat',
        'stall_suspected', 'stall_trip', 'gpu_health_trip', 'budget_trip',
        'requeued', 'reassigned', 'lease_renew', 'completed', 'failed',
        'cancelled', 'error'
    ));
