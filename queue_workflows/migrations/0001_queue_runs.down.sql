-- Reverse of 0001. Drop the run file table first (FK), then runs.
DROP TABLE IF EXISTS workflow_run_files CASCADE;
DROP TABLE IF EXISTS workflow_runs CASCADE;
