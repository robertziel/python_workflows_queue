DROP TRIGGER IF EXISTS worker_llm_config_notify ON worker_controls;
DROP FUNCTION IF EXISTS notify_worker_llm_config();
ALTER TABLE worker_controls
    DROP COLUMN IF EXISTS vllm_idle_ttl_s,
    DROP COLUMN IF EXISTS llm_parallelism,
    DROP COLUMN IF EXISTS llm_server_type;
