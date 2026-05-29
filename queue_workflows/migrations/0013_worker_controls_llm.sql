-- queue_workflows 0013 — per-machine LLM server config on worker_controls.
--
-- WHY columns on worker_controls (not a new table). The LLM server a host runs
-- is DESIRED, operator-set state with the SAME key as the ON/OFF switch —
-- (host_label, queue) — and is read by the SAME worker that reads desired_state.
-- It belongs next to it, written by the same Rails/operator path. Other consumer
-- projects read worker_controls only through the engine accessors (never
-- SELECT *), so widening the row is safe for them.
--
--   * llm_server_type — 'ollama' | 'vllm'. A stable two-value enum ⇒ a DB CHECK.
--                       DEFAULT 'ollama' so every pre-existing / freshly-written
--                       row keeps today's behaviour with no backfill.
--   * llm_parallelism — concurrent requests the SIDECAR serves (ollama
--                       OLLAMA_NUM_PARALLEL / vllm --max-num-seqs). This is NOT
--                       the claim-worker concurrency (which is 1 by contract) —
--                       it is a property of the LLM server process, surfaced in
--                       the queue UI as an editable "par N". DEFAULT 1, CHECK >= 1.
--   * vllm_idle_ttl_s — seconds of zero LLM requests before the supervisor
--                       SIGTERMs the vllm sidecar to free VRAM (docker
--                       restart: unless-stopped brings it back on next request).
--                       Ignored for ollama (its own KEEP_ALIVE handles idle).
--                       DEFAULT 60, CHECK >= 0 (0 disables the idle reap).
--
-- Each ADD COLUMN carries IF NOT EXISTS so the whole clause (column + its inline
-- CHECK) is a no-op on re-run — idempotent like the rest of the chain.
--
-- A SECOND trigger NOTIFYs a DEDICATED channel ``worker_llm_config_changed``
-- (payload ``host_label|queue`` — note the ``|`` separator, distinct from the
-- 0012 ``worker_control`` channel's ``:``). The backend factory LISTENs this for
-- instant refresh. It is deliberately a different channel from ``worker_control``
-- so an LLM-config edit does NOT look like an ON/OFF change to the hard-stop
-- WorkerControlWatcher; the function stays quiet on an UPDATE that changes none
-- of the three LLM columns, so unrelated desired_state writes don't spam it.

ALTER TABLE worker_controls
    ADD COLUMN IF NOT EXISTS llm_server_type TEXT NOT NULL DEFAULT 'ollama'
        CHECK (llm_server_type IN ('ollama', 'vllm')),
    ADD COLUMN IF NOT EXISTS llm_parallelism INTEGER NOT NULL DEFAULT 1
        CHECK (llm_parallelism >= 1),
    ADD COLUMN IF NOT EXISTS vllm_idle_ttl_s INTEGER NOT NULL DEFAULT 60
        CHECK (vllm_idle_ttl_s >= 0);

CREATE OR REPLACE FUNCTION notify_worker_llm_config() RETURNS trigger AS $$
BEGIN
    -- Stay quiet on an UPDATE that touches none of the LLM columns (e.g. a plain
    -- desired_state on/off write) so the config channel only carries real config
    -- changes. INSERTs always fire (a new row IS a config event).
    IF TG_OP = 'UPDATE'
       AND NEW.llm_server_type IS NOT DISTINCT FROM OLD.llm_server_type
       AND NEW.llm_parallelism IS NOT DISTINCT FROM OLD.llm_parallelism
       AND NEW.vllm_idle_ttl_s IS NOT DISTINCT FROM OLD.vllm_idle_ttl_s THEN
        RETURN NEW;
    END IF;
    PERFORM pg_notify('worker_llm_config_changed', NEW.host_label || '|' || NEW.queue);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS worker_llm_config_notify ON worker_controls;
CREATE TRIGGER worker_llm_config_notify
    AFTER INSERT OR UPDATE ON worker_controls
    FOR EACH ROW EXECUTE FUNCTION notify_worker_llm_config();
