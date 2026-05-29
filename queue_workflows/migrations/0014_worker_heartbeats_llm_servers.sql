-- queue_workflows 0014 — advertise each worker's AVAILABLE LLM server types.
--
-- WHY on worker_heartbeats (not worker_controls). This is OBSERVED capability —
-- "which LLM servers can THIS machine actually run" — the direct analog of the
-- existing ``known_models`` advertisement, and it belongs next to it on the
-- heartbeat (observed), NOT on worker_controls (operator-DESIRED state). A worker
-- computes it at startup and re-advertises every ~10 s.
--
-- WHY it matters. vllm here is the CUDA ``vllm/vllm-openai`` sidecar — it runs
-- ONLY on the NVIDIA GB10 hubs, never on the AMD/ROCm control box (host-c). The
-- queue UI reads this list to gate its per-machine server-type control: a host
-- that can't run vllm shows the vllm option DISABLED, so an operator can't toggle
-- a machine onto a server it has no sidecar for (which would route every VLM
-- call at a non-existent endpoint and silently no-op the lane).
--
--   * llm_servers_available — text[] of server types this host can serve, e.g.
--     ``{ollama}`` (the universal baseline) or ``{ollama,vllm}`` (an NVIDIA host
--     with the vllm sidecar rendered). DEFAULT ``{ollama}`` so every existing /
--     freshly-written row is ollama-only until a worker advertises otherwise —
--     no backfill, and consumers that never set it (lm_flood, lm_content_generator)
--     are unaffected.
--
-- IF NOT EXISTS ⇒ idempotent on re-run, like the rest of the chain.

ALTER TABLE worker_heartbeats
    ADD COLUMN IF NOT EXISTS llm_servers_available text[] NOT NULL DEFAULT '{ollama}';
