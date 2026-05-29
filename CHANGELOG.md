# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] — 2026-05-30

### Added
- **Docs: ollama vs vLLM request flow + how a diffusion model shares the GPU** —
  `docs/llm_backends.md` gains two Mermaid request-flow graphs (ollama's
  always-up, self-managed daemon vs vLLM's engine-managed, idle-stopping,
  batching server), a side-by-side comparison table, and a "When a diffusion
  model runs on the same host" section: the two-lane split (the in-process
  diffusion inline lane + the PAR-sized VLM pool), the PAR cap, and why a
  compute-bound diffusion model never goes through ollama/vLLM. The README gains
  a copy-pasteable **Ansible deployment example** (inventory + playbook) that
  wires the per-machine ollama/vLLM choice, capability advertisement, and PAR —
  secrets kept in `ansible-vault`, not the inventory.
- **GPU node-job capacity capped at PAR total (was 1+PAR)** — a GPU machine runs
  1 inline diffusion (concurrency-1) + a PAR-sized VLM pool; the pool feeder now
  budgets `PAR - 1` while the inline diffusion runs (`_pool_budget` +
  an `_inline_running` flag set around the inline `run_once`), so the diffusion
  occupies one of the PAR slots and the machine's TOTAL concurrent node-jobs
  never exceeds PAR. Idle inline ⇒ the full PAR pool. Best-effort (a transient
  race may briefly allow 1+PAR until a pool job drains). Consumer-safe: a worker
  with no inline diffusion keeps the full PAR (flag stays False) — byte-identical
  to before. Makes a consumer's `used/PAR node-jobs` gauge honest (used ≤ PAR).
- **FILL-BEFORE-SPILL packing for the no-model GPU (VLM) pool lane** — VLM jobs now
  bin-pack onto the highest-ranked vLLM machine before spilling, instead of every
  vLLM box claiming independently (which SPREAD VLM work across the fleet). New
  `node_queue.vlm_pool_should_defer(host_label, par, *, stale_s=30)` runs one cheap
  `EXISTS` over `worker_heartbeats` (fresh gpu rows only) LEFT JOINed to a
  per-`claimed_by` COUNT of running no-model gpu jobs, returning `True` iff a FRESH
  gpu peer ranked strictly ABOVE M — by `(concurrency DESC, host_label ASC)` —
  still has free VLM capacity (`R.concurrency > M.par OR (R.concurrency = M.par AND
  R.host_label < M.host)` AND its running-no-model count `< R.concurrency`). The
  GPU pool feeder (`_pool_feeder_loop`) consults it before each no-model claim and
  DEFERS the cycle (does not claim — no claim-then-release) when it returns `True`,
  letting the higher-ranked machine fill its PAR slots first. Under light load this
  consolidates VLM onto one box (freeing the others for diffusion / idle-unload);
  under heavy load it still spills (no throughput loss). Invariants: the top-ranked
  machine has no peer above it → never defers → fills first (no global starvation);
  a single box / no fresh higher peer → `False` ⇒ behaviour byte-identical to today
  (SAFE default for single-box fleets + other library consumers); a STALE higher
  peer (heartbeat older than `stale_s`) is ignored so a dead top box can't block
  everyone; capacity counts `queue='gpu' AND status='running' AND required_model IS
  NULL` only — the inline diffusion lane is neither counted nor affected. A
  defer-query blip in the feeder falls through to the legacy spread behaviour
  (claim now), never crashes the lane. Additive: the inline diffusion lane
  (`_claim`, `require_model=True`) is UNCHANGED and never consults the gate.
- **PAR-driven GPU two-lane concurrency** — the GPU claim worker now runs no-model
  GPU jobs (VLM facade work: HTTP to a per-host vLLM server that batches up to
  `--max-num-seqs` = PAR requests on the GPU) in a PAR-sized `ThreadPoolExecutor`
  pool lane, alongside the UNCHANGED concurrency-1 inline lane for model-backed
  diffusion jobs. A dedicated feeder thread (gpu only, started in `run_forever`)
  claims `require_model=False` jobs and submits them to the pool, gating
  submissions on the live PAR (`worker_control.llm_config_for(host,'gpu').parallelism`,
  clamped ≥ 1, re-read each cycle so a UI change takes effect without a restart)
  so it never exceeds PAR in flight. The pool path (`ClaimWorker._run_pool_node`)
  is a lighter sibling of `_run_node`: it keeps the `__input__` outbox park, the
  run-cancel watcher, a per-job `LeaseRenewer`, and the `JobStatusWatcher`, but
  OMITS the warm-model cache busy-bracket and the `GpuHealthWatchdog`/`StallWatchdog`
  (those police an in-worker diffusion hang; a VLM job's GPU work is in the server,
  HTTP-bound in the worker). Additive: the inline diffusion path is byte-identical
  except the new `require_model=True` claim filter. Fixes "the vLLM toggle gives
  nothing over ollama" — a concurrency-1 worker could only issue one VLM request at
  a time, so vLLM never batched.
- `node_queue.claim_next_gpu_job(..., require_model: bool | None = None)` — a
  model-presence claim filter that splits the GPU queue into two disjoint sets:
  `None` (default) = existing claim-any; `True` adds `AND c.required_model IS NOT
  NULL` (diffusion); `False` adds `AND c.required_model IS NULL` (no-model VLM).
  ANDs with the existing capability gate + keeps the affinity/priority ordering.
  The two GPU lanes use `True`/`False` so they never over-claim or steal each
  other's rows.
- Worker LLM-server capability advertisement (migration 0014): a worker publishes
  `worker_heartbeats.llm_servers_available text[]` (default `{ollama}`) — which
  server types it can actually run — set via `set_llm_servers_available([...])` /
  `config.llm_servers_available` and emitted in the heartbeat alongside
  `known_models`. Lets a UI gate its per-machine server-type control (an AMD box
  that can't run the CUDA vllm sidecar advertises `{ollama}` only). Additive +
  default-safe: other consumers and CPU/ingest workers keep the `{ollama}` baseline.
- `queue_workflows.set_vllm_lifecycle(stop_fn, start_fn)` + the
  `config.vllm_stop_fn` / `vllm_start_fn` hooks — a host that runs vllm as a
  SEPARATE container wires how to stop/start it (ai_leads → docker Engine API
  over the unix socket). The backend factory threads them into `VLLMBackend`'s
  `kill_fn` / `ensure_up_fn`, so the in-worker idle supervisor can free the
  sibling sidecar's VRAM WITHOUT a docker restart policy (which would re-trigger
  the consumer's NFS boot race). `None` (the default) keeps the backend's
  built-in pkill / no-op seams, so an unconfigured deployment is unchanged.
- `BackendFactory` + `get_backend(host, queue)` — the per-`(host, queue)` owner
  of the live `LLMBackend` + its `LLMSupervisor`, kept in sync with the DB config
  (0013). A snapshot read-through cache preserves backend identity (request
  counters + vllm state survive) while config is unchanged, rebuilds on an actual
  change, and refreshes on a 10 s TTL or a `worker_llm_config_changed` NOTIFY
  (gated by `AI_LEADS_DISABLE_LLM_CONFIG_LISTENER`). The gpu claim worker starts
  the config listener at boot.
- `VLLMBackend` (+ `VLLMState` enum) — the vllm concrete backend. A lifecycle
  state machine (`DEAD`/`LOADING`/`SERVING`/`SLEEPING_L2`/`RELOADING`/
  `UNSUPPORTED_SLEEP`) over fully-injected I/O seams (kill / ensure-up / health /
  served-model probes — httpx lazy-imported only inside the defaults). `chat_url`
  → `/v1/chat/completions`, `health_url` → `/health`. Model switch tries Sleep-L2
  reload (stubbed `NotImplementedError` → sticky `UNSUPPORTED_SLEEP`) then falls
  back to stop+bring-up. Satisfies the `LLMSupervisor` duck-typed surface so an
  idle vllm sidecar gets SIGTERMed to free VRAM.
- `queue_workflows.llm_backends` — the per-machine LLM server abstraction.
  `LLMBackend` ABC owns RLock-guarded request accounting (`mark_request_start`/
  `mark_request_end`/`inflight`/`idle_seconds`) so the idle supervisor can decide
  when to free VRAM, and exposes `chat_url`/`health_url` for the HOST to POST
  against (the library never makes the LLM call itself). `OllamaBackend` is the
  trivial reference backend — lifecycle is inert (the daemon self-manages idle via
  `OLLAMA_KEEP_ALIVE`). `LLMSupervisor` + the pure `vllm_should_stop(...)` decision
  mirror `ModelCache`'s idle reaper: a daemon that SIGTERMs an idle vllm sidecar
  (inert for ollama; gated by `AI_LEADS_DISABLE_LLM_SUPERVISOR`).
- Per-machine LLM server config on `worker_controls` (migration 0013): new
  `llm_server_type` (`'ollama'` | `'vllm'`, default `ollama`), `llm_parallelism`
  (sidecar concurrent-request capacity — NOT the claim-worker concurrency, which
  stays 1 by contract; default 1) and `vllm_idle_ttl_s` (idle window before the
  vllm supervisor frees VRAM; default 60) columns. `worker_control.set_llm_config`
  (partial COALESCE upsert, soft — never touches `desired_state`/`stop_policy`) +
  `worker_control.llm_config_for -> LLMConfig` (default-safe on a pre-0012/pre-0013
  DB). A dedicated `worker_llm_config_changed` NOTIFY channel (payload
  `host|queue`, distinct from `worker_control`) lets a worker refresh its backend
  without the hard-stop watcher mistaking a config edit for an OFF switch.
- `paint_mask` gains an optional `initial_mask_opacity` field (0.0-1.0) —
  when a workflow step ships it, the dispatcher passes it through on the
  input spec so the host widget can render the pre-painted mask at that
  alpha (visual cue: "this is a suggestion, your strokes overwrite it").
  Default unset preserves the 100%-opaque pre-paint.
- New `pick_fence` input widget — `_build_input_spec` resolves both `source`
  (the image to display as background) and `detections` (a `detections.json`
  index of per-detection masks) into `source_options` / `source_rel_path` /
  `source_abs_path` plus `detections_options` / `detections_rel_path` /
  `detections_abs_path`. Lets a host-side widget render each detection as a
  colored overlay the operator clicks to toggle in/out. Same `$from`/`$filter`
  resolution shape as `paint_mask` + `choose_one` — no new mini-language.
- `node_queue.delete_non_terminal_jobs_for_run(run_id) -> list[node_id]` —
  restart primitive that deletes every job whose status is NOT
  `completed` / `skipped`, returning the deleted `node_id`s so the host can
  cascade cleanup into its own artefacts (on-disk dirs, input submissions).
  Paired with the engine's existing `_find_ready_nodes` (which treats
  surviving `completed` / `skipped` rows as cursors), a follow-up
  `dispatcher.start_run` resumes a failed run from exactly the failed
  branch instead of re-doing the completed prefix. Caller-policy
  in-flight guard is documented in the docstring.
- `paint_mask` widget gains an optional `initial_mask` ref — when a workflow
  step declares it (same `$from`/`$filter` shape as `source`), the
  dispatcher resolves it into `initial_mask_options` /
  `initial_mask_rel_path` / `initial_mask_abs_path` on the input spec so
  the host widget can pre-paint the overlay from an upstream auto-mask
  (e.g. a GroundingDINO+SAM2 detection pre-pass). Default unset preserves
  the existing blank-canvas behaviour.
- `paint_mask` input widget — the dispatcher's `_build_input_spec` resolves
  the step's `source` ref into `source_options` / `source_rel_path` /
  `source_abs_path` so a host-side canvas widget can display the upstream
  image and upload a binary mask PNG back through the standard multipart
  pipe.
- Opt-in orphan-cancel sweep: with
  `configure(cancel_orphan_queued_jobs=True)` the `NodePool` periodically
  flips `queued` `workflow_node_jobs` whose parent run is already
  `cancelled` / `failed` to `cancelled`. The claim SQL's run-cancel guard
  already refuses such jobs, but they linger in `queued` and pollute the
  operator-facing queue gauges; this is the cleanup. Default `False`
  preserves pre-0.4 behaviour byte-for-byte. New
  `node_queue.cancel_orphaned_queued_jobs()` exposes the underlying join
  UPDATE for direct use. Interval-gated by
  `AI_LEADS_ORPHAN_CANCEL_SWEEP_INTERVAL_S` (default 30 s).

### Changed
- The GPU worker's capacity heartbeat advertises `concurrency = 1` (the single
  structural warm-model diffusion slot), same as CPU/ingest. The PAR-sized VLM
  pool's capacity is a per-machine VLM-request-batching property surfaced to the
  consumer UI via `worker_controls.llm_parallelism` (a "PAR" field) — deliberately
  NOT folded into this gauge, so the consumer's GPU pill counts the heavy
  warm-model slot (1/box), not the lightweight no-model VLM pool that rides
  beside it. (A `HeartbeatEmitter(concurrency_fn=...)` seam remains for a future
  caller wanting a live per-tick value; unused today.)

### Fixed
- `VLLMBackend.ensure_ready` now FAILS LOUD on a served-model mismatch. After the
  bring-up + health wait it reconciles the `/v1/models` probe against the
  requested id and raises `RuntimeError` if they differ — because the default
  bring-up (docker `restart: unless-stopped` + the BAKED `--model` flag) can
  resurrect the sidecar serving the wrong model on a cross-model switch. Failing
  loudly lets the consumer soft-degrade (e.g. fall back to ollama) instead of
  silently POSTing prompts to the wrong model. A blank probe still optimistically
  trusts the requested id (no false trip on the existing fallback path).

## [0.3.0] — 2026-05-27

### Added
- **Pluggable storage backends** — a `StorageBackend` SPI
  (`queue_workflows.backends`) makes the durable-queue store selectable via
  `configure(db_backend="pg"|"redis"|"mongodb")`, one provider per file. `pg`
  (default) is byte-compatible and unchanged; `redis` (atomic claim/terminal via
  Lua + pub/sub wake) and `mongodb` (`find_one_and_update` + multi-doc-txn outbox
  + change-stream wake, replica set) reproduce the same contract — claim
  exactly-once, lease/reclaim, idempotent terminals, the atomic outbox, and
  per-namespace tenant isolation — pinned by a parametrized contract suite run
  against all three live servers. Optional drivers: `queue_workflows[redis]` /
  `[mongodb]`. See `docs/storage_backends.md`.
- Operator worker ON/OFF control plane: a `worker_controls` table + a
  per-`(host, queue)` `WorkerControlWatcher` that hard-stops (kill in-flight +
  free RAM/VRAM) or parks a worker on command (migration 0012). See
  `docs/worker_control.md`.
- Durable `workflow_node_events` store with lifecycle/trip/terminal event
  writers emitted from the worker, plus a NodePool retention sweep
  (migration 0011).
- Orchestrator-side dead-worker sweep that flags a worker whose heartbeat is
  stale while it still holds a `running` job; health-gated stall trip with
  re-queue-and-retry (migrations 0009/0010).
- No-progress `StallWatchdog` for GPU nodes; re-queue all running jobs on
  restart and worker self-kill when reassigned; always re-queue orphaned runs.
- `invoke_context` host hook — a per-node setup/teardown context manager whose
  `finalize(context_delta)` is applied only on success.
- MIT `LICENSE` and PEP 639 license metadata, keywords, and trove classifiers
  in `pyproject.toml`.
- `README.md`: a "local-cluster swift management" purpose note, an Architecture
  diagram (Mermaid), an "Example dashboard" section, a "Turning workers on/off"
  section, and a Docs index.
- `AGENTS.md` as a byte-identical mirror of `CLAUDE.md`, and a documented
  changelog convention.
- This `CHANGELOG.md`.

### Changed
- GPU guard is now health-driven (GPU-idle **and** static-RAM) instead of a
  fixed wall-clock cap.

### Fixed
- GPU claim: restored the capability-routing gate.
- `__input__` nodes are parked via the dispatch outbox instead of importing the
  sentinel.

### Removed
- `uv.lock` is no longer tracked — this is a library; consumers resolve against
  `pyproject.toml`. The lockfile stays local for dev/CI reproducibility.

## [0.2.0] — 2026-05-25

### Added
- Multi-tenant ingest path: host-defined ingest queues and per-job `args`, with
  host-side `task_name`/queue validation (migration 0008). A second consumer
  (e.g. a non-DAG forecast service) can route its own queue names and carry
  per-job arguments without forking the schema.

## [0.1.0] — 2026-05-25

### Added
- Initial standalone release — a Postgres-as-queue workflow engine extracted
  from the `ai_leads` stack (Phase 6): a `SELECT … FOR UPDATE SKIP LOCKED` claim
  loop woken by `LISTEN`/`NOTIFY`, lease reclaim, a DAG dispatcher with a durable
  dispatch-event outbox, a GPU warm-model cache, periodic ingest work + a
  PG-native scheduler, and per-host hw-metrics telemetry. Migrations 0001–0007.

[Unreleased]: https://github.com/robertziel/python_workflows_queue/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/robertziel/python_workflows_queue/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/robertziel/python_workflows_queue/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/robertziel/python_workflows_queue/releases/tag/v0.2.0
[0.1.0]: https://github.com/robertziel/python_workflows_queue/commit/9ddaf4ae80d906e9d286403bab015e56ba9899ed
