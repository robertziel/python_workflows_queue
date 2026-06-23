# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**broker_parrot** (Python package `queue_workflows` — the import/distribution name is unchanged for compatibility) is a standalone, pip-installable **Postgres-as-queue workflow engine**: a `SELECT … FOR UPDATE SKIP LOCKED` claim loop woken by `LISTEN`, lease reclaim, a DAG dispatcher with a durable outbox, a GPU warm-model cache, periodic ingest work, per-host hw-metrics telemetry, a durable per-node event log, and an operator worker ON/OFF control plane (`worker_control` — hard-stop/park a `(host, queue)` worker; see `docs/worker_control.md` and `docs/watchdogs.md`). **Postgres (via `psycopg` 3) is the only hard runtime dependency.**

## Who uses it / why the `ai_leads` defaults

The engine was extracted from the **`ai_leads`** stack (its "Phase 6") so the ~35 sibling projects in that stack can share one DRY source instead of each carrying a copy. `ai_leads` is the origin and first consumer; it lives in a separate repo (not a sibling of this checkout). A **second consumer** — a non-DAG forecast service — drove the v0.2.0 multi-tenant-ingest generalization (host-defined ingest queues + per-job args; see *Two job families* below), so treat "host" as **≥2 distinct apps**, not just `ai_leads`, when generalizing.

This explains a pattern you'll see everywhere: defaults are **`ai_leads`-byte-compatible** so the live deploy needs zero `.env` changes at cutover — the DSN env var defaults to `AI_LEADS_DB_URL`, `container_prefix` to `"ai_leads-"`, runtime knobs are `AI_LEADS_*`, and tests honor `AI_LEADS_DISABLE_*` gates. **These are configurable defaults, not couplings.** The package imports *nothing* from any host application (enforced by `tests/test_no_ai_leads_import.py`); a consumer overrides the names via `queue_workflows.configure(db_url_env=..., container_prefix=..., …)`. When adding a new tunable, follow the same shape: read an env *name* off `EngineConfig`, default it to the `ai_leads` name.

## Commands

```bash
# Setup
pip install -e '.[test]'        # add [metrics] for the psutil-based hw_metrics CPU/RAM probe

# Tests — REQUIRE a reachable Postgres; the suite forces a *_test DB and
# creates it if missing (see tests/conftest.py).
QUEUE_WORKFLOWS_TEST_DB_URL=postgresql://user:pw@host:port/queue_workflows_test python -m pytest
#   (falls back to AI_LEADS_DB_URL with its db-name suffixed _test if the above is unset)
#   The multi-backend contract suite (tests/test_backend_contract.py) also reads
#   QUEUE_WORKFLOWS_TEST_REDIS_URL / _MONGO_URL (mongo needs a replica set); each
#   backend SKIPS if its server is unset/unreachable. `.[test]` pulls redis +
#   pymongo; standalone consumers add `.[redis]` / `.[mongodb]`. See docs/storage_backends.md.

python -m pytest tests/test_node_queue.py            # one module
python -m pytest tests/test_node_queue.py::test_name # one test
python -m pytest -k lease                            # by keyword

# Console scripts (installed by pyproject; also `python -m queue_workflows.<mod>`)
queue-orchestrator                  # bootstrap migrations + NodePool (dispatch/outbox/reclaim/input)
queue-claim-worker --queue=gpu      # one worker process; --queue ∈ {cpu,gpu} (DAG) ∪ config.ingest_queues (default {fetch,load})
queue-scheduler                     # PG-native ingest ticker
queue-worker-control --queue=gpu --off   # operator ON/OFF for a (host,queue) worker (migration 0012; docs/worker_control.md)
```

There is no linter/formatter config and no CI in this repo; match the surrounding style (heavy module/function docstrings explaining *why*, `from __future__ import annotations` everywhere).

## Changelog

`CHANGELOG.md` tracks all notable changes ([Keep a Changelog](https://keepachangelog.com/) format + [SemVer](https://semver.org/)). **When you make a user-visible change (new feature, behavior change, fix, migration, or removal), add a bullet under `## [Unreleased]`** in the right group (`Added` / `Changed` / `Fixed` / `Removed`) as part of the same change — don't leave it for "later." On a release, rename `[Unreleased]` to the new version + date, bump `version` in `pyproject.toml` to match, tag `vX.Y.Z`, and update the link footer. Pure-internal refactors with no observable effect don't need an entry. (This file and `AGENTS.md` are kept byte-identical — edit one, copy to the other.)

## Development workflow — TDD

This codebase is built test-first, and the test suite is the spec. **Write the failing test before the implementation**, then make it pass with the minimal change.

- **Encode behavioral contracts as `tests/test_invariant_*.py`.** The existing invariant tests (idempotent `mark_completed`/`mark_failed`, cancel semantics, the dispatch-event outbox, late `$from` input resolution, startup health, input-listener reclaim) define guarantees the engine must never break — when you add or change a guarantee, add/adjust an invariant test for it. Other contracts live in topic modules (`test_node_queue_lease.py`, `test_dispatcher_skip_if.py`, etc.).
- **Tests run against a real Postgres**, not a mock — `conftest.py` creates the `*_test` DB, applies the engine migration chain, `TRUNCATE`s the engine tables between tests, and resets injected config so a hook one test wires doesn't leak. Pure logic (schedule math, ref resolution, the idle-unload decision) is written with injectable `now_fn`/`sleep_fn`/`on_exit` seams so it's unit-testable with a virtual clock and no real waiting — preserve those seams.
- **Two guard tests must stay green** for every change: `test_no_ai_leads_import.py` (no module imports a host package) and `test_standalone_import.py` (`import` + `configure()` + a real end-to-end round-trip works with only psycopg + Postgres, using an in-test fake node module/workflow).

## Architecture

### Three process roles, one Postgres

All three run as separate processes against the same database; the DB *is* the message bus.

1. **Orchestrator** (`orchestrator.py` → `node_pool.NodePool`) — the only process that bootstraps migrations. Its `NodePool` runs background threads: the **dispatch loop** (`_tick`) expands freshly-`queued` `mode='node'` runs into node-jobs via `dispatcher.start_run`, **drains the dispatch-event outbox**, and runs the **lease-reclaim sweeps** (node + ingest); plus an **`InputListener`** that polls `workflow_input_submissions` and resumes parked input nodes. No node bodies run here.
2. **Claim worker** (`claim_worker.ClaimWorker`) — **one process == one worker, concurrency-1 by contract.** `run_forever` does `LISTEN <channel>` then drains the queue greedily on each wake (1 s safety poll covers a dropped NOTIFY). `cpu`/`gpu` draw DAG node-jobs from `workflow_node_jobs`; the **ingest-family** queues (`config.ingest_queues`, default `fetch`/`load`) draw standalone ingest jobs from `ingest_jobs`. The GPU worker owns the process-wide warm `ModelCache` (cache logic lives in `model_cache.py`, DB-decoupled; `gpu_model_cache.py` wires the one process-wide instance and publishes `current_model` to `worker_heartbeats` for affinity routing). Every claimed job is bracketed by a `LeaseRenewer` + a wall-clock `Watchdog`; a GPU node additionally gets the health-driven `GpuHealthWatchdog` (and a `StallWatchdog` for non-video). A node-job also runs a `JobStatusWatcher` (run-cancel/reassign self-kill), and the worker process runs one `WorkerControlWatcher` (operator ON/OFF). See *Lease + reclaim + watchdog* below.
3. **Scheduler** (`scheduler.Ticker`) — a Python loop (not pg_cron) that sleeps to the next scheduled minute and enqueues `ingest_jobs` rows; an ingest claim worker picks them up.

### The queue mechanism

`INSERT`ing a row *is* enqueuing the work. The claim is a single statement — a `FOR UPDATE SKIP LOCKED` subselect picks the next claimable row, the outer `UPDATE` flips `queued → running` and stamps `claimed_by` + `lease_expires_at` (see `node_queue._CLAIM_SQL`). A trigger (migrations 0006/0007) fires `pg_notify('node_job_ready' | 'ingest_job_ready', <queue>)` **inside the writer's transaction**, so there's no "row queued but no wake" window. The claim's `ORDER BY` is built only from validated ints/fixed fragments (never caller strings); GPU claims add a **warm-model affinity** tiebreak (`required_model IS NOT DISTINCT FROM current_model` sorts first) and a `host_priority` direction term.

### Lease + reclaim + watchdog (the liveness model)

A live worker renews `lease_expires_at` (~every 10 s) while a job runs, so lease length is independent of job duration. A **dead/wedged** worker stops renewing → its lease lapses → the orchestrator's reclaim sweep flips the row back to `queued` (re-firing the NOTIFY); this sweep is the **sole** recovery path for an orphaned `running` row.

**Three daemon watchdogs** can bracket a claimed job (all in `claim_worker.py`), each hard-exiting with a **distinct code** so the cause is readable from the exit status:

- the wall-clock **`Watchdog`** (exit `75`) trips on `elapsed ≥ budget_for(job)`;
- the no-progress **`StallWatchdog`** (exit `76`) is *opt-in* (a non-video GPU node whose `run(...)` declares a `status_callback`), **inert until the first per-step `beat()` arms it after the model load** (so a minutes-long cold load is never policed), then trips on a beat gap ≥ `STALL_TIMEOUT_S` (120 s);
- the **`GpuHealthWatchdog`** (exit `78`) is the GPU guard — **health-driven, not wall-clock**: every `interval_s` (default 300 s) it trips only when the per-container GPU stayed idle **and** RAM was static across the window (no GPU work *and* no memory movement ⇒ wedged), arming at job start with a 20-min `load_grace_s` first window (safe because a healthy load *moves* RAM and a healthy render *keeps the GPU busy*). It replaced the old fixed GPU wall-clock cap, which couldn't catch the Blackwell qwen 0 %-GPU stall yet false-killed long-but-healthy renders. Per-container GPU%/RAM samplers (`gpu_health.py`) use pmon `sm%` scoped to this container (excludes a co-tenant ollama sidecar) + cgroup RAM.

All three trips funnel through one policy point — **`_watchdog_trip`**, which does **re-queue-and-retry, not fail** (migration `0010`): for a DAG node-job it flips the row `running → queued`, bumps `watchdog_retries`, and writes **no** dispatch event (so the *run* stays `running` — only this node re-runs; `_requeue_job_and_exit`), then hard-exits so a fresh worker re-claims it. Only once `watchdog_retries` hits `AI_LEADS_WATCHDOG_MAX_RETRIES` (default 3) does it fall back to **`_fail_job_and_exit`** (mark failed **+** the `failed` dispatch event in one txn — the outbox-atomicity contract, coded in exactly one place). An **ingest** job (no run, no retry counter) always takes the fail path. So a single transient wedge no longer kills the whole workflow — and a GPU worker stays one process holding one model: a hard exit kills exactly the hung job. See **`docs/watchdogs.md`** for the full design.

Two state-watcher threads sit alongside the watchdogs: **`JobStatusWatcher`** (exit `77`) self-kills a worker whose `running` row was cancelled/reassigned out from under it (avoids a double-run), and **`WorkerControlWatcher`** (exit `79`) enforces the operator ON/OFF control plane (below).

**Last-resort recovery — the orchestrator-side dead-worker detector.** Every watchdog above is an in-process *thread*; a GPU **hardware-hang** can defeat all of them (the trip signal becomes unobservable from inside — e.g. on ROCm the box-level GPU probe still reads non-idle while *this* render is wedged — or, on a GIL-holding hang, the threads can't run at all). The worker then sits wedged while its `worker_heartbeats.last_seen` freezes. The orchestrator is a **separate process** (GIL-independent of the worker), so `NodePool._tick` adds `_sweep_dead_workers` → `node_queue.flag_stale_workers_holding_running_jobs`: it flags any worker whose heartbeat is stale (>30 s, 3× the cadence) **while it still owns a `running` job** (join `claimed_by = host_label`), stamping `worker_heartbeats.last_flagged_dead_at` (migration 0009) + an actionable `DEAD WORKER:` ERROR. The JOB is recovered by the lease-reclaim as usual; this flags the dead **process** for a host-supervisor to bounce (the orchestrator can't safely cross-host-kill it). A fresh heartbeat clears the flag. See **`docs/watchdogs.md` → "last-resort layer"** for the root-cause and the host-supervisor hook.

### Durable node-event history (migration 0011)

`workflow_node_jobs` is one *mutable* row per `(run_id, node_id)` — a watchdog re-queue overwrites `claimed_by`/timing and only bumps `watchdog_retries`, so the prior attempt's worker, timing, and trip reason are lost. `workflow_node_events` is an **append-only** forensic log of the per-attempt lifecycle (`claimed`, `model_load_*`, `progress_beat`, `stall_*`, `gpu_health_trip`, `budget_trip`, `requeued`, `reassigned`, terminal …; `attempt` = `watchdog_retries` at emit ties the tries of one node together). Writers in `node_queue`: `record_node_event` (best-effort — own connection, swallow-on-failure, so an event blip can **never** fail the load-bearing claim/terminal/watchdog path) and `record_node_event_in_txn` (terminal + `requeued` events ride the **same txn** as the state change — the dispatch-outbox atomicity pattern). The worker emits via `claim_worker._emit_node_event`; `NodePool` prunes old rows on a sweep (`prune_node_events`, default 30-day retention). **Append-only — no UPDATE path**, so it adds no new mutation invariant.

### Operator worker ON/OFF control plane (migration 0012)

`worker_controls` is **desired** state (an operator or host UI writes a `(host_label, queue)` row), kept deliberately separate from the *observed* `worker_heartbeats` — an OFF state must persist precisely while the worker isn't beating. A row trigger fires `pg_notify('worker_control', '<host>:<queue>')` inside the writer's txn, so a plain SQL write from any DB consumer wakes the worker with no app-side NOTIFY. `worker_control.WorkerControlWatcher` (LISTEN + 5 s safety poll) enforces it; on OFF it dispatches the row's `stop_policy` through the `STOP_POLICIES` registry — only `"hard"` exists today (`"drain"`/`"pause"` are reserved names that slot in with no schema change, which is why `stop_policy` is free-form TEXT, not a CHECK). **Hard stop = process exit (`os._exit(79)`)**: the node body runs inline on the worker's main thread, so killing the process is the only thing that reliably stops a wedged CUDA kernel and frees VRAM (it re-queues the in-flight job first, resume-style, with **no** `watchdog_retries` bump — an operator stop isn't a fault). The supervisor restarts the container; on boot the worker re-reads `worker_controls` and **parks** (idle, not claiming) while still OFF. A worker absent from the table — or a DB predating 0012 (`get_worker_control` swallows `UndefinedTable`) — is treated as **ON**, so the engine runs unchanged before 0012 (claim workers gate on schema 6/8, not 12). See `docs/worker_control.md`.

### DAG dispatch + the durable outbox (key decoupling)

`dispatcher.py` is **pure DAG-walk logic** (unit-testable without a worker pool): expand a run's initial nodes, and on each node terminal event find downstream nodes whose deps are all `completed`/`skipped` and enqueue (or insert a `skipped` marker per `skip_if`). The worker→dispatcher handoff is an **outbox**: when a worker finalizes a node it writes the terminal status **and** a `workflow_dispatch_events` row in **one transaction** (`node_executor.execute_node`). The orchestrator drains that outbox and calls `on_node_completed`/`on_node_failed`/`on_node_awaiting_input`. So fan-out is retryable and never synchronously coupled to the worker; a failing callback is retried next tick (and poison-flagged after `_DISPATCH_MAX_ATTEMPTS`).

### Two job families: DAG node-jobs vs ingest jobs (multi-tenant)

The engine runs **two independent job shapes**, each with its own table and claim path:

- **DAG node-jobs** (`workflow_node_jobs`, queues `cpu`/`gpu`) — fanned out from a `mode='node'` run by the dispatcher (above); this is `ai_leads`' path.
- **Ingest jobs** (`ingest_jobs`, host-defined queues) — standalone periodic/parametrised work with **no DAG**, enqueued by the scheduler ticker or directly by a host, executed by `ingest_executor`.

`config.ingest_queues` names the ingest-family queues (default `{fetch, load}`; `configure(ingest_queues=...)` **rejects** reuse of the reserved `cpu`/`gpu` names). Migration `0008` moved the queue allow-list from a DB `CHECK` to **host-side validation** in `node_queue.enqueue_ingest_job` (mirroring the `task_name` gate `0007` added), so the second consumer (a non-DAG forecast service) routes its own queue names without forking the schema. A registered ingest task is `fn(reason)` **or** `fn(reason, args)` returning a JSON-able dict; `enqueue_ingest_job(task_name=, queue=, args=, conn=)` accepts a caller connection so the NOTIFY **rides the caller's transaction** (atomic with the host's own row insert). `budget_for` gives host-defined ingest queues `config.ingest_default_budget_s` (default 3600 s). Ingest workers now emit `worker_heartbeats` too, so `node_queue.ingest_snapshot()` reports `{queued, running, completed, failed, workers}` per queue.

### The host-agnostic seam — the single most important design fact

Everything domain-specific is an **injected hook** on a process-wide `EngineConfig` singleton (`config.py`), wired once at startup via `queue_workflows.configure(...)` + the `set_*`/`register_*` helpers in `__init__.py`. The hooks:

- **workflow/pipeline provider** — `load_workflow(name)` / `pipeline_schema(name)`: where the dispatcher reads the DAG from (pipeline schemas own the `nodes` list).
- **node-module resolver** — maps a stored `node_module` string to an imported module exposing `run(...)` (`set_node_module_package` builds `"<pkg>.<node_module>"`, or `set_node_resolver` for full control).
- **builtin-model registrar** — idempotently registers the host's `ModelSpec`s into `model_registry` (the GPU empty-registry fallback + once-at-startup call).
- **ingest tasks + schedule + queues** — `register_ingest_task(name, fn)` (`fn(reason)` or `fn(reason, args)`), `set_ingest_schedule([...])`, and `configure(ingest_queues=…, ingest_default_budget_s=…)` for the multi-tenant ingest path (see *Two job families* above).
- **per-node invoke wrapper** — `set_invoke_context(factory)`: a `Callable[[job, run], ContextManager]` whose CM brackets each node invoke. `__enter__` does host setup (e.g. pin a run-context `ContextVar`, capture a live mock flag) and yields a `finalize(context_delta) -> context_delta` callable that `execute_node` applies **only on success**; `__exit__` tears down on every path. Default unset ⇒ nodes run directly. Lets a host thread per-node state (e.g. a `_mocked` stamp) without forking `node_executor.execute_node`.
- **ref resolver** — defaults to the engine's own `refs.resolve_ref` (the `$value`/`$from`/`$filter`/`$eq`/`$ne` mini-language).

**Every hook has a safe default**, so `import queue_workflows` + `configure()` + a reachable Postgres runs standalone. When working in any engine module, never reach "up" into a host — add a config hook with a default instead. `config.py` is a **leaf** (imports nothing from other engine modules) to keep the dependency graph acyclic; respect that (e.g. it lazily imports `refs` only inside `get_resolve_ref`).

### Pluggable storage backends (the `db_backend` seam — additive, v0.3.0)

Beyond the host hooks, the **storage layer itself** is selectable:
`configure(db_backend="pg"|"redis"|"mongodb")` resolves a `StorageBackend`
(`queue_workflows/backends/`, **one provider per file**) — a generic durable-queue
SPI (enqueue / claim-exactly-once / lease+reclaim / idempotent terminals / the
**atomic outbox** `complete_with_event`/`fail_with_event` / wake / heartbeat /
ON-OFF control). It is **additive and opt-in**: `pg` (default) is byte-compatible
and the legacy engine modules still talk to Postgres directly — selecting
redis/mongo does **not** re-home the orchestrator/worker (a later milestone), and
the redis/pymongo drivers import lazily so a pg-only deploy needs neither. Two
invariants make it honest: the port leaks **no** driver object (no
cursor/pipeline/session in any signature — the anti-leakage rule), and every
backend is **namespace-bound** so two tenants on one redis/mongo server can't see
each other's jobs. pg uses `SKIP LOCKED`/`RETURNING`/one-txn-outbox/`LISTEN`;
redis uses **Lua** (atomic claim+terminal+event) + pub/sub; mongodb uses
`find_one_and_update` + a **multi-doc txn** + a change stream (**replica set
required**). The contract is one parametrized suite (`tests/test_backend_contract.py`)
green against all three live servers. See `docs/storage_backends.md`.

### Migrations — the engine owns one chain, hosts run a second

The engine owns `queue_workflows/migrations/NNNN_*.sql` (+ paired `.down.sql`), shipped as package data, tracked in the `queue_schema_version` ledger. `db.bootstrap()` applies the chain idempotently; `db.downgrade()` reverses it. A host with its own domain tables runs a **second** chain via `db.bootstrap(migrations_dir=..., version_table=...)` against its own ledger — "two ORMs / two chains, one Postgres." **Only the orchestrator bootstraps** (`db.bootstrap` takes no advisory lock); claim workers call `db.wait_for_schema(min_version)` and block until the schema is ready rather than racing the migration run (`_REQUIRED_SCHEMA_VERSION` maps each queue to its minimum version).

The chain: `0001` `workflow_runs` → `0002` `workflow_node_jobs` → `0003` `workflow_input_submissions` → `0004` `workflow_dispatch_events` → `0005` `worker_heartbeats` → `0006` lease columns + `node_job_ready` trigger → `0007` `ingest_jobs` + `ingest_job_ready` trigger → `0008` multi-tenant ingest (adds per-job `args JSONB`; drops the `fetch`/`load` queue CHECK and the `cpu`/`gpu`-only `worker_heartbeats` CHECK so those allow-lists move host-side — all additive/idempotent) → `0009` `worker_heartbeats.last_flagged_dead_at` (dead-worker flag) → `0010` `workflow_node_jobs.watchdog_retries` (watchdog re-queue counter) → `0011` `workflow_node_events` (append-only per-attempt event log) → `0012` `worker_controls` (+ `worker_control` NOTIFY trigger). Ingest queues require schema version ≥ 8; worker-control is read-optional — claim workers gate on 6/8, not 12, and treat a pre-0012 DB as all-ON. `run_store` treats `parcel_id` as an opaque nullable column (the engine drops the host's parcels FK) so the engine never knows the host's domain.

### Idempotency contracts to preserve

`mark_completed`/`mark_failed`/`mark_awaiting_input` (and the ingest twins) all `UPDATE … WHERE status NOT IN ('completed','failed','cancelled') RETURNING *` and return `None` when the row was already terminal. This `WHERE` is load-bearing: it makes duplicate deliveries and claim-race losers safe, and stops a stray second call from clobbering a finalized `context_delta`. JSON columns are pre-validated (`json.dumps`) before any state mutation so a bad payload fails before the write. Keep this shape for any new state transition. (The one deliberate exception is `workflow_node_events` — **append-only, no UPDATE path** — whose terminal/`requeued` rows instead ride the state-change txn, like the dispatch outbox.)

### Telemetry (hw_metrics + cgroup attribution)

`hw_metrics.py` samples per-host CPU/GPU/RAM and `pg_notify('hw_metrics', …)`s a snapshot for a dashboard (the GPU probe shells out to `rocm-smi`/`nvidia-smi`, no Python dep; CPU/RAM needs the optional `[metrics]` `psutil` extra). `cgroup_attribution.py` reads the host cgroup-v2 tree to split the CPU/RAM slice owned by **our** containers (those whose name starts with `config.container_prefix`, default `ai_leads-`) from everything else on the box; it needs the host `/sys/fs/cgroup` + docker socket mounted read-only and returns `None` (graceful fallback) without them. CPU/RAM only — GPU attribution is intentionally skipped (ROCm doesn't expose the per-PID counters it would need).
