# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`queue_workflows` is a standalone, pip-installable **Postgres-as-queue workflow engine**: a `SELECT … FOR UPDATE SKIP LOCKED` claim loop woken by `LISTEN`, lease reclaim, a DAG dispatcher with a durable outbox, a GPU warm-model cache, periodic ingest work, and per-host hw-metrics telemetry. **Postgres (via `psycopg` 3) is the only hard runtime dependency.**

## Who uses it / why the `ai_leads` defaults

The engine was extracted from the **`ai_leads`** stack (its "Phase 6") so the ~35 sibling projects in that stack can share one DRY source instead of each carrying a copy. `ai_leads` is the origin and first consumer; it lives in a separate repo (not a sibling of this checkout).

This explains a pattern you'll see everywhere: defaults are **`ai_leads`-byte-compatible** so the live deploy needs zero `.env` changes at cutover — the DSN env var defaults to `AI_LEADS_DB_URL`, `container_prefix` to `"ai_leads-"`, runtime knobs are `AI_LEADS_*`, and tests honor `AI_LEADS_DISABLE_*` gates. **These are configurable defaults, not couplings.** The package imports *nothing* from any host application (enforced by `tests/test_no_ai_leads_import.py`); a consumer overrides the names via `queue_workflows.configure(db_url_env=..., container_prefix=..., …)`. When adding a new tunable, follow the same shape: read an env *name* off `EngineConfig`, default it to the `ai_leads` name.

## Commands

```bash
# Setup
pip install -e '.[test]'        # add [metrics] for the psutil-based hw_metrics CPU/RAM probe

# Tests — REQUIRE a reachable Postgres; the suite forces a *_test DB and
# creates it if missing (see tests/conftest.py).
QUEUE_WORKFLOWS_TEST_DB_URL=postgresql://user:pw@host:port/queue_workflows_test python -m pytest
#   (falls back to AI_LEADS_DB_URL with its db-name suffixed _test if the above is unset)

python -m pytest tests/test_node_queue.py            # one module
python -m pytest tests/test_node_queue.py::test_name # one test
python -m pytest -k lease                            # by keyword

# Console scripts (installed by pyproject; also `python -m queue_workflows.<mod>`)
queue-orchestrator                  # bootstrap migrations + NodePool (dispatch/outbox/reclaim/input)
queue-claim-worker --queue=gpu      # one worker process; queue ∈ {cpu,gpu,fetch,load}
queue-scheduler                     # PG-native ingest ticker
```

There is no linter/formatter config and no CI in this repo; match the surrounding style (heavy module/function docstrings explaining *why*, `from __future__ import annotations` everywhere).

## Development workflow — TDD

This codebase is built test-first, and the test suite is the spec. **Write the failing test before the implementation**, then make it pass with the minimal change.

- **Encode behavioral contracts as `tests/test_invariant_*.py`.** The existing invariant tests (idempotent `mark_completed`/`mark_failed`, cancel semantics, the dispatch-event outbox, late `$from` input resolution, startup health, input-listener reclaim) define guarantees the engine must never break — when you add or change a guarantee, add/adjust an invariant test for it. Other contracts live in topic modules (`test_node_queue_lease.py`, `test_dispatcher_skip_if.py`, etc.).
- **Tests run against a real Postgres**, not a mock — `conftest.py` creates the `*_test` DB, applies the engine migration chain, `TRUNCATE`s the engine tables between tests, and resets injected config so a hook one test wires doesn't leak. Pure logic (schedule math, ref resolution, the idle-unload decision) is written with injectable `now_fn`/`sleep_fn`/`on_exit` seams so it's unit-testable with a virtual clock and no real waiting — preserve those seams.
- **Two guard tests must stay green** for every change: `test_no_ai_leads_import.py` (no module imports a host package) and `test_standalone_import.py` (`import` + `configure()` + a real end-to-end round-trip works with only psycopg + Postgres, using an in-test fake node module/workflow).

## Architecture

### Three process roles, one Postgres

All three run as separate processes against the same database; the DB *is* the message bus.

1. **Orchestrator** (`orchestrator.py` → `node_pool.NodePool`) — the only process that bootstraps migrations. Its `NodePool` runs background threads: the **dispatch loop** (`_tick`) expands freshly-`queued` `mode='node'` runs into node-jobs via `dispatcher.start_run`, **drains the dispatch-event outbox**, and runs the **lease-reclaim sweeps** (node + ingest); plus an **`InputListener`** that polls `workflow_input_submissions` and resumes parked input nodes. No node bodies run here.
2. **Claim worker** (`claim_worker.ClaimWorker`) — **one process == one worker, concurrency-1 by contract.** `run_forever` does `LISTEN <channel>` then drains the queue greedily on each wake (1 s safety poll covers a dropped NOTIFY). `cpu`/`gpu` draw DAG node-jobs from `workflow_node_jobs`; `fetch`/`load` draw standalone ingest jobs from `ingest_jobs`. The GPU worker owns the process-wide warm `ModelCache`. Every claimed job is bracketed by a `LeaseRenewer` + a `Watchdog` (and node-jobs also by a run-cancel watcher).
3. **Scheduler** (`scheduler.Ticker`) — a Python loop (not pg_cron) that sleeps to the next scheduled minute and enqueues `ingest_jobs` rows; a fetch/load claim worker picks them up.

### The queue mechanism

`INSERT`ing a row *is* enqueuing the work. The claim is a single statement — a `FOR UPDATE SKIP LOCKED` subselect picks the next claimable row, the outer `UPDATE` flips `queued → running` and stamps `claimed_by` + `lease_expires_at` (see `node_queue._CLAIM_SQL`). A trigger (migrations 0006/0007) fires `pg_notify('node_job_ready' | 'ingest_job_ready', <queue>)` **inside the writer's transaction**, so there's no "row queued but no wake" window. The claim's `ORDER BY` is built only from validated ints/fixed fragments (never caller strings); GPU claims add a **warm-model affinity** tiebreak (`required_model IS NOT DISTINCT FROM current_model` sorts first) and a `host_priority` direction term.

### Lease + reclaim + watchdog (the liveness model)

A live worker renews `lease_expires_at` (~every 10 s) while a job runs, so lease length is independent of job duration. A **dead/wedged** worker stops renewing → its lease lapses → the orchestrator's reclaim sweep flips the row back to `queued` (re-firing the NOTIFY). The `Watchdog` enforces a per-job wall-clock budget (`budget_for`): on trip it marks the row failed (writing the dispatch event in the same txn for node-jobs) and **hard-exits the process** (`os._exit`), letting reclaim re-queue the work. This is why a GPU worker is one process holding one model — a hard exit kills exactly the hung job.

### DAG dispatch + the durable outbox (key decoupling)

`dispatcher.py` is **pure DAG-walk logic** (unit-testable without a worker pool): expand a run's initial nodes, and on each node terminal event find downstream nodes whose deps are all `completed`/`skipped` and enqueue (or insert a `skipped` marker per `skip_if`). The worker→dispatcher handoff is an **outbox**: when a worker finalizes a node it writes the terminal status **and** a `workflow_dispatch_events` row in **one transaction** (`node_executor.execute_node`). The orchestrator drains that outbox and calls `on_node_completed`/`on_node_failed`/`on_node_awaiting_input`. So fan-out is retryable and never synchronously coupled to the worker; a failing callback is retried next tick (and poison-flagged after `_DISPATCH_MAX_ATTEMPTS`).

### The host-agnostic seam — the single most important design fact

Everything domain-specific is an **injected hook** on a process-wide `EngineConfig` singleton (`config.py`), wired once at startup via `queue_workflows.configure(...)` + the `set_*`/`register_*` helpers in `__init__.py`. The hooks:

- **workflow/pipeline provider** — `load_workflow(name)` / `pipeline_schema(name)`: where the dispatcher reads the DAG from (pipeline schemas own the `nodes` list).
- **node-module resolver** — maps a stored `node_module` string to an imported module exposing `run(...)` (`set_node_module_package` builds `"<pkg>.<node_module>"`, or `set_node_resolver` for full control).
- **builtin-model registrar** — idempotently registers the host's `ModelSpec`s into `model_registry` (the GPU empty-registry fallback + once-at-startup call).
- **ingest task map + schedule** — `register_ingest_task(name, fn)` and `set_ingest_schedule([...])`.
- **ref resolver** — defaults to the engine's own `refs.resolve_ref` (the `$value`/`$from`/`$filter`/`$eq`/`$ne` mini-language).

**Every hook has a safe default**, so `import queue_workflows` + `configure()` + a reachable Postgres runs standalone. When working in any engine module, never reach "up" into a host — add a config hook with a default instead. `config.py` is a **leaf** (imports nothing from other engine modules) to keep the dependency graph acyclic; respect that (e.g. it lazily imports `refs` only inside `get_resolve_ref`).

### Migrations — the engine owns one chain, hosts run a second

The engine owns `queue_workflows/migrations/NNNN_*.sql` (+ paired `.down.sql`), shipped as package data, tracked in the `queue_schema_version` ledger. `db.bootstrap()` applies the chain idempotently; `db.downgrade()` reverses it. A host with its own domain tables runs a **second** chain via `db.bootstrap(migrations_dir=..., version_table=...)` against its own ledger — "two ORMs / two chains, one Postgres." **Only the orchestrator bootstraps** (`db.bootstrap` takes no advisory lock); claim workers call `db.wait_for_schema(min_version)` and block until the schema is ready rather than racing the migration run (`_REQUIRED_SCHEMA_VERSION` maps each queue to its minimum version).

The chain: `0001` `workflow_runs` → `0002` `workflow_node_jobs` → `0003` `workflow_input_submissions` → `0004` `workflow_dispatch_events` → `0005` `worker_heartbeats` → `0006` lease columns + `node_job_ready` trigger → `0007` `ingest_jobs` + `ingest_job_ready` trigger. `run_store` treats `parcel_id` as an opaque nullable column (the engine drops the host's parcels FK) so the engine never knows the host's domain.

### Idempotency contracts to preserve

`mark_completed`/`mark_failed`/`mark_awaiting_input` (and the ingest twins) all `UPDATE … WHERE status NOT IN ('completed','failed','cancelled') RETURNING *` and return `None` when the row was already terminal. This `WHERE` is load-bearing: it makes duplicate deliveries and claim-race losers safe, and stops a stray second call from clobbering a finalized `context_delta`. JSON columns are pre-validated (`json.dumps`) before any state mutation so a bad payload fails before the write. Keep this shape for any new state transition.
