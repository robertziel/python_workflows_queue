# Client API — `import queue_workflows`

The **client** is the per-project data plane: the Python package a host application
imports to enqueue work, run the orchestrator/workers/scheduler, and read queue
state. Everything domain-specific is an **injected hook** on a process-wide
`EngineConfig` singleton, wired once at startup with `queue_workflows.configure(...)`
plus the `set_*` / `register_*` helpers — so the package imports *nothing* from any
host application, yet a host can thread its workflows, node modules, models, ingest
tasks, and per-node setup into the engine without forking it. Every hook has a safe
default, so `import queue_workflows` + `configure()` + a reachable store (SQLite by
default, Postgres by opt-in) runs the engine standalone. This document is the
reference for that public surface: `configure()`, the hooks, the enqueue + state-
transition + introspection helpers (`node_queue` / `run_store` / `ingest_store`),
the worker ON/OFF control plane, the `HwFeed` telemetry reader, bootstrap/schema,
the `db_backend` storage seam, and the console scripts.

> Sibling references: [storage_backends.md](storage_backends.md) (the `redis`/`mongodb`
> SPI), [multitenant_broker.md](multitenant_broker.md) (`project` pooling),
> [worker_control.md](worker_control.md) (ON/OFF design), [conductor.md](conductor.md)
> (fleet observability), [watchdogs.md](watchdogs.md) (the liveness model),
> [gpu_pool.md](gpu_pool.md), [llm_backends.md](llm_backends.md),
> [sqlite_engine_port.md](sqlite_engine_port.md),
> [client_db_schema.md](client_db_schema.md) (the SQLite per-column schema),
> [broker_api.md](broker_api.md) (the broker/DB-protocol side of this SDK).

---

## 1. Overview — the host-agnostic seam

The engine has three process roles (orchestrator, claim worker, scheduler), all
talking to one store which *is* the message bus. What keeps the engine reusable is
that none of those roles knows your domain: every place the original `ai_leads`
stack reached "up" into the application is now a function pointer held on
`EngineConfig` (`queue_workflows/config.py`), a `@dataclass` of which exactly one
instance exists per process.

```python
from queue_workflows import get_config, EngineConfig
cfg: EngineConfig = get_config()      # the process-wide singleton (also re-exported as EngineConfig)
```

`EngineConfig` is mutated **only** through the public helpers in
`queue_workflows/__init__.py` (`configure`, `set_*`, `register_*`), never by engine
modules at import time — so a host can configure *after* import. `config.py` is a
**leaf** module (it imports nothing else from the engine, lazily importing `refs`
only inside `get_resolve_ref`) to keep the dependency graph acyclic; respect that
when adding a tunable. Every mutation is guarded by `cfg._lock` (a `threading.RLock`),
so calling `configure()` from a host thread is safe.

Two design rules everything follows:

- **Defaults are `ai_leads`-byte-compatible.** Env-var *names* default to the
  `ai_leads` names (`AI_LEADS_DB_URL`, `AI_LEADS_HOST_LABEL`, …) so the live deploy
  needs zero `.env` changes. These are configurable defaults, not couplings.
- **The one deliberate exception (v1.0.0, BREAKING):** `db_backend` defaults to
  `"sqlite"`, not `"pg"` — the friendliest zero-config default for a library. A
  Postgres consumer must opt in (see §11).

> Version note: the installable distribution (`pyproject.toml`) is `1.0.0`;
> `queue_workflows.__version__` reads `"0.5.0"`.

---

## 2. Install & quickstart

```bash
pip install -e '.[test]'        # add [metrics] (psutil) for the hw_metrics CPU/RAM probe;
                                # [redis] / [mongodb] for those StorageBackend SPI drivers
```

`psycopg[binary,pool]>=3.1` is the only hard runtime dependency.

### Zero-config round-trip (SQLite default)

A minimal ingest enqueue → claim → execute → complete cycle with no Postgres and no
host wiring — just a registered task and the default SQLite backend:

```python
import os, queue_workflows
from queue_workflows import db, node_queue, get_config

os.environ["QUEUE_WORKFLOWS_DB_URL"] = "/tmp/broker_parrot.db"   # a SQLite file path
queue_workflows.configure(db_url_env="QUEUE_WORKFLOWS_DB_URL")   # db_backend defaults to "sqlite"

db.bootstrap()                                                   # apply the engine migration chain (idempotent)

# register the work, then enqueue it
queue_workflows.register_ingest_task("hello", lambda reason, args=None: {"ok": True, "reason": reason})
job_id = node_queue.enqueue_ingest_job(task_name="hello", queue="fetch", reason="manual")

# claim + run + finalize (what a claim worker does in run_forever)
job = node_queue.claim_next_ingest_job("fetch", host="box-1")
result = get_config().ingest_task_map[job["task_name"]](job["reason"], job["args"])
node_queue.mark_ingest_completed(job["id"], result=result, seconds=0.01)

print(node_queue.ingest_snapshot())   # {'queues': {'fetch': {'completed': 1, 'workers': 0, ...}}}
```

In production you don't hand-run the claim loop — you launch `queue-claim-worker
--queue=fetch` and the scheduler enqueues on a cron. The cycle above is the same
contract, inlined.

### Postgres opt-in

```python
queue_workflows.configure(db_backend="pg", db_url_env="AI_LEADS_DB_URL")
# or, for the console scripts that have no configure() call:  export QUEUE_WORKFLOWS_DB_BACKEND=pg
```

---

## 3. `configure()` — the one-call startup entry point

```python
def configure(
    *,
    db_url_env: str | None = None,
    metrics_db_url_env: str | None = None,
    video_model_ids: frozenset[str] | None = None,
    node_module_package: str | None = None,
    host_label_env: str | None = None,
    host_priority_env: str | None = None,
    container_prefix: str | None = None,
    project: str | None = None,
    ingest_queues: frozenset[str] | None = None,
    ingest_default_budget_s: int | None = None,
    db_backend: str | None = None,
    db_namespace: str | None = None,
    cancel_orphan_queued_jobs: bool | None = None,
    vlm_pool_node_modules: frozenset[str] | None = None,
    gpu_self_load_node_modules: frozenset[str] | None = None,
    gpu_pool_backend: str | None = None,
    gpu_pool_url_env: str | None = None,
    gpu_pool_namespace: str | None = None,
) -> EngineConfig
```

Keyword-only. **Only the kwargs you pass are mutated**; the rest keep their
defaults. Returns the live `EngineConfig` for chaining/inspection. Idempotent and
additive — call again to adjust a subset. Each kwarg maps 1:1 to an `EngineConfig`
field.

**Validation performed in the call:**
- `ingest_queues` rejects reuse of the reserved DAG names `cpu`/`gpu` → `ValueError`
  (those queues draw from `workflow_node_jobs`, not `ingest_jobs`).
- `db_backend == "sqlite"` bypasses SPI-registry validation (SQLite is a relational
  engine backend, not a flat-queue SPI provider). Any other value is normalized via
  `backends.canonical_backend_name` (so `"postgres"`/`"mongo"` aliases resolve, and a
  typo raises). `gpu_pool_backend` is normalized the same way.
- `project` and `db_namespace` are coerced to `str`.

### Parameter reference

| kwarg | `EngineConfig` field / default | meaning |
|---|---|---|
| `db_url_env` | `"AI_LEADS_DB_URL"` | env-var **name** holding the DSN (or, for SQLite, the file path). `db.db_url()` / `db.sqlite_path()` read `os.environ[this]`. |
| `metrics_db_url_env` | `None` | env-var name for the shared-broker DSN hw-metrics publishes to / reads from. `None` ⇒ falls back to `db_url_env`. **Always a pg DSN** (hw-metrics is NOTIFY/Postgres-only) regardless of `db_backend`. |
| `video_model_ids` | `frozenset()` | GPU model ids on the tight per-job video-render budget (`claim_worker.budget_for`). |
| `node_module_package` | `""` | dotted package the node-module resolver imports under (e.g. `"workflows.nodes"`). Empty ⇒ the stored `node_module` is imported as a fully-qualified name. |
| `host_label_env` | `"AI_LEADS_HOST_LABEL"` | env-var name for this host's label (worker identity, heartbeat key). Falls back to `socket.gethostname()`. |
| `host_priority_env` | `"AI_LEADS_GPU_CONSUMER_PRIORITY"` | env-var name for this host's GPU-consumer priority (`< 0` ⇒ overflow host claims the queue *tail*). |
| `container_prefix` | `"ai_leads-"` | cgroup-attribution container-name prefix (the `hw_metrics` per-container CPU/RAM slice). |
| `project` | `""` (env `QUEUE_WORKFLOWS_PROJECT`) | tenant tag of **this** client on a shared broker (migration 0017). Enqueues/claims only rows whose `project` matches **by exact equality**. `""` = single-tenant sentinel (matches all). See [multitenant_broker.md](multitenant_broker.md). |
| `ingest_queues` | `frozenset({"fetch", "load"})` | host-side allow-list of ingest-family queue names (validated before enqueue). Rejects `cpu`/`gpu`. |
| `ingest_default_budget_s` | `3600` | wall-clock budget (s) for ingest queues other than the built-in `fetch`/`load`. |
| `db_backend` | `"sqlite"` (env `QUEUE_WORKFLOWS_DB_BACKEND`) | `"sqlite"` / `"pg"` (relational, full DAG engine via the dialect seam) or `"redis"` / `"mongodb"` (flat-queue StorageBackend SPI). **BREAKING v1.0.0:** was `"pg"`. |
| `db_namespace` | `""` | logical namespace isolating this tenant's jobs on a shared redis/mongodb server (`""` ⇒ literal `"default"`); scopes pg SPI rows via a `namespace` column. The **inverse** of `project` (isolate vs pool). |
| `cancel_orphan_queued_jobs` | `False` | when `True`, the `NodePool` periodically flips `queued` jobs of an already-terminal run to `cancelled` (gauge hygiene). Default off = pre-0.4 byte-compat. |
| `vlm_pool_node_modules` | `frozenset()` | node modules that are genuine VLM-facade (HTTP to a per-host vLLM/ollama server) and therefore pool-safe for the PAR concurrency lane. Non-empty routes every *other* no-model GPU job to the conc-1 inline lane. |
| `gpu_self_load_node_modules` | `frozenset()` | GPU node modules that intentionally run without a cache-managed model, exempt from the required-model guard at DAG expansion. |
| `gpu_pool_backend` | `"redis"` | shared-GPU-pool store backend (addressed **independently** of `db_backend`). See [gpu_pool.md](gpu_pool.md). |
| `gpu_pool_url_env` | `"QUEUE_WORKFLOWS_GPU_POOL_URL"` | env-var name holding the shared-pool DSN (deployment topology). |
| `gpu_pool_namespace` | `"gpu_pool"` | logical tenant namespace for the shared pool (every app + box that should share a fleet uses the same value). |

### Env knobs (read directly, no `configure()` needed)

These reach entrypoints that hand-roll their own `configure()` (standalone scripts,
console tooling). An explicit `configure(...)` value still wins.

| env var | read by | effect |
|---|---|---|
| `QUEUE_WORKFLOWS_DB_BACKEND` | `config._default_db_backend` | default `db_backend` (validated/normalized; typo ⇒ `ValueError`). |
| `QUEUE_WORKFLOWS_PROJECT` | `config._default_project` | default `project` tenant tag. |
| `AI_LEADS_DB_URL` (name configurable) | `db.db_url` / `db.sqlite_path` | the DSN / SQLite path. |
| `AI_LEADS_HOST_LABEL` (name configurable) | worker / sampler / control | this host's label. |
| `AI_LEADS_GPU_CONSUMER_PRIORITY` (name configurable) | GPU claim | host priority direction. |
| `QUEUE_WORKFLOWS_REDIS_URL` / `QUEUE_WORKFLOWS_MONGO_URL` | redis/mongo backends | SPI DSNs (names configurable via `redis_url_env`/`mongo_url_env`). |
| `QUEUE_WORKFLOWS_GPU_POOL_URL` | gpu pool | shared-pool DSN. |
| `AI_LEADS_DB_POOL_MAX` | `db.get_pool` | psycopg pool `max_size` (default 10). |
| `QUEUE_WORKFLOWS_SQLITE_TIMEOUT_S` | `db._get_sqlite_conn` | SQLite busy timeout (default 30 s). |
| `AI_LEADS_STALE_WORKER_AFTER_S` | `node_queue` | dead-worker / freshness window (default 30 s). |
| `AI_LEADS_WORKER_CONTROL_POLL_S` | `worker_control` | ON/OFF safety-poll cadence (default 5 s). |
| `AI_LEADS_DISABLE_WORKER_CONTROL` | `WorkerControlWatcher` | keep the watcher inert (tests). |
| `AI_LEADS_DISABLE_HW_METRICS` | `hw_metrics` | don't start the sampler (tests). |
| `AI_LEADS_GPU_VRAM_TOTAL_MB` | `hw_metrics.total_vram_mb` | operator-declared total VRAM (reliable on unified-memory GPUs). |

---

## 4. Injectable hooks

Each hook is a field on `EngineConfig` with a **safe default**, wired by a helper in
`queue_workflows/__init__.py` (all thread-safe under `cfg._lock`). Because every hook
has a default, `import queue_workflows` + `configure()` + a reachable store runs
standalone — when working in an engine module, never reach up into a host; add a hook
with a default instead.

| helper | signature | default | when it fires |
|---|---|---|---|
| `set_node_module_package` | `set_node_module_package(package: str) -> None` | `""` | resolving a stored `node_module` string: `resolve_node_module` imports `"<pkg>.<node_module>"` (or the bare name if empty). Same as `configure(node_module_package=)`. |
| `set_node_resolver` | `set_node_resolver(resolver: Callable[[str], Any]) -> None` | `None` | fully custom node-module resolver (`str -> module exposing run(...)`); **overrides** `set_node_module_package`. |
| `set_builtin_model_registrar` | `set_builtin_model_registrar(registrar: Callable[[], None]) -> None` | no-op `lambda` | idempotently register the host's `ModelSpec`s into `queue_workflows.model_registry`; called once at startup (claim worker / orchestrator) and by the GPU empty-registry fallback in `model_cache`. |
| `set_workflow_provider` | `set_workflow_provider(load_workflow: Callable[[str], dict], pipeline_schema: Callable[[str], dict], *, resolve_ref: Callable[[Any, dict], Any] | None = None) -> None` | loaders `None` (raise if used); `resolve_ref` ⇒ engine's `refs.resolve_ref` | the DAG source the dispatcher reads: `load_workflow(name)`, `pipeline_schema(name)` (owns the `nodes` list), and the optional `$value`/`$from`/`$filter`/`$eq`/`$ne` ref resolver. |
| `set_invoke_context` | `set_invoke_context(factory: Callable[[dict, dict], Any]) -> None` | `None` ⇒ nodes run directly | per-node invoke wrapper: `(job, run) -> ContextManager`. `__enter__` does host setup and yields `finalize(context_delta) -> context_delta` applied **only on success**; `__exit__` tears down on every path. Lets a host thread per-node state (e.g. a `_mocked` stamp) without forking `node_executor.execute_node`. |
| `set_llm_servers_available` | `set_llm_servers_available(servers: list[str]) -> None` | `["ollama"]` | declare which LLM server types this host can run; published in the worker heartbeat (migration 0014) for the queue UI's per-machine gate. |
| `set_vllm_lifecycle` | `set_vllm_lifecycle(stop_fn: Callable[[], bool], start_fn: Callable[[str], None]) -> None` | both `None` ⇒ the vllm backend's built-in pkill / no-op seams | wire the vllm-sidecar stop/start the idle supervisor + model-switch drive: `stop_fn()` frees VRAM (True iff it stopped one), `start_fn(model_id)` (re)starts it. See [llm_backends.md](llm_backends.md). |
| `register_ingest_task` | `register_ingest_task(name: str, callable_: Callable[[str], dict]) -> None` | `ingest_task_map = {}` | register a periodic/parametrised ingest callable under `name`; `fn(reason)` **or** `fn(reason, args)` returns a JSON-able dict. The registered names are the valid `task_name` set `enqueue_ingest_job` validates against. |
| `set_ingest_schedule` | `set_ingest_schedule(schedule: list) -> None` | `[]` | the scheduler's periodic schedule (list of `queue_workflows.scheduler.ScheduleEntry`); the `Ticker` fires it. |
| `register_pool_handler` | `register_pool_handler(name: str, callable_: Callable[..., dict]) -> None` | `gpu_pool_handlers = {}` | register a shared-GPU-pool handler keyed by a task's `handler` name; `fn(*, inputs, output_dir, params) -> dict`, run on a pooled GPU worker. A submit-only app registers none. |

`get_config() -> EngineConfig` returns the singleton (also re-exported as
`EngineConfig`; both are in `__all__`). `config.reset_for_tests()` restores all
defaults (test-only).

---

## 5. Enqueuing work

`INSERT`ing a row *is* enqueuing — a trigger (migrations 0006/0007) fires the wake
NOTIFY (`node_job_ready` / `ingest_job_ready`) **inside the writer's transaction**, so
there's no "queued but no wake" window. Two job families, two tables, two claim paths
(see [multitenant_broker.md](multitenant_broker.md) and the CLAUDE.md "Two job
families" section).

`node_queue.*` is the direct relational (pg/sqlite) path. `ingest_store.*` is the
**backend-agnostic** ingest facade that delegates to `node_queue` for pg/sqlite and
maps onto the StorageBackend SPI for redis/mongodb.

### DAG node-jobs (`workflow_node_jobs`, queues `cpu`/`gpu`)

```python
node_queue.enqueue_node_job(
    *, run_id: str, node_id: str, node_module: str, queue: str,
    required_model: str | None = None, inputs: dict | None = None,
    priority: int = 100, pipeline_name: str | None = None,
    project: str | None = None,
) -> str            # returns the row id
```
Fail-before-write `ValueError` if `queue not in {"cpu","gpu"}` or a `cpu` row carries
a `required_model` (CPU has no model cache). `(run_id, node_id)` is UNIQUE — one row
per DAG cell. `project=None` ⇒ `config.project`. Normally the **dispatcher** calls
this when a node's deps are satisfied; a host rarely enqueues node-jobs directly.

```python
node_queue.insert_skipped_job(*, run_id, node_id, pipeline_name=None, project=None) -> str
```
Insert a `status='skipped'` marker (when `skip_if` is true) so dependents see a
satisfied predecessor. `queue='cpu'`, `required_model` NULL, empty `node_module` —
pure bookkeeping, no worker touches it.

```python
node_queue.prioritize_node_job(job_id: str) -> dict | None
```
Flag a **queued** node-job "run next" (`is_priority = TRUE`, which sorts first in the
claim `ORDER BY`, ahead of the priority band and the GPU warm-model affinity
tiebreak). No-op on a non-queued row ⇒ returns `None`.

### Ingest jobs (`ingest_jobs`, host-defined queues)

```python
node_queue.enqueue_ingest_job(
    *, task_name: str, queue: str, reason: str = "tick", priority: int = 100,
    args: dict | None = None, conn: Any = None, project: str | None = None,
) -> str
```
Fail-before-write `ValueError` on an unknown `queue` (not in `config.ingest_queues`)
or an unregistered `task_name`. `args` (migration 0008) is the per-job JSON-able dict
handed to the registered callable, so a host can enqueue a *parametrised* task, not
just a parameterless sweep.

The `conn` parameter is the atomicity lever: pass a host psycopg connection and the
INSERT runs on it, so the job row + the host's own domain row + the `ingest_job_ready`
NOTIFY **commit in the caller's transaction**:

```python
with my_pool.connection() as conn:
    scenario_id = insert_my_scenario_row(conn, ...)         # host domain write
    node_queue.enqueue_ingest_job(                          # rides the same txn
        task_name="run_scenario", queue="ingest",
        args={"scenario_id": scenario_id}, conn=conn,
    )
    # commit → the job and the scenario row appear together; the NOTIFY fires in-txn
```
`conn=None` borrows a pooled connection that autocommits.

`ingest_store.enqueue_ingest_job(...)` has the same signature and is what you call to
stay backend-agnostic: for pg/sqlite it delegates to `node_queue`; for redis/mongodb
it maps onto the SPI (`payload = {task_name, reason, args}`, priority **negated** so
"lower ingest number runs first" holds on a DESC-claiming SPI). `conn=` and `project=`
are pg-path only and ignored by the SPI (whose tenancy is `db_namespace`). The
`ingest_store` module's `__all__` also exports `get_ingest_job`,
`claim_next_ingest_job`, `renew_ingest_lease`, `mark_ingest_completed`,
`mark_ingest_failed`, `reclaim_expired_ingest_leases`, `ingest_snapshot`.

### Runs (`workflow_runs`)

A `mode='node'` run is what the dispatcher fans out into node-jobs:

```python
run_store.insert_run(
    *, run_id: str, workflow_name: str, parcel_id: str | None = None,
    out_dir: str | None = None, status: str = "queued", priority: int = 100,
    mode: str = "node", context: dict | None = None, project: str | None = None,
) -> dict
run_store.get_run(run_id) -> dict | None
run_store.update_run(run_id, **fields) -> dict | None     # whitelisted columns only
run_store.delete_run(run_id) -> None                       # children cascade via FK
```
`update_run` accepts only the whitelisted columns (`status`, `priority`,
`current_step_id`, `progress_pct`, `steps_done`, `context`, `input_spec`, `error`,
`out_dir`, `resume_count`, `parcel_id`, `mode`, `queued_at`, `started_at`,
`finished_at`); an unknown column raises `ValueError` (typos surface at call time).
`steps_done`/`context`/`input_spec` are JSONB (wrapped on write); `updated_at` is
always bumped. The engine treats `parcel_id` as an opaque nullable column (it drops
the host's parcels FK), so `run_store` never knows your domain.

### Per-job input snapshots

```python
node_queue.set_input_spec(run_id, node_id, spec: dict | None) -> None
node_queue.set_resolved_inputs(job_id, resolved_inputs: dict) -> None
```
Persist a per-job `input_spec` (for awaiting-input rendering) / the execution-time
resolved-inputs snapshot. Both pre-validate `json.dumps` before any write.

---

## 6. State transitions & idempotency

Every terminal/await transition does

```sql
UPDATE … SET status = '<terminal>' … WHERE … status NOT IN ('completed','failed','cancelled') RETURNING *
```

and returns `None` when the row was already terminal. This `WHERE` is **load-bearing**:
it makes duplicate deliveries and claim-race losers safe and stops a stray second call
from clobbering a finalized `context_delta`. JSON columns are `json.dumps`-validated
*before* any state mutation, so a bad payload fails before the write. Keep this shape
for any new state transition. (The one deliberate exception is `workflow_node_events`
— append-only, no UPDATE path — whose rows instead ride the state-change txn.)

The `*_in_txn` variants run on a **caller-supplied cursor** so the dispatch-event /
node-event row rides the **same transaction** as the state change — the outbox-
atomicity pattern (`node_executor.execute_node` uses these).

### Node-jobs

| function | transition / effect |
|---|---|
| `mark_completed(job_id, *, context_delta: dict, seconds: float, vm_rss_mb_peak: int | None = None) -> dict | None` | non-terminal → `completed`; stamps `context_delta`/`seconds`/`vm_rss_mb_peak`, `COALESCE`s `host_label = claimed_by`. |
| `mark_completed_in_txn(cur, job_id, *, context_delta, seconds, vm_rss_mb_peak=None) -> dict | None` | same, on the caller's cursor (write the dispatch event in the same txn). |
| `mark_failed(job_id, *, error: str, seconds: float | None = None) -> dict | None` | non-terminal → `failed`; `error` truncated to 8000 chars, `host_label` COALESCE. |
| `mark_failed_in_txn(cur, job_id, *, error, seconds=None) -> dict | None` | same, on the caller's cursor. |
| `mark_awaiting_input(job_id) -> dict | None` | non-terminal → `awaiting_input`. |
| `mark_awaiting_input_in_txn(cur, job_id) -> dict | None` | same, on the caller's cursor. |

### Ingest jobs (the twins)

| function | transition |
|---|---|
| `mark_ingest_completed(job_id, *, result: dict | None = None, seconds: float | None = None) -> dict | None` | non-terminal → `completed` (stamps `result`/`seconds`). |
| `mark_ingest_failed(job_id, *, error: str, seconds: float | None = None) -> dict | None` | non-terminal → `failed` (`error` truncated to 8000 chars). |

Same WHERE-not-terminal idempotency contract; `None` when already terminal.
`ingest_store.mark_ingest_completed/failed` wrap these and also cover the SPI backends.

### Claim / lease / recovery (what workers and the orchestrator call)

These are the live queue mechanics; a host rarely calls them directly, but they're
part of the surface:

- `claim_next_cpu_job(...)` / `claim_next_gpu_job(...)` / `claim_next_ingest_job(...)`
  — the `SELECT … FOR UPDATE SKIP LOCKED` atomic `queued → running` claim (stamps lease +
  `claimed_by`, applies the run-cancel guard; GPU adds capability gate + warm-model
  affinity + lane filters). All project-scoped.
- `reclaim_expired_leases()` / `reclaim_expired_ingest_leases()` — re-queue `running`
  rows whose lease lapsed (the sole recovery path for an orphaned row). Intentionally
  broker-wide (the `project` travels with the row).
- `requeue_job_for_retry(job_id)` / `requeue_job_for_retry_in_txn(cur, job_id)` — the
  watchdog-retry mechanic: `running → queued`, bump `watchdog_retries`, jump the queue,
  **no** dispatch event. See [watchdogs.md](watchdogs.md).
- `reclaim_all_running_for_resume(*, project=None)` — orchestrator-boot recovery of `running`
  **node-jobs** whose claiming worker has no fresh heartbeat (heartbeat-scoped — it won't yank a
  still-beating worker's job; the `NOT EXISTS (… worker_heartbeats … last_seen > now() - STALE)` guard).
- `run_store.reenqueue_running_for_resume(*, project=None)` — flips **every** `running` *run* for the
  project back to `queued` **unconditionally** (a full resume — NOT heartbeat-scoped; pairs with the
  node-job reclaim above, which is the one that respects live workers).
- `requeue_running_for_worker(host_label, queue, *, project=None)` — re-queue every
  `running` row a *specific* worker owns (the operator hard-stop path; resume-style, no
  retry bump).
- Lifecycle helpers: `cancel_queued_jobs_for_run(run_id)`,
  `cancel_siblings_after_failure(run_id)`, `delete_non_terminal_jobs_for_run(run_id)`
  (restart primitive), `cancel_orphaned_queued_jobs(*, project=None)`.

### The dispatch-event outbox

`enqueue_dispatch_event_in_txn(cur, run_id, node_id, kind)` writes a
`workflow_dispatch_events` row (`kind ∈ {completed, failed, awaiting_input}`,
DB-CHECK-enforced) in the worker's terminal txn; the orchestrator drains it
(`list_unprocessed_dispatch_events`, `mark_dispatch_event_processed`,
`record_dispatch_event_failure`, `count_unprocessed_dispatch_events`). So fan-out is
retryable and never synchronously coupled to the worker.

### The append-only node-event log (migration 0011)

```python
node_queue.record_node_event(*, run_id, node_id, event_type, job_id=None, attempt=0,
    host_label=None, queue=None, model=None, elapsed_s=None, error=None, detail=None) -> int | None
node_queue.record_node_event_in_txn(cur, *, run_id, node_id, event_type, ...) -> int
```
`record_node_event` opens its **own** connection and is **best-effort** — it swallows
every error and returns `None` on failure, so an event blip can never fail the
load-bearing claim/terminal/watchdog path. `record_node_event_in_txn` rides the
caller's txn (terminal + `requeued` events land atomically with the state change).
`event_type` must be in `NODE_EVENT_TYPES`:

```
claimed, model_load_start, model_load_done, progress_beat, stall_suspected,
stall_trip, gpu_health_trip, budget_trip, requeued, reassigned, lease_renew,
completed, failed, cancelled, error
```
`attempt` is the node-job's `watchdog_retries` at emit time — the cross-attempt key
that ties one node's tries together. `prune_node_events(older_than_days=30) -> int`
trims old rows (the `NodePool` calls it on a sweep).

---

## 7. Introspection / snapshots

All read-only; no host coupling. The `project` filter is the migration-0017
multi-tenant seam (`None` = broker-wide, byte-compatible).

| function | returns |
|---|---|
| `node_queue.snapshot(*, project=None) -> dict` | per-queue counts + up to 50 running/queued rows per `cpu`/`gpu` queue: `{"cpu": {...}, "gpu": {...}, "counts": {"cpu_running": n, …}}`. |
| `node_queue.ingest_snapshot(*, project=None) -> dict` | `{"queues": {q: {queued, running, completed, failed, workers}}}` for the ingest queues (`workers` = fresh `worker_heartbeats` rows < 30 s). `ingest_store.ingest_snapshot()` is the backend-agnostic twin. |
| `node_queue.fleet_snapshot(*, stale_after_s=30.0, project=None) -> list[dict]` | every `worker_heartbeats` row (incl. stale / dead-flagged) ordered by `(queue, host_label)`, each augmented with derived `fresh` and `flagged_dead` flags. The observability read the conductor consumes — see [conductor.md](conductor.md). |
| `node_queue.recent_jobs(*, project=None, status=None, min_retries=0, limit=40) -> list[dict]` | unified newest-first feed across **both** families; each row carries `kind` (`'node'`/`'ingest'`), `name`, `queue`, `status`, `project`, `worker`, timing, `seconds`, `retries`, `error`. `status='failed'` ⇒ dead-jobs view; `min_retries>=1` ⇒ retries view (ingest jobs drop out, no retry counter). |
| `node_queue.list_node_events(job_id, *, limit=200) -> list[dict]` | the per-attempt `workflow_node_events` timeline for one node-job, oldest-first (`[]` for an ingest job / unknown id). |
| `node_queue.list_projects() -> list[str]` | distinct `project` tags across runs/node-jobs/ingest/heartbeats — the option list for a multi-tenant filter. |
| `node_queue.get_node_job(job_id)` / `list_jobs_for_run(run_id)` / `get_ingest_job(job_id)` | single-row / per-run reads. |

```python
snap = node_queue.ingest_snapshot()
for q, s in snap["queues"].items():
    if s["queued"] and not s["workers"]:
        print(f"WARNING: {q} has {s['queued']} queued and no live consumer")
```

---

## 8. Worker ON/OFF control

`worker_controls` (migration 0012) is operator-written **desired** state per
`(host_label, queue)`, deliberately separate from the *observed* `worker_heartbeats`.
A row trigger fires `pg_notify('worker_control', '<host>:<queue>')` in the writer's
txn, so a plain SQL write from any consumer wakes the worker. Full design + the
"why a process exit" rationale: [worker_control.md](worker_control.md).

```python
from queue_workflows import worker_control

worker_control.disable_worker("host-a", "gpu")        # hard stop + stay off
worker_control.enable_worker("host-a", "gpu")         # resume
worker_control.set_worker_control("host-a", "gpu",
    desired_state="off", stop_policy="hard", requested_by="ops", conn=None)
worker_control.get_worker_control("host-a", "gpu")    # row or None
worker_control.desired_state_for("host-a", "gpu")     # 'on' | 'off'  (no row / table absent ⇒ 'on')
```

- `set_worker_control(host_label, queue, *, desired_state, stop_policy="hard",
  requested_by=None, conn=None)` validates `desired_state ∈ {on, off}` and
  `stop_policy` against the `STOP_POLICIES` registry **before** the write; `conn`
  threads the caller's transaction (row + wake NOTIFY commit with the caller's work).
- `get_worker_control` swallows `UndefinedTable`, so a pre-0012 DB is treated as
  all-ON (default-on) — claim workers gate on schema 6/8, not 12.
- `STOP_POLICIES: dict[str, Callable[..., None]]` is the extensibility seam; only
  `"hard"` exists today (`EXIT_CONTROL_HARD_STOP = 79` — re-queue in-flight work, clear
  the GPU busy-ghost, `os._exit(79)`). `"drain"`/`"pause"` are reserved names that slot
  in with no schema change (`stop_policy` is free-form TEXT).
- `WorkerControlWatcher(*, worker, on_exit=None, poll_s=None)` is the daemon the worker
  process runs (LISTEN + 5 s safety poll). Inert when `AI_LEADS_DISABLE_WORKER_CONTROL`
  is set.

Per-machine **LLM server config** (migration 0013) rides the same table via
`set_llm_config(host_label, queue, *, server_type=None, parallelism=None,
vllm_idle_ttl_s=None, conn=None)` (partial, COALESCE-on-update) and
`llm_config_for(host_label, queue) -> LLMConfig` (defaults on a pre-0013 DB). This is a
**soft** config change that never touches `desired_state` and fires the dedicated
`worker_llm_config_changed` channel — see [llm_backends.md](llm_backends.md).

CLI: `queue-worker-control --queue gpu --off [--host H] [--policy hard] [--requested-by W]`
/ `--on` (§12).

---

## 9. Telemetry (HwFeed + hw_metrics)

The GPU claim worker starts one **`HwMetricsSampler`** per host (the call site
guarantees one-per-host; a flock is a secondary guard). Every `SAMPLE_INTERVAL_S`
(5 s) it samples CPU/RAM/swap (psutil, optional `[metrics]` extra), per-GPU
util/VRAM (shells out to `nvidia-smi`/`rocm-smi`, no Python dep), and the
per-container cgroup slice, then fires `NOTIFY hw_metrics, <json>`.

```python
def metrics_dsn() -> str | None
```
Resolves the DSN hw-metrics publishes to / reads from: `config.metrics_db_url_env` if
set (the shared broker), else `config.db_url_env`; `None` if that env var is unset.
**Always a pg DSN** — hw-metrics is NOTIFY/Postgres-only — even when `db_backend` is
something else. If the metrics DSN differs from the queue pool DSN, the publisher
opens its own short autocommit connection per broadcast.

The matching reader is **`HwFeed`** — one wrapper every project imports so they all
show the same broker-sourced, fleet-wide view:

```python
from queue_workflows import hw_feed
feed = hw_feed.HwFeed(stale_after_s=15.0, dsn=None).start()   # background daemon; reads metrics_dsn()
...
return {"hosts": feed.latest_by_host()}     # {host: {**latest_sample, "stale": bool}}
```
`HwFeed` is a daemon thread on a dedicated autocommit LISTEN connection (not the
engine pool), **never fatal** to the host (it logs + reconnects with capped backoff if
the DSN is missing or the connection drops), and holds the latest sample per host in
memory. `latest_by_host()` returns a read-only copy, each entry marked `stale` past
`stale_after_s`. `stop()` ends the thread.

Sample payload keys: `sampled_at`, `host`, `cpu_percent`, `cpu_ours_percent`,
`ram_percent`, `ram_used_mb`, `ram_total_mb`, `ram_ours_used_mb`, `ram_ours_percent`,
`swap_used_mb`, `gpus` (a list of `{id, use_pct, vram_used_mb, vram_total_mb}`). GPU
attribution is intentionally omitted (ROCm doesn't expose per-PID counters).
`total_vram_mb()` resolves the machine's capacity (env override
`AI_LEADS_GPU_VRAM_TOTAL_MB` first, else the largest single probed card ≥ 2048 MB, else
`None` = "unknown" ⇒ callers fail open).

---

## 10. Bootstrap & schema (the two-chain pattern)

The engine owns one migration chain; a host runs a second against its own ledger —
"two ORMs / two chains, one Postgres." `db.bootstrap()` applies the chain idempotently;
`db.downgrade()` reverses it (each step needs a paired `.down.sql`).

```python
from queue_workflows import db

db.bootstrap()                                  # engine chain → queue_schema_version ledger
db.bootstrap(migrations_dir=my_dir,             # the host's domain chain, second ledger
             version_table="my_app_schema_version")

db.current_schema_version() -> int              # 0 on a brand-new DB (never raises UndefinedTable)
db.wait_for_schema(min_version, *, timeout_s=120.0, poll_s=0.5) -> int
db.downgrade(*, to_version=0) -> list[int]
```

| function | role |
|---|---|
| `bootstrap(*, migrations_dir=None, version_table=ENGINE_VERSION_TABLE)` | apply pending migrations. **Only the orchestrator bootstraps.** Concurrency-safe on Postgres via a `pg_advisory_xact_lock` (many orchestrators booting one shared broker is fine); on SQLite the lock is a no-op (single-machine, only the orchestrator bootstraps). `migrations_dir` defaults to the chain for the active relational backend (`migrations/` for pg, `migrations_sqlite/` for sqlite). |
| `wait_for_schema(min_version, …)` | claim workers / scheduler **block** here until the schema is ready rather than racing the migration run. Raises `TimeoutError` if not reached. |
| `current_schema_version(*, version_table=…)` | highest applied version, or 0. |
| `downgrade(*, to_version=0, …)` | roll back > `to_version` (highest-first); raises if a step lacks a `.down.sql`. |
| `db.connection()` / `db.cursor()` | borrow a relational connection (pooled psycopg, or the shared SQLite connection); auto-commit on clean exit, rollback on exception. |
| `db.db_url()` / `db.sqlite_path()` | resolve the DSN / SQLite path from `os.environ[config.db_url_env]` (raises a clear error if unset). |
| `db.listen_with_reconnect(channel, stop_event, loop_body, …)` | the durable LISTEN-with-reconnect helper all five engine LISTEN sites funnel through (survives a PG bounce; a poll-only fake on SQLite). |

`version_table` is the one interpolated identifier and is pinned to a plain SQL
identifier (`^[A-Za-z_][A-Za-z0-9_]*$`). The shipped engine chain
(`queue_workflows/migrations/`, also `queue_workflows.migrations.dir()`):

`0001` runs → `0002` node-jobs → `0003` input submissions → `0004` dispatch events →
`0005` worker heartbeats → `0006` lease + `node_job_ready` trigger → `0007` ingest jobs
+ `ingest_job_ready` trigger → `0008` multi-tenant ingest (per-job `args`, drops the
queue/heartbeat CHECKs) → `0009` dead-worker flag → `0010` `watchdog_retries` → `0011`
`workflow_node_events` → `0012` `worker_controls` + NOTIFY trigger → `0013` per-machine
LLM config → `0014` `worker_heartbeats.llm_servers_available` → `0015` capacity-aware
GPU assignment (`fits_models` + `unassignable`) → `0016` per-node run-next priority flag
→ `0017` `project` tenant tag + 3-column heartbeat PK.

---

## 11. Storage backends (`db_backend`)

The storage layer is selectable: `configure(db_backend="sqlite"|"pg"|"redis"|"mongodb")`.

- **`sqlite`** (default, v1.0.0) and **`pg`** are the two **relational** engine
  backends — they run the *full DAG engine* via the dialect seam
  (`queue_workflows/dialect.py`), speaking the same SQL. Everything in §§5–7 works.
- **`redis`** / **`mongodb`** resolve a `StorageBackend` (`queue_workflows/backends/`,
  one provider per file) — a generic durable-queue SPI (enqueue / claim-exactly-once /
  lease+reclaim / idempotent terminals / the atomic outbox / wake / heartbeat / ON-OFF).
  The SPI is **additive and opt-in**: selecting redis/mongo does **not** re-home the
  orchestrator/worker (a later milestone), and the redis/pymongo drivers import lazily.
  Today `ingest_store.*` is the engine path that routes onto the SPI; use the SPI
  directly (`from queue_workflows.backends import get_backend`) as a standalone
  pluggable durable queue.

Two tenancy seams, opposite intents: **`project`** (relational backends — pg/sqlite) *pools* tenants into one
queue with an exact-match filter; **`db_namespace`** *isolates* tenants on a shared
redis/mongo server. Full contract, caveats, and the parametrized test suite:
[storage_backends.md](storage_backends.md). The SQLite port specifics:
[sqlite_engine_port.md](sqlite_engine_port.md).

**A Postgres consumer MUST opt in** — `configure(db_backend="pg")` **or**
`export QUEUE_WORKFLOWS_DB_BACKEND=pg` (the env knob also reaches the console scripts
that have no host `configure()`). Without it, an `AI_LEADS_DB_URL` pg DSN is read as a
SQLite path.

---

## 12. Console scripts

Installed by `pyproject.toml` (also runnable as `python -m queue_workflows.<mod>`).
All honour `QUEUE_WORKFLOWS_DB_BACKEND` / the configured `db_url_env`.

| script | module | purpose |
|---|---|---|
| `queue-broker` | `queue_workflows.broker:main` | stand up / inspect **THE** shared broker. `--db-backend pg` `--db-url-env BROKER_DSN` `--status` (print the consolidated, broker-wide view without bootstrapping; default action = `db.bootstrap()` then print status). |
| `queue-orchestrator` | `queue_workflows.orchestrator:main` | bootstrap migrations + run the `NodePool` (dispatch / outbox drain / lease-reclaim / input listener). No args. |
| `queue-claim-worker` | `queue_workflows.claim_worker:main` | one worker process (concurrency-1). `--queue` (required) ∈ `{cpu, gpu}` ∪ `config.ingest_queues`; `--lease-seconds`. Custom ingest queue names require a prior `configure(ingest_queues=...)`. |
| `queue-scheduler` | `queue_workflows.scheduler:main` | PG-native ingest ticker (sleeps to the next scheduled minute, enqueues `ingest_jobs`). No args. |
| `queue-worker-control` | `queue_workflows.worker_control:main` | operator ON/OFF for a `(host, queue)` worker. `--queue` (required); mutually exclusive `--on` / `--off`; `--host` (default this box's label), `--policy` (default `hard`), `--requested-by`. |

```bash
export QUEUE_WORKFLOWS_DB_BACKEND=pg
BROKER_DSN=postgresql://…/broker  queue-broker --db-url-env BROKER_DSN          # bootstrap + status
queue-orchestrator &                                                            # dispatch/outbox/reclaim
queue-claim-worker --queue=gpu &                                               # one GPU worker
queue-scheduler &                                                              # ingest cron
queue-worker-control --queue gpu --off                                        # hard-stop + park this box's GPU worker
```

The conductor (`queue-conductor`, `queue-conductor-web`) ships in the separate
`queue-workflows-conductor` distribution — see [conductor.md](conductor.md).
