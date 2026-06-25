# Client database schema (SQLite)

The **client** database is the per-project, embedded data plane a host application
gets out of the box: `import queue_workflows` + `configure()` + `db.bootstrap()`
against a single SQLite file (the zero-config **v1.0.0 default** backend). It holds
the same **ten logical tables** as the shared broker — the same runs, node-jobs,
ingest-jobs, heartbeats, control plane, and event log — but rendered in the SQLite
dialect: JSON-as-`TEXT`, `TEXT` UTC timestamps, no `LISTEN`/`NOTIFY` triggers.

SQLite is **single-writer** (the WAL journal + a busy-timeout serialize writers),
which is exactly right for an embedded, single-process client — but it is *not* the
multi-tenant shared bus. A production client that wants to pool onto the broker opts
into Postgres with `configure(db_backend="pg")` (or `export
QUEUE_WORKFLOWS_DB_BACKEND=pg`) and then shares the broker's **identical logical
schema** — see the Postgres rendering in
[broker_db_schema.md](broker_db_schema.md), its API in
[broker_api.md](broker_api.md), and the engine port notes in
[sqlite_engine_port.md](sqlite_engine_port.md).

> Sibling references: [client_api.md](client_api.md) (the `import queue_workflows`
> public surface that reads/writes these tables), [broker_db_schema.md](broker_db_schema.md)
> (the Postgres rendering of the same ten tables), [multitenant_broker.md](multitenant_broker.md)
> (the `project` tenant tag), [worker_control.md](worker_control.md) (the ON/OFF
> control plane), [watchdogs.md](watchdogs.md) (the liveness model behind the
> lease/retry columns).

The facts below were introspected from a freshly-bootstrapped SQLite file at schema
version **17** (`PRAGMA table_info` / `index_list`). They are ground truth.

**Reading the column tables.** `Null` is whether the column accepts `NULL`, exactly
as `PRAGMA table_info` reports it (`no` = `NOT NULL`, `yes` = nullable). One SQLite
quirk surfaces here: an `INTEGER PRIMARY KEY` rowid-alias reports `Null = yes` even
though it is the primary key (SQLite does not imply `NOT NULL` from an inline
`PRIMARY KEY`); a `TEXT PRIMARY KEY` reports `yes` too **unless** its DDL adds an
explicit `NOT NULL` — e.g. `ingest_jobs.id` reports `no`. The engine never inserts a
`NULL` id, so this is cosmetic either way. `Default` is the literal stored default (`—` = none).

---

## Dialect differences vs Postgres

The same engine runs on Postgres *or* a single SQLite file without forking every
module. The divergence is realized in **two** places:

1. **The DDL** — `db.bootstrap()` applies a *parallel* migration chain selected by
   the active relational backend: `queue_workflows/migrations/` for `pg`,
   `queue_workflows/migrations_sqlite/` for `sqlite`. The SQLite chain mirrors the
   pg chain table-for-table (same names, same columns, same composite PKs, same
   `project` tenant tag) but renders Postgres types into SQLite affinities and
   **omits every `pg_notify` function + trigger** (SQLite has no `LISTEN`/`NOTIFY`).
2. **The hot-path SQL** — the runtime claim/lease/reclaim queries are spliced from
   fragments produced by a process-wide `Dialect` (`queue_workflows/dialect.py`,
   chosen from `config.db_backend`), so the *same* engine code emits Postgres SQL or
   SQLite SQL at execute time. `db.py`'s string-literal-aware translator then
   converts the pyformat (`%s`) statement to SQLite's qmark/named paramstyle.

### Type mapping (DDL)

| Postgres type | SQLite rendering | Notes |
|---|---|---|
| `jsonb` | `TEXT` | JSON stored as text; the engine `json.dumps`/`json.loads` at the boundary. Defaults render as `'{}'` / `'[]'`. |
| `timestamptz` / `timestamp` | `TEXT` | ISO-8601 UTC strings, lexically comparable. The `now()` default becomes `strftime('%Y-%m-%d %H:%M:%f', 'now')` (UTC by convention — SQLite has no tz type). |
| `smallint` | `INTEGER` | `priority`, `resume_count`, `attempt`, `worker_lane`. |
| `double precision` | `REAL` | `seconds`, `progress_pct`, `elapsed_s`. |
| `boolean` | `INTEGER` | `0` / `1`; `is_primary`, `is_priority` default `0`. |
| `bigserial` | `INTEGER PRIMARY KEY` | the rowid alias auto-increments; no separate sequence, no unique autoindex. |
| `text[]` | `TEXT` | JSON array text (`Dialect.array_literal` → `json.dumps`); `known_models`/`fits_models` default `'[]'`, `llm_servers_available` defaults `'["ollama"]'` (vs pg `'{ollama}'`). |

### No NOTIFY triggers — the wake is POLL-based, not push

On Postgres a row trigger fires `pg_notify(...)` **inside the writer's
transaction** so idle workers can block on `LISTEN` and wake the instant a row is
enqueued or a control flips. SQLite has no such mechanism, so **none of these
triggers exist** in the client schema:

| Postgres trigger (dropped on SQLite) | Channel it fired |
|---|---|
| `node_job_ready_notify` (migration 0006) | `node_job_ready` |
| `ingest_job_ready_notify` (migration 0007) | `ingest_job_ready` |
| `worker_control_notify` (migration 0012) | `worker_control` |
| `worker_llm_config_notify` (migration 0013) | `worker_llm_config_changed` |

With no push channel, the client wake is **poll-based**: the claim loop falls back
to its safety-poll cadence rather than a `LISTEN` wake, and `db.listen_with_reconnect`
degrades to a poll-only fake. Correctness is unchanged — only latency-to-wake.

### Hot-path SQL fragments (`dialect.py`)

| Operation | Postgres (`PgDialect`) | SQLite (`SqliteDialect`) |
|---|---|---|
| current time | `now()` | `datetime('now')` |
| future / past offset | `now() + make_interval(secs => %s)` | `datetime('now', ('+' \|\| %s \|\| ' seconds'))` |
| seconds-since-epoch | `EXTRACT(EPOCH FROM col)` | `CAST(strftime('%s', col) AS REAL)` |
| FIFO tiebreak order | `EXTRACT(EPOCH FROM a.created_at)` | `a.rowid` (monotonic with INSERT, never ties) |
| claim concurrency | `FOR UPDATE SKIP LOCKED` | *(empty)* — writers serialize via WAL + busy-timeout, so the single-statement `UPDATE … WHERE id=(SELECT … LIMIT 1)` claim is already atomic |
| null-safe equality | `a IS NOT DISTINCT FROM b` | `a IS b` |
| scalar minimum | `LEAST(…)` | `MIN(…)` |
| array membership | `val = ANY(arr::text[])` | `val IN (SELECT value FROM json_each(arr))` |
| `RETURNING` columns | `alias.col, …` (qualified) | `col, …` (SQLite RETURNING can't alias-qualify) |
| table exists | `to_regclass('public.' \|\| %s)` | `SELECT name FROM sqlite_master WHERE type='table' AND name = %s` |

The `project` tenant column (migration 0017) and every **composite primary key**
(`worker_heartbeats` = `(host_label, queue, project)`, `worker_controls` =
`(host_label, queue)`) are preserved **identically** to Postgres — tenancy and
worker identity behave the same on either backend; only the dialect of the SQL
around them changes.

---

## `workflow_runs`

The queue's substrate: one row per workflow run. Workers claim runnable runs by
`(status='queued', priority, queued_at)`; a `mode='node'` run is what the DAG
dispatcher fans out into node-jobs. **Engine-owned but parcel-agnostic** — the
engine drops the host's `parcels` foreign key, so `parcel_id` is a plain nullable
opaque tag and the queue schema stands alone on a parcel-less DB. `context` /
`steps_done` / `input_spec` are JSON (stored as `TEXT`). Carries the `project`
tenant tag (migration 0017).

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | TEXT | yes | — | primary key (run id). |
| `parcel_id` | TEXT | yes | — | opaque host domain tag; the engine drops the FK. |
| `workflow_name` | TEXT | no | — | which workflow this run executes. |
| `status` | TEXT | no | — | `queued` / `running` / `completed` / `failed` / `cancelled` / `awaiting_input`. |
| `priority` | INTEGER | no | `100` | lower = sooner (pg `smallint`). |
| `current_step_id` | TEXT | yes | — | step/node cursor for `mode='step'`. |
| `progress_pct` | REAL | no | `0.0` | 0–100 progress gauge. |
| `steps_done` | TEXT | no | `'[]'` | JSON array of completed step ids. |
| `context` | TEXT | no | `'{}'` | JSON run context (accumulated `context_delta`s). |
| `input_spec` | TEXT | yes | — | JSON awaiting-input widget spec. |
| `error` | TEXT | yes | — | failure text. |
| `out_dir` | TEXT | yes | — | output directory for this run's artifacts. |
| `mode` | TEXT | no | `'step'` | `'step'` or `'node'` (DAG fan-out). |
| `resume_count` | INTEGER | no | `0` | times the run was resumed (pg `smallint`). |
| `created_at` | TEXT | no | `strftime('%Y-%m-%d %H:%M:%f', 'now')` | UTC TEXT timestamp. |
| `updated_at` | TEXT | no | `strftime('%Y-%m-%d %H:%M:%f', 'now')` | bumped on every `update_run`. |
| `queued_at` | TEXT | yes | — | when it entered `queued` (claim ordering). |
| `started_at` | TEXT | yes | — | first transition to `running`. |
| `finished_at` | TEXT | yes | — | terminal timestamp. |
| `project` | TEXT | no | `''` | tenant tag; `''` = single-tenant sentinel (migration 0017). |

- **Primary key:** `(id)`.
- **Indexes:** `workflow_runs_claim_idx (priority, queued_at)` (partial: `status='queued'`); `workflow_runs_parcel_created_idx (parcel_id, created_at)`; `workflow_runs_status_idx (status)`; `workflow_runs_project_idx (project, status)`; `sqlite_autoindex_workflow_runs_1 (id)` unique (the TEXT-PK autoindex).
- **Triggers:** none.

---

## `workflow_node_jobs`

The node-per-job queue: one **mutable** row per `(run_id, node_id)` DAG cell. The
engine dispatches one *node* at a time onto the `cpu` queue (short-lived workers) or
the `gpu` queue (long-lived workers with a warm-model cache). Carries the
claim/lease columns (`claimed_by`, `lease_expires_at`), the watchdog re-queue
counter (`watchdog_retries`, migration 0010), the capacity-aware `unassignable_*`
red-flag (migration 0015), the `is_priority` "run next" flag (migration 0016), and
the `project` tenant tag (migration 0017). A watchdog re-queue overwrites
`claimed_by`/timing in place and only bumps `watchdog_retries` — the prior attempt's
forensics live in `workflow_node_events`.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | TEXT | yes | — | primary key (job id). |
| `run_id` | TEXT | no | — | parent run; cascade-deletes with the run. |
| `node_id` | TEXT | no | — | logical id inside the workflow JSON. |
| `node_module` | TEXT | no | — | importable node-module name (resolved via the host package). |
| `pipeline_name` | TEXT | yes | — | parent pipeline ref. |
| `queue` | TEXT | no | — | `'cpu'` or `'gpu'`. |
| `required_model` | TEXT | yes | — | GPU model id (CPU rows must leave this NULL). |
| `status` | TEXT | no | — | `queued`/`running`/`completed`/`failed`/`cancelled`/`awaiting_input`/`skipped`. |
| `priority` | INTEGER | no | `100` | lower = sooner (pg `smallint`). |
| `worker_lane` | INTEGER | yes | — | concurrency-lane hint (pg `smallint`). |
| `inputs` | TEXT | no | `'{}'` | JSON declared inputs (may contain `$from`/`$value` refs). |
| `resolved_inputs` | TEXT | yes | — | JSON execute-time `$from` snapshot. |
| `input_spec` | TEXT | yes | — | JSON per-job awaiting-input spec. |
| `context_delta` | TEXT | no | `'{}'` | JSON result merged into the run context on success. |
| `host_label` | TEXT | yes | — | claiming host (COALESCEd from `claimed_by` at terminal). |
| `celery_task_id` | TEXT | yes | — | legacy, unused post-Phase-5 (kept for fidelity). |
| `error` | TEXT | yes | — | failure text (truncated to 8000 chars). |
| `vm_rss_mb_peak` | INTEGER | yes | — | peak RSS telemetry. |
| `seconds` | REAL | yes | — | execution wall time (pg `double precision`). |
| `created_at` | TEXT | no | `strftime('%Y-%m-%d %H:%M:%f', 'now')` | UTC TEXT timestamp. |
| `started_at` | TEXT | yes | — | first transition to `running`. |
| `finished_at` | TEXT | yes | — | terminal timestamp. |
| `claimed_by` | TEXT | yes | — | worker host holding the in-flight lease (migration 0006). |
| `lease_expires_at` | TEXT | yes | — | lease expiry; a reclaim sweep re-queues past this (migration 0006). |
| `watchdog_retries` | INTEGER | no | `0` | watchdog re-queue counter (migration 0010; max default 3). |
| `unassignable_at` | TEXT | yes | — | stamped when no live machine can fit `required_model` (migration 0015). |
| `unassignable_reason` | TEXT | yes | — | human-readable red-flag reason (migration 0015). |
| `is_priority` | INTEGER | no | `0` | "run next" flag, sorts first in the claim order (migration 0016; pg `boolean`). |
| `project` | TEXT | no | `''` | tenant tag (migration 0017). |

- **Primary key:** `(id)`. **Unique:** `(run_id, node_id)` — one row per DAG cell.
- **Indexes:** `workflow_node_jobs_claim_idx (queue, priority, created_at)`; `workflow_node_jobs_project_claim_idx (queue, project, priority, created_at)` (both partial: `status='queued'`); `workflow_node_jobs_run_idx (run_id)`; `workflow_node_jobs_status_idx (status)`; `workflow_node_jobs_model_idx (required_model)`; `workflow_node_jobs_pipeline_idx (pipeline_name)`; `workflow_node_jobs_host_label_idx (host_label)`; `workflow_node_jobs_celery_task_id_idx (celery_task_id)`; `workflow_node_jobs_lease_idx (lease_expires_at)`; `workflow_node_jobs_unassignable_idx (queue, status)`; `sqlite_autoindex_workflow_node_jobs_2 (run_id, node_id)` unique; `sqlite_autoindex_workflow_node_jobs_1 (id)` unique.
- **Triggers:** none (Postgres has `node_job_ready_notify`; the SQLite client polls).

---

## `workflow_input_submissions`

A durable user-input store: the host inserts a row (`status='pending'`) when a user
submits a value for an `awaiting_input` node; the Python `InputListener` polls,
claims, resumes the run via the dispatcher, then marks the row processed. This
replaced a transient `pg_notify` channel that dropped submissions on listener
restart. The partial-unique index enforces uniqueness only for **in-flight**
(`pending`/`processing`) rows so legitimate re-submissions across retries don't
collide.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | TEXT | yes | — | primary key (submission id). |
| `run_id` | TEXT | no | — | parent run; cascade-deletes with the run. |
| `node_id` | TEXT | no | — | the awaiting-input node this value is for. |
| `value` | TEXT | yes | — | JSON submitted value. |
| `status` | TEXT | no | `'pending'` | `pending` / `processing` / `processed` / `failed`. |
| `error` | TEXT | yes | — | processing failure text. |
| `claimed_at` | TEXT | yes | — | when the listener claimed it (reclaim a stuck `processing` row). |
| `created_at` | TEXT | no | `strftime('%Y-%m-%d %H:%M:%f', 'now')` | UTC TEXT timestamp. |
| `processed_at` | TEXT | yes | — | when it reached a terminal state. |

- **Primary key:** `(id)`.
- **Indexes:** `workflow_input_submissions_pending_idx (created_at)` (partial: `status='pending'`); `workflow_input_submissions_inflight_unique (run_id, node_id)` **unique** (partial: `status IN ('pending','processing')`); `sqlite_autoindex_workflow_input_submissions_1 (id)` unique.
- **Triggers:** none.

---

## `workflow_dispatch_events`

The durable dispatcher **outbox**. A worker writes one row here in the **same
transaction** as its terminal `mark_completed`/`mark_failed`/`mark_awaiting_input`,
so fan-out is never synchronously coupled to the worker. The orchestrator drains
unprocessed rows each tick and invokes the dispatcher callback; on callback failure
the row stays `processed_at IS NULL` with `attempts++` and the next tick retries
(poison-flagged after the max). `kind` is one of `completed` / `failed` /
`awaiting_input` (a DB CHECK on Postgres).

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | INTEGER | yes | — | primary key, rowid alias (pg `bigserial`). |
| `run_id` | TEXT | no | — | parent run; cascade-deletes with the run. |
| `node_id` | TEXT | no | — | which node terminalized. |
| `kind` | TEXT | no | — | `completed` / `failed` / `awaiting_input`. |
| `processed_at` | TEXT | yes | — | NULL until the orchestrator drains it. |
| `error` | TEXT | yes | — | last callback failure text. |
| `attempts` | INTEGER | no | `0` | outbox-drain retry count (pg `smallint`). |
| `created_at` | TEXT | no | `strftime('%Y-%m-%d %H:%M:%f', 'now')` | UTC TEXT timestamp. |

- **Primary key:** `(id)` (rowid alias — no unique autoindex).
- **Indexes:** `workflow_dispatch_events_unprocessed_idx (created_at)` (partial: `processed_at IS NULL`).
- **Triggers:** none.

---

## `worker_heartbeats`

The per-host fleet **capacity ledger** — *observed* state. Each claim worker upserts
its `(host_label, queue, project)` row at startup and refreshes `last_seen` every
~10 s; a stopped worker simply ages out of the freshness window (no DELETE on
shutdown needed). The orchestrator stamps `last_flagged_dead_at` when a worker's
heartbeat goes stale **while it still owns a `running` job** (migration 0009). The
LLM-capability and capacity columns (`llm_servers_available` migration 0014;
`vram_total_mb` / `fits_models` migration 0015) advertise what the machine can
actually run, gating model assignment. The 3-column PK (migration 0017) lets two
projects' workers share one `(host_label, queue)` without clobbering each other.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `host_label` | TEXT | no | — | worker host identity (primary key part). |
| `queue` | TEXT | no | — | `cpu` / `gpu` / ingest queue (primary key part). |
| `concurrency` | INTEGER | no | — | capacity this worker contributes (1 by contract). |
| `current_model` | TEXT | yes | — | warm GPU model, for affinity routing. |
| `known_models` | TEXT | no | `'[]'` | JSON array of registered model ids this host knows (pg `text[]`). |
| `last_seen` | TEXT | no | `strftime('%Y-%m-%d %H:%M:%f', 'now')` | freshness heartbeat (stale > 30 s). |
| `last_flagged_dead_at` | TEXT | yes | — | dead-process flag set by the orchestrator (migration 0009). |
| `llm_servers_available` | TEXT | no | `'["ollama"]'` | JSON array of LLM server types this host can run (migration 0014; pg `text[]` default `'{ollama}'`). |
| `vram_total_mb` | INTEGER | yes | — | total GPU VRAM (MB), for capacity-fit (migration 0015). |
| `fits_models` | TEXT | no | `'[]'` | JSON array of model ids that fit this machine's VRAM (migration 0015; pg `text[]`). |
| `project` | TEXT | no | `''` | tenant tag (primary key part, migration 0017). |

- **Primary key:** `(host_label, queue, project)`.
- **Indexes:** `worker_heartbeats_last_seen_idx (last_seen)`; `worker_heartbeats_flagged_dead_idx (last_flagged_dead_at)` (partial: `last_flagged_dead_at IS NOT NULL`); `sqlite_autoindex_worker_heartbeats_1 (host_label, queue, project)` unique (the composite-PK autoindex). The Postgres GIN index over `known_models` has no SQLite analog.
- **Triggers:** none.

---

## `ingest_jobs`

The second job family: standalone, periodic/parametrised **ingest** work with no
DAG, no parent run, no dispatch-event outbox. A scheduler ticker (or a host
directly) inserts rows; an ingest claim worker drains them. It carries the **same**
claim/lease columns as `workflow_node_jobs` so the lease-renew/reclaim machinery is
reused at the SQL-shape level. `queue` is **host-defined** (the `fetch`/`load` CHECK
was dropped in migration 0008; the host validates the name before enqueue), and
`args` (migration 0008) carries the per-job JSON handed to the registered callable
so a host can enqueue a *parametrised* task. Carries the `project` tenant tag
(migration 0017).

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | TEXT | no | — | primary key (job id). |
| `task_name` | TEXT | no | — | registered host ingest callable to run. |
| `queue` | TEXT | no | — | host-defined ingest queue name (e.g. `fetch`/`load`). |
| `reason` | TEXT | no | `'tick'` | provenance: `tick` / `boot` / `manual`. |
| `status` | TEXT | no | `'queued'` | `queued`/`running`/`completed`/`failed`/`cancelled`. |
| `priority` | INTEGER | no | `100` | lower = sooner (pg `smallint`). |
| `result` | TEXT | yes | — | JSON return value of the task. |
| `error` | TEXT | yes | — | failure text (truncated to 8000 chars). |
| `seconds` | REAL | yes | — | execution wall time (pg `double precision`). |
| `claimed_by` | TEXT | yes | — | worker host holding the lease. |
| `lease_expires_at` | TEXT | yes | — | lease expiry; reclaim sweep re-queues past this. |
| `created_at` | TEXT | no | `strftime('%Y-%m-%d %H:%M:%f', 'now')` | UTC TEXT timestamp. |
| `started_at` | TEXT | yes | — | first transition to `running`. |
| `finished_at` | TEXT | yes | — | terminal timestamp. |
| `args` | TEXT | no | `'{}'` | JSON per-job arguments (migration 0008). |
| `project` | TEXT | no | `''` | tenant tag (migration 0017). |

- **Primary key:** `(id)`.
- **Indexes:** `ingest_jobs_claim_idx (queue, priority, created_at)`; `ingest_jobs_project_claim_idx (queue, project, priority, created_at)` (both partial: `status='queued'`); `ingest_jobs_lease_idx (lease_expires_at)`; `sqlite_autoindex_ingest_jobs_1 (id)` unique.
- **Triggers:** none (Postgres has `ingest_job_ready_notify`; the SQLite client polls).

---

## `workflow_node_events`

An **append-only** forensic log of the per-node, per-attempt lifecycle. Because
`workflow_node_jobs` is one mutable row per cell, a watchdog re-queue overwrites the
prior attempt's worker/timing/trip reason — this table keeps them durably so
cross-attempt failure/stall history is queryable after the fact (surfaced as a
per-node timeline). Terminal + `requeued` events ride the **same transaction** as
the state change (the outbox-atomicity pattern); every other event is best-effort.
`attempt` (= `watchdog_retries` at emit time) is the cross-attempt key tying one
node's tries together; `detail` carries the free-form trip metrics. **No UPDATE
path** — it adds no new mutation invariant. `event_type` is one of `claimed`,
`model_load_start`, `model_load_done`, `progress_beat`, `stall_suspected`,
`stall_trip`, `gpu_health_trip`, `budget_trip`, `requeued`, `reassigned`,
`lease_renew`, `completed`, `failed`, `cancelled`, `error`, `unassignable`.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | INTEGER | yes | — | primary key, rowid alias (pg `bigserial`). |
| `run_id` | TEXT | no | — | parent run; cascade-deletes with the run. |
| `node_id` | TEXT | no | — | logical id inside the workflow JSON. |
| `job_id` | TEXT | yes | — | `workflow_node_jobs.id` at emit (nullable: survives row churn). |
| `attempt` | INTEGER | no | `0` | `watchdog_retries` at emit — cross-attempt key (pg `smallint`). |
| `event_type` | TEXT | no | — | lifecycle event (see the set above). |
| `host_label` | TEXT | yes | — | emitting / claiming worker. |
| `queue` | TEXT | yes | — | `cpu` / `gpu` / `fetch` / `load`. |
| `model` | TEXT | yes | — | `required_model`, when relevant. |
| `elapsed_s` | REAL | yes | — | seconds in this attempt (pg `double precision`). |
| `error` | TEXT | yes | — | trip reason / failure text (truncated). |
| `detail` | TEXT | no | `'{}'` | JSON free-form trip metrics. |
| `created_at` | TEXT | no | `strftime('%Y-%m-%d %H:%M:%f', 'now')` | UTC TEXT timestamp. |

- **Primary key:** `(id)` (rowid alias — no unique autoindex).
- **Indexes:** `workflow_node_events_node_idx (run_id, node_id, created_at)` (the hot per-node timeline read); `workflow_node_events_created_idx (created_at)` (the retention-sweep predicate).
- **Triggers:** none.

---

## `worker_controls`

The operator worker ON/OFF control plane — *desired* state, deliberately separate
from the *observed* `worker_heartbeats` (an OFF state must persist precisely while
the worker is **not** beating). An operator (or a host UI / the
`queue-worker-control` CLI) writes a `(host_label, queue)` row; the worker's
`WorkerControlWatcher` enforces it. On Postgres a trigger NOTIFYs the worker
immediately; on the SQLite client the watcher relies on its safety poll instead.
`stop_policy` is free-form TEXT (only `"hard"` is wired today; `"drain"`/`"pause"`
slot in with no schema change). The LLM-config columns (migration 0013) ride the
same row: `desired`, operator-set, read by the same worker. See
[worker_control.md](worker_control.md).

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `host_label` | TEXT | no | — | worker host (primary key part). |
| `queue` | TEXT | no | — | `cpu` / `gpu` / ingest queue (primary key part). |
| `desired_state` | TEXT | no | `'on'` | `'on'` or `'off'` (a DB CHECK on Postgres). |
| `stop_policy` | TEXT | no | `'hard'` | how to transition on→off; free-form TEXT, validated in Python. |
| `requested_by` | TEXT | yes | — | provenance (operator / service name). |
| `updated_at` | TEXT | no | `strftime('%Y-%m-%d %H:%M:%f', 'now')` | UTC TEXT timestamp. |
| `llm_server_type` | TEXT | no | `'ollama'` | `'ollama'` or `'vllm'` (migration 0013). |
| `llm_parallelism` | INTEGER | no | `1` | concurrent requests the LLM sidecar serves (migration 0013; CHECK ≥ 1 on pg). |
| `vllm_idle_ttl_s` | INTEGER | no | `60` | idle seconds before the vllm sidecar is reaped (migration 0013; CHECK ≥ 0 on pg). |

- **Primary key:** `(host_label, queue)`.
- **Indexes:** `sqlite_autoindex_worker_controls_1 (host_label, queue)` unique (the composite-PK autoindex).
- **Triggers:** none (Postgres has `worker_control_notify` + `worker_llm_config_notify`; the SQLite client polls).

---

## `workflow_run_files`

An index of the output **files** a run produced — one row per artifact, keyed by run
+ relative path. Created alongside `workflow_runs` in migration 0001; rows
cascade-delete with their run. `is_primary` flags the headline artifact of a step;
`kind` lets a UI group/filter by file type.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | INTEGER | yes | — | primary key, rowid alias (pg `bigserial`). |
| `run_id` | TEXT | no | — | parent run; cascade-deletes with the run. |
| `step_id` | TEXT | no | — | which step/node produced the file. |
| `rel_path` | TEXT | no | — | path relative to the run's `out_dir`. |
| `kind` | TEXT | no | — | artifact type/category. |
| `size_bytes` | INTEGER | no | `0` | file size (pg `bigint`). |
| `is_primary` | INTEGER | no | `0` | headline-artifact flag (pg `boolean`). |
| `created_at` | TEXT | no | `strftime('%Y-%m-%d %H:%M:%f', 'now')` | UTC TEXT timestamp. |

- **Primary key:** `(id)` (rowid alias — no unique autoindex). **Unique:** `(run_id, rel_path)`.
- **Indexes:** `workflow_run_files_run_idx (run_id)`; `workflow_run_files_kind_idx (kind)`; `sqlite_autoindex_workflow_run_files_1 (run_id, rel_path)` unique.
- **Triggers:** none.

---

## `queue_schema_version`

The engine's migration **ledger** — one row per applied migration. `db.bootstrap()`
inserts a row as each `NNNN_*.sql` step applies; `db.current_schema_version()` reads
`MAX(version)` (0 on a brand-new DB), and claim workers `db.wait_for_schema(n)`
against it instead of racing the migration run. A host with its own domain tables
runs a *second* chain against its own ledger table — "two chains, one database."

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `version` | INTEGER | yes | — | primary key; the applied migration number (rowid alias). |
| `applied_at` | TIMESTAMPTZ | no | `strftime('%Y-%m-%d %H:%M:%f', 'now')` | when this migration applied. (Declared type is literally `TIMESTAMPTZ` here — the ledger keeps the pg type name even on SQLite, where it carries NUMERIC affinity; the value is a UTC TEXT timestamp.) |

- **Primary key:** `(version)` (rowid alias — no unique autoindex).
- **Indexes:** none.
- **Triggers:** none.
