# The broker API — Postgres as the message bus

`broker_parrot` (Python package `queue_workflows`) is a **Postgres-as-queue
workflow engine**: the database *is* the broker. `INSERT`ing a row is enqueuing
work; a single `SELECT … FOR UPDATE SKIP LOCKED` claim flips it to `running` and
stamps a lease; an in-trigger `pg_notify` wakes idle listeners; a renewed lease
keeps a long job alive while a lapsed one is reclaimed; and a terminal state plus
its dispatch event are written in **one transaction** (the durable outbox). This
document is the contract reference for that broker — the tables, columns, NOTIFY
channels, claim protocol, outbox atomicity, the additive `StorageBackend` SPI,
the `project` tenant tag, the migration chain, the conductor HTTP surface, and
how a non-Python consumer can talk to the broker over plain SQL. Everything here
is read out of the source (`queue_workflows/migrations/*.sql`, `node_queue.py`,
`node_executor.py`, `dialect.py`, `backends/`, `conductor/.../web.py`) — the
signatures and SQL match exactly.

> **See also:** [`broker_db_schema.md`](broker_db_schema.md) — the full per-column
> Postgres schema (one markdown table per DB table); [`client_api.md`](client_api.md) —
> the client SDK that drives this broker.

> Sibling docs: [`storage_backends.md`](storage_backends.md) (the redis/mongo
> SPI), [`multitenant_broker.md`](multitenant_broker.md) (the shared-broker
> `project` model), [`conductor.md`](conductor.md) (the fleet view + the
> client/conductor split), [`worker_control.md`](worker_control.md) (operator
> ON/OFF), [`watchdogs.md`](watchdogs.md) (the liveness / reclaim layers).

---

## 1. Overview — the DB is the bus

### 1.1 Three process roles, one Postgres

All three roles run as separate processes against the **same** database; there is
no broker daemon, no message-queue server — Postgres is the entire transport.

| Role | Module | What it does | Bootstraps migrations? |
|---|---|---|---|
| **Orchestrator** | `orchestrator.py` → `node_pool.NodePool` | Expands `mode='node'` runs into node-jobs (the DAG dispatcher), drains the dispatch-event outbox, runs the lease-reclaim + dead-worker sweeps, and resumes parked input nodes (`InputListener`). Runs no node bodies. | **Yes** — the only role that does |
| **Claim worker** | `claim_worker.ClaimWorker` | One process == one worker, concurrency-1 by contract. `LISTEN`s its wake channel, then drains its queue greedily. `cpu`/`gpu` draw DAG node-jobs from `workflow_node_jobs`; ingest-family queues draw from `ingest_jobs`. | No — calls `db.wait_for_schema(...)` |
| **Scheduler** | `scheduler.Ticker` | A plain Python loop (not `pg_cron`) that sleeps to the next scheduled minute and enqueues `ingest_jobs`. | No |

The orchestrator owns the migration run (`db.bootstrap` takes a
`pg_advisory_xact_lock`); claim workers and the scheduler block on
`db.wait_for_schema(min_version)` rather than racing the migration. The required
floor is per-queue: cpu/gpu wait for schema **≥ 6**, ingest for **≥ 8**
(`claim_worker._NODE_REQUIRED_VERSION = 6`, `_INGEST_REQUIRED_VERSION = 8`).

### 1.2 The queue mechanism in one breath

`INSERT`ing a `workflow_node_jobs` (or `ingest_jobs`) row *is* enqueuing the
work. The claim is a **single statement** — a `FOR UPDATE SKIP LOCKED` subselect
picks the next claimable row and the outer `UPDATE` flips `queued → running`,
stamping `claimed_by` + `lease_expires_at` (see [§4](#4-the-claim-protocol)). A
trigger fires `pg_notify(...)` **inside the writer's transaction** (see
[§3](#3-notify-channels)), so there is no "row queued but no wake" window. A 1 s
safety poll on each listener covers a dropped NOTIFY.

### 1.3 Two relational dialects (the engine), three flat backends (the SPI)

The engine's own connection runs on one of **two relational dialects** selected
by `config.db_backend` via `queue_workflows/dialect.py`:

- `db_backend="sqlite"` (the **default** since v1.0.0) → `SqliteDialect`
- `db_backend="pg"` → `PgDialect` — emits **exactly** the SQL the engine has
  always used, so a live Postgres deploy is byte-identical.

`redis`/`mongodb` do **not** host the relational DAG engine — they select the
flat-queue `StorageBackend` SPI ([§6](#6-the-storagebackend-spi)). The SQL in
this document is the Postgres rendering; the dialect produces the SQLite
equivalents from the same call sites (`now()` → `datetime('now')`,
`make_interval(secs => …)` → `datetime('now', '+N seconds')`,
`FOR UPDATE SKIP LOCKED` → `""` because SQLite's WAL serializes writers, etc.).

---

## 2. Schema / tables reference

The tables below are the broker's **language-agnostic interface** — any consumer
that can speak SQL can enqueue, claim, and inspect work by reading and writing
them directly ([§10](#10-direct-db-interop)). The engine owns one migration chain
(`queue_workflows/migrations/NNNN_*.sql`, [§8](#8-migrations)); the migration
that introduced each column is noted in parentheses.

Four claim/identity tables carry a `project TEXT NOT NULL DEFAULT ''` tenant tag
(migration 0017): `workflow_runs`, `workflow_node_jobs`, `ingest_jobs`,
`worker_heartbeats`. `''` is the single-tenant sentinel ([§7](#7-multi-tenancy)).

### 2.1 `workflow_runs` (0001) — the DAG run substrate

One row per workflow run. The orchestrator expands a `mode='node'` run into
node-jobs.

| Column | Type / notes |
|---|---|
| `id` | `TEXT PRIMARY KEY` |
| `parcel_id` | `TEXT` — engine-agnostic, **nullable**; the host's parcels FK is intentionally dropped so the engine never knows the host's domain |
| `workflow_name` | `TEXT NOT NULL` |
| `status` | `TEXT NOT NULL` |
| `priority` | `SMALLINT NOT NULL DEFAULT 100` |
| `current_step_id`, `progress_pct`, `steps_done`, `error`, `resume_count` | run bookkeeping |
| `context` | `JSONB NOT NULL DEFAULT '{}'` |
| `input_spec` | `JSONB` |
| `out_dir` | `TEXT` |
| `mode` | `TEXT NOT NULL DEFAULT 'step' CHECK (mode IN ('step','node'))` |
| `created_at` / `updated_at` / `queued_at` / `started_at` / `finished_at` | `TIMESTAMPTZ` |
| `project` | `TEXT NOT NULL DEFAULT ''` (0017) |

Indexes: partial claim index `(priority, queued_at) WHERE status='queued'`;
`(parcel_id, created_at DESC)`; `(status)`; `workflow_runs_project_idx
(project, status)` (0017). (The vestigial `current_step_idx` column is omitted
from fresh engine DBs.)

### 2.2 `workflow_run_files` (0001) — per-run output ledger

| Column | Type / notes |
|---|---|
| `id` | `BIGSERIAL PRIMARY KEY` |
| `run_id` | `TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE` |
| `step_id`, `rel_path`, `kind` | `TEXT NOT NULL` |
| `size_bytes` | `BIGINT NOT NULL DEFAULT 0` |
| `is_primary` | `BOOLEAN NOT NULL DEFAULT FALSE` |
| `created_at` | `TIMESTAMPTZ` |
| | `UNIQUE (run_id, rel_path)` |

`node_executor._update_run_thumbnail` upserts the `is_primary` run-card thumbnail
here on each node completion.

### 2.3 `workflow_node_jobs` (0002) — the DAG node-per-job queue

One **mutable** row per `(run_id, node_id)` — the cell a watchdog re-queue
overwrites (`claimed_by`/timing change, `watchdog_retries` bumps; the prior
attempt's detail is preserved only in `workflow_node_events`, §2.7). Queues
`cpu`/`gpu`.

| Column | Type / notes |
|---|---|
| `id` | `TEXT PRIMARY KEY` |
| `run_id` | `TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE` |
| `node_id` | `TEXT NOT NULL` — logical id inside the workflow JSON |
| `node_module` | `TEXT NOT NULL` — importable name (host node-module package) |
| `pipeline_name` | `TEXT` |
| `queue` | `TEXT NOT NULL CHECK (queue IN ('cpu','gpu'))` |
| `required_model` | `TEXT` |
| `status` | `TEXT NOT NULL`, `CHECK status IN (queued, running, completed, failed, cancelled, awaiting_input, skipped)` |
| `priority` | `SMALLINT NOT NULL DEFAULT 100` |
| `worker_lane` | `SMALLINT` |
| `inputs` / `resolved_inputs` / `input_spec` / `context_delta` | `JSONB` (`resolved_inputs` = the execute-time `$from` snapshot) |
| `host_label` | `TEXT` |
| `claimed_by` | `TEXT` (0006) — the lease owner; **the real host identity** the engine stamps |
| `lease_expires_at` | `TIMESTAMPTZ` (0006) |
| `vm_rss_mb_peak`, `seconds`, `error`, `celery_task_id` | worker telemetry / legacy |
| `watchdog_retries` | `INTEGER NOT NULL DEFAULT 0` (0010) — watchdog re-queue counter |
| `unassignable_at` / `unassignable_reason` | `TIMESTAMPTZ` / `TEXT` (0015) — capacity red-flag |
| `is_priority` | `BOOLEAN NOT NULL DEFAULT FALSE` (0016) — operator "run next" |
| `created_at` / `started_at` / `finished_at` | `TIMESTAMPTZ` |
| `project` | `TEXT NOT NULL DEFAULT ''` (0017) |
| | `UNIQUE (run_id, node_id)` |
| | `CHECK (queue = 'gpu' OR required_model IS NULL)` — CPU rows carry no model |

Indexes: claim index `(queue, priority, created_at) WHERE status='queued'`; lease
index `(lease_expires_at) WHERE status='running'`; a GPU model index, a host /
pipeline index, the unassignable partial index `(queue, status) WHERE
required_model IS NOT NULL` (0015); and the 0017 project claim index
`(queue, project, priority, created_at) WHERE status='queued'`.

### 2.4 `workflow_input_submissions` (0003) — durable user-input store

A host (e.g. Rails) inserts a `pending` row when a user answers an
`awaiting_input` node; the orchestrator's `InputListener` claims it, resumes the
run, and marks it `processed`. Replaced a transient `pg_notify('input_submitted')`
channel that dropped submissions on listener restart.

| Column | Type / notes |
|---|---|
| `id` | `TEXT PRIMARY KEY` |
| `run_id` | `TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE` |
| `node_id` | `TEXT NOT NULL` |
| `value` | `JSONB` |
| `status` | `TEXT NOT NULL DEFAULT 'pending'`, `CHECK status IN (pending, processing, processed, failed)` |
| `error`, `claimed_at`, `created_at`, `processed_at` | bookkeeping |
| | partial `UNIQUE (run_id, node_id) WHERE status IN ('pending','processing')` — only in-flight rows are unique, so re-submissions across retries don't 409 |

### 2.5 `workflow_dispatch_events` (0004) — the durable dispatcher outbox

A worker writes a row here **in the same transaction** as its terminal mark; the
orchestrator drains unprocessed rows on every tick and invokes the dispatcher
callback. See [§5](#5-the-durable-outbox).

| Column | Type / notes |
|---|---|
| `id` | `BIGSERIAL PRIMARY KEY` |
| `run_id` | `TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE` |
| `node_id` | `TEXT NOT NULL` |
| `kind` | `TEXT NOT NULL CHECK (kind IN ('completed','failed','awaiting_input'))` |
| `processed_at` | `TIMESTAMPTZ` — NULL until drained |
| `error` | `TEXT` |
| `attempts` | `SMALLINT NOT NULL DEFAULT 0` — **orchestrator-side drain** retries (distinct from `workflow_node_jobs.watchdog_retries`) |
| `created_at` | `TIMESTAMPTZ` |

Index: `(created_at) WHERE processed_at IS NULL`.

### 2.6 `worker_heartbeats` (0005) — observed fleet capacity ledger

Each claim worker upserts every ~10 s; a stopped worker simply ages out of the
30 s freshness window (no `DELETE` on shutdown). This is **observed** state —
kept deliberately separate from the **desired** `worker_controls` (§2.8).

| Column | Type / notes |
|---|---|
| `host_label` | `TEXT NOT NULL` |
| `queue` | `TEXT NOT NULL` (the cpu/gpu-only CHECK was dropped in 0008 so ingest workers heartbeat too) |
| `concurrency` | `INTEGER NOT NULL` |
| `current_model` | `TEXT` — GPU busy / affinity hint (NULL for cpu/ingest) |
| `known_models` | `text[] NOT NULL DEFAULT '{}'` (0005) |
| `last_seen` | `TIMESTAMPTZ NOT NULL DEFAULT now()` |
| `last_flagged_dead_at` | `TIMESTAMPTZ` (0009) — set by the dead-worker sweep, cleared by a fresh heartbeat |
| `llm_servers_available` | `text[] NOT NULL DEFAULT '{ollama}'` (0014) — observed LLM capability |
| `vram_total_mb` | `INTEGER` (0015) |
| `fits_models` | `text[] NOT NULL DEFAULT '{}'` (0015) — worker-computed VRAM-fit set |
| `project` | `TEXT NOT NULL DEFAULT ''` (0017) |
| | `PRIMARY KEY (host_label, queue, project)` — 3-col since 0017 |

> **BREAKING (0017):** the PK widened from `(host_label, queue)` to
> `(host_label, queue, project)`. Any raw-SQL writer using
> `INSERT … ON CONFLICT (host_label, queue)` must move to
> `node_queue.upsert_worker_heartbeat` (3-col `ON CONFLICT`) or it errors with
> *"no unique or exclusion constraint matching the ON CONFLICT specification."*

### 2.7 `ingest_jobs` (0007) — standalone periodic / parametrised work

No DAG, no parent run, no `$from`, no outbox. Host-defined queues. Carries the
**same** lease columns as node-jobs so the reclaim machinery is reused.

| Column | Type / notes |
|---|---|
| `id` | `TEXT`, `PRIMARY KEY (id)` |
| `task_name` | `TEXT NOT NULL` — **no DB CHECK**; the host validates against `config.ingest_task_map` before enqueue |
| `queue` | `TEXT NOT NULL` — **no DB CHECK** (the `fetch`/`load` CHECK was dropped in 0008); host validates against `config.ingest_queues` |
| `reason` | `TEXT NOT NULL DEFAULT 'tick'` — provenance (`tick`/`boot`/`manual`) |
| `status` | `TEXT NOT NULL DEFAULT 'queued'`, `CHECK status IN (queued, running, completed, failed, cancelled)` |
| `priority` | `SMALLINT NOT NULL DEFAULT 100` |
| `result` / `error` / `seconds` | terminal bookkeeping |
| `args` | `JSONB NOT NULL DEFAULT '{}'` (0008) — per-job arguments for parametrised tasks |
| `claimed_by` / `lease_expires_at` | identical lease bookkeeping to node-jobs |
| `created_at` / `started_at` / `finished_at` | `TIMESTAMPTZ` |
| `project` | `TEXT NOT NULL DEFAULT ''` (0017) |

Indexes: claim index `(queue, priority, created_at) WHERE status='queued'`; lease
index `(lease_expires_at) WHERE status='running'`; 0017 project claim index
`(queue, project, priority, created_at) WHERE status='queued'`.

### 2.8 `workflow_node_events` (0011) — append-only per-attempt forensic log

The mutable node-job row loses prior-attempt detail on a watchdog re-queue; this
**append-only** log keeps the per-attempt lifecycle durably. **No UPDATE path** —
so it adds no new mutation invariant; terminal / `requeued` events ride the
state-change txn, everything else is best-effort. `attempt` (= `watchdog_retries`
at emit) ties the tries of one node together.

| Column | Type / notes |
|---|---|
| `id` | `BIGSERIAL PRIMARY KEY` |
| `run_id` | `TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE` |
| `node_id` | `TEXT NOT NULL` |
| `job_id` | `TEXT` (nullable — survives the mutable row's churn) |
| `attempt` | `SMALLINT NOT NULL DEFAULT 0` — the cross-attempt key |
| `event_type` | `TEXT NOT NULL`, CHECK set below |
| `host_label`, `queue`, `model` | emitting context |
| `elapsed_s` | `DOUBLE PRECISION` |
| `error` | `TEXT` |
| `detail` | `JSONB NOT NULL DEFAULT '{}'` — free-form trip metrics (`max_sm_pct`, `ram_anchor_mb`, `model_load_s`, `exit_code`, …) |
| `created_at` | `TIMESTAMPTZ` |

`event_type` CHECK set: `claimed`, `model_load_start`, `model_load_done`,
`progress_beat`, `stall_suspected`, `stall_trip`, `gpu_health_trip`,
`budget_trip`, `requeued`, `reassigned`, `lease_renew`, `completed`, `failed`,
`cancelled`, `error`, **`unassignable`** (added 0015). Indexes:
`(run_id, node_id, created_at)` (the per-node timeline) and `(created_at)` (the
retention sweep, `prune_node_events`, default 30-day retention).

### 2.9 `worker_controls` (0012, +0013) — operator desired ON/OFF state

**Desired** state, written by an operator / host and read by the worker. Kept
apart from the observed `worker_heartbeats` precisely because an OFF state must
persist while the worker is **not** beating. Not project-tagged (worker identity
is `host_label:queue`). See [`worker_control.md`](worker_control.md).

| Column | Type / notes |
|---|---|
| `host_label` | `TEXT NOT NULL` |
| `queue` | `TEXT NOT NULL` |
| `desired_state` | `TEXT NOT NULL DEFAULT 'on' CHECK (desired_state IN ('on','off'))` |
| `stop_policy` | `TEXT NOT NULL DEFAULT 'hard'` — **free-form, no CHECK** (host validates vs the `STOP_POLICIES` registry; only `"hard"` exists today, `"drain"`/`"pause"` are reserved) |
| `requested_by` | `TEXT` |
| `updated_at` | `TIMESTAMPTZ` |
| `llm_server_type` | `TEXT NOT NULL DEFAULT 'ollama' CHECK (llm_server_type IN ('ollama','vllm'))` (0013) |
| `llm_parallelism` | `INTEGER NOT NULL DEFAULT 1 CHECK (llm_parallelism >= 1)` (0013) |
| `vllm_idle_ttl_s` | `INTEGER NOT NULL DEFAULT 60 CHECK (vllm_idle_ttl_s >= 0)` (0013) |
| | `PRIMARY KEY (host_label, queue)` |

An **absent row** — or a DB predating 0012 — is treated as **ON** (the default-on
contract; `get_worker_control` swallows `UndefinedTable`). Claim workers gate on
schema 6/8, never 12, so the engine runs unchanged before 0012.

### 2.10 `queue_schema_version` — the migration ledger

The engine's version ledger (`db.ENGINE_VERSION_TABLE = "queue_schema_version"`),
one `version INTEGER PRIMARY KEY` row per applied migration. `db.bootstrap()`
applies the chain idempotently; `db.downgrade()` reverses it. A host runs a
**second** chain against its own ledger (`bootstrap(migrations_dir=...,
version_table=...)`) — "two chains, one Postgres." See [§8](#8-migrations).

---

## 3. NOTIFY channels

Wakes are **best-effort** so an idle listener can block instead of polling. Every
trigger fires `pg_notify` **inside the writer's transaction**, so there is no
"row written but no wake" window — and a plain SQL write from *any* DB consumer
wakes the worker with no app-side NOTIFY code. A safety poll (1 s on the claim
loop, 5 s on the control watcher) covers a dropped NOTIFY.

| Channel | Payload | Fired by | Migration |
|---|---|---|---|
| `node_job_ready` | the queue name (`cpu`/`gpu`) | trigger `node_job_ready_notify` `AFTER INSERT OR UPDATE OF status` on `workflow_node_jobs`, `WHEN NEW.status='queued'` | 0006 |
| `ingest_job_ready` | the host-defined ingest queue name | trigger `ingest_job_ready_notify`, same INSERT/UPDATE-of-status shape on `ingest_jobs` | 0007 |
| `worker_control` | `host_label \|\| ':' \|\| queue` (**`:`** separator) | trigger `worker_control_notify` `AFTER INSERT OR UPDATE` on `worker_controls` (fires on **every** write) | 0012 |
| `worker_llm_config_changed` | `host_label \|\| '\|' \|\| queue` (**`\|`** separator) | trigger `worker_llm_config_notify`; stays **quiet** on an UPDATE touching none of the three LLM columns | 0013 |
| `hw_metrics` | a JSON telemetry snapshot | `hw_metrics.py` (not a queue wake — a dashboard feed) | — |

The two trigger functions for the queues share this shape:

```sql
CREATE OR REPLACE FUNCTION notify_node_job_ready() RETURNS trigger AS $$
BEGIN
    IF NEW.status = 'queued' THEN
        PERFORM pg_notify('node_job_ready', NEW.queue);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

So a status flip *back* to `queued` (a lease reclaim or a watchdog re-queue)
re-fires the wake exactly like a fresh `INSERT`. The dedicated
`worker_llm_config_changed` channel exists so an LLM-config edit does **not** look
like an ON/OFF change to the hard-stop watcher (note the distinct `:` vs `|`
separators).

---

## 4. The claim protocol

### 4.1 One statement: `SELECT … FOR UPDATE SKIP LOCKED`

The exactly-once claim is a single statement. A `SKIP LOCKED` subselect picks the
next claimable row; the outer `UPDATE` flips `queued → running` and stamps the
lease. The `{order}` fragment is the **only** interpolation and is built **only**
from validated ints / fixed dialect fragments — never caller strings.

```sql
-- node_queue._CLAIM_SQL (the {…} are dialect/validated fragments)
UPDATE workflow_node_jobs AS j
SET status = 'running',
    started_at = now(),
    worker_lane = %(worker_lane)s,
    claimed_by = %(host)s,
    lease_expires_at = {lease_expr}        -- now() + make_interval(secs => %(lease_s)s)
WHERE j.id = (
    SELECT c.id FROM workflow_node_jobs c
    WHERE c.queue = %(queue)s
      AND c.status = 'queued'
      AND c.project = %(project)s          -- exact-match tenant filter (0017)
      AND EXISTS (                         -- run-cancel guard
          SELECT 1 FROM workflow_runs r
          WHERE r.id = c.run_id
            AND r.status NOT IN ('cancelled', 'failed')
      )
      {capability}                         -- GPU capability / lane / pool terms
    ORDER BY {order}
    {skip_locked}                          -- 'FOR UPDATE SKIP LOCKED' on pg, '' on sqlite
    LIMIT 1
)
RETURNING *
```

The **run-cancel guard** means a worker can't even *claim* a job whose parent run
was cancelled or failed out from under it. `{skip_locked}` is empty on SQLite
because its WAL serializes writers, so the single-statement
`UPDATE … WHERE id=(SELECT … LIMIT 1)` is already atomic.

### 4.2 The three claim helpers

| Function | Signature (keyword-only after `*`) | Ordering / extras |
|---|---|---|
| `claim_next_cpu_job` | `(worker_lane=0, *, host=None, lease_s=DEFAULT_LEASE_S, host_priority=0, project=None)` | `is_priority DESC, priority ASC, (creation × host_dir) ASC`. No model cache ⇒ no affinity. |
| `claim_next_gpu_job` | `(worker_lane=0, current_model=None, *, host, lease_s, host_priority=0, known_models=None, require_model=None, pool_modules=None, project=None)` | `is_priority DESC, (affinity) DESC, priority ASC, (creation × host_dir) ASC` — affinity outranks the priority band; only `is_priority` outranks affinity. |
| `claim_next_ingest_job` | `(queue, *, host=None, lease_s=600, project=None)` | `priority ASC, created_at ASC`. **No** run-cancel join (no parent run). Targets `ingest_jobs`. |

All return the claimed row `dict` or `None`. The GPU claim adds, on top of the
base statement:

- **Capability gate** — only claim a job whose `required_model` is in
  `known_models` (rendered `c.required_model = ANY(%(known_models)s::text[])`, or
  the SQLite `json_each` form), or that needs no model. With an **empty**
  `known_models` it falls back to claim-any so a cold worker can't wedge the
  queue.
- **Warm-model affinity tiebreak** — `c.required_model IS NOT DISTINCT FROM
  %(current_model)s` (null-safe) sorts first within the band, so consecutive
  same-model jobs don't reload.
- **`host_priority` direction** — `_host_dir(host_priority)` returns `+1`
  (baseline, oldest-first → claim the head) or `-1` (an explicit overflow host
  with `host_priority < 0`, newest-first → claim the tail), multiplied into the
  creation-order term.
- **Model-presence lane filter** (`require_model`: `True` ⇒ model-backed only,
  `False` ⇒ no-model only, `None` ⇒ either) and **VLM-pool eligibility**
  (`pool_modules`) — split a two-lane GPU worker's claim sets.

### 4.3 The lease and its renewal

`DEFAULT_LEASE_S = 600`. **Lease length is independent of job duration** — a live
worker renews `lease_expires_at` (~every `HEARTBEAT_INTERVAL_S = 10 s`, via a
`LeaseRenewer`) while the job runs, so a multi-hour job survives. Only a
dead/wedged worker lets the lease lapse. `STALE_WORKER_AFTER_S = 30` (3× the
cadence) is the freshness window the gauges and the dead-worker detector use
(env-overridable via `AI_LEADS_STALE_WORKER_AFTER_S`). The lease expression is
`now() + make_interval(secs => %(lease_s)s)` on pg / `datetime('now', '+N
seconds')` on SQLite.

### 4.4 Reclaim — the sole recovery path

A lapsed lease means the owner died or wedged. The reclaim sweep is the **only**
recovery path for an orphaned `running` row.

| Function | What it does |
|---|---|
| `reclaim_expired_leases() -> list[{id, run_id, node_id}]` | One CASE-on-target `UPDATE`: a **running** parent ⇒ flip `running → queued`, clear the lease, jump priority (`LEAST(priority, 10)`), which re-fires `node_job_ready`; a **terminal** parent (cancelled/failed/completed) ⇒ flip to `cancelled` instead (avoids a ghost queued row no worker can ever claim, since the claim filters out non-running parents). |
| `reclaim_expired_ingest_leases() -> list[{id, task_name, queue}]` | The ingest twin: flip lapsed `running → queued`, clear lease, bump priority, re-fire `ingest_job_ready`. |

Both are **intentionally broker-wide** (not project-scoped): reclaim is a pure
status-driven recovery that leaves each row's `project` tag intact, so a re-queued
row is still claimed only by its own project's worker — and running it broker-wide
makes any orchestrator a recovery backstop for any project's orphaned lease.
Idempotent under concurrency (`WHERE status='running'`).

### 4.5 Watchdog re-queue vs operator stop vs dead-worker flag

Three more transitions sit alongside the lease path (full design in
[`watchdogs.md`](watchdogs.md)):

| Function | Behavior |
|---|---|
| `requeue_job_for_retry(job_id)` / `requeue_job_for_retry_in_txn(cur, job_id)` | **Watchdog re-queue** of ONE running node-job by id: flip `running → queued`, clear lease, `LEAST(priority, 10)`, **increment `watchdog_retries`**, write **no** dispatch event (the run stays `running`; only this node re-runs). CAS-guarded `WHERE status='running'`. |
| `requeue_running_for_worker(host, queue)` | **Operator stop** re-queue: redistribute a worker's in-flight rows, resume-style, with **no** `watchdog_retries` bump (an operator turning a machine off is not a fault). |
| `flag_stale_workers_holding_running_jobs(*, stale_after_s=None, project=None) -> list[{host_label, queue, project, last_seen, running_jobs}]` | **Last-resort dead-worker detector**, run by the orchestrator (a separate, GIL-independent process). Flags any worker whose heartbeat is stale **while it still owns a running job** (join `claimed_by = host_label AND queue AND project`). Stamps `last_flagged_dead_at`; does **not** touch jobs or kill anything — the lease-reclaim recovers the job; this flags the dead *process* for a host-supervisor to bounce. Idempotent (re-flags only after recover + re-stale). |

### 4.6 Idempotent terminal transitions

`mark_completed` / `mark_failed` / `mark_awaiting_input` (and the ingest twins
`mark_ingest_completed` / `mark_ingest_failed`) all share the load-bearing shape:

```sql
UPDATE workflow_node_jobs
SET status = 'completed', finished_at = now(), context_delta = %s, seconds = %s, ...
WHERE id = %s
  AND status NOT IN ('completed', 'failed', 'cancelled')   -- the guard
RETURNING *
```

They return `None` when the row was already terminal. This `WHERE` makes
duplicate deliveries and claim-race losers safe and stops a stray second call
from clobbering a finalized `context_delta`. JSON columns are pre-validated
(`json.dumps`) before the write so a bad payload fails before any state mutation.
The `_in_txn` variants run on a caller cursor — the basis of the outbox below.

---

## 5. The durable outbox — one-transaction atomicity

The worker→dispatcher handoff is an **outbox**, not a synchronous call. When a
worker finalizes a node it writes the terminal status **and** a
`workflow_dispatch_events` row in **one transaction**, in
`node_executor.execute_node`:

```python
with _db_connection() as conn, conn.cursor() as cur:
    row = node_queue.mark_completed_in_txn(
        cur, job_id, context_delta=context_delta, seconds=seconds,
        vm_rss_mb_peak=_rss_mb(),
    )
    if row is None:                       # already terminal → duplicate delivery
        return "skipped"                  # …and NO event is written
    node_queue.enqueue_dispatch_event_in_txn(
        cur, job["run_id"], job["node_id"], "completed",
    )
# the forensic node event is best-effort, AFTER this txn commits
```

Because both writes ride one transaction, the system can never be in the state
"node terminal but no event" or "event but node not terminal." The orchestrator
(`NodePool._tick`) drains unprocessed events and calls
`on_node_completed`/`on_node_failed`/`on_node_awaiting_input`; a failing callback
leaves `processed_at IS NULL`, bumps `attempts`, and is retried next tick (poison-
flagged after the drain's max attempts). The drain helpers:

| Function | Purpose |
|---|---|
| `enqueue_dispatch_event_in_txn(cur, run_id, node_id, kind) -> int` | Insert the outbox row in the caller's txn. `kind ∈ {completed, failed, awaiting_input}` (DB CHECK). |
| `list_unprocessed_dispatch_events(*, limit=50)` | The `processed_at IS NULL` backlog, oldest-first. |
| `mark_dispatch_event_processed(event_id)` | Stamp `processed_at = now()`. |
| `record_dispatch_event_failure(event_id, error)` | `attempts += 1` + record error, leaving `processed_at` NULL so the next tick retries. |
| `count_unprocessed_dispatch_events() -> int` | Cheap COUNT for the snapshot / startup health. |

The same atomicity pattern backs the append-only event log:
`record_node_event_in_txn(cur, …)` rides the state-change txn for terminal /
`requeued` events, while `record_node_event(…)` opens its own connection and is
**best-effort** (swallows every error) for the non-terminal emit sites — an
event-history blip can never fail the load-bearing claim/terminal/watchdog path.

---

## 6. The `StorageBackend` SPI

Beyond the relational engine, the **storage layer itself** is selectable:
`configure(db_backend="sqlite"|"pg"|"redis"|"mongodb")`. `sqlite`/`pg` run the
full DAG engine via the dialect seam (§1.3); `redis`/`mongodb` resolve a flat
`StorageBackend` — a generic durable-queue SPI in `queue_workflows/backends/`
(one provider per file). The SPI is **additive and opt-in**: the legacy engine
modules talk to the relational store directly, so selecting redis/mongo does
**not** re-home the orchestrator/worker (a later milestone). Full treatment in
[`storage_backends.md`](storage_backends.md); the contract reference follows.

A backend is constructed `(url, namespace)` and is thread-safe (each call borrows
its own connection / pipeline). Resolve the process-wide instance with
`backends.get_backend(*, namespace=None)` (cached per `(backend, namespace,
url)`), or build one explicitly with `backends.build_backend(name, *, url,
namespace="")`.

### 6.1 The port (`backends/base.py`, `StorageBackend`)

| Method | Contract |
|---|---|
| `ensure_schema()` | Idempotently create durable structures (PG tables / Mongo indexes; no-op for Redis). |
| `close()` | Release pooled connections / clients. |
| `enqueue(queue, payload, *, job_id=None, priority=0) -> str` | Append a `queued` job and fire the wake — same durable write (no "queued but no wake"). |
| `claim(queue, worker, *, lease_s) -> Job \| None` | Atomically take the oldest claimable (highest-priority) job: flip `running`, stamp `claimed_by` + `lease_expires_at = now+lease_s`, bump `attempts`. **Exactly-once under contention.** |
| `renew_lease(job_id, worker, *, lease_s) -> bool` | Extend the lease iff still `running` and owned by `worker`. |
| `reclaim_expired(*, queue=None) -> list[str]` | Re-queue every `running` job whose lease lapsed; re-fire the wake. The sole recovery path. |
| `requeue_for_retry(job_id) -> Job \| None` | Watchdog re-queue: `running → queued`, keep `attempts`, write **no** event. |
| `mark_completed(job_id, *, result=None) -> Job \| None` | Flip → `completed` iff not terminal; `None` on already-terminal (idempotency guard). |
| `mark_failed(job_id, *, error=None) -> Job \| None` | Flip → `failed` iff not terminal. |
| `complete_with_event(job_id, event_type, *, result=None, detail=None) -> Job \| None` | **Atomic outbox**: go `completed` **and** append one event, both-or-neither; `None` ⇒ already terminal ⇒ **no event** (second-delivery no-op). |
| `fail_with_event(job_id, event_type, *, error=None, detail=None) -> Job \| None` | The failure-path twin. |
| `get(job_id) -> Job \| None` | By id, within the namespace. |
| `counts(queue) -> {queued, running, completed, failed}` | Snapshot. |
| `events(*, since=0, limit=1000) -> list[Event]` | Outbox events with `seq > since`, oldest-first, namespace-scoped. |
| `notify(queue)` / `subscribe(*queues) -> WakeListener` | Out-of-band wake / a `with sub: sub.wait(timeout)` listener (timeout doubles as the safety poll). |
| `heartbeat(host, queue, *, current_model=None, stale_after_s=30.0)` / `workers(queue)` | Upsert liveness / list fresh workers. |
| `set_control(host, queue, *, desired_state, stop_policy="hard", requested_by=None)` / `desired_state(host, queue) -> str` | Operator ON/OFF; `desired_state` returns `"off"` only on an explicit OFF row, else `"on"` (default-on). |

The canonical `Job` shape is backend-neutral (a plain dict): `id`, `queue`,
`namespace`, `status`, `payload`, `priority`, `attempts`, `claimed_by`,
`lease_expires_at`, `result`, `error`, `created_at`, `updated_at` — times are
**epoch seconds (float)**. Terminal statuses (`TERMINAL_STATUSES =
{completed, failed}`) are the idempotency guard.

### 6.2 The three backends

| Guarantee | `pg` | `redis` | `mongodb` |
|---|---|---|---|
| Storage | own tables `qw_jobs` / `qw_events` / `qw_workers` / `qw_controls` (each row carries a `namespace` column) | key prefix `qw:<namespace>:` | one **database** per namespace (`qw_<namespace>`), collections `jobs`/`events`/`workers`/`controls`/`counters`/`wake` |
| Claim exactly-once | `FOR UPDATE SKIP LOCKED` | a registered **Lua** script (`ZPOPMIN`-style) | `find_one_and_update` |
| Atomic outbox | one **transaction** | one **Lua** script | one **multi-doc transaction** |
| Wake | `pg_notify` on a per-namespace channel + `LISTEN` | **pub/sub** on a per-namespace channel | **change stream** on a capped `wake` collection |
| Namespace isolation | `namespace` column filter on every query | key prefix | separate database |

### 6.3 The two invariants that keep the SPI honest

1. **No driver leaks.** No method takes or returns a cursor / pipeline / session.
   The outbox atomicity is exposed as one high-level call
   (`complete_with_event`/`fail_with_event`) each backend implements in its own
   idiom — so PG internals can't bleed into redis/mongo call sites.
2. **Namespace-bound.** Each instance is bound to one namespace (constructor arg,
   `"" → "default"`) and scopes every key/row/collection by it, so two tenants on
   one redis/mongo server can't see each other's jobs.

> Caveats (from `storage_backends.md`): Redis has no cross-key ACID txn —
> atomicity is Lua on a **single instance** (not Cluster); MongoDB transactions
> **and** change streams need a **replica set** (single-node RS is fine), else
> `*_with_event` and the wake fail loudly. The `redis`/`pymongo` drivers import
> lazily, so a sqlite/pg-only deploy needs neither installed
> (`pip install 'queue_workflows[redis]'` / `[mongodb]`).

---

## 7. Multi-tenancy — the `project` tenant tag

Two distinct, **inverse** mechanisms exist; don't conflate them:

| Mechanism | Where | Effect |
|---|---|---|
| `project` (migration 0017) | the **relational engine** (`workflow_runs`, `workflow_node_jobs`, `ingest_jobs`, `worker_heartbeats`) | **Pools** many projects onto one shared broker queue; each client claims only its own rows by exact match. |
| `db_namespace` (config) | the **`StorageBackend` SPI** (redis/mongo) | **Isolates** tenants on a shared server so they can't see each other. |

The `project` design rule is **exact-match-always**: every queue row carries
`project TEXT NOT NULL DEFAULT ''`; enqueue stamps it (from the parent run or
`config.project`); claim filters `AND project = <this client's project>`
unconditionally (see the `_CLAIM_SQL` in §4.1). `''` is the single-tenant
sentinel — every row is `''`, the filter `project=''` matches them all, so a
pre-0017 single-tenant deploy is byte-compatible with **zero** host wiring. A
multi-tenant client sets `configure(project="X")` (or exports
`QUEUE_WORKFLOWS_PROJECT=X`) and only ever sees its own rows; claiming another
tenant's row is not even expressible.

Recovery/telemetry helpers that join on `host_label` are **project-scoped** (on a
shared broker a machine name is no longer globally unique):
`upsert_worker_heartbeat` (3-col upsert), `flag_stale_workers_holding_running_jobs`,
`flag_unassignable_gpu_jobs`, `vlm_pool_should_defer`, `clear_worker_current_model`,
`reclaim_all_running_for_resume`, `requeue_running_for_worker`. The lease reclaims
(`reclaim_expired_leases` / `reclaim_expired_ingest_leases`) are deliberately
**un**scoped — they act on the row in place and the project travels with it.
`snapshot` / `ingest_snapshot` / `fleet_snapshot` / `recent_jobs` take an optional
`project` filter (`None` = broker-wide). `list_projects()` returns the distinct
tags across all four tables for a UI filter. Full design (cutover, deployment,
phases): [`multitenant_broker.md`](multitenant_broker.md).

---

## 8. Migrations

The engine owns one chain, shipped as package data and tracked in the
`queue_schema_version` ledger (§2.10). `db.bootstrap()` applies pending
migrations idempotently; on Postgres it takes a `pg_advisory_xact_lock` keyed on
the version table, so many processes (every project's orchestrator booting against
one shared broker) can call it concurrently and safely — the lock holder applies
the chain, every waiter re-reads the ledger and finds nothing to do.
`db.downgrade()` runs the paired `.down.sql` steps; `db.wait_for_schema(
min_version)` blocks a non-bootstrapping process until the schema is ready.

The chain:

| # | Migration | Adds |
|---|---|---|
| 0001 | `queue_runs` | `workflow_runs` + `workflow_run_files` |
| 0002 | `node_jobs` | `workflow_node_jobs` (consolidated final shape) |
| 0003 | `input_submissions` | `workflow_input_submissions` |
| 0004 | `dispatch_events` | `workflow_dispatch_events` (the outbox) |
| 0005 | `worker_heartbeats` | `worker_heartbeats` (+ `current_model`, `known_models`) |
| 0006 | `pg_queue_lease` | lease columns + `node_job_ready` trigger |
| 0007 | `ingest_jobs` | `ingest_jobs` + `ingest_job_ready` trigger |
| 0008 | `multitenant_ingest` | per-job `args JSONB`; drops the `fetch`/`load` queue CHECK and the cpu/gpu `worker_heartbeats` CHECK (allow-lists move host-side) |
| 0009 | `worker_heartbeats_dead_flag` | `worker_heartbeats.last_flagged_dead_at` |
| 0010 | `node_job_watchdog_retries` | `workflow_node_jobs.watchdog_retries` |
| 0011 | `node_events` | `workflow_node_events` (append-only log) |
| 0012 | `worker_controls` | `worker_controls` + `worker_control` trigger |
| 0013 | `worker_controls_llm` | per-machine LLM config columns + `worker_llm_config_changed` trigger |
| 0014 | `worker_heartbeats_llm_servers` | `worker_heartbeats.llm_servers_available` |
| 0015 | `capacity_aware_assignment` | `vram_total_mb` / `fits_models`, `unassignable_at`/`_reason`, `unassignable` event type |
| 0016 | `node_priority_flag` | `workflow_node_jobs.is_priority` |
| 0017 | `project_tenant` | `project` tag on the four claim/identity tables + the 3-col `worker_heartbeats` PK + project-aware indexes |

**Two chains, one Postgres.** A host with its own domain tables runs a *second*
chain via `db.bootstrap(migrations_dir=..., version_table=...)` against its own
ledger — the engine and the host each own an independent chain on the same DB.

```python
from queue_workflows import db
db.bootstrap()                                   # engine chain → queue_schema_version
db.bootstrap(migrations_dir=MY_DIR,              # host chain → its own ledger
             version_table="my_schema_version")
```

Every migration is additive + idempotent (`IF NOT EXISTS` / drop-then-add), so
re-running on an already-migrated DB is a no-op. (The 0017 down-migration is only
safe on a single-tenant all-`''` DB — see `multitenant_broker.md`.)

---

## 9. Conductor HTTP API

`queue-conductor-web` (in the separately-installable
`queue-workflows-conductor` distribution; `conductor/queue_workflows_conductor/
web.py`) serves a read-only operator view of one shared broker over pure stdlib
`http.server` — **no web framework, no JS, zero new runtime deps**. It reads
whatever DB the client's `db_url_env` points at via the project-aware engine
primitives (`node_queue.snapshot` / `ingest_snapshot` / `fleet_snapshot` /
`recent_jobs` / `list_projects` / `list_node_events`). See
[`conductor.md`](conductor.md).

```bash
queue-conductor-web                                   # http://127.0.0.1:8787, read-only
queue-conductor-web --db-backend pg --db-url-env BROKER_DSN
queue-conductor-web --host 0.0.0.0 --port 9000 --enable-writes
```

### 9.1 Endpoints

| Method | Path | Query | Purpose |
|---|---|---|---|
| `GET` | `/`, `/index.html` | `project` (omit ⇒ all), `view ∈ {all, retries, dead}` | The dashboard: KPI strip, shared cpu/gpu cards, ingest cards, recent-activity feed, fleet table. |
| `GET` | `/job/<id>` | `kind ∈ {node, ingest}` (default `node`) | Job-detail + the per-attempt `workflow_node_events` timeline (`[]` for ingest). 404 on unknown id. |
| `POST` | `/control` | form: `host`, `queue`, `desired_state` | **Opt-in** worker ON/OFF (`worker_control.set_worker_control`, `requested_by="conductor-web"`). |
| `POST` | `/requeue` | form: `job_id` | **Opt-in** re-queue a running node-job (`node_queue.requeue_job_for_retry`). |

`GET` is always read-only. The dashboard auto-refreshes via
`<meta http-equiv="refresh">` every `REFRESH_S = 5` s. A DB blip renders an error
page (500) rather than killing the server.

### 9.2 Writes are opt-in and CSRF-guarded

The two `POST` actions exist **only** with `--enable-writes`
(`ConductorWebHandler.writes_enabled`); otherwise a `POST` returns **403** ("writes
disabled"). There is **no** cancel/delete — the surface is exactly ON/OFF +
re-queue. When writes are enabled:

- **CSRF / same-origin guard.** A `POST` whose `Origin` header is present and
  whose netloc doesn't match the `Host` header is rejected **403** ("cross-origin
  POST rejected"). A request with no `Origin` (curl / a same-origin form) is
  allowed. Cheap defence so enabling writes + binding `0.0.0.0` can't be driven by
  another tab the operator has open.
- **POST/redirect/GET.** A successful write replies **303** to the `Referer`'s
  **path+query only** (never its scheme/host — no open redirect; falls back to
  `/`).

---

## 10. Direct DB interop — talking to the broker over plain SQL

Because the DB *is* the bus, any consumer that can speak SQL + `LISTEN`/`NOTIFY`
is a first-class broker client — no Python required. The contract:

### 10.1 Enqueue by `INSERT`

Insert a `queued` row; the trigger fires the wake inside your transaction.

```sql
-- A DAG node-job (queue ∈ {cpu, gpu}; gpu may carry a required_model, cpu must not)
INSERT INTO workflow_node_jobs
    (id, run_id, pipeline_name, node_id, node_module,
     queue, required_model, project, status, priority, inputs, context_delta, created_at)
VALUES
    (gen_random_uuid()::text, :run_id, :pipeline, :node_id, :module,
     'gpu', :model, '', 'queued', 100, '{}'::jsonb, '{}'::jsonb, now());
-- → trigger fires pg_notify('node_job_ready', 'gpu') in THIS txn

-- A standalone ingest job (queue + task_name are host-validated, not DB-checked)
INSERT INTO ingest_jobs (id, task_name, queue, reason, args, project, status, priority, created_at)
VALUES (gen_random_uuid()::text, :task, :queue, 'manual', '{}'::jsonb, '', 'queued', 100, now());
-- → trigger fires pg_notify('ingest_job_ready', :queue) in THIS txn
```

The Python helpers `node_queue.enqueue_node_job(...)` and
`enqueue_ingest_job(... , conn=...)` do exactly this (the latter accepts a caller
connection so the NOTIFY rides the host's own transaction). They fail-before-write
on a bad queue/task name — a raw SQL consumer should likewise only use registered
names, since the queue/task allow-lists moved host-side (0007/0008).

### 10.2 Wait for work with `LISTEN`

```sql
LISTEN node_job_ready;   -- payload is the queue name ('cpu' / 'gpu')
LISTEN ingest_job_ready; -- payload is the ingest queue name
```

Block on the notification, but keep a periodic safety poll (the engine uses 1 s)
because a wake is best-effort and a subscriber that was down misses it.

### 10.3 Claim atomically

Run the single-statement claim. A non-Python consumer reproduces `_CLAIM_SQL`
(§4.1) — the `FOR UPDATE SKIP LOCKED` subselect makes it exactly-once under
contention, and the run-cancel guard + `project` filter must be preserved:

```sql
UPDATE workflow_node_jobs AS j
SET status = 'running', started_at = now(),
    claimed_by = :worker, lease_expires_at = now() + make_interval(secs => 600)
WHERE j.id = (
    SELECT c.id FROM workflow_node_jobs c
    WHERE c.queue = 'gpu' AND c.status = 'queued' AND c.project = ''
      AND EXISTS (SELECT 1 FROM workflow_runs r
                  WHERE r.id = c.run_id AND r.status NOT IN ('cancelled','failed'))
    ORDER BY c.is_priority DESC, c.priority ASC, EXTRACT(EPOCH FROM c.created_at) ASC
    FOR UPDATE SKIP LOCKED
    LIMIT 1)
RETURNING *;
```

### 10.4 Keep the lease alive, then finalize atomically

While running, renew the lease faster than it expires (the engine renews every
10 s on a 600 s lease):

```sql
UPDATE workflow_node_jobs
SET lease_expires_at = now() + make_interval(secs => 600)
WHERE id = :job_id AND status = 'running' AND claimed_by = :worker;
```

Finalize the node and its outbox event in **one transaction** so a downstream
DAG fan-out is never lost (§5):

```sql
BEGIN;
  UPDATE workflow_node_jobs
  SET status = 'completed', finished_at = now(), context_delta = :delta, seconds = :secs
  WHERE id = :job_id AND status NOT IN ('completed','failed','cancelled');
  -- if the UPDATE returned a row (not already terminal):
  INSERT INTO workflow_dispatch_events (run_id, node_id, kind)
  VALUES (:run_id, :node_id, 'completed');
COMMIT;
```

If you don't write the dispatch event, the orchestrator never expands the
downstream nodes — the run stalls. A consumer that doesn't participate in the DAG
(standalone `ingest_jobs`) skips the event: ingest jobs have no run, no outbox.

### 10.5 Heartbeat (so the fleet view / claim affinity see you)

Heartbeats must go through the 3-col `ON CONFLICT` (the 0017 PK):

```sql
INSERT INTO worker_heartbeats
    (host_label, queue, project, concurrency, last_seen, current_model, known_models,
     llm_servers_available, vram_total_mb, fits_models)
VALUES (:host, 'gpu', '', 1, now(), :model, '{}', '{ollama}', :vram, '{}')
ON CONFLICT (host_label, queue, project) DO UPDATE
SET concurrency = EXCLUDED.concurrency, current_model = EXCLUDED.current_model,
    last_seen = EXCLUDED.last_seen, last_flagged_dead_at = NULL;
```

A live refresh clears any `last_flagged_dead_at` the orchestrator's dead-worker
sweep had set. A worker that stops simply ages out of the 30 s window — no
`DELETE` needed.

### 10.6 Flip a worker ON/OFF over plain SQL

A row write to `worker_controls` from any consumer wakes the worker's
`WorkerControlWatcher` via the `worker_control` trigger — no app-side NOTIFY code:

```sql
INSERT INTO worker_controls (host_label, queue, desired_state, requested_by)
VALUES (:host, 'gpu', 'off', 'ops')
ON CONFLICT (host_label, queue) DO UPDATE
SET desired_state = EXCLUDED.desired_state, requested_by = EXCLUDED.requested_by,
    updated_at = now();
-- → trigger fires pg_notify('worker_control', :host || ':gpu') in THIS txn
```

A worker absent from the table is treated as **ON** (default-on), so you only ever
write a row to turn one OFF or back ON. See
[`worker_control.md`](worker_control.md).
