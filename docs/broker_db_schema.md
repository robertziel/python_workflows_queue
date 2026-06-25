# The broker database schema — Postgres

`broker_parrot` (Python package `queue_workflows`) runs the **broker** on
Postgres: the database *is* the message bus. There is no broker daemon and no
message-queue server — `INSERT`ing a row is enqueuing work, a single
`SELECT … FOR UPDATE SKIP LOCKED` claim flips it `running`, an in-trigger
`pg_notify` wakes idle listeners inside the writer's transaction, and a terminal
state plus its dispatch event are written in one transaction (the durable
outbox). This document is the column-level reference for that database — one
section per table, each with a purpose paragraph, the full column list, and its
primary key / indexes / triggers.

> Sibling docs: [`broker_api.md`](broker_api.md) (the broker contract — claim
> protocol, outbox atomicity, NOTIFY channels, the `StorageBackend` SPI, the
> conductor HTTP surface), [`multitenant_broker.md`](multitenant_broker.md) (the
> shared-broker `project` model), [`worker_control.md`](worker_control.md)
> (operator ON/OFF), [`watchdogs.md`](watchdogs.md) (liveness / reclaim).

## How the schema is owned and applied

The engine owns **one migration chain** —
`queue_workflows/migrations/NNNN_*.sql` (`0001` … `0017`, each with a paired
`.down.sql`), shipped as package data. `db.bootstrap()` applies it idempotently
under a Postgres advisory lock (only the orchestrator bootstraps; claim workers
call `db.wait_for_schema(min_version)` and block rather than racing the run).
Every applied step is recorded in the **`queue_schema_version` ledger** (one row
per version), which is how `wait_for_schema` and the per-queue
`_REQUIRED_SCHEMA_VERSION` gate know the schema is ready.

A host with its own domain tables runs a **second** chain against its own ledger
(`db.bootstrap(migrations_dir=…, version_table=…)`) — *"two chains, one
Postgres."* That pattern is visible in the live `workflow_runs` table below: the
last six columns (`target_kind` … `parent_run_id`) and the
`workflow_runs_region_created_idx` / `workflow_runs_parent_run_id_idx` indexes
are **not** part of the engine chain — a host added them to the shared table via
its own chain. They are documented here because they exist in the live broker,
flagged as host-chain (not engine-owned).

This is the Postgres backend specifically (`configure(db_backend="pg")` or
`QUEUE_WORKFLOWS_DB_BACKEND=pg`). It leans on Postgres-native features
throughout: **`jsonb`** for every structured payload (`context`, `inputs`,
`context_delta`, `args`, `result`, `value`, `detail`, …), **`timestamptz`** for
all timing, partial + GIN indexes on the hot paths, and four **NOTIFY triggers**
(`node_job_ready_notify`, `ingest_job_ready_notify`, `worker_control_notify`,
`worker_llm_config_notify`) that call `pg_notify(...)` **inside the writer's
transaction** — so a plain SQL write from any DB consumer wakes the right
listener with no app-side NOTIFY code and with no "row written but no wake"
window. Migration `0017` threads a **`project`** tenant tag (`TEXT NOT NULL
DEFAULT ''`) through the queue tables and widens the `worker_heartbeats`
primary key to three columns so one shared broker can pool ≥2 projects.

> Notation: `Type` uses the canonical Postgres spelling (`timestamptz` for
> `timestamp with time zone`, `text[]` for the array columns); `Null` is `no`
> for `NOT NULL`. `bigint` PK columns are `BIGSERIAL` (default
> `nextval(<table>_id_seq)`).

---

## `workflow_runs` (migration 0001)

Runs are the queue's substrate and the parent of the DAG. A `mode='node'` run is
expanded by the orchestrator's dispatcher into per-node jobs; workers never claim
runs directly in the node path, but the table carries the run-level status,
progress, accumulated `context`, and lifecycle timing. It is engine-owned and
**parcel-agnostic**: the original host FK from `parcel_id` to a `parcels` table
is dropped, so `parcel_id` is a plain nullable `text` and the queue schema stands
alone on a parcel-less DB.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `text` | no | | PK; the run id |
| `parcel_id` | `text` | yes | | opaque host tag; the engine dropped the `parcels` FK |
| `workflow_name` | `text` | no | | which workflow/pipeline this run executes |
| `status` | `text` | no | | `queued` / `running` / `completed` / `failed` / `cancelled` / … |
| `priority` | `smallint` | no | `100` | lower = sooner; claim `ORDER BY` band |
| `current_step_id` | `text` | yes | | progress pointer |
| `progress_pct` | `real` | no | `0.0` | 0–100 progress |
| `steps_done` | `jsonb` | no | `'[]'::jsonb` | completed-step list |
| `context` | `jsonb` | no | `'{}'::jsonb` | accumulated run context (node `context_delta`s merged in) |
| `input_spec` | `jsonb` | yes | | run-level awaiting-input widget spec |
| `error` | `text` | yes | | failure text |
| `out_dir` | `text` | yes | | run output directory |
| `mode` | `text` | no | `'step'::text` | `CHECK (mode IN ('step','node'))`; `node` = DAG-dispatched |
| `resume_count` | `smallint` | no | `0` | resume bookkeeping |
| `created_at` | `timestamptz` | no | `now()` | |
| `updated_at` | `timestamptz` | no | `now()` | |
| `queued_at` | `timestamptz` | yes | | set when enqueued; part of the claim order |
| `started_at` | `timestamptz` | yes | | |
| `finished_at` | `timestamptz` | yes | | |
| `project` | `text` | no | `''::text` | **tenant tag (0017)**; `''` = single-tenant. Enqueue stamps it, claim filters by it (exact match) |
| `target_kind` | `text` | yes | | host-chain column (not engine-owned) |
| `target_ref` | `text` | yes | | host-chain column (not engine-owned) |
| `region` | `text` | yes | | host-chain column (not engine-owned) |
| `language` | `text` | yes | | host-chain column (not engine-owned) |
| `time_window` | `text` | yes | | host-chain column (not engine-owned) |
| `parent_run_id` | `text` | yes | | host-chain column (not engine-owned) |

- **Primary key:** `(id)` — `workflow_runs_pkey`.
- **Indexes:**
  - `workflow_runs_claim_idx` — `(priority, queued_at) WHERE status = 'queued'` (partial; keeps the hot claim tiny).
  - `workflow_runs_status_idx` — `(status)`.
  - `workflow_runs_project_idx` — `(project, status)` (per-project run history / snapshot filter, 0017).
  - `workflow_runs_parcel_created_idx` — `(parcel_id, created_at DESC)` (per-parcel history view).
  - `workflow_runs_region_created_idx` — host-chain index (not engine-owned).
  - `workflow_runs_parent_run_id_idx` — host-chain index (not engine-owned).
- **Triggers:** none.

---

## `workflow_node_jobs` (migration 0002)

The engine dispatches one **node** at a time, not a whole pipeline step. Each
node-job is the DAG work unit, living on either the `cpu` queue (short-lived
workers) or the `gpu` queue (long-lived workers holding a warm `ModelCache`).
It is one **mutable** row per `(run_id, node_id)`: a watchdog re-queue overwrites
`claimed_by` / timing and bumps `watchdog_retries` in place (the per-attempt
forensic history lives in `workflow_node_events`). The terminal mark and its
dispatch-outbox row are written in one transaction.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `text` | no | | PK; the job id |
| `run_id` | `text` | no | | FK → `workflow_runs(id)` `ON DELETE CASCADE` |
| `node_id` | `text` | no | | logical node id inside the workflow JSON |
| `node_module` | `text` | no | | importable name resolved via the host node-module hook |
| `pipeline_name` | `text` | yes | | parent pipeline ref |
| `queue` | `text` | no | | `CHECK (queue IN ('cpu','gpu'))` |
| `required_model` | `text` | yes | | GPU model id; `CHECK (queue='gpu' OR required_model IS NULL)` |
| `status` | `text` | no | | `CHECK status IN ('queued','running','completed','failed','cancelled','awaiting_input','skipped')` |
| `priority` | `smallint` | no | `100` | claim-order band |
| `worker_lane` | `smallint` | yes | | optional lane hint |
| `inputs` | `jsonb` | no | `'{}'::jsonb` | declared node inputs (may carry `$from`/`$value` refs) |
| `input_spec` | `jsonb` | yes | | per-job awaiting-input widget spec |
| `context_delta` | `jsonb` | no | `'{}'::jsonb` | the node's contribution to run context (merged on success) |
| `host_label` | `text` | yes | | claiming host |
| `celery_task_id` | `text` | yes | | legacy, unused post-Phase-5 (kept for fidelity) |
| `error` | `text` | yes | | failure text |
| `vm_rss_mb_peak` | `integer` | yes | | worker RSS telemetry |
| `seconds` | `double precision` | yes | | wall time |
| `created_at` | `timestamptz` | no | `now()` | part of the claim order |
| `started_at` | `timestamptz` | yes | | |
| `finished_at` | `timestamptz` | yes | | |
| `claimed_by` | `text` | yes | | lease owner (worker host); cleared on reclaim/requeue (0006) |
| `lease_expires_at` | `timestamptz` | yes | | lease lapse time; reclaim sweep re-queues `running` rows past it (0006) |
| `watchdog_retries` | `integer` | no | `0` | watchdog re-queue counter; fail only at `AI_LEADS_WATCHDOG_MAX_RETRIES` (0010) |
| `unassignable_at` | `timestamptz` | yes | | red-flag stamp: no live machine can fit `required_model` (0015) |
| `unassignable_reason` | `text` | yes | | human-readable unassignable reason (0015) |
| `is_priority` | `boolean` | no | `false` | "run next" flag; sorted first in the claim `ORDER BY` (0016) |
| `project` | `text` | no | `''::text` | **tenant tag (0017)** |

> Note: the live table does **not** carry the old `resolved_inputs` column (it
> was dropped from this DB; ordinal 12 is vacant) — the execute-time `$from`
> snapshot is reconstructed at run time rather than persisted here.

- **Primary key:** `(id)` — `workflow_node_jobs_pkey`.
- **Indexes:**
  - `workflow_node_jobs_run_id_node_id_key` — **UNIQUE** `(run_id, node_id)` (one job per node per run).
  - `workflow_node_jobs_claim_idx` — `(queue, priority, created_at) WHERE status = 'queued'` (partial; the original hot claim).
  - `workflow_node_jobs_project_claim_idx` — `(queue, project, priority, created_at) WHERE status = 'queued'` (project-aware claim, 0017).
  - `workflow_node_jobs_lease_idx` — `(lease_expires_at) WHERE status = 'running'` (the reclaim predicate, 0006).
  - `workflow_node_jobs_model_idx` — `(required_model) WHERE queue = 'gpu' AND status = 'queued'` (warm-model affinity ordering).
  - `workflow_node_jobs_unassignable_idx` — `(queue, status) WHERE required_model IS NOT NULL` (the fleet capacity sweep, 0015).
  - `workflow_node_jobs_run_idx` — `(run_id)`.
  - `workflow_node_jobs_status_idx` — `(status)`.
  - `workflow_node_jobs_pipeline_idx` — `(pipeline_name)`.
  - `workflow_node_jobs_host_label_idx` — `(host_label) WHERE host_label IS NOT NULL` (partial).
  - `workflow_node_jobs_celery_task_id_idx` — `(celery_task_id) WHERE celery_task_id IS NOT NULL` (partial, legacy).
- **Triggers:**
  - **`node_job_ready_notify`** `AFTER INSERT OR UPDATE OF status` → `notify_node_job_ready()`: when `NEW.status = 'queued'`, fires `pg_notify('node_job_ready', NEW.queue)` inside the writer's txn — the LISTEN/NOTIFY wake for idle `cpu`/`gpu` claim workers, covering both a fresh enqueue and a status flip back to `queued` (e.g. a lease reclaim).

---

## `workflow_input_submissions` (migration 0003)

The durable user-input store. When a user submits a value for an
`awaiting_input` node, a consumer (e.g. a host's Rails over the shared DB)
`INSERT`s a `pending` row; the orchestrator's `InputListener` polls, claims,
calls `dispatcher.resume_after_input`, then marks the row `processed`. It
replaces a transient `pg_notify('input_submitted', …)` channel that dropped
submissions on a listener restart.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `text` | no | | PK |
| `run_id` | `text` | no | | FK → `workflow_runs(id)` `ON DELETE CASCADE` |
| `node_id` | `text` | no | | the awaiting-input node |
| `value` | `jsonb` | yes | | the submitted value |
| `status` | `text` | no | `'pending'::text` | `CHECK status IN ('pending','processing','processed','failed')` |
| `error` | `text` | yes | | failure text |
| `claimed_at` | `timestamptz` | yes | | set on claim; lets a row stuck in `processing` be reclaimed |
| `created_at` | `timestamptz` | no | `now()` | |
| `processed_at` | `timestamptz` | yes | | |

- **Primary key:** `(id)` — `workflow_input_submissions_pkey`.
- **Indexes:**
  - `workflow_input_submissions_pending_idx` — `(created_at) WHERE status = 'pending'` (the listener's poll).
  - `workflow_input_submissions_inflight_unique` — **UNIQUE** `(run_id, node_id) WHERE status IN ('pending','processing')` (partial; one in-flight submission per node, but legitimate re-submissions across retries don't conflict).
- **Triggers:** none (the orchestrator polls this table).

---

## `workflow_dispatch_events` (migration 0004)

The durable dispatcher **outbox**. When a worker finalizes a node it writes the
terminal status **and** a `workflow_dispatch_events` row in the **same
transaction** (the atomicity contract). `NodePool._tick` drains unprocessed
events and invokes the dispatcher callback
(`on_node_completed` / `on_node_failed` / `on_node_awaiting_input`); a failing
callback leaves `processed_at IS NULL` with `attempts++` and is retried next
tick, and exhausting retries flips the run to `failed` so the user sees a result
instead of a stall. So fan-out is retryable and never synchronously coupled to
the worker.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `bigint` | no | `nextval(workflow_dispatch_events_id_seq)` | PK (`BIGSERIAL`) |
| `run_id` | `text` | no | | FK → `workflow_runs(id)` `ON DELETE CASCADE` |
| `node_id` | `text` | no | | the node that terminated |
| `kind` | `text` | no | | `CHECK (kind IN ('completed','failed','awaiting_input'))` |
| `processed_at` | `timestamptz` | yes | | `NULL` = unprocessed (the drain predicate) |
| `error` | `text` | yes | | last callback error |
| `attempts` | `smallint` | no | `0` | outbox-drain retry counter (distinct from `workflow_node_jobs.watchdog_retries`) |
| `created_at` | `timestamptz` | no | `now()` | drain order |

- **Primary key:** `(id)` — `workflow_dispatch_events_pkey`.
- **Indexes:**
  - `workflow_dispatch_events_unprocessed_idx` — `(created_at) WHERE processed_at IS NULL` (partial; the orchestrator's hot drain).
- **Triggers:** none (the orchestrator polls the outbox).

---

## `worker_heartbeats` (migration 0005)

The per-worker fleet **capacity + capability** ledger (observed, ephemeral
state). Each claim worker upserts its row at startup and refreshes `last_seen`
every ~10 s; a stopped worker simply ages out of the freshness window (no
`DELETE` on shutdown). Consumers SUM `concurrency` over fresh rows for the
capacity gauge, read `current_model` for GPU warm-model affinity routing,
`known_models` / `fits_models` for the capability + VRAM-fit claim gate, and
`last_flagged_dead_at` for the dead-worker recovery flag. **All heartbeat writes
must go through `node_queue.upsert_worker_heartbeat`** — the 0017 PK is 3-column.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `host_label` | `text` | no | | PK; the machine name |
| `queue` | `text` | no | | PK; `cpu` / `gpu` / ingest queue (the cpu/gpu `CHECK` was dropped in 0008) |
| `concurrency` | `integer` | no | | this worker's concurrency (1 by contract); summed for the gauge |
| `current_model` | `text` | yes | | GPU warm-model affinity hint (sticky routing) |
| `known_models` | `text[]` | no | `'{}'::text[]` | registered model ids this host advertises |
| `last_seen` | `timestamptz` | no | `now()` | freshness; consumers filter `last_seen > now() - 30s` |
| `last_flagged_dead_at` | `timestamptz` | yes | | orchestrator's stale-worker flag; cleared by a fresh heartbeat (0009) |
| `llm_servers_available` | `text[]` | no | `'{ollama}'::text[]` | observed LLM-server capability of this host (0014) |
| `vram_total_mb` | `integer` | yes | | total GPU VRAM (MB), sampled per heartbeat (0015) |
| `fits_models` | `text[]` | no | `'{}'::text[]` | model ids that fit this machine's VRAM (worker-computed; the capacity claim gate) (0015) |
| `project` | `text` | no | `''::text` | **tenant tag (0017)**; part of the PK |

- **Primary key:** `(host_label, queue, project)` — `worker_heartbeats_pkey`. **Widened to 3 columns in 0017** so two projects' workers can share one `(host_label, queue)` machine without clobbering each other's heartbeat.
- **Indexes:**
  - `worker_heartbeats_last_seen_idx` — `(last_seen)` (the staleness filter).
  - `worker_heartbeats_known_models_gin` — **GIN** `(known_models)` (so `known_models @> ARRAY['x']` is O(log n)).
  - `worker_heartbeats_flagged_dead_idx` — `(last_flagged_dead_at) WHERE last_flagged_dead_at IS NOT NULL` (partial; the supervisor's "recently flagged dead" poll).
- **Triggers:** none.

---

## `ingest_jobs` (migration 0007)

Standalone periodic / parametrised work with **no DAG**: no parent
`workflow_runs` row, no `$from` inputs, no dispatch-event outbox. A dedicated
table carries the **same** claim/lease columns as `workflow_node_jobs` so the
lease-renew / reclaim machinery is reused at the SQL-shape level, plus its own
NOTIFY trigger. Enqueued by the scheduler ticker or directly by a host; executed
by `ingest_executor`. The queue name is **host-defined** (the `fetch`/`load`
`CHECK` was dropped in 0008 — the host validates `queue` and `task_name` against
its registered sets before enqueue), and per-job `args` (0008) carry parameters
for parametrised tasks.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `text` | no | | PK |
| `task_name` | `text` | no | | the host ingest callable to run (validated host-side, not by a DB CHECK) |
| `queue` | `text` | no | | host-defined ingest queue (default set `fetch`/`load`); the worker LISTENs per queue |
| `reason` | `text` | no | `'tick'::text` | provenance: `tick` / `boot` / `manual` |
| `status` | `text` | no | `'queued'::text` | `CHECK status IN ('queued','running','completed','failed','cancelled')` |
| `priority` | `smallint` | no | `100` | claim-order band |
| `result` | `jsonb` | yes | | the task's JSON-able return |
| `error` | `text` | yes | | failure text |
| `seconds` | `double precision` | yes | | wall time |
| `claimed_by` | `text` | yes | | lease owner |
| `lease_expires_at` | `timestamptz` | yes | | lease lapse; `reclaim_expired_ingest_leases` re-queues past it |
| `created_at` | `timestamptz` | no | `now()` | claim order |
| `started_at` | `timestamptz` | yes | | |
| `finished_at` | `timestamptz` | yes | | |
| `args` | `jsonb` | no | `'{}'::jsonb` | per-job arguments for parametrised tasks (0008) |
| `project` | `text` | no | `''::text` | **tenant tag (0017)** |

- **Primary key:** `(id)` — `ingest_jobs_pkey`.
- **Indexes:**
  - `ingest_jobs_claim_idx` — `(queue, priority, created_at) WHERE status = 'queued'` (partial; per-queue claim for any queue name).
  - `ingest_jobs_project_claim_idx` — `(queue, project, priority, created_at) WHERE status = 'queued'` (project-aware claim, 0017).
  - `ingest_jobs_lease_idx` — `(lease_expires_at) WHERE status = 'running'` (the reclaim predicate).
- **Triggers:**
  - **`ingest_job_ready_notify`** `AFTER INSERT OR UPDATE OF status` → `notify_ingest_job_ready()`: when `NEW.status = 'queued'`, fires `pg_notify('ingest_job_ready', NEW.queue)` inside the writer's txn — the wake for ingest-family claim workers, exactly mirroring `node_job_ready`.

---

## `workflow_node_events` (migration 0011)

The **append-only** per-node, per-attempt forensic log. `workflow_node_jobs` is
a single mutable row, so a watchdog re-queue overwrites attempt N-1's worker,
timing, and trip reason the instant attempt N is claimed; this table keeps the
rich lifecycle signals (claim, model-load, stall, GPU-health / budget trip,
requeue, reassign, terminal, unassignable) durably for a queryable per-node
timeline. Terminal + `requeued` events ride the **same txn** as the state change
(the outbox-atomicity pattern); every other event is best-effort (own
connection, swallow-on-failure) so an event blip can never fail the load-bearing
claim/terminal/watchdog path. **No UPDATE path** — it adds no new mutation
invariant.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `bigint` | no | `nextval(workflow_node_events_id_seq)` | PK (`BIGSERIAL`) |
| `run_id` | `text` | no | | FK → `workflow_runs(id)` `ON DELETE CASCADE` |
| `node_id` | `text` | no | | logical node id |
| `job_id` | `text` | yes | | `workflow_node_jobs.id` at emit (nullable: survives row churn) |
| `attempt` | `smallint` | no | `0` | `= watchdog_retries` at emit — the cross-attempt key tying one node's tries |
| `event_type` | `text` | no | `CHECK` one of: `claimed`, `model_load_start`, `model_load_done`, `progress_beat`, `stall_suspected`, `stall_trip`, `gpu_health_trip`, `budget_trip`, `requeued`, `reassigned`, `lease_renew`, `completed`, `failed`, `cancelled`, `error`, `unassignable` (0015) | |
| `host_label` | `text` | yes | | emitting / claiming worker |
| `queue` | `text` | yes | | `cpu` / `gpu` / `fetch` / `load` |
| `model` | `text` | yes | | `required_model`, when relevant |
| `elapsed_s` | `double precision` | yes | | seconds in this attempt (trips / terminal) |
| `error` | `text` | yes | | trip reason / failure text (truncated) |
| `detail` | `jsonb` | no | `'{}'::jsonb` | free-form trip metrics (`max_sm_pct`, `ram_anchor_mb`, `budget_s`, `exit_code`, `model_load_s`, …) |
| `created_at` | `timestamptz` | no | `now()` | timeline order |

- **Primary key:** `(id)` — `workflow_node_events_pkey`.
- **Indexes:**
  - `workflow_node_events_node_idx` — `(run_id, node_id, created_at)` (the hot read: one node's timeline, oldest→newest).
  - `workflow_node_events_created_idx` — `(created_at)` (the `prune_node_events` retention sweep, default 30-day).
- **Triggers:** none (append-only; events are written directly by the writers).

---

## `worker_controls` (migration 0012, extended 0013)

The operator worker ON/OFF control plane — **desired** state, deliberately a
separate table from the *observed* `worker_heartbeats` because an OFF state must
persist precisely while the worker is **not** beating (exactly when its
heartbeat is aging out). An operator / host UI / the `queue-worker-control` CLI
writes a `(host_label, queue)` row; `worker_control.WorkerControlWatcher`
(LISTEN + safety poll) enforces it, dispatching `stop_policy` through the
in-code `STOP_POLICIES` registry (only `hard` = `os._exit(79)` exists today,
hence `stop_policy` is free-form TEXT, not a CHECK). Migration 0013 adds the
per-machine LLM-server config (desired, operator-set) read by the same worker.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `host_label` | `text` | no | | PK; the machine name |
| `queue` | `text` | no | | PK; `cpu` / `gpu` / ingest queue (control is per-queue) |
| `desired_state` | `text` | no | `'on'::text` | `CHECK (desired_state IN ('on','off'))` |
| `stop_policy` | `text` | no | `'hard'::text` | how to transition on→off; **free-form** (no CHECK) so `drain`/`pause` slot in with no migration; validated against `STOP_POLICIES` in Python |
| `requested_by` | `text` | yes | | provenance (operator / service name); informational |
| `updated_at` | `timestamptz` | no | `now()` | |
| `llm_server_type` | `text` | no | `'ollama'::text` | `CHECK (llm_server_type IN ('ollama','vllm'))` (0013) |
| `llm_parallelism` | `integer` | no | `1` | sidecar concurrency (`OLLAMA_NUM_PARALLEL` / vllm `--max-num-seqs`); `CHECK >= 1` (0013) |
| `vllm_idle_ttl_s` | `integer` | no | `60` | seconds idle before the supervisor SIGTERMs the vllm sidecar; `CHECK >= 0` (0 disables); ignored for ollama (0013) |

- **Primary key:** `(host_label, queue)` — `worker_controls_pkey`. The same identity `worker_heartbeats` and the claim's `claimed_by`/`queue` use; a host runs several workers under one `host_label`, so control is per-queue.
- **Indexes:** the primary-key unique index only.
- **Triggers:**
  - **`worker_control_notify`** `AFTER INSERT OR UPDATE` → `notify_worker_control()`: fires `pg_notify('worker_control', '<host_label>:<queue>')` on **every** write, inside the txn — so a plain SQL write (e.g. a host's Rails `INSERT … ON CONFLICT`) wakes the `WorkerControlWatcher` with no app-side NOTIFY code.
  - **`worker_llm_config_notify`** `AFTER INSERT OR UPDATE` → `notify_worker_llm_config()`: fires `pg_notify('worker_llm_config_changed', '<host_label>|<queue>')` — a **dedicated** channel (note the `|` separator vs the `worker_control` channel's `:`) so an LLM-config edit isn't read as an ON/OFF change. It stays quiet on an UPDATE that changes none of the three LLM columns (INSERTs always fire).

---

## `workflow_run_files` (migration 0001)

The per-run output-file manifest — each row records one artifact a step
produced (relative path, kind, size, and whether it's the run's primary
output). Created alongside `workflow_runs` in migration 0001, cascade-deleted
with the run.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `bigint` | no | `nextval(workflow_run_files_id_seq)` | PK (`BIGSERIAL`) |
| `run_id` | `text` | no | | FK → `workflow_runs(id)` `ON DELETE CASCADE` |
| `step_id` | `text` | no | | the step that produced the file |
| `rel_path` | `text` | no | | path relative to the run's output dir |
| `kind` | `text` | no | | artifact kind/category |
| `size_bytes` | `bigint` | no | `0` | file size |
| `is_primary` | `boolean` | no | `false` | the run's headline output |
| `created_at` | `timestamptz` | no | `now()` | |

- **Primary key:** `(id)` — `workflow_run_files_pkey`.
- **Indexes:**
  - `workflow_run_files_run_id_rel_path_key` — **UNIQUE** `(run_id, rel_path)` (one row per file per run).
  - `workflow_run_files_run_idx` — `(run_id)`.
  - `workflow_run_files_kind_idx` — `(kind)`.
- **Triggers:** none.

---

## `queue_schema_version` (the migration ledger)

The engine's migration ledger — one row per applied migration version.
`db.bootstrap()` inserts a row as each `NNNN_*.sql` step succeeds;
`db.wait_for_schema(min_version)` reads `MAX(version)` to block a claim worker
until the schema it needs is present (the per-queue `_REQUIRED_SCHEMA_VERSION`
map: e.g. ingest needs ≥ 8, claim workers gate on 6/8, not 12). A host running
its own second chain points `db.bootstrap(version_table=…)` at its **own**
ledger table — two ledgers, one Postgres.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `version` | `integer` | no | | PK; the migration number (1 … 17) |
| `applied_at` | `timestamptz` | no | `now()` | when this step was applied |

- **Primary key:** `(version)` — `queue_schema_version_pkey`.
- **Indexes:** the primary-key unique index only.
- **Triggers:** none.
