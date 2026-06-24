# Multi-tenant broker — one shared queue across all projects

## The problem this solves

Historically each project ran its **own** Postgres + its own engine instance, so
a fleet of N projects was N isolated single-tenant deployments. The engine's
queue was *already* partitioned by **resource** — one `cpu` queue + one `gpu`
queue (`workflow_node_jobs.queue`, migration 0002), never per-project — but you
could only see one project's queue per DB. The operator wanted **one queue per
CPU/GPU shared across all projects**, each record tagged with its project for
filtering, the queue living on the **broker** side, with the broker telling each
**project client** when a node-job may start / should abort.

## The model (operator decision): shared DB + per-project clients

```
                 Broker  =  ONE shared Postgres
        workflow_node_jobs(queue=gpu, project=ai_leads, …)
        workflow_node_jobs(queue=gpu, project=pic_to_3d, …)
        ingest_jobs(queue=fetch, project=ai_leads, …)
        worker_heartbeats(host_label=spark2, queue=gpu, project=ai_leads, …)
        worker_heartbeats(host_label=spark2, queue=gpu, project=pic_to_3d, …)
                              │
            LISTEN/NOTIFY + worker_control (project-scoped)
                              │
        ┌─────────────────┬──┴──────────────┬─────────────────┐
   [ai_leads client]  [pic_to_3d client]  [lm_cg client]   …each holds its
    orchestrator +      orchestrator +      orchestrator +   OWN code, claims
    cpu/gpu workers     cpu/gpu workers     cpu/gpu workers   ONLY its project's
                                                              rows.
```

- **The broker is the shared Postgres.** It holds *all* projects' jobs on the
  shared `cpu`/`gpu` (+ ingest) queues — one place to see the whole fleet, one
  set of counts, one future cross-project arbiter.
- **Each project runs its own client** (orchestrator + claim workers + scheduler)
  pointed at the shared broker DB. Workers stay project-local because they must
  import that project's node code (`config.node_module_package` /
  `node_resolver`). A physical GPU box can run several projects' workers as
  separate processes/containers sharing the hardware; each claims only its own
  project's rows.
- **No process imports another project.** The tenant boundary is the `project`
  tag + the per-client `config.project`, not code sharing.

This is the *inverse* of `db_namespace` (config.py): that **isolates** tenants on
a shared redis/mongo so they can't see each other; `project` **pools** them into
one pg queue with a filter.

## Phase 1 — the tenant core (DELIVERED, migration 0017)

A `project` identity threaded end-to-end through the queue records and the live
claim path.

### Design rule — exact-match-always

Every queue row carries `project TEXT NOT NULL DEFAULT ''`. **Enqueue** stamps it
(from the parent run, or `config.project`). **Claim** filters
`AND project = <this client's project>` *unconditionally* (resolved from
`config.project`, default `""`). There is no "claim-any-on-empty" special case to
reason about:

- **Single-tenant (default `""`):** every row is `''` and the filter `project=''`
  matches them all → byte-compatible with the pre-0017 one-Postgres-per-project
  deploy, **zero host wiring**.
- **Multi-tenant:** each client sets `config.project=X` (via
  `configure(project="X")`), so it enqueues X and claims X. Claiming another
  tenant's row is not even expressible.

### What changed

| Area | Change |
|---|---|
| schema (0017) | `project` column on `workflow_runs`, `workflow_node_jobs`, `ingest_jobs`, `worker_heartbeats`; `worker_heartbeats` PK → `(host_label, queue, project)` (two projects' workers can share a machine without clobbering heartbeats); project-aware claim indexes. |
| `config` / `configure` | `config.project` + `configure(project=…)`. |
| `node_queue` enqueue | `enqueue_node_job` / `insert_skipped_job` / `enqueue_ingest_job` stamp `project` (default → `config.project`). |
| `node_queue` claim | `claim_next_{cpu,gpu,ingest}_job` filter by `project` (default → `config.project`). |
| `node_queue` recovery / telemetry | `upsert_worker_heartbeat` (3-col upsert), `clear_worker_current_model`, `flag_stale_workers_holding_running_jobs`, `reclaim_all_running_for_resume`, `vlm_pool_should_defer`, `flag_unassignable_gpu_jobs`, `requeue_running_for_worker` are **project-scoped** — their `claimed_by`/`host_label` joins (and the unassignable fleet read) also match `project`, because on a shared broker `host_label` (a machine name) is no longer globally unique. (`reclaim_expired_leases` needs no project term — it acts on the row in place; the project travels with the row.) |
| `node_queue` snapshots | `snapshot` / `ingest_snapshot` / `fleet_snapshot` take an optional `project` filter (`None` = broker-wide, byte-compatible); rows already expose `project`. |
| `run_store` | `insert_run` stamps `project`; `list_queued_node_run_ids(project=…)` is the dispatch loop's **project-scoped** work-list. |
| `dispatcher` | propagates `run["project"]` onto every node-job + skipped-marker it expands. |
| orchestrator pickup paths | A "client" = orchestrator + workers + scheduler, so **every** row-pickup path is scoped, not only the worker `claim_next_*`. Scoped: `NodePool._tick` run expansion (`list_queued_node_run_ids`), `_drain_dispatch_events` (outbox drain, correlated to the run's `project`), `_sweep_stuck_runs` (`list_stuck_node_run_ids` → `reconcile_run`), `_sweep_unassignable_jobs`, `_sweep_dead_workers`, `_sweep_orphan_queued_jobs`, `InputListener._claim_pending` (input resume, run-correlated), and the orchestrator **startup** hooks `reenqueue_running_for_resume` (runs) + `reclaim_all_running_for_resume` (node-jobs — the *outer* UPDATE is filtered, not just the heartbeat sub-join, so A's restart can't clear B's `claimed_by` and kill B's live render). |
| genuinely **un**scoped (and safe) | `reclaim_expired_leases` / `reclaim_expired_ingest_leases` (act on the row in place — the project travels with it), `cancel_queued_jobs_for_run` / `delete_non_terminal_jobs_for_run` / the cancel-watcher / terminal marks (keyed by `run_id`/`job_id`, project inherent), `prune_node_events` (age-based retention), and the conductor's `fleet_snapshot()` with no filter (the deliberate broker-wide cross-project view). |

### Deployment requirement

A host wires `configure(project="<name>")` **once at startup for every process**
of that project (orchestrator, claim workers, scheduler). The default `""` keeps
existing single-tenant deploys unchanged. Mixing a configured client with `""`
rows in the same DB is a misconfiguration — a `""` client would see only `""`
rows, a configured client only its own.

### Standing up THE broker (one queue for all projects)

Consolidation is a **config flip**, not new code. Three steps:

```bash
# 1. Stand up the shared broker schema ONCE (idempotent), against the broker DSN.
#    `queue-broker` is the explicit "own the migration chain" entry point. You do
#    NOT strictly have to run it first: db.bootstrap() takes a Postgres advisory
#    lock, so every project's orchestrator can also boot against the shared broker
#    concurrently and safely — the lock serializes, and a late bootstrap that
#    finds the chain already applied is a no-op. queue-broker just makes the
#    "bootstrap once, independent of any app" step explicit + inspectable.
BROKER_DSN=postgresql://…/broker   queue-broker
```

```python
# 2. Point EVERY process of EVERY project at that broker + name the project.
#    (orchestrator, claim workers, scheduler — all of them.)
queue_workflows.configure(project="ai_leads",   db_url_env="BROKER_DSN")
queue_workflows.configure(project="pic_to_3d",  db_url_env="BROKER_DSN")
# … each then enqueues + claims ONLY its own project's rows on the ONE shared
#   cpu/gpu (+ ingest) queue. Cross-project isolation is enforced by the engine
#   (exact-match-always; proven in tests/test_broker_consolidation.py).
```

```bash
# 3. Watch the CONSOLIDATED queue across all projects.
BROKER_DSN=…   queue-broker --status      # schema version + per-project depth
BROKER_DSN=…   queue-conductor-web         # the web view, filterable by project
```

That is the whole consolidation: one broker DB, one cpu + one gpu queue, every
record tagged with `project`, each client scoped to its own — the headline of
this design. (Migrating the *existing* per-project deploys onto the broker is the
operational **Cutover** below — it backfills each app's in-flight rows to its
project tag and repoints its env at `BROKER_DSN`.)

### Cutover — adopting a project name on an existing deploy

Migration 0017 backfills every existing row to `project=''`. Because claiming is
**exact-match**, the instant you switch a running deploy to `configure(project=
"ai_leads")` its `project='ai_leads'` clients stop seeing the backfilled `''`
rows — any in-flight queued/running work at cutover would be **stranded**, and
old heartbeats orphaned. So adopt a project name **only on a drained queue**, or
run a one-time backfill in the same window:

```sql
UPDATE workflow_runs        SET project = 'ai_leads' WHERE project = '';
UPDATE workflow_node_jobs   SET project = 'ai_leads' WHERE project = '';
UPDATE ingest_jobs          SET project = 'ai_leads' WHERE project = '';
UPDATE worker_heartbeats    SET project = 'ai_leads' WHERE project = '';  -- or let stale rows age out
```

A deploy that simply stays single-tenant (`project` unset) needs none of this.

### Operational notes

- **`worker_heartbeats` writes must go through `node_queue.upsert_worker_heartbeat`.**
  The 0017 PK widened to `(host_label, queue, project)`, so any consumer that
  upserts heartbeats with its own `INSERT … ON CONFLICT (host_label, queue)` will
  fail (`no unique or exclusion constraint matching the ON CONFLICT
  specification`). In-repo every write already goes through that function; the
  note is for external consumers carrying their own heartbeat SQL.
- **The `project` tag applies to the legacy direct-pg engine path only.** The
  pluggable `StorageBackend` SPI (`backends/{postgres,redis,mongodb}.py`) has no
  `project` concept — its multi-tenancy is `db_namespace` (isolation, the inverse
  of pooling). Selecting redis/mongo does not re-home the orchestrator/worker, so
  `project` filtering does not apply there.
- **The 0017 down-migration is only safe on a single-tenant (all-`''`) DB.**
  Dropping `project` and re-adding the 2-col heartbeat PK collapses two projects'
  rows that share `(host_label, queue)` into duplicates and the PK re-add throws —
  inherent to reversing multi-tenant data into a single-tenant shape. The forward
  path and the empty-DB migration roundtrip are unaffected.

## Later phases (NOT in this change)

1. **Broker→client start/abort signalling, project-scoped.** The pieces exist —
   `worker_control` (operator ON/OFF), `JobStatusWatcher` (cancel/reassign
   self-kill), `cancel_watcher` — but `worker_controls` is still keyed
   `(host_label, queue)`. To steer *a specific project's* worker on a shared box,
   add `project` to `worker_controls` + the `worker_control` NOTIFY payload. This
   is the literal "tell the project client to abort" path.
2. **Cross-project GPU arbiter.** Today each project's workers claim their own
   rows independently; a true *shared* GPU queue could let the broker arbitrate
   *across* projects (fair-share, priority, preemption) — the shared-DB view in
   Phase 1 is the prerequisite read model.
3. **`project` on the forensic tables** (`workflow_node_events`,
   `workflow_dispatch_events`, `workflow_input_submissions`) — currently
   derivable via a `run_id` join; denormalize if direct per-project filtering of
   the event log is wanted.
4. **Conductor multi-project view + dashboard.** `fleet_snapshot()` /
   `snapshot()` already return `project` and accept a filter; the conductor and
   the fleet panel can group by it.

## Tests

`tests/test_multitenant_project.py` pins the contract: cpu/gpu/ingest claim
isolation between two projects on one DB, the `""` single-tenant back-compat
round-trip, `config.project` as the implicit enqueue+claim tag, and two projects
sharing a `(host_label, queue)` heartbeat without clobber.
