# queue_workflows

A standalone, pip-installable **Postgres-as-queue workflow engine**, extracted
from the ai_leads stack so the ~35 sibling projects can share one DRY source.

Postgres is the only hard dependency. The engine provides:

- a `SELECT … FOR UPDATE SKIP LOCKED` claim loop woken by `LISTEN node_job_ready`
  (`claim_worker`), with a `LeaseRenewer` + per-job `Watchdog`;
- lease reclaim (a dead/wedged worker's row is re-queued);
- a DAG dispatcher with a durable dispatch-event outbox (`dispatcher` + `node_pool`);
- a GPU warm-model cache that keeps one model loaded across same-model jobs
  (`model_cache` / `gpu_model_cache` / `model_registry`);
- periodic "ingest" work on a dedicated `ingest_jobs` table + a PG-native
  ticker (`scheduler` / `ingest_executor`);
- per-host CPU/GPU/RAM telemetry → `pg_notify('hw_metrics', …)` (`hw_metrics`).

## Quick start

```python
import queue_workflows
from queue_workflows import model_registry
from queue_workflows.model_registry import ModelSpec
from queue_workflows.scheduler import ScheduleEntry

# 1. configure (all keys optional — defaults keep ai_leads byte-compat)
queue_workflows.configure(
    db_url_env="MY_DB_URL",                 # env var holding the DSN
    video_model_ids=frozenset({"wan_i2v"}), # GPU models on the tight render budget
    node_module_package="myapp.nodes",      # node-module resolver prefix
    container_prefix="myapp-",              # cgroup attribution
)

# 2. wire the host seams
queue_workflows.set_workflow_provider(load_workflow_fn, pipeline_schema_fn)
queue_workflows.set_builtin_model_registrar(register_my_models)
queue_workflows.register_ingest_task("run_fetch_all", run_fetch_all)
queue_workflows.set_ingest_schedule([ScheduleEntry("fetch", 37, "run_fetch_all", "fetch")])

# 3. apply the engine's migration chain (idempotent), then launch
queue_workflows.db.bootstrap()             # queue tables → queue_schema_version
queue_workflows.claim_worker.main(["--queue", "gpu"])
```

Console scripts (for standalone / other-project use):

```
queue-claim-worker --queue=gpu
queue-scheduler
queue-orchestrator
```

## Migrations

The engine owns its schema as `queue_workflows/migrations/NNNN_*.sql` (shipped
as package data). `queue_workflows.migrations.dir()` returns the directory;
`queue_workflows.db.bootstrap()` applies the chain against the
`queue_schema_version` ledger. A host with its own domain tables runs a second
chain via `db.bootstrap(migrations_dir=..., version_table=...)`.

## Tests

```
pip install -e '.[test]'
QUEUE_WORKFLOWS_TEST_DB_URL=postgresql://user:pw@host:port/queue_workflows_test \
  python -m pytest
```

The suite forces a `*_test` DB and applies the engine migration chain only. See
`tests/conftest.py`.
