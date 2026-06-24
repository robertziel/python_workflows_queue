# Pluggable storage backends (`pg` · `redis` · `mongodb`)

`queue_workflows` is, by design, a **Postgres-as-queue** engine — its claim loop,
lease/reclaim, dispatch outbox and wake are written directly in SQL. That stays
the reference path. This document describes the **`StorageBackend` SPI**, an
additive seam (since v0.3.0) that makes the *queue storage* selectable so the
same durable-queue semantics can run on Redis or MongoDB.

```python
import queue_workflows
queue_workflows.configure(db_backend="redis")          # or "mongodb" / "pg" / "sqlite" (default)
from queue_workflows.backends import get_backend
be = get_backend()                                      # bound to config.db_namespace
jid = be.enqueue("cpu", {"task": "render"})
job = be.claim("cpu", worker="box-1", lease_s=30)
be.complete_with_event(job["id"], "completed", result={"ok": True})
```

Selecting `pg` (the default) imports **nothing** new and changes nothing — the
seam is opt-in and the legacy engine is untouched.

## What the SPI is — and is not

The port (`queue_workflows/backends/base.py`, `StorageBackend`) is a **generic
durable work-queue with a transactional outbox**:

- `enqueue` / `claim` (exactly-once) / `renew_lease` / `reclaim_expired` / `requeue_for_retry`
- idempotent terminals (`mark_completed` / `mark_failed`) and the **atomic outbox**
  (`complete_with_event` / `fail_with_event` — go terminal *and* append the event,
  both-or-neither)
- best-effort wake (`notify` / `subscribe`), `heartbeat` / `workers`, and the
  operator `set_control` / `desired_state` (ON/OFF, default-ON)

It is deliberately **not leaky**: no method takes or returns a driver handle
(psycopg cursor, redis pipeline, pymongo session). Each backend is **bound to one
namespace** and scopes every key/row/collection by it, so two tenants on one
server are isolated.

> **Integration boundary (read this).** The SPI is **additive**. Selecting
> `redis`/`mongodb` does **not** re-home the existing orchestrator / claim-worker
> / dispatcher — those still run on Postgres. v0.3.0 ships the SPI + three
> backends that pass an identical contract suite; wiring the DAG orchestrator
> end-to-end onto a non-PG backend is a later milestone. Use the SPI today as a
> standalone pluggable durable queue.

## How each backend reproduces the contract

| Guarantee | `pg` | `redis` | `mongodb` |
|---|---|---|---|
| Claim exactly-once | `FOR UPDATE SKIP LOCKED` | `ZPOPMIN` inside a **Lua** script | `find_one_and_update` |
| Priority + FIFO | `ORDER BY priority DESC, created_at` | sorted set, score `-priority`, FIFO tiebreak | sort `priority desc, seq asc` |
| Lease / reclaim | lease column + `UPDATE … lease < now()` | per-queue running ZSET scored by expiry | `lease_expires_at < now` sweep |
| Idempotent terminal | `UPDATE … WHERE status NOT IN (terminal) RETURNING *` | Lua status guard | `find_one_and_update` status guard |
| **Atomic outbox** | one **transaction** | one **Lua script** | one **multi-doc transaction** |
| Wake | `LISTEN` / `pg_notify` (in-txn) | **pub/sub** (fire-and-forget) | **change stream** on a capped coll. |
| Namespace isolation | `namespace` column | key prefix `qw:<ns>:` | one **database** per namespace |

### Caveats (be honest about the weakenings)

- **Redis** has no cross-key ACID transaction; atomicity comes from **Lua**
  (single server-side step), so this targets a **single Redis instance**, not
  Cluster (Cluster needs all keys in one hash slot). The wake is pub/sub —
  fire-and-forget — so a subscriber that is down misses it; the worker's safety
  poll covers that, exactly as it does behind PG `LISTEN`.
- **MongoDB** transactions *and* change streams require a **replica set** (a
  single-node RS is fine); on a standalone `mongod`, `complete_with_event` /
  `fail_with_event` and the wake will fail loudly. `ensure_schema()` pings on
  connect.
- **pg** computes `counts()` from the live `status`, so it can never drift;
  redis keeps maintained counters (decremented against the *prior* status — see
  the audit below); mongo counts live documents.

## Configuration

| `configure(...)` key | env (default) | meaning |
|---|---|---|
| `db_backend` | `QUEUE_WORKFLOWS_DB_BACKEND` (`sqlite`) | `"sqlite"` (default) / `"pg"` / `"redis"` / `"mongodb"` (aliases `postgres`, `mongo`) |
| `db_namespace` | — | tenant scope on a shared server (`""` ⇒ `"default"`) |
| — | `QUEUE_WORKFLOWS_REDIS_URL` | redis DSN (`redis_url_env` renames it) |
| — | `QUEUE_WORKFLOWS_MONGO_URL` | mongo DSN, incl. `?replicaSet=…` / `directConnection=true` |

Install the optional driver: `pip install 'queue_workflows[redis]'` or
`'queue_workflows[mongodb]'`. Selecting a backend whose driver is missing raises
a clear `ImportError` naming the extra.

## Tests

`tests/test_backend_contract.py` is one parametrized suite run against **all three
live servers** (a backend whose server is unreachable is skipped, not failed):

```bash
docker run -d --name qw_pg    -e POSTGRES_PASSWORD=postgres -p 5433:5432 postgres:16
docker run -d --name qw_redis  -p 6380:6379 redis:7
docker run -d --name qw_mongo  -p 27018:27017 mongo:7 --replSet rs0 --bind_ip_all
docker exec qw_mongo mongosh --quiet --eval 'rs.initiate()'

export QUEUE_WORKFLOWS_TEST_DB_URL="postgresql://postgres:postgres@localhost:5433/queue_workflows_test"
export QUEUE_WORKFLOWS_TEST_REDIS_URL="redis://localhost:6380/0"
export QUEUE_WORKFLOWS_TEST_MONGO_URL="mongodb://localhost:27018/?directConnection=true"
python -m pytest tests/test_backend_contract.py
```

The suite pins claim-exactly-once (incl. an 8-thread contention stress test),
lease/renew/reclaim, idempotent terminals, the atomic outbox (both-or-neither +
no duplicate on re-delivery), the wake, heartbeat/control, and cross-namespace
isolation.

## Audit (v0.3.0)

The SPI was audited across design / regressions / safety / internal data leakage:

- **A1 (safety, fixed).** Redis `counts()` used maintained counters that
  decremented `:running` unconditionally; a terminal/requeue applied to a job
  that wasn't `running` would drift them. Fixed: the Lua now decrements the
  job's **prior** status. (pg/mongo derive counts from live status — immune.)
- **A2 (safety, added).** Added a multi-threaded contention test that proves
  claim-exactly-once on all three backends, not just sequentially.
- **A3 (design/leakage).** The SPI is additive (see *Integration boundary*); it
  does not silently re-home the engine. No driver objects appear in the port, so
  PG internals can't leak into redis/mongo call sites. Cross-namespace
  claim/read/count/wake are isolated and tested.
- **Regressions:** none — the full engine suite (415 tests) stays green; the new
  drivers are imported lazily so a `pg`-only deploy needs neither installed.
