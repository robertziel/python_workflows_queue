# Running the engine on SQLite (`db_backend="sqlite"`)

`queue_workflows` is a Postgres-as-queue engine, but the **whole engine** can run
against a single **SQLite file** — a local, daemon-less, low-RAM/low-disk deploy
with no separate database server — selected entirely in config. **As of v1.0.0
SQLite is the DEFAULT** (`configure()` with no `db_backend` ⇒ sqlite); the
Postgres path is unchanged and byte-identical but now opt-in via
`configure(db_backend="pg")`.

> Why SQLite for a local box: Postgres needs a running daemon (tens–hundreds of MB
> RAM) and tuning; SQLite is an embedded file (≈0 daemon, a few MB), trivially
> backed up (copy one file) and reset (delete it). For a single machine with a few
> worker processes it is the lighter footprint. (It is **not** for multi-host
> fleets — a SQLite file is local to one machine; use Postgres there.)

## Usage

```python
import queue_workflows

queue_workflows.configure(
    db_backend="sqlite",
    db_url_env="MYAPP_DB_URL",     # the env var that holds the SQLite location
)
# MYAPP_DB_URL may be: /var/lib/myapp/queue.db  |  sqlite:////abs/path.db  |  :memory:
```

Then the orchestrator bootstraps and everything else works the same:

```bash
MYAPP_DB_URL=/var/lib/myapp/queue.db queue-orchestrator     # applies migrations_sqlite/
MYAPP_DB_URL=/var/lib/myapp/queue.db queue-claim-worker --queue=cpu
```

`configure(db_backend=...)` selects the engine's **relational** store: `"sqlite"`
(default) or `"pg"`. (`"redis"`/`"mongodb"` select the separate *flat-queue*
`StorageBackend` SPI — see `storage_backends.md` — and do **not** host the
relational DAG engine.)

## Architecture — one dialect seam, no forked modules

The engine's SQL is written once (in pyformat, `%s` / `%(name)s`). Two small,
well-contained pieces make it run on either backend; **the Postgres rendering is
exactly today's SQL**, so the live pg deploy is byte-identical.

### 1. `queue_workflows/dialect.py` — the structural divergences
`get_dialect()` → `PgDialect` | `SqliteDialect`, chosen from `config.db_backend`.
It produces the fragments that genuinely differ in *structure* (not just syntax):

| concept | Postgres | SQLite |
|---|---|---|
| interval arithmetic | `now() + make_interval(secs => …)` | `datetime('now', ('+' \|\| … \|\| ' seconds'))` |
| epoch of a timestamp | `EXTRACT(EPOCH FROM c.created_at)` | `CAST(strftime('%s', c.created_at) AS REAL)` |
| skip-locked claim | `FOR UPDATE SKIP LOCKED` | *(none — see concurrency)* |
| null-safe equals (affinity) | `a IS NOT DISTINCT FROM b` | `a IS b` |
| array-column membership | `x = ANY(arr)` | `EXISTS (SELECT 1 FROM json_each(arr) WHERE value = x)` |
| param-list membership | `x = ANY(%(p)s::text[])` | `x IN (SELECT value FROM json_each(%(p)s))` |
| array param/column value | a python `list` (→ `text[]`) | a JSON string |
| table-exists probe | `to_regclass('public.'\|\|name)` | `sqlite_master` lookup |

### 2. `queue_workflows/db.py` — the SQLite connection + translator
When `db_backend="sqlite"`, `connection()` yields a psycopg-shaped wrapper over a
per-process shared `sqlite3` connection (WAL + `busy_timeout` for cross-process
safety; an RLock serializes threads). Each statement passes through a
**string-literal-aware** translator that handles the *mechanical* differences
without touching call sites:

- pyformat → sqlite paramstyle (`%s`→`?`, `%(name)s`→`:name`), **skipping string
  literals** — so a real placeholder converts but `strftime('%s')` survives;
- `now()` → `(datetime('now'))` (parenthesized → valid even in a column DEFAULT);
- `now() ± make_interval(<unit> => <n>)` → a fused `datetime('now', modifier)`;
- strips `::type` casts and `FOR UPDATE [SKIP LOCKED]`; `LEAST`/`GREATEST` →
  `MIN`/`MAX`.

**psycopg type parity** is restored by an explicit row factory keyed on the
engine's **known column names** (robust under `RETURNING *` / joins, where
`PARSE_DECLTYPES` is not): JSON columns → `dict`, array columns → `list`,
`TIMESTAMPTZ` columns → aware-UTC `datetime`. Write adapters turn psycopg `Jsonb`
and `datetime` params into the matching SQLite text.

### 3. `queue_workflows/migrations_sqlite/` — the DDL twin
The same 17-version chain (paired `.down.sql`), translated to SQLite: `TIMESTAMPTZ`
/`JSONB`/`text[]`→`TEXT`, `BIGSERIAL`→`INTEGER … AUTOINCREMENT`, `DEFAULT now()`→
`DEFAULT (datetime('now'))`, triggers/NOTIFY/plpgsql and gin indexes omitted, one
column per `ALTER`, and the migration-0017 `worker_heartbeats` primary-key change
done via the create-copy-drop-rename idiom (SQLite can't `ALTER` a PK).
`db.bootstrap()`/`downgrade()` pick the dir per backend automatically.

### Concurrency model
SQLite serializes writers (WAL + `busy_timeout`), so the single-statement
`UPDATE … WHERE id = (SELECT … LIMIT 1)` claim is atomic without `SKIP LOCKED`:
two worker processes contend on the file lock and one wins, exactly-once. This is
right for a local box with a handful of workers; it does **not** scale to the
high-concurrency, many-host fleet Postgres handles.

### Wake
SQLite has no `LISTEN/NOTIFY`. Workers fall back to the existing safety **poll**
(claims already poll on a short interval); the NOTIFY triggers are simply omitted
from the SQLite migrations.

## Status

**Working + tested on SQLite** (`tests/test_sqlite_backend.py`): the connection/
dialect seam, the full migration chain (bootstrap to v17 + downgrade roundtrip),
and the core queue round-trips — cpu enqueue→claim→complete, gpu claim with
warm-model affinity + capability gating, ingest enqueue→claim→complete, heartbeat
upsert + `fleet_snapshot`, and the unassignable sweep. The Postgres suite stays
green (byte-identical pg path).

**Remaining before the engine is fully SQLite-complete** (tracked in
`worklog/sqlite-engine-port.md`):
1. The two `UPDATE … FROM … RETURNING <alias>.col` sweeps —
   `reclaim_expired_leases` and `flag_stale_workers_holding_running_jobs` — need a
   SQLite-safe form (SQLite RETURNING can't alias-qualify or return FROM-table
   columns); plus a sweep of `run_store`/`dispatcher`/`node_pool`/`cancel_watcher`
   for any residual pg-only SQL.
2. The poll-only wake stub for the LISTEN sites on SQLite.
3. Cross-backend test parity — run the entire engine suite against a SQLite temp
   file (the full "engine runs on SQLite" proof), then an independent audit.
