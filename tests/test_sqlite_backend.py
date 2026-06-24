"""SQLite engine backend — Phase 1: the connection/dialect seam.

Proves the SQLite compatibility layer in ``db.py`` gives psycopg parity for the
constructs the engine relies on: pyformat (``%s`` / ``%(name)s``) paramstyle,
JSON-obj→dict, JSON-array→list, TIMESTAMPTZ→aware ``datetime``, ``now()``
translation, the string-literal-aware translator (``strftime('%s')`` survives
while ``%s`` placeholders convert), ``::cast`` strip, ``LEAST``→``MIN``,
``FOR UPDATE [SKIP LOCKED]`` strip, ``RETURNING *``, and ``rowcount``.

These run against a throwaway SQLite FILE (per test, via tmp_path), independent
of the Postgres test DB the rest of the suite uses.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

import queue_workflows
from queue_workflows import db, dialect


@pytest.fixture
def sqlite_engine(tmp_path):
    """Point the engine's relational store at a fresh SQLite file, then restore
    the Postgres test config on teardown (this fixture tears down before the
    autouse pg-truncate fixture, so it must hand the engine back to Postgres)."""
    path = str(tmp_path / "qw.db")
    os.environ["QUEUE_WORKFLOWS_SQLITE_SMOKE_URL"] = path
    db.close_pool()
    queue_workflows.configure(
        db_backend="sqlite", db_url_env="QUEUE_WORKFLOWS_SQLITE_SMOKE_URL",
    )
    yield path
    db.close_pool()
    queue_workflows.configure(db_backend="pg", db_url_env="QUEUE_WORKFLOWS_TEST_DB_URL")


@pytest.fixture
def sqlite_ready(sqlite_engine):
    """A SQLite engine with the full migration chain applied (v17)."""
    db.bootstrap()
    return sqlite_engine


def test_translator_literal_awareness_unit():
    # Direct unit cover of the string-literal-aware translator: a real %s
    # placeholder converts to ?, but a '%s' INSIDE a string literal (strftime)
    # must survive untouched; a named param converts; now() -> strftime.
    from queue_workflows.db import _translate_sql_for_sqlite as T
    out = T("SELECT strftime('%s', created_at) AS e, %s AS a, %(b)s AS b, now() AS n")
    assert "strftime('%s', created_at)" in out      # literal preserved
    assert "? AS a" in out                          # positional placeholder -> ?
    assert ":b AS b" in out                         # named placeholder -> :name
    assert "now()" not in out                       # now() rewritten away


def test_dialect_selected_for_sqlite(sqlite_engine):
    assert dialect.is_sqlite() is True
    assert dialect.get_dialect().name == "sqlite"
    assert db.sqlite_path() == sqlite_engine


def _create(cur):
    # Real engine column NAMES so the row factory's by-name converters fire.
    cur.execute(
        """
        CREATE TABLE t (
            id           TEXT PRIMARY KEY,
            context      TEXT,
            known_models TEXT,
            priority     INTEGER NOT NULL DEFAULT 100,
            created_at   TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )


def test_paramstyle_json_array_timestamp_roundtrip(sqlite_engine):
    from psycopg.types.json import Jsonb

    with db.connection() as conn, conn.cursor() as cur:
        _create(cur)
    # positional %s + Jsonb (adapted to JSON text) + a JSON-array string + now()
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO t (id, context, known_models, created_at) "
            "VALUES (%s, %s, %s, now())",
            ("a", Jsonb({"k": 1, "nested": [1, 2]}), json.dumps(["sdxl", "qwen"])),
        )
        assert cur.rowcount == 1
    # named %(id)s + RETURNING-style read
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM t WHERE id = %(id)s", {"id": "a"})
        row = cur.fetchone()

    assert row["id"] == "a"
    assert row["context"] == {"k": 1, "nested": [1, 2]}     # JSON-obj → dict
    assert row["known_models"] == ["sdxl", "qwen"]          # JSON-array → list
    assert isinstance(row["created_at"], datetime)          # TIMESTAMPTZ → datetime
    assert row["created_at"].tzinfo is not None             # aware (UTC), psycopg parity


def test_translator_is_string_literal_aware(sqlite_engine):
    # strftime('%s', …) must SURVIVE (literal), while the %s placeholder converts.
    with db.connection() as conn, conn.cursor() as cur:
        _create(cur)
        cur.execute("INSERT INTO t (id) VALUES (%s)", ("a",))
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT CAST(strftime('%s', created_at) AS REAL) AS epoch, "
            "%s AS passthru FROM t WHERE id = %s",
            ("hello", "a"),
        )
        row = cur.fetchone()
    assert isinstance(row["epoch"], float) and row["epoch"] > 0
    assert row["passthru"] == "hello"


def test_mechanical_rewrites(sqlite_engine):
    # LEAST→MIN, ::cast strip, FOR UPDATE SKIP LOCKED strip in one statement.
    with db.connection() as conn, conn.cursor() as cur:
        _create(cur)
        cur.execute("INSERT INTO t (id, priority) VALUES (%s, %s)", ("a", 50))
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE t SET priority = LEAST(priority, 10) "
            "WHERE id = (SELECT id FROM t WHERE priority = 50 "
            "            ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1) "
            "RETURNING priority, '{}'::jsonb AS j",
            (),
        )
        row = cur.fetchone()
    assert row["priority"] == 10        # LEAST→MIN applied
    assert row["j"] == "{}"             # ::jsonb stripped; 'j' not a known JSON col → text


def test_dialect_fragments_render_for_sqlite(sqlite_engine):
    d = dialect.get_dialect()
    assert d.skip_locked == ""
    assert "datetime('now'" in d.future_seconds("%(s)s")
    assert "strftime" in d.epoch("created_at")
    assert "json_each" in d.value_in_param_array("x", "%(p)s")
    assert d.array_param(["a", "b"]) == '["a", "b"]'   # list → JSON text for sqlite


def test_migration_chain_bootstraps_and_roundtrips(sqlite_engine):
    # The SQLite migration chain (migrations_sqlite/) applies to v17, creates the
    # full engine schema, and survives a full downgrade→0→re-bootstrap roundtrip.
    db.bootstrap()
    assert db.current_schema_version() == 17

    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = {r["name"] for r in cur.fetchall()}
        cur.execute("PRAGMA table_info(worker_heartbeats)")
        hb_pk = [r["name"] for r in cur.fetchall() if r["pk"]]

    assert {
        "workflow_runs", "workflow_node_jobs", "ingest_jobs", "worker_heartbeats",
        "worker_controls", "workflow_dispatch_events", "workflow_node_events",
        "workflow_input_submissions", "workflow_run_files",
    } <= tables
    assert hb_pk == ["host_label", "queue", "project"]   # migration 0017 PK rebuild

    reverted = db.downgrade(to_version=0)
    assert len(reverted) == 17
    assert db.current_schema_version() == 0
    db.bootstrap()
    assert db.current_schema_version() == 17


# ── Phase 3: real engine queue round-trips on SQLite ────────────────────────


def _mk_run(project="", status="running"):
    import uuid
    from queue_workflows import run_store
    rid = str(uuid.uuid4())
    run_store.insert_run(run_id=rid, workflow_name="_wf", out_dir="/tmp/o",
                         status=status, mode="node", project=project)
    return rid


def test_cpu_enqueue_claim_complete_roundtrip(sqlite_ready):
    from datetime import datetime, timezone
    from queue_workflows import node_queue
    rid = _mk_run()
    jid = node_queue.enqueue_node_job(
        run_id=rid, node_id="n1", node_module="m", queue="cpu",
        inputs={"a": 1},
    )
    claimed = node_queue.claim_next_cpu_job(host="boxA", lease_s=600)
    assert claimed is not None and claimed["id"] == jid
    assert claimed["status"] == "running" and claimed["claimed_by"] == "boxA"
    assert claimed["inputs"] == {"a": 1}                       # JSON parity
    assert isinstance(claimed["lease_expires_at"], datetime)   # TS parity
    assert claimed["lease_expires_at"] > datetime.now(timezone.utc)
    # nothing left to claim
    assert node_queue.claim_next_cpu_job(host="boxA") is None
    done = node_queue.mark_completed(jid, context_delta={"out": 7}, seconds=1.0)
    assert done["status"] == "completed" and done["context_delta"] == {"out": 7}
    # idempotent terminal
    assert node_queue.mark_completed(jid, context_delta={}, seconds=0.0) is None


def test_gpu_claim_with_model_affinity_and_capability(sqlite_ready):
    from queue_workflows import node_queue
    rid = _mk_run()
    jid = node_queue.enqueue_node_job(
        run_id=rid, node_id="g", node_module="m", queue="gpu", required_model="sdxl",
    )
    # capability gate: a worker that knows sdxl claims it; warm-model affinity ok
    claimed = node_queue.claim_next_gpu_job(
        0, current_model="sdxl", host="gpu1", known_models=["sdxl"],
    )
    assert claimed is not None and claimed["id"] == jid


def test_ingest_enqueue_claim_roundtrip(sqlite_ready):
    import queue_workflows
    from queue_workflows import node_queue
    queue_workflows.register_ingest_task("noop", lambda reason: {})
    iid = node_queue.enqueue_ingest_job(task_name="noop", queue="fetch")
    got = node_queue.claim_next_ingest_job("fetch", host="w")
    assert got is not None and got["id"] == iid and got["status"] == "running"
    done = node_queue.mark_ingest_completed(iid, result={"n": 3}, seconds=0.5)
    assert done["status"] == "completed" and done["result"] == {"n": 3}


def test_heartbeat_and_fleet_snapshot_array_parity(sqlite_ready):
    from queue_workflows import node_queue
    node_queue.upsert_worker_heartbeat(
        host_label="gpu1", queue="gpu", concurrency=2,
        known_models=["sdxl", "qwen"], fits_models=["sdxl"],
        llm_servers_available=["ollama", "vllm"],
    )
    fleet = node_queue.fleet_snapshot()
    assert len(fleet) == 1
    row = fleet[0]
    assert row["known_models"] == ["sdxl", "qwen"]            # text[]→list parity
    assert row["fits_models"] == ["sdxl"]
    assert row["llm_servers_available"] == ["ollama", "vllm"]
    assert row["fresh"] in (True, 1)                          # derived flag


def test_unassignable_sweep_on_sqlite(sqlite_ready):
    from queue_workflows import node_queue
    # a gpu worker that fits nothing; a queued model-job needing 'big'
    node_queue.upsert_worker_heartbeat(
        host_label="small", queue="gpu", concurrency=1, fits_models=[],
        vram_total_mb=8000,
    )
    rid = _mk_run()
    jid = node_queue.enqueue_node_job(
        run_id=rid, node_id="g", node_module="m", queue="gpu", required_model="big",
    )
    flagged = node_queue.flag_unassignable_gpu_jobs()
    assert [r["id"] for r in flagged] == [jid]                 # json_each ANY path


def test_commit_and_rollback_semantics(sqlite_engine):
    with db.connection() as conn, conn.cursor() as cur:
        _create(cur)
    # rollback on exception: the row must NOT persist
    with pytest.raises(RuntimeError):
        with db.connection() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO t (id) VALUES (%s)", ("rollme",))
            raise RuntimeError("boom")
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM t WHERE id = %s", ("rollme",))
        assert cur.fetchone()["n"] == 0
