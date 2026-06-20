"""``ingest_jobs`` claim helpers (the periodic-work queue).

A periodic ingest unit is a STANDALONE callable, not a DAG node — so it gets a
dedicated ``ingest_jobs`` table (migration 0007). These pin the thin SQL layer
over that table, mirroring ``test_node_queue_lease.py`` for cpu/gpu:

  * ``enqueue_ingest_job`` inserts a ``queued`` row (validating task_name
    against the host-registered task map);
  * ``claim_next_ingest_job`` atomically claims, stamping the lease;
  * a claim is scoped to its queue;
  * ``mark_ingest_completed`` / ``mark_ingest_failed`` are idempotent;
  * ``reclaim_expired_ingest_leases`` re-queues an expired-lease running row;
  * the NOTIFY trigger fires ``ingest_job_ready``.
"""

from __future__ import annotations

import pytest
import psycopg

import queue_workflows
from queue_workflows import db, node_queue
from queue_workflows.db import connection


@pytest.fixture(autouse=True)
def _register_ingest_tasks():
    """Register the fake ingest task names so enqueue_ingest_job accepts them
    (the engine's task set is host-configurable — empty by default)."""
    for name in ("run_fetch_all", "run_load_all", "audit_freshness"):
        queue_workflows.register_ingest_task(name, lambda reason: {"ok": True})
    yield


def _row(job_id: str) -> dict:
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM ingest_jobs WHERE id = %s", (job_id,))
        r = cur.fetchone()
    assert r is not None
    return r


def _force_ingest_lease(job_id: str, *, expires_in_s: float) -> None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE ingest_jobs "
            "SET status='running', claimed_by='dead-worker', "
            "    lease_expires_at = now() + make_interval(secs => %s) "
            "WHERE id = %s",
            (expires_in_s, job_id),
        )


# ── enqueue ──────────────────────────────────────────────────────────────


def test_enqueue_ingest_job_inserts_queued_row():
    jid = node_queue.enqueue_ingest_job(
        task_name="run_fetch_all", queue="fetch", reason="tick",
    )
    row = _row(jid)
    assert row["status"] == "queued"
    assert row["task_name"] == "run_fetch_all"
    assert row["queue"] == "fetch"
    assert row["reason"] == "tick"
    assert row["claimed_by"] is None
    assert row["lease_expires_at"] is None


def test_enqueue_ingest_job_rejects_bad_queue():
    with pytest.raises(ValueError):
        node_queue.enqueue_ingest_job(task_name="run_fetch_all", queue="cpu")


def test_enqueue_ingest_job_rejects_unregistered_task():
    with pytest.raises(ValueError):
        node_queue.enqueue_ingest_job(task_name="rm_rf", queue="fetch")


# ── claim ──────────────────────────────────────────────────────────────────


def test_claim_next_ingest_job_stamps_lease():
    jid = node_queue.enqueue_ingest_job(task_name="run_load_all", queue="load")
    claimed = node_queue.claim_next_ingest_job("load", host="h1", lease_s=600)
    assert claimed is not None
    assert claimed["id"] == jid
    assert claimed["status"] == "running"
    assert claimed["claimed_by"] == "h1"
    assert claimed["lease_expires_at"] is not None
    assert claimed["started_at"] is not None


def test_claim_is_scoped_to_its_queue():
    fetch_id = node_queue.enqueue_ingest_job(task_name="run_fetch_all", queue="fetch")
    node_queue.enqueue_ingest_job(task_name="run_load_all", queue="load")

    got = node_queue.claim_next_ingest_job("fetch", host="hf")
    assert got["id"] == fetch_id
    assert node_queue.claim_next_ingest_job("fetch", host="hf") is None
    assert node_queue.claim_next_ingest_job("load", host="hl") is not None


def test_claim_returns_none_when_empty():
    assert node_queue.claim_next_ingest_job("fetch", host="h") is None


def test_claim_orders_by_priority_then_creation():
    older = node_queue.enqueue_ingest_job(task_name="run_fetch_all", queue="fetch")
    newer = node_queue.enqueue_ingest_job(task_name="run_fetch_all", queue="fetch")
    assert node_queue.claim_next_ingest_job("fetch", host="h")["id"] == older
    assert node_queue.claim_next_ingest_job("fetch", host="h")["id"] == newer


# ── terminal marks (idempotent) ──────────────────────────────────────────────


def test_mark_ingest_completed_sets_terminal_and_result():
    jid = node_queue.enqueue_ingest_job(task_name="run_fetch_all", queue="fetch")
    node_queue.claim_next_ingest_job("fetch", host="h")
    row = node_queue.mark_ingest_completed(jid, result={"dispatched": 7}, seconds=1.5)
    assert row is not None
    assert row["status"] == "completed"
    assert row["result"] == {"dispatched": 7}
    assert row["finished_at"] is not None
    assert node_queue.mark_ingest_completed(jid, result={}, seconds=0.0) is None


def test_mark_ingest_failed_sets_terminal_and_error():
    jid = node_queue.enqueue_ingest_job(task_name="run_load_all", queue="load")
    node_queue.claim_next_ingest_job("load", host="h")
    row = node_queue.mark_ingest_failed(jid, error="boom", seconds=2.0)
    assert row is not None
    assert row["status"] == "failed"
    assert "boom" in (row["error"] or "")
    assert node_queue.mark_ingest_failed(jid, error="again") is None


# ── reclaim ──────────────────────────────────────────────────────────────────


def test_reclaim_expired_lease_requeues_ingest_job():
    jid = node_queue.enqueue_ingest_job(task_name="run_fetch_all", queue="fetch")
    _force_ingest_lease(jid, expires_in_s=-5)
    reclaimed = node_queue.reclaim_expired_ingest_leases()
    assert any(r["id"] == jid for r in reclaimed)
    row = _row(jid)
    assert row["status"] == "queued"
    assert row["claimed_by"] is None
    assert row["lease_expires_at"] is None


def test_reclaim_leaves_fresh_lease_ingest_job_alone():
    jid = node_queue.enqueue_ingest_job(task_name="run_fetch_all", queue="fetch")
    _force_ingest_lease(jid, expires_in_s=600)
    reclaimed = node_queue.reclaim_expired_ingest_leases()
    assert all(r["id"] != jid for r in reclaimed)
    assert _row(jid)["status"] == "running"


# ── NOTIFY trigger ───────────────────────────────────────────────────────────


def test_notify_fires_on_queued_insert_and_reclaim():
    with psycopg.connect(db.db_url(), autocommit=True) as listen_conn:
        listen_conn.execute("LISTEN ingest_job_ready")

        jid = node_queue.enqueue_ingest_job(task_name="run_fetch_all", queue="fetch")
        notifies = []
        for n in listen_conn.notifies(timeout=2.0, stop_after=1):
            notifies.append(n)
        assert notifies, "queued INSERT must fire ingest_job_ready"
        assert notifies[0].payload == "fetch"

        _force_ingest_lease(jid, expires_in_s=-5)
        node_queue.reclaim_expired_ingest_leases()
        reclaim_notifies = []
        for n in listen_conn.notifies(timeout=2.0, stop_after=1):
            reclaim_notifies.append(n)
        assert reclaim_notifies, "reclaim UPDATE must fire ingest_job_ready"
        assert reclaim_notifies[0].payload == "fetch"


# ── per-job args payload (migration 0008 / G2) ───────────────────────────────


def test_enqueue_persists_args():
    jid = node_queue.enqueue_ingest_job(
        task_name="run_fetch_all", queue="fetch", args={"scenario_id": 42},
    )
    assert _row(jid)["args"] == {"scenario_id": 42}


def test_enqueue_defaults_args_to_empty():
    jid = node_queue.enqueue_ingest_job(task_name="run_fetch_all", queue="fetch")
    assert _row(jid)["args"] == {}


def test_execute_passes_args_to_two_arg_callable():
    seen: dict = {}

    def scenario_task(reason, args):
        seen.update(args)
        return {"ran": reason}

    queue_workflows.register_ingest_task("scenario_task", scenario_task)
    jid = node_queue.enqueue_ingest_job(
        task_name="scenario_task", queue="load", args={"scenario_id": 7},
    )
    job = node_queue.claim_next_ingest_job("load", host="h")
    from queue_workflows import ingest_executor
    assert ingest_executor.execute_ingest_job(job) == "completed"
    assert seen == {"scenario_id": 7}
    assert node_queue.get_ingest_job(jid)["status"] == "completed"


def test_execute_one_arg_callable_ignores_args():
    # run_fetch_all is registered (autouse) as a 1-arg lambda; args are dropped.
    jid = node_queue.enqueue_ingest_job(
        task_name="run_fetch_all", queue="fetch", args={"ignored": 1},
    )
    job = node_queue.claim_next_ingest_job("fetch", host="h")
    from queue_workflows import ingest_executor
    assert ingest_executor.execute_ingest_job(job) == "completed"


# ── caller-supplied transaction (G3) ─────────────────────────────────────────


def test_enqueue_with_conn_commits_with_caller():
    with connection() as conn:
        jid = node_queue.enqueue_ingest_job(
            task_name="run_fetch_all", queue="fetch", conn=conn,
        )
        with conn.cursor() as cur:  # visible inside the caller's own txn
            cur.execute("SELECT status FROM ingest_jobs WHERE id = %s", (jid,))
            assert cur.fetchone()["status"] == "queued"
    assert node_queue.get_ingest_job(jid) is not None  # committed on clean exit


def test_enqueue_with_conn_rolls_back_with_caller():
    holder: dict = {}
    with pytest.raises(RuntimeError):
        with connection() as conn:
            holder["id"] = node_queue.enqueue_ingest_job(
                task_name="run_fetch_all", queue="fetch", conn=conn,
            )
            raise RuntimeError("boom")  # context exit rolls the txn back
    assert node_queue.get_ingest_job(holder["id"]) is None


# ── host-configurable queue names (G1) ───────────────────────────────────────


def test_enqueue_accepts_host_configured_queue():
    queue_workflows.configure(ingest_queues=frozenset({"hydraulic", "corrdiff"}))
    queue_workflows.register_ingest_task("run_scenario", lambda r, a: {"ok": True})
    jid = node_queue.enqueue_ingest_job(
        task_name="run_scenario", queue="hydraulic", args={"scenario_id": 1},
    )
    assert _row(jid)["queue"] == "hydraulic"


def test_enqueue_rejects_queue_outside_configured_set():
    queue_workflows.configure(ingest_queues=frozenset({"hydraulic"}))
    queue_workflows.register_ingest_task("run_scenario", lambda r, a: {"ok": True})
    with pytest.raises(ValueError):
        node_queue.enqueue_ingest_job(task_name="run_scenario", queue="fetch")


# ── ingest snapshot for the host queue-indicator UI (G5) ──────────────────────


def test_ingest_snapshot_reports_depth_and_workers():
    queue_workflows.configure(ingest_queues=frozenset({"hydro", "hydraulic"}))
    queue_workflows.register_ingest_task("t", lambda r, a: {})
    node_queue.enqueue_ingest_job(task_name="t", queue="hydro")
    node_queue.enqueue_ingest_job(task_name="t", queue="hydro")
    node_queue.enqueue_ingest_job(task_name="t", queue="hydraulic")
    node_queue.claim_next_ingest_job("hydraulic", host="h")  # → running
    node_queue.upsert_worker_heartbeat(
        host_label="host-c", queue="hydro", concurrency=1,
    )

    snap = node_queue.ingest_snapshot()
    assert snap["queues"]["hydro"]["queued"] == 2
    assert snap["queues"]["hydro"]["workers"] == 1
    assert snap["queues"]["hydraulic"]["running"] == 1
    assert snap["queues"]["hydraulic"]["workers"] == 0


# ── ingest_executor terminal/idempotency contract (rank 9) ───────────────────


def test_execute_ingest_job_failed_when_task_raises_and_skipped_when_terminal():
    """``execute_ingest_job`` is the ingest twin of ``execute_node``'s
    terminal+idempotency contract, and its failure/skip branches are the part
    that the happy-path tests never reach:

      * a task that RAISES must be caught, the row marked ``failed`` with the
        exception text preserved, and the call must return ``"failed"`` —
        deleting the ``except`` handler would otherwise leak the exception and
        leave the row stuck ``running``;
      * an UNKNOWN ``task_name`` (no registered callable) raises inside
        ``_run_task`` and must funnel through the same fail path;
      * if the row was already finalized out-of-band (a raced/duplicate claim),
        ``mark_ingest_completed`` returns ``None`` and the executor must return
        ``"skipped"`` WITHOUT clobbering the already-stored result.
    """
    from queue_workflows import ingest_executor

    # FAILED: a registered task that raises.
    def boom_task(reason):
        raise RuntimeError("kaboom")

    queue_workflows.register_ingest_task("boom_task", boom_task)
    jid = node_queue.enqueue_ingest_job(task_name="boom_task", queue="fetch")
    job = node_queue.claim_next_ingest_job("fetch", host="h")
    assert ingest_executor.execute_ingest_job(job) == "failed"
    failed_row = node_queue.get_ingest_job(jid)
    assert failed_row["status"] == "failed"
    assert "kaboom" in (failed_row["error"] or "")
    assert "RuntimeError" in (failed_row["error"] or "")

    # FAILED: an unknown task_name (raises ValueError inside _run_task).
    jid2 = node_queue.enqueue_ingest_job(task_name="run_fetch_all", queue="fetch")
    job2 = node_queue.claim_next_ingest_job("fetch", host="h")
    assert job2["id"] == jid2
    job2["task_name"] = "not_registered"  # simulate a stale/unmapped row
    assert ingest_executor.execute_ingest_job(job2) == "failed"
    unknown_row = node_queue.get_ingest_job(jid2)
    assert unknown_row["status"] == "failed"
    assert "unknown ingest task_name" in (unknown_row["error"] or "")

    # SKIPPED: row finalized out-of-band before the executor's terminal write.
    queue_workflows.register_ingest_task("ok_task", lambda reason: {"ok": True})
    jid3 = node_queue.enqueue_ingest_job(task_name="ok_task", queue="load")
    job3 = node_queue.claim_next_ingest_job("load", host="h")
    node_queue.mark_ingest_completed(jid3, result={"first": 1}, seconds=0.0)
    assert ingest_executor.execute_ingest_job(job3) == "skipped"
    # the pre-existing result must NOT be clobbered by the skipped completion.
    assert node_queue.get_ingest_job(jid3)["result"] == {"first": 1}


# ── ingest terminal twins: clobber / cross-terminal / JSON-prevalidation (14) ─


def test_mark_ingest_terminals_no_clobber_cross_terminal_and_json_prevalidated():
    """The ingest terminal twins must carry the SAME load-bearing guarantees as
    their node twins (``mark_completed``/``mark_failed``), not merely "the 2nd
    same-status call returns None":

      * NO-CLOBBER: a 2nd ``mark_ingest_completed`` returns ``None`` AND leaves
        the first call's stored ``result`` intact (a stray duplicate delivery
        can't overwrite a finalized payload);
      * CROSS-TERMINAL: ``mark_ingest_completed`` on an already-``failed`` row
        returns ``None`` and leaves ``status='failed'`` + the error intact —
        proving the ``WHERE status NOT IN (...)`` shape, not just a
        ``status='running'`` guard;
      * JSON-PREVALIDATION: a non-JSON ``result`` raises BEFORE the UPDATE, so
        the row is never mutated (it stays ``running``).
    """
    # NO-CLOBBER — second completed call must not overwrite the result.
    jid = node_queue.enqueue_ingest_job(task_name="run_fetch_all", queue="fetch")
    node_queue.claim_next_ingest_job("fetch", host="h")
    assert node_queue.mark_ingest_completed(jid, result={"dispatched": 7}) is not None
    assert node_queue.mark_ingest_completed(jid, result={"x": 99}) is None
    assert node_queue.get_ingest_job(jid)["result"] == {"dispatched": 7}

    # CROSS-TERMINAL — completed-on-failed returns None, error untouched.
    jid2 = node_queue.enqueue_ingest_job(task_name="run_load_all", queue="load")
    node_queue.claim_next_ingest_job("load", host="h")
    assert node_queue.mark_ingest_failed(jid2, error="boom") is not None
    assert node_queue.mark_ingest_completed(jid2, result={"ok": 1}) is None
    cross = node_queue.get_ingest_job(jid2)
    assert cross["status"] == "failed"
    assert cross["error"] == "boom"

    # JSON-PREVALIDATION — bad payload raises before any mutation.
    jid3 = node_queue.enqueue_ingest_job(task_name="run_fetch_all", queue="fetch")
    node_queue.claim_next_ingest_job("fetch", host="h")
    with pytest.raises((TypeError, ValueError)):
        node_queue.mark_ingest_completed(jid3, result={"k": {1, 2, 3}})
    assert node_queue.get_ingest_job(jid3)["status"] == "running"
