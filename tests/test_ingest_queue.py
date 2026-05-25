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
