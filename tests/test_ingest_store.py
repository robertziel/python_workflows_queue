"""``ingest_store`` — the backend-agnostic ingest-job seam (``db_backend``).

The same data-layer contract must hold on BOTH backends:

* ``pg`` (default) is **byte-identical** to today — the seam just delegates to the
  existing ``node_queue.*ingest*`` functions against ``ingest_jobs``.
* ``redis`` runs the ingest family on the StorageBackend SPI with **no**
  ``ingest_jobs`` table at all.

The redis cases SKIP when ``QUEUE_WORKFLOWS_TEST_REDIS_URL`` is unset/unreachable,
so the suite still runs Postgres-only; CI must show redis green.
"""

from __future__ import annotations

import os
import threading
import time
import uuid

import pytest

import queue_workflows
from queue_workflows import ingest_store
from queue_workflows.db import connection


def _task(reason, args=None):
    return {"ok": True, "reason": reason}


def _ingest_jobs_count() -> int:
    """Rows in the Postgres ``ingest_jobs`` table (the pg-only engine table)."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM ingest_jobs")
        return int(cur.fetchone()["n"])


# ── pg path: the seam is a transparent delegate to node_queue ───────────────


def test_pg_path_delegates_to_node_queue_end_to_end():
    queue_workflows.register_ingest_task("probe_task", _task)
    from queue_workflows import node_queue

    jid = ingest_store.enqueue_ingest_job(
        task_name="probe_task", queue="fetch", reason="tick", args={"k": 1},
    )
    # Visible through the engine's own ingest_jobs reader (same row, same table).
    row = node_queue.get_ingest_job(jid)
    assert row is not None and row["status"] == "queued" and row["task_name"] == "probe_task"
    assert ingest_store.get_ingest_job(jid)["args"] == {"k": 1}

    claimed = ingest_store.claim_next_ingest_job("fetch", host="w1", lease_s=30)
    assert claimed is not None and claimed["id"] == jid and claimed["status"] == "running"
    assert ingest_store.renew_ingest_lease(jid, "w1", lease_s=30) is True

    done = ingest_store.mark_ingest_completed(jid, result={"n": 7}, seconds=0.1)
    assert done is not None and done["status"] == "completed"
    # Idempotent: a second terminal call is a no-op.
    assert ingest_store.mark_ingest_completed(jid, result={"n": 99}) is None
    assert node_queue.get_ingest_job(jid)["result"] == {"n": 7}


def test_pg_snapshot_matches_node_queue():
    queue_workflows.register_ingest_task("probe_task", _task)
    ingest_store.enqueue_ingest_job(task_name="probe_task", queue="fetch")
    from queue_workflows import node_queue

    assert ingest_store.ingest_snapshot() == node_queue.ingest_snapshot()


@pytest.mark.parametrize("backend_env", [None])  # pg path (default)
def test_enqueue_validation_rejects_before_write(backend_env):
    queue_workflows.register_ingest_task("probe_task", _task)
    before = _ingest_jobs_count()
    with pytest.raises(ValueError, match="ingest queue must be in"):
        ingest_store.enqueue_ingest_job(task_name="probe_task", queue="nope")
    with pytest.raises(ValueError, match="registered ingest task"):
        ingest_store.enqueue_ingest_job(task_name="ghost", queue="fetch")
    assert _ingest_jobs_count() == before  # nothing written on a rejected enqueue


# ── redis path: ingest runs on the SPI, no ingest_jobs table ────────────────


@pytest.fixture
def redis_ingest(monkeypatch):
    """Configure the seam onto a fresh redis namespace; skip if redis is down."""
    url = os.environ.get("QUEUE_WORKFLOWS_TEST_REDIS_URL")
    if not url:
        pytest.skip("set QUEUE_WORKFLOWS_TEST_REDIS_URL to exercise the redis path")
    # The seam's redis backend reads its DSN from cfg.redis_url_env (default env).
    monkeypatch.setenv("QUEUE_WORKFLOWS_REDIS_URL", url)
    ns = f"ing_{uuid.uuid4().hex[:12]}"
    queue_workflows.configure(db_backend="redis", db_namespace=ns)
    queue_workflows.register_ingest_task("probe_task", _task)
    from queue_workflows import backends

    try:
        backends.get_backend().ensure_schema()
    except Exception as exc:  # driver missing / server down
        pytest.skip(f"redis unreachable: {type(exc).__name__}: {exc}")
    yield
    backends.close_all()  # conftest _reset_engine_config restores db_backend=pg


def test_redis_end_to_end_and_no_postgres_rows(redis_ingest):
    jid = ingest_store.enqueue_ingest_job(
        task_name="probe_task", queue="fetch", reason="manual", args={"scenario": 5},
    )
    got = ingest_store.get_ingest_job(jid)
    assert got is not None and got["status"] == "queued"
    assert got["task_name"] == "probe_task" and got["reason"] == "manual"
    assert got["args"] == {"scenario": 5}

    claimed = ingest_store.claim_next_ingest_job("fetch", host="w1", lease_s=30)
    assert claimed is not None and claimed["id"] == jid
    assert claimed["status"] == "running" and claimed["task_name"] == "probe_task"
    assert claimed["args"] == {"scenario": 5}
    # Nothing claimable left.
    assert ingest_store.claim_next_ingest_job("fetch", host="w1", lease_s=30) is None

    assert ingest_store.renew_ingest_lease(jid, "w1", lease_s=30) is True
    assert ingest_store.renew_ingest_lease(jid, "someone_else", lease_s=30) is False

    done = ingest_store.mark_ingest_completed(jid, result={"n": 7}, seconds=0.1)
    assert done is not None and done["status"] == "completed"
    assert ingest_store.mark_ingest_completed(jid, result={"n": 99}) is None  # idempotent
    # No-clobber: the raced 2nd terminal must not overwrite the finalized result.
    assert ingest_store.get_ingest_job(jid)["result"] == {"n": 7}

    # The whole round-trip touched ZERO Postgres ingest_jobs rows.
    assert _ingest_jobs_count() == 0


def test_redis_mark_failed_idempotent(redis_ingest):
    jid = ingest_store.enqueue_ingest_job(task_name="probe_task", queue="fetch")
    ingest_store.claim_next_ingest_job("fetch", host="w1", lease_s=30)
    failed = ingest_store.mark_ingest_failed(jid, error="boom", seconds=0.2)
    assert failed is not None and failed["status"] == "failed"
    assert ingest_store.mark_ingest_failed(jid, error="again") is None
    # Cannot resurrect a failed job into completed.
    assert ingest_store.mark_ingest_completed(jid, result={}) is None


def test_redis_claim_exactly_once_under_contention(redis_ingest):
    n = 25
    ids = {
        ingest_store.enqueue_ingest_job(task_name="probe_task", queue="fetch")
        for _ in range(n)
    }
    assert len(ids) == n

    claimed: list[str] = []
    lock = threading.Lock()
    start = threading.Event()

    def worker(w: int):
        start.wait()
        while True:
            job = ingest_store.claim_next_ingest_job("fetch", host=f"w{w}", lease_s=30)
            if job is None:
                return
            with lock:
                claimed.append(job["id"])

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(timeout=30)

    assert sorted(claimed) == sorted(ids)        # every job claimed
    assert len(claimed) == len(set(claimed))     # none claimed twice


def test_redis_reclaim_expired_requeues_lapsed_lease(redis_ingest):
    jid = ingest_store.enqueue_ingest_job(task_name="probe_task", queue="fetch")
    # Claim with a tiny lease, let it lapse, then reclaim.
    ingest_store.claim_next_ingest_job("fetch", host="w1", lease_s=1)
    assert ingest_store.get_ingest_job(jid)["status"] == "running"
    time.sleep(1.2)

    reclaimed = ingest_store.reclaim_expired_ingest_leases()
    assert any(r["id"] == jid for r in reclaimed)
    assert {r["id"]: r["task_name"] for r in reclaimed}[jid] == "probe_task"
    assert ingest_store.get_ingest_job(jid)["status"] == "queued"
    # Reclaimed job is claimable again.
    again = ingest_store.claim_next_ingest_job("fetch", host="w2", lease_s=30)
    assert again is not None and again["id"] == jid


def test_redis_priority_ordering_lower_number_first(redis_ingest):
    """The seam negates priority for the SPI's DESC claim order, so the engine's
    ingest convention holds: a LOWER priority number is claimed first. Guards the
    negation mapping (a dropped negation would invert the order)."""
    low_first = ingest_store.enqueue_ingest_job(
        task_name="probe_task", queue="fetch", priority=10,
    )
    later = ingest_store.enqueue_ingest_job(
        task_name="probe_task", queue="fetch", priority=100,
    )
    first = ingest_store.claim_next_ingest_job("fetch", host="w1", lease_s=30)
    second = ingest_store.claim_next_ingest_job("fetch", host="w1", lease_s=30)
    assert first["id"] == low_first and first["priority"] == 10
    assert second["id"] == later and second["priority"] == 100


def test_redis_snapshot_reflects_state(redis_ingest):
    ingest_store.enqueue_ingest_job(task_name="probe_task", queue="fetch")
    ingest_store.enqueue_ingest_job(task_name="probe_task", queue="fetch")
    ingest_store.claim_next_ingest_job("fetch", host="w1", lease_s=30)
    from queue_workflows import backends

    backends.get_backend().heartbeat("w1", "fetch")

    snap = ingest_store.ingest_snapshot()["queues"]["fetch"]
    assert snap["queued"] == 1 and snap["running"] == 1
    assert snap["workers"] >= 1
