"""The LIVE ingest work path runs on ``db_backend="redis"``.

The prior slice proved the `ingest_store` seam in isolation; this proves the
ENGINE's real call sites are wired through it — so with `db_backend="redis"` the
scheduler enqueues to redis and a claim worker's `run_once()` claims → executes →
finalizes the ingest job on redis, with no Postgres `ingest_jobs` rows touched.

Redis cases SKIP when `QUEUE_WORKFLOWS_TEST_REDIS_URL` is unset/unreachable.

Out of scope (next slice): `run_forever`'s daemon bootstrap (await_schema / park /
the LISTEN wake loop / worker_heartbeats) is still PG-coupled, so this drives the
unit of work via `run_once()` directly — the same handle the existing claim-worker
tests use.
"""

from __future__ import annotations

import os
import uuid

import pytest

import queue_workflows
from queue_workflows import claim_worker, ingest_store, scheduler
from queue_workflows.db import connection


def _ingest_jobs_count() -> int:
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM ingest_jobs")
        return int(cur.fetchone()["n"])


# ── pg path: node_pool reclaim sweep routes through the seam ────────────────


def test_node_pool_ingest_reclaim_routes_through_seam(monkeypatch):
    """`NodePool._sweep_expired_ingest_leases` must call the db_backend seam (so a
    redis deploy reclaims on redis), not `node_queue` directly. Verified on the pg
    default with a spy — proving the wire without needing a live redis here."""
    from queue_workflows import node_pool as _np

    calls: list[str] = []
    monkeypatch.setattr(
        ingest_store, "reclaim_expired_ingest_leases",
        lambda: calls.append("seam") or [],
    )
    pool = _np.NodePool()
    pool._ingest_reclaim_last_run = 0.0          # bypass the interval gate
    pool._ingest_reclaim_interval_s = 0.0
    pool._sweep_expired_ingest_leases()
    assert calls == ["seam"]


# ── redis path: the wired work path runs end-to-end on redis ────────────────


@pytest.fixture
def redis_ingest():
    """Configure the engine onto a fresh redis namespace; register a probe task
    that records each run. Skips if redis is down."""
    url = os.environ.get("QUEUE_WORKFLOWS_TEST_REDIS_URL")
    if not url:
        pytest.skip("set QUEUE_WORKFLOWS_TEST_REDIS_URL to exercise the redis path")
    os.environ["QUEUE_WORKFLOWS_REDIS_URL"] = url  # the env the seam's backend reads
    ns = f"ingw_{uuid.uuid4().hex[:12]}"
    queue_workflows.configure(db_backend="redis", db_namespace=ns)

    ran: list[str] = []

    def probe_task(reason, args=None):
        ran.append(reason)
        return {"ok": True, "reason": reason}

    queue_workflows.register_ingest_task("probe_task", probe_task)
    from queue_workflows import backends

    try:
        backends.get_backend().ensure_schema()
    except Exception as exc:
        pytest.skip(f"redis unreachable: {type(exc).__name__}: {exc}")
    yield ran
    backends.close_all()  # conftest _reset_engine_config restores db_backend=pg
    os.environ.pop("QUEUE_WORKFLOWS_REDIS_URL", None)


def test_scheduler_enqueue_then_worker_run_once_on_redis(redis_ingest):
    ran = redis_ingest
    entry = scheduler.ScheduleEntry(
        name="probe", minute=0, task_name="probe_task", queue="fetch",
    )
    ids = scheduler.enqueue_due([entry], reason="tick")
    assert len(ids) == 1
    jid = ids[0]
    assert ingest_store.get_ingest_job(jid)["status"] == "queued"

    worker = claim_worker.ClaimWorker(queue="fetch", host="w1", lease_s=30)
    assert worker.run_once() is True          # claimed + executed one job
    assert ran == ["tick"]                    # the registered task actually ran

    done = ingest_store.get_ingest_job(jid)
    assert done["status"] == "completed"
    assert done["result"] == {"ok": True, "reason": "tick"}

    assert worker.run_once() is False         # queue now empty
    assert _ingest_jobs_count() == 0          # ran entirely on redis, zero PG rows


def test_lease_renewer_renews_ingest_on_redis(redis_ingest):
    jid = ingest_store.enqueue_ingest_job(task_name="probe_task", queue="fetch")
    ingest_store.claim_next_ingest_job("fetch", host="w1", lease_s=30)

    owner = claim_worker.LeaseRenewer(
        job_id=jid, claimed_by="w1", lease_s=30, table="ingest_jobs",
    )
    assert owner._renew_once() is True        # renews on redis via the seam
    stranger = claim_worker.LeaseRenewer(
        job_id=jid, claimed_by="someone_else", lease_s=30, table="ingest_jobs",
    )
    assert stranger._renew_once() is False     # not the owner → no renew


def test_watchdog_fail_path_marks_redis_ingest_job_failed(redis_ingest):
    jid = ingest_store.enqueue_ingest_job(task_name="probe_task", queue="fetch")
    ingest_store.claim_next_ingest_job("fetch", host="w1", lease_s=30)

    exits: list[int] = []
    claim_worker._fail_job_and_exit(
        job_id=jid, table="ingest_jobs", error="watchdog boom",
        on_exit=exits.append, exit_code=75,
    )
    assert exits == [75]                        # hard-exit hook still fired
    failed = ingest_store.get_ingest_job(jid)
    assert failed["status"] == "failed"         # marked failed ON REDIS
    assert _ingest_jobs_count() == 0
