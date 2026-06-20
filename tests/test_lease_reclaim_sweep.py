"""Lease-reclaim sweep — ``node_queue.reclaim_expired_leases`` (+ ingest twin)
wired into :class:`NodePool` as a periodic sweep.

Covered:
- the pool tick invokes ``reclaim_expired_leases`` on its cadence (spied) — and
  the interval-gate suppresses back-to-back ticks;
- an expired-lease ``running`` row is re-queued through the sweep;
- with NO leased rows, the sweep is a clean no-op and leaves a lease-less
  ``running`` row untouched;
- the same three for the ``ingest_jobs`` twin.
"""

from __future__ import annotations

import logging
import time

import pytest

import queue_workflows
from queue_workflows import node_pool, node_queue
from queue_workflows.db import connection
from tests._helpers import force_lease, make_run


@pytest.fixture(autouse=True)
def _register_ingest_tasks():
    for name in ("run_fetch_all", "run_load_all"):
        queue_workflows.register_ingest_task(name, lambda reason: {"ok": True})
    yield


def _make_run(run_id: str | None = None) -> str:
    return make_run(run_id, workflow_name="_reclaim_sweep_test")


def _reclaim_pool() -> node_pool.NodePool:
    """NodePool whose reclaim sweep we drive directly without ``start()``.
    Interval forced to 0 so the gate never suppresses in tight loops."""
    pool = node_pool.NodePool(cpu_workers=0, gpu_workers=0, register_builtins=None)
    pool._reclaim_interval_s = 0.0
    pool._ingest_reclaim_interval_s = 0.0
    return pool


# ── cadence / spy ──────────────────────────────────────────────────────────


def test_sweep_invokes_reclaim_expired_leases(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(
        node_queue, "reclaim_expired_leases",
        lambda: calls.append(time.time()) or [],
    )
    pool = _reclaim_pool()
    pool._sweep_expired_leases()
    assert len(calls) == 1


def test_sweep_gated_by_interval(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(
        node_queue, "reclaim_expired_leases",
        lambda: calls.append(time.time()) or [],
    )
    pool = _reclaim_pool()
    pool._reclaim_interval_s = 5.0
    pool._sweep_expired_leases()
    pool._sweep_expired_leases()
    pool._sweep_expired_leases()
    assert len(calls) == 1


def test_tick_runs_the_reclaim_sweep(monkeypatch):
    fired: list[str] = []
    monkeypatch.setattr(node_pool.NodePool, "_drain_dispatch_events", lambda self: None)
    monkeypatch.setattr(node_pool.NodePool, "_sweep_expired_ingest_leases", lambda self: None)
    monkeypatch.setattr(
        node_pool.NodePool, "_sweep_expired_leases",
        lambda self: fired.append("reclaim"),
    )
    pool = node_pool.NodePool(cpu_workers=0, gpu_workers=0, register_builtins=None)
    pool._tick()
    assert fired == ["reclaim"]


# ── integration ─────────────────────────────────────────────────────────────


def test_sweep_requeues_expired_lease_row():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="x", queue="cpu", priority=100,
    )
    force_lease(job_id, expires_in_s=-30)

    pool = _reclaim_pool()
    pool._sweep_expired_leases()

    row = node_queue.get_node_job(job_id)
    assert row["status"] == "queued"
    assert row["claimed_by"] is None
    assert row["lease_expires_at"] is None
    assert row["started_at"] is None
    assert row["priority"] <= 10


def test_sweep_logs_reclaimed_rows(caplog):
    run_id = _make_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="the-node", node_module="x", queue="gpu",
        required_model="sdxl",
    )
    job_id = node_queue.list_jobs_for_run(run_id)[0]["id"]
    force_lease(job_id, expires_in_s=-5)

    pool = _reclaim_pool()
    with caplog.at_level(logging.INFO):
        pool._sweep_expired_leases()
    msgs = [r.getMessage() for r in caplog.records]
    assert any(run_id in m and "the-node" in m for m in msgs), msgs


# ── no-op safety ────────────────────────────────────────────────────────────


def test_sweep_noop_with_no_leased_rows():
    pool = _reclaim_pool()
    pool._sweep_expired_leases()  # must not raise


def test_sweep_leaves_legacy_running_row_untouched():
    """A ``running`` row carrying NO lease is left exactly as-is — the reclaim
    WHERE requires a non-null, expired lease."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="legacy", node_module="x", queue="cpu",
    )
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs SET status='running', started_at=now() "
            "WHERE id=%s",
            (job_id,),
        )

    pool = _reclaim_pool()
    pool._sweep_expired_leases()

    row = node_queue.get_node_job(job_id)
    assert row["status"] == "running"
    assert row["lease_expires_at"] is None


def test_sweep_does_not_reclaim_fresh_lease():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="fresh", node_module="x", queue="cpu",
    )
    force_lease(job_id, expires_in_s=600)

    pool = _reclaim_pool()
    pool._sweep_expired_leases()

    row = node_queue.get_node_job(job_id)
    assert row["status"] == "running"
    assert row["claimed_by"] == "host-x"


# ── ingest lease reclaim (the ingest_jobs twin) ─────────────────────────────


def _force_ingest_lease(job_id: str, *, expires_in_s: float) -> None:
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE ingest_jobs "
            "SET status='running', started_at=now(), claimed_by='host-x', "
            "    lease_expires_at = now() + make_interval(secs => %s) "
            "WHERE id=%s",
            (float(expires_in_s), job_id),
        )


def test_ingest_sweep_invokes_reclaim_expired_ingest_leases(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(
        node_queue, "reclaim_expired_ingest_leases",
        lambda: calls.append(time.time()) or [],
    )
    pool = _reclaim_pool()
    pool._sweep_expired_ingest_leases()
    assert len(calls) == 1


def test_ingest_sweep_gated_by_interval(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(
        node_queue, "reclaim_expired_ingest_leases",
        lambda: calls.append(time.time()) or [],
    )
    pool = _reclaim_pool()
    pool._ingest_reclaim_interval_s = 5.0
    pool._sweep_expired_ingest_leases()
    pool._sweep_expired_ingest_leases()
    pool._sweep_expired_ingest_leases()
    assert len(calls) == 1


def test_tick_runs_the_ingest_reclaim_sweep(monkeypatch):
    fired: list[str] = []
    monkeypatch.setattr(node_pool.NodePool, "_drain_dispatch_events", lambda self: None)
    monkeypatch.setattr(node_pool.NodePool, "_sweep_expired_leases", lambda self: None)
    monkeypatch.setattr(
        node_pool.NodePool, "_sweep_expired_ingest_leases",
        lambda self: fired.append("ingest-reclaim"),
    )
    pool = node_pool.NodePool(cpu_workers=0, gpu_workers=0, register_builtins=None)
    pool._tick()
    assert fired == ["ingest-reclaim"]


def test_ingest_sweep_requeues_expired_lease_row():
    job_id = node_queue.enqueue_ingest_job(
        task_name="run_fetch_all", queue="fetch", priority=100,
    )
    _force_ingest_lease(job_id, expires_in_s=-30)

    pool = _reclaim_pool()
    pool._sweep_expired_ingest_leases()

    row = node_queue.get_ingest_job(job_id)
    assert row["status"] == "queued"
    assert row["claimed_by"] is None
    assert row["lease_expires_at"] is None
    assert row["started_at"] is None
    assert row["priority"] <= 10


def test_ingest_sweep_logs_reclaimed_rows(caplog):
    job_id = node_queue.enqueue_ingest_job(task_name="run_load_all", queue="load")
    _force_ingest_lease(job_id, expires_in_s=-5)

    pool = _reclaim_pool()
    with caplog.at_level(logging.WARNING):
        pool._sweep_expired_ingest_leases()
    msgs = [r.getMessage() for r in caplog.records]
    assert any(job_id in m and "run_load_all" in m for m in msgs), msgs


def test_ingest_sweep_noop_with_no_leased_rows():
    pool = _reclaim_pool()
    pool._sweep_expired_ingest_leases()  # must not raise


def test_ingest_sweep_does_not_reclaim_fresh_lease():
    job_id = node_queue.enqueue_ingest_job(task_name="run_fetch_all", queue="fetch")
    _force_ingest_lease(job_id, expires_in_s=600)

    pool = _reclaim_pool()
    pool._sweep_expired_ingest_leases()

    row = node_queue.get_ingest_job(job_id)
    assert row["status"] == "running"
    assert row["claimed_by"] == "host-x"


# ── reclaim ALL running (the restart-resume hook, NOT lease-expiry) ──────────


def test_reclaim_all_requeues_dead_workers_running_row():
    """``reclaim_all_running_for_resume`` re-queues a running row whose claiming
    worker is GONE — even with a FRESH lease the expiry sweep leaves alone.
    ``host-x`` has no heartbeat here, so it counts as dead and the row is
    reclaimed at once instead of waiting out the 600s lease."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="x", queue="gpu",
        required_model="qwen_edit", priority=100,
    )
    force_lease(job_id, expires_in_s=600)  # fresh lease, but host-x has no heartbeat → dead
    assert node_queue.reclaim_expired_leases() == []  # sanity: not yet expired

    rows = node_queue.reclaim_all_running_for_resume()
    assert any(r["id"] == job_id for r in rows)
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "queued"
    assert row["claimed_by"] is None
    assert row["lease_expires_at"] is None
    assert row["started_at"] is None
    assert row["priority"] <= 10


def test_reclaim_all_leaves_live_workers_running_row():
    """Regression — "box-a2 stopped taking GPU tasks". A running row whose
    claiming worker is STILL ALIVE (fresh heartbeat on the job's queue) must NOT
    be reclaimed on orchestrator boot: re-queuing it clears ``claimed_by`` and
    trips the live worker's JobStatusWatcher into a hard-exit. The orchestrator/
    dispatcher container restarts independently of the GPU claim workers, so its
    boot recovery must only touch genuinely-orphaned (dead-worker) rows; the
    lease sweep backstops anything it skips."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="x", queue="gpu",
        required_model="qwen_edit", priority=100,
    )
    force_lease(job_id, expires_in_s=600)  # claimed_by='host-x', running
    node_queue.upsert_worker_heartbeat(
        host_label="host-x", queue="gpu", concurrency=1,
    )  # host-x is ALIVE

    rows = node_queue.reclaim_all_running_for_resume()
    assert all(r["id"] != job_id for r in rows), "a live worker's job must not be reclaimed"
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "running"
    assert row["claimed_by"] == "host-x"


def test_reclaim_all_requeues_gpu_job_when_only_cpu_heartbeat_is_fresh():
    """The dead-worker scope is per-QUEUE, not per-host. A fresh heartbeat on a
    DIFFERENT queue of the same host must NOT shield an orphaned job.

    ``reclaim_all_running_for_resume`` skips a running row only when its claiming
    host has a fresh heartbeat ON THE JOB'S OWN QUEUE
    (``h.host_label = j.claimed_by AND h.queue = j.queue``). A host can run a
    live cpu worker while its gpu worker has died: the gpu job is genuinely
    orphaned and must be reclaimed. Without the ``AND h.queue = j.queue`` term
    the live cpu heartbeat would wrongly protect the dead gpu worker's job,
    leaving it stuck ``running`` until its lease lapses (here, never — fresh
    600 s lease). Pairs with ``test_reclaim_all_leaves_live_workers_running_row``
    (same-queue heartbeat ⇒ correctly left alone)."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="x", queue="gpu",
        required_model="qwen_edit", priority=100,
    )
    force_lease(job_id, expires_in_s=600)  # claimed_by='host-x', fresh gpu lease
    node_queue.upsert_worker_heartbeat(
        host_label="host-x", queue="cpu", concurrency=1,
    )  # only the CPU worker on host-x is alive; its GPU worker is gone

    rows = node_queue.reclaim_all_running_for_resume()
    assert any(r["id"] == job_id for r in rows), (
        "a gpu job must be reclaimed when only a cpu heartbeat is fresh"
    )
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "queued"
    assert row["claimed_by"] is None


def test_reclaim_all_noop_when_nothing_running():
    assert node_queue.reclaim_all_running_for_resume() == []


# ── cancel-aware reclaim ────────────────────────────────────────────────────


def _cancel_run(run_id: str) -> None:
    """Flip ``workflow_runs.status`` to ``cancelled`` — same effect as
    the Rails ``WorkflowsController#cancel`` endpoint, just the
    state-transition step (the cascade is what we're testing isn't
    needed for the lease-reclaim path)."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_runs SET status='cancelled', finished_at=now() "
            "WHERE id::text=%s",
            (run_id,),
        )


def test_lease_reclaim_cancels_running_row_on_cancelled_parent():
    """Regression: when a worker dies mid-job on a CANCELLED run, the
    lease expires, the sweep re-queues the row, the parent run is
    cancelled so no worker will EVER claim it (the claim SQL filters
    cancelled parents) — and the row sits in ``queued`` as a ghost
    forever. The fix: when the parent run is in a terminal state, the
    sweep marks the row ``cancelled`` instead of re-queueing.

    Reproduction: run ``41570ecd-566e-4281-8b12-4e925fceebd1`` —
    ``reconstruct/render_hunyuan_i2v`` sat ``queued`` for 20+ minutes
    after the run was cancelled. The orphan was discovered by the
    operator in the queue popover and reported as
    "I cancelled this run but it's still in the GPU queue."
    """
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="ghost", node_module="x", queue="gpu",
        priority=100,
    )
    force_lease(job_id, expires_in_s=-30)
    _cancel_run(run_id)

    pool = _reclaim_pool()
    pool._sweep_expired_leases()

    row = node_queue.get_node_job(job_id)
    assert row["status"] == "cancelled", (
        f"row whose parent run is cancelled must NOT be re-queued; "
        f"got status={row['status']!r}. This is the ghost-job bug."
    )
    # Bookkeeping should still be cleared so a future debugger can tell
    # this was a reclaim-into-cancel, not a stale ``running`` row.
    assert row["claimed_by"] is None
    assert row["lease_expires_at"] is None


def test_lease_reclaim_still_requeues_when_parent_is_running():
    """The cancel-aware branch must NOT regress the normal case. An
    expired lease on a row whose parent is still ``running`` must
    re-queue, exactly as before."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="alive", node_module="x", queue="cpu",
        priority=100,
    )
    force_lease(job_id, expires_in_s=-30)

    pool = _reclaim_pool()
    pool._sweep_expired_leases()

    row = node_queue.get_node_job(job_id)
    assert row["status"] == "queued"
