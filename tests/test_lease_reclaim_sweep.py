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
