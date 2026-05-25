"""Durable dispatch-event outbox.

Three invariants:
1. Atomicity — ``mark_completed`` + ``enqueue_dispatch_event`` either both land
   or both roll back.
2. Replay — a failing callback is retried on the next drain cycle.
3. Eventual terminal — exhausted retries flip the run to ``failed``.
Plus the concurrent-drainers race test.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

import queue_workflows
from queue_workflows import dispatcher, node_pool, node_queue
from queue_workflows.db import connection
from tests._dispatch_driver import dispatch_driver  # noqa: F401
from tests._helpers import make_run


@pytest.fixture(autouse=True)
def _fake_node_pkg():
    queue_workflows.set_node_module_package("qwf_de_nodes")
    mod = types.ModuleType("qwf_de_nodes.smoke_heartbeat")
    mod.run = lambda **kw: {"context_delta": {"ok": True}}
    sys.modules["qwf_de_nodes.smoke_heartbeat"] = mod
    yield


def _make_run() -> str:
    return make_run(status="queued", workflow_name="_dispatch_events_test")


def _make_job(run_id: str, node_id: str = "n", queue: str = "cpu") -> str:
    return node_queue.enqueue_node_job(
        run_id=run_id, node_id=node_id, node_module="x", queue=queue,
    )


def _list_events(run_id: str | None = None) -> list[dict]:
    with connection() as c, c.cursor() as cur:
        if run_id is None:
            cur.execute("SELECT * FROM workflow_dispatch_events ORDER BY id")
        else:
            cur.execute(
                "SELECT * FROM workflow_dispatch_events WHERE run_id = %s ORDER BY id",
                (run_id,),
            )
        return list(cur.fetchall())


# ── Atomicity ─────────────────────────────────────────────────────────────


def test_invariant_mark_completed_and_event_atomic():
    run_id = _make_run()
    job_id = _make_job(run_id)

    with connection() as conn, conn.cursor() as cur:
        row = node_queue.mark_completed_in_txn(
            cur, job_id, context_delta={"k": "v"}, seconds=1.0,
        )
        assert row is not None
        node_queue.enqueue_dispatch_event_in_txn(cur, run_id, "n", "completed")

    job = node_queue.get_node_job(job_id)
    assert job["status"] == "completed"
    events = _list_events(run_id)
    assert len(events) == 1
    assert events[0]["kind"] == "completed"
    assert events[0]["processed_at"] is None


def test_invariant_atomic_rollback_on_jsonable_failure():
    run_id = _make_run()
    job_id = _make_job(run_id)

    with pytest.raises((TypeError, ValueError)):
        with connection() as conn, conn.cursor() as cur:
            node_queue.mark_completed_in_txn(
                cur, job_id, context_delta={"bad": {1, 2}}, seconds=1.0,
            )
            node_queue.enqueue_dispatch_event_in_txn(cur, run_id, "n", "completed")

    job = node_queue.get_node_job(job_id)
    assert job["status"] == "queued"
    assert _list_events(run_id) == []


# ── Replay & retry ────────────────────────────────────────────────────────


def test_crash_dispatch_event_replays_after_callback_failure(dispatch_driver, monkeypatch):
    run_id = _make_run()
    job_id = _make_job(run_id)

    with connection() as conn, conn.cursor() as cur:
        node_queue.mark_completed_in_txn(cur, job_id, context_delta={"k": "v"}, seconds=1.0)
        node_queue.enqueue_dispatch_event_in_txn(cur, run_id, "n", "completed")

    calls = {"n": 0}

    def flaky(*args, **kwargs) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated callback failure")
        return 0

    monkeypatch.setattr(dispatcher, "on_node_completed", flaky)

    dispatch_driver.pool._drain_dispatch_events()
    events = _list_events(run_id)
    assert len(events) == 1
    assert events[0]["processed_at"] is None
    assert events[0]["attempts"] == 1
    assert "simulated" in (events[0]["error"] or "")

    dispatch_driver.pool._drain_dispatch_events()
    events = _list_events(run_id)
    assert len(events) == 1
    assert events[0]["processed_at"] is not None
    assert calls["n"] == 2


def test_invariant_run_terminal_after_callbacks_exhaust_retries(dispatch_driver, monkeypatch):
    run_id = _make_run()
    job_id = _make_job(run_id)

    with connection() as conn, conn.cursor() as cur:
        node_queue.mark_completed_in_txn(cur, job_id, context_delta={}, seconds=0.0)
        node_queue.enqueue_dispatch_event_in_txn(cur, run_id, "n", "completed")

    def always_fails(*args, **kwargs) -> int:
        raise RuntimeError("permanent")

    monkeypatch.setattr(dispatcher, "on_node_completed", always_fails)

    cap = node_pool.NodePool._DISPATCH_MAX_ATTEMPTS
    for _ in range(cap):
        dispatch_driver.pool._drain_dispatch_events()

    from queue_workflows import run_store
    run = run_store.get_run(run_id)
    assert run["status"] == "failed", run.get("error")
    assert "permanent" in (run.get("error") or "")
    events = _list_events(run_id)
    assert events[0]["processed_at"] is not None


# ── Concurrent drain ─────────────────────────────────────────────────────


def test_race_concurrent_drainers_dont_double_dispatch(monkeypatch):
    run_id = _make_run()
    _make_job(run_id, node_id="n1")
    with connection() as conn, conn.cursor() as cur:
        node_queue.enqueue_dispatch_event_in_txn(cur, run_id, "n1", "completed")

    invocations: list[str] = []
    invocations_lock = threading.Lock()

    def counting(rid: str, nid: str) -> int:
        with invocations_lock:
            invocations.append(nid)
        return 0

    monkeypatch.setattr(dispatcher, "on_node_completed", counting)

    pool_a = node_pool.NodePool(register_builtins=None)
    pool_b = node_pool.NodePool(register_builtins=None)

    barrier = threading.Barrier(2)

    def drain(pool):
        barrier.wait()
        pool._drain_dispatch_events()

    t1 = threading.Thread(target=drain, args=(pool_a,))
    t2 = threading.Thread(target=drain, args=(pool_b,))
    t1.start(); t2.start()
    t1.join(); t2.join()

    events = _list_events(run_id)
    assert len(events) == 1
    assert events[0]["processed_at"] is not None
    assert len(invocations) == 1


# ── End-to-end via execute_node ──────────────────────────────────────────


def test_execute_node_writes_dispatch_event_in_same_txn(monkeypatch, dispatch_driver):
    from queue_workflows import node_executor

    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="smoke_heartbeat", queue="cpu",
    )
    job = node_queue.get_node_job(job_id)

    monkeypatch.setattr(
        node_executor, "_invoke", lambda **kw: {"context_delta": {"ok": True}},
    )
    monkeypatch.setattr(dispatcher, "on_node_completed", lambda *a, **k: 0)

    result = node_executor.execute_node(job)
    assert result == "completed"

    job = node_queue.get_node_job(job_id)
    assert job["status"] == "completed"
    events = _list_events(run_id)
    assert len(events) == 1
    assert events[0]["kind"] == "completed"
    assert events[0]["processed_at"] is None

    dispatch_driver.pool._drain_dispatch_events()
    events = _list_events(run_id)
    assert events[0]["processed_at"] is not None
