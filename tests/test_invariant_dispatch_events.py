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

import psycopg
import pytest

import queue_workflows
from queue_workflows import dispatcher, node_pool, node_queue, run_store
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


def test_invariant_event_insert_failure_rolls_back_terminal_mark():
    """The outbox is atomic in BOTH directions.

    ``test_invariant_atomic_rollback_on_jsonable_failure`` only proves the
    *first* statement (the ``mark_completed`` JSON validation) aborting the txn.
    The reverse — the *dispatch-event INSERT* aborting the txn after a
    successful terminal mark — is the one that actually matters for the outbox
    guarantee: if the terminal mark could commit independently of (or before)
    the event insert, a regression would leave a node row ``completed`` with NO
    outbox event — a node the dispatcher never fans out from (a permanently
    un-dispatched, silently-stuck workflow). Here we make the event INSERT fail
    via the migration-0004 CHECK ``kind IN ('completed','failed',
    'awaiting_input')`` and assert the terminal mark rolled back WITH it.
    """
    run_id = _make_run()
    job_id = _make_job(run_id)

    # The terminal mark lands first (status -> 'completed' inside this txn),
    # then the CHECK-violating kind aborts the whole transaction.
    from tests._helpers import INTEGRITY_ERRORS
    with pytest.raises(INTEGRITY_ERRORS):
        with connection() as conn, conn.cursor() as cur:
            marked = node_queue.mark_completed_in_txn(
                cur, job_id, context_delta={"k": "v"}, seconds=1.0,
            )
            assert marked is not None and marked["status"] == "completed"
            node_queue.enqueue_dispatch_event_in_txn(cur, run_id, "n", "BOGUS_KIND")

    # Both halves rolled back: the row is back to 'queued' and no event exists.
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


# ── Drain routing: non-'completed' branches ──────────────────────────────


_VALID_KINDS = ("completed", "failed", "awaiting_input")
_KIND_CHECK = "workflow_dispatch_events_kind_check"


def test_drain_routes_awaiting_input_to_its_own_callback(dispatch_driver, monkeypatch):
    """The drain loop must route each ``kind`` to its OWN dispatcher callback.

    Every other test in this module enqueues ``kind='completed'``, so the
    ``awaiting_input`` branch (node_pool ``_drain_dispatch_events`` ->
    ``dispatcher.on_node_awaiting_input``) is otherwise dead in the suite. A
    regression that routed ``awaiting_input`` to ``on_node_completed`` (which
    would wrongly fan out a node that's actually parked waiting for human
    input) would pass every existing test. Here we prove the awaiting_input
    event invokes on_node_awaiting_input — and ONLY that callback — and is then
    marked processed.
    """
    run_id = _make_run()
    _make_job(run_id, node_id="n1")
    with connection() as conn, conn.cursor() as cur:
        node_queue.enqueue_dispatch_event_in_txn(cur, run_id, "n1", "awaiting_input")

    awaited: list[tuple[str, str]] = []
    monkeypatch.setattr(
        dispatcher, "on_node_awaiting_input",
        lambda rid, nid: awaited.append((rid, nid)),
    )

    def _must_not_fire(*a, **k):
        raise AssertionError("awaiting_input routed to the wrong callback")

    monkeypatch.setattr(dispatcher, "on_node_completed", _must_not_fire)
    monkeypatch.setattr(dispatcher, "on_node_failed", _must_not_fire)

    dispatch_driver.pool._drain_dispatch_events()

    assert awaited == [(run_id, "n1")]
    events = _list_events(run_id)
    assert len(events) == 1
    assert events[0]["processed_at"] is not None


@pytest.mark.pg_only
def test_drain_unknown_kind_never_silently_processed(dispatch_driver, monkeypatch):
    """An unrecognised ``kind`` must NOT be silently marked processed.

    The migration-0004 CHECK normally makes an unknown kind unreachable, so the
    ``else`` branch of ``_drain_dispatch_events`` (250-260) is pure defensive
    code — but it is the last guard against silently *dropping* fan-out: if a
    regression made that branch ``SET processed_at = now()`` (treating an
    unknown event as handled), a node's downstream fan-out would vanish with no
    error. We exercise the branch by temporarily dropping the CHECK, inserting a
    bogus-kind row, and asserting the drain (a) calls NONE of the three
    callbacks, (b) bumps ``attempts`` and records the kind in ``error``, and
    (c) leaves ``processed_at IS NULL`` — even after many cycles, the event is
    retried forever rather than dropped.

    NOTE on the asymmetry: unlike the callback-failure path, this branch does
    ``continue`` *before* the poison-flag code, so it never flips the run to
    'failed'. We assert that TRUE behavior (not the run-failed the callback
    path produces) — see the reported discrepancy.
    """
    for name in ("on_node_completed", "on_node_failed", "on_node_awaiting_input"):
        monkeypatch.setattr(
            dispatcher, name,
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("unknown kind routed to a real callback")
            ),
        )

    run_id = _make_run()
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"ALTER TABLE workflow_dispatch_events "
            f"DROP CONSTRAINT IF EXISTS {_KIND_CHECK}",
        )
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO workflow_dispatch_events (run_id, node_id, kind) "
                "VALUES (%s, %s, %s)",
                (run_id, "n1", "bogus_kind"),
            )

        cap = node_pool.NodePool._DISPATCH_MAX_ATTEMPTS

        dispatch_driver.pool._drain_dispatch_events()
        events = _list_events(run_id)
        assert len(events) == 1
        assert events[0]["processed_at"] is None        # NOT silently dropped
        assert events[0]["attempts"] == 1
        assert "bogus_kind" in (events[0]["error"] or "")

        # Many more cycles: still never processed, attempts keep climbing, and
        # (per the actual code) the run is NOT poison-flagged failed.
        for _ in range(cap + 2):
            dispatch_driver.pool._drain_dispatch_events()
        events = _list_events(run_id)
        assert events[0]["processed_at"] is None
        assert events[0]["attempts"] >= cap
        assert run_store.get_run(run_id)["status"] != "failed"
    finally:
        # Restore the schema for every later test in this session: drop the
        # bogus rows so the re-added CHECK validates against a clean table.
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM workflow_dispatch_events WHERE kind <> ALL(%s)",
                (list(_VALID_KINDS),),
            )
            cur.execute(
                f"ALTER TABLE workflow_dispatch_events "
                f"DROP CONSTRAINT IF EXISTS {_KIND_CHECK}",
            )
            cur.execute(
                f"ALTER TABLE workflow_dispatch_events ADD CONSTRAINT {_KIND_CHECK} "
                "CHECK (kind IN ('completed', 'failed', 'awaiting_input'))",
            )
