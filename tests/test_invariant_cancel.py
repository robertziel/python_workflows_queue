"""Cancel propagation (run-cancel guard + cancel-watcher + dispatcher
early-returns).

The contract: no node-job transitions to ``running`` after the parent run is
cancelled, and downstream nodes never get enqueued once a run is terminal.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

import queue_workflows
from queue_workflows import dispatcher, node_queue, run_store
from queue_workflows.db import connection
from tests._cancel_helper import cancel_run_via_rails
from tests._helpers import make_run


def _make_run() -> str:
    return make_run(status="queued", workflow_name="_cancel_test")


# ── Claim SQL run-cancel guard ──────────────────────────────────────────


def test_claim_skips_job_when_run_cancelled():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    cancel_run_via_rails(run_id)

    assert node_queue.claim_next_cpu_job(0, host="h") is None
    assert node_queue.get_node_job(job_id)["status"] == "queued"


def test_claim_skips_job_when_run_failed():
    run_id = _make_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    run_store.update_run(run_id, status="failed", error="upstream blew up")

    assert node_queue.claim_next_cpu_job(0, host="h") is None


def test_claim_succeeds_when_run_active():
    run_id = _make_run()
    run_store.update_run(run_id, status="running")
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    claimed = node_queue.claim_next_cpu_job(0, host="h")
    assert claimed is not None and claimed["id"] == job_id
    assert node_queue.get_node_job(job_id)["status"] == "running"


# ── dispatcher early-return ──────────────────────────────────────────────


def test_invariant_no_dependents_enqueued_after_cancel():
    run_id = _make_run()
    a_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="x", queue="cpu",
    )
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs SET status='completed', "
            "context_delta='{}'::jsonb WHERE id=%s",
            (a_id,),
        )
    cancel_run_via_rails(run_id)

    n = dispatcher.on_node_completed(run_id, "a")
    assert n == 0


def test_invariant_on_node_failed_short_circuits_on_cancelled():
    run_id = _make_run()
    cancel_run_via_rails(run_id)

    dispatcher.on_node_failed(run_id, "some_node")

    row = run_store.get_run(run_id)
    assert row["status"] == "cancelled"
    assert row["error"] == "cancelled by user"


def test_invariant_on_node_awaiting_input_short_circuits_on_cancelled():
    run_id = _make_run()
    cancel_run_via_rails(run_id)

    dispatcher.on_node_awaiting_input(run_id, "pick_pano")

    assert run_store.get_run(run_id)["status"] == "cancelled"


# ── Cancel watcher ────────────────────────────────────────────────────────


def test_cancel_watcher_observes_status_change():
    from queue_workflows.cancel_watcher import _start_run_cancel_watcher

    run_id = _make_run()
    cancel_event = threading.Event()
    t = _start_run_cancel_watcher(run_id, cancel_event, interval_s=0.1)

    cancel_event.wait(0.2)
    assert not cancel_event.is_set()

    cancel_run_via_rails(run_id)

    fired = cancel_event.wait(2.0)
    assert fired
    t.join(timeout=1.0)


def test_cancel_watcher_exits_on_event_set():
    from queue_workflows.cancel_watcher import _start_run_cancel_watcher

    run_id = _make_run()
    cancel_event = threading.Event()
    t = _start_run_cancel_watcher(run_id, cancel_event, interval_s=5.0)

    cancel_event.set()
    t.join(timeout=2.0)
    assert not t.is_alive()


def test_cancel_watcher_swallows_db_errors(monkeypatch):
    from queue_workflows import cancel_watcher as _cw

    run_id = _make_run()
    cancel_event = threading.Event()
    calls = {"n": 0}

    def flaky_get_run(rid: str):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated DB blip")
        return {"id": rid, "status": "cancelled"}

    monkeypatch.setattr(_cw.run_store, "get_run", flaky_get_run)

    t = _cw._start_run_cancel_watcher(run_id, cancel_event, interval_s=0.05)
    fired = cancel_event.wait(2.0)
    assert fired
    t.join(timeout=1.0)


# ── _invoke threads cancel_event through to opt-in nodes ─────────────────


def test_invoke_passes_cancel_event_to_opting_in_node():
    queue_workflows.set_node_module_package("qwf_cancel_nodes")
    captured: dict = {}

    def opt_in_run(*, inputs=None, out=None, model_handle=None,
                   status_callback=None, cancel_event=None):
        captured["got_cancel"] = cancel_event is not None
        captured["event_set"] = cancel_event.is_set() if cancel_event else None
        return {"context_delta": {}}

    mod = types.ModuleType("qwf_cancel_nodes._cancel_test_node")
    mod.run = opt_in_run
    sys.modules["qwf_cancel_nodes._cancel_test_node"] = mod

    from queue_workflows import node_executor

    sentinel_event = threading.Event()
    sentinel_event.set()

    node_executor._invoke(
        module_name="_cancel_test_node",
        inputs={}, out=None, handle=None,
        cancel_event=sentinel_event,
    )

    assert captured == {"got_cancel": True, "event_set": True}
