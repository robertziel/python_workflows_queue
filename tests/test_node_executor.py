"""Unit tests for the shared node-execution body ``execute_node``.

``execute_node(job, *, model_cache=None, cancel_event=None)`` owns:
re-resolving ``$from`` inputs at execute time, building the node out_dir,
invoking the node module (threading the model handle + cancel_event), and the
terminal ``mark_*_in_txn`` + dispatch-event outbox write in one transaction.

The node-module resolver is wired at a fake test package via
``set_node_module_package`` so the engine suite stays domain-free.

Contract pinned here:
  * a mocked node → row flips ``completed`` + a ``completed`` event in the same
    txn;
  * a raising node → row flips ``failed`` + a ``failed`` event;
  * a GPU job with a ``required_model`` calls ``model_cache.require_model``
    (once) and threads the handle into the node;
  * an already-terminal row → no-op (returns "skipped"), writes no event.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

import queue_workflows
from queue_workflows import node_executor, node_queue
from queue_workflows.db import connection
from tests._helpers import make_run


@pytest.fixture(autouse=True)
def _fake_node_pkg():
    queue_workflows.set_node_module_package("qwf_ne_nodes")
    yield


def _make_run() -> str:
    return make_run(workflow_name="_node_exec_test", out_dir=None)


def _install_fake_node(name: str, run_fn) -> None:
    mod = types.ModuleType(f"qwf_ne_nodes.{name}")
    mod.run = run_fn
    sys.modules[f"qwf_ne_nodes.{name}"] = mod


def _events(run_id: str) -> list[dict]:
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT * FROM workflow_dispatch_events WHERE run_id=%s ORDER BY id",
            (run_id,),
        )
        return list(cur.fetchall())


# ── happy path ───────────────────────────────────────────────────────────


def test_execute_node_marks_completed_and_enqueues_event():
    run_id = _make_run()
    captured: dict = {}

    def run(*, inputs=None, out=None, model_handle=None, status_callback=None):
        captured["inputs"] = inputs
        return {"context_delta": {"ok": True}}

    _install_fake_node("_ne_ok", run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="_ne_ok", queue="cpu",
        inputs={"a": 1},
    )

    result = node_executor.execute_node(node_queue.get_node_job(job_id))
    assert result == "completed"

    row = node_queue.get_node_job(job_id)
    assert row["status"] == "completed"
    assert row["context_delta"] == {"ok": True}
    assert row["resolved_inputs"] == {"a": 1}

    evts = _events(run_id)
    assert len(evts) == 1
    assert evts[0]["kind"] == "completed"
    assert evts[0]["processed_at"] is None


def test_execute_node_marks_failed_and_enqueues_event():
    run_id = _make_run()

    def run(**_kw):
        raise RuntimeError("boom in node body")

    _install_fake_node("_ne_boom", run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="_ne_boom", queue="cpu",
    )

    result = node_executor.execute_node(node_queue.get_node_job(job_id))
    assert result == "failed"

    row = node_queue.get_node_job(job_id)
    assert row["status"] == "failed"
    assert "boom in node body" in (row["error"] or "")

    evts = _events(run_id)
    assert len(evts) == 1
    assert evts[0]["kind"] == "failed"


def test_execute_node_threads_model_cache_for_gpu_job():
    run_id = _make_run()
    seen: dict = {}

    def run(*, inputs=None, out=None, model_handle=None, status_callback=None,
            cancel_event=None, model_load_seconds=None):
        seen["handle"] = model_handle
        seen["load_s_is_float"] = isinstance(model_load_seconds, float)
        return {"context_delta": {}}

    _install_fake_node("_ne_gpu", run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="_ne_gpu", queue="gpu",
        required_model="some_model",
    )

    sentinel_handle = object()
    require_calls: list[str] = []

    class _FakeCache:
        current_model = None

        def require_model(self, model_id):
            require_calls.append(model_id)
            return sentinel_handle

    result = node_executor.execute_node(
        node_queue.get_node_job(job_id), model_cache=_FakeCache(),
    )
    assert result == "completed"
    assert require_calls == ["some_model"]
    assert seen["handle"] is sentinel_handle
    assert seen["load_s_is_float"] is True


def test_execute_node_model_load_failure_marks_failed():
    run_id = _make_run()

    def run(**_kw):
        raise AssertionError("node body must not run when model load fails")

    _install_fake_node("_ne_gpu_loadfail", run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="_ne_gpu_loadfail", queue="gpu",
        required_model="broken_model",
    )

    class _FakeCache:
        current_model = None

        def require_model(self, model_id):
            raise RuntimeError("cold load exploded")

    result = node_executor.execute_node(
        node_queue.get_node_job(job_id), model_cache=_FakeCache(),
    )
    assert result == "failed"
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "failed"
    assert "cold load exploded" in (row["error"] or "")
    evts = _events(run_id)
    assert len(evts) == 1 and evts[0]["kind"] == "failed"


def test_execute_node_noop_on_already_terminal_row():
    run_id = _make_run()
    ran = {"n": 0}

    def run(**_kw):
        ran["n"] += 1
        return {"context_delta": {}}

    _install_fake_node("_ne_terminal", run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="_ne_terminal", queue="cpu",
    )
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs SET status='completed', "
            "context_delta='{}'::jsonb WHERE id=%s",
            (job_id,),
        )

    result = node_executor.execute_node(node_queue.get_node_job(job_id))
    assert result == "skipped"
    evts = _events(run_id)
    assert evts == []


def test_execute_node_passes_cancel_event_to_opting_in_node():
    run_id = _make_run()
    captured: dict = {}

    def run(*, inputs=None, out=None, model_handle=None, status_callback=None,
            cancel_event=None):
        captured["got"] = cancel_event is not None
        return {"context_delta": {}}

    _install_fake_node("_ne_cancel", run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="_ne_cancel", queue="cpu",
    )
    ev = threading.Event()
    node_executor.execute_node(node_queue.get_node_job(job_id), cancel_event=ev)
    assert captured["got"] is True
