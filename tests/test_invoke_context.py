"""``invoke_context`` hook (Phase 6 upstream from lm_content_generator).

A host wraps each node invoke with setup/teardown + a success-only
``context_delta`` finalize, WITHOUT forking ``execute_node``: ``__enter__`` does
host setup (here: pin a ContextVar to the run id), yields a
``finalize(delta) -> delta`` applied only on success, and ``__exit__`` tears down
on every exit path. Default unset ⇒ identical behavior (``test_node_executor``
covers the no-hook path).
"""
from __future__ import annotations

import contextlib
import contextvars
import sys
import types

import pytest

import queue_workflows
from queue_workflows import node_executor, node_queue
from tests._helpers import make_run

_CV: contextvars.ContextVar = contextvars.ContextVar("qw_ic_test_ctx", default=None)


@pytest.fixture(autouse=True)
def _fake_pkg():
    # conftest._reset_engine_config resets injected config (incl. invoke_context)
    # after each test, keeping the test DSN wired — so no teardown needed here.
    queue_workflows.set_node_module_package("qwf_ic_nodes")


def _install(name, run_fn):
    mod = types.ModuleType(f"qwf_ic_nodes.{name}")
    mod.run = run_fn
    sys.modules[f"qwf_ic_nodes.{name}"] = mod


def _ctx_factory(events):
    @contextlib.contextmanager
    def factory(job, run):
        token = _CV.set(run.get("id"))
        events.append(("enter", _CV.get()))

        def finalize(delta):
            d = dict(delta)
            d["_stamped_run"] = _CV.get()
            return d

        try:
            yield finalize
        finally:
            _CV.reset(token)
            events.append(("exit", _CV.get()))

    return factory


def test_pins_during_run_stamps_on_success_and_tears_down():
    events: list = []
    queue_workflows.set_invoke_context(_ctx_factory(events))

    seen: dict = {}

    def run(**_kw):
        seen["cv"] = _CV.get()  # pinned by __enter__ for the duration of the run
        return {"context_delta": {"ok": True}}

    _install("_ic_ok", run)

    run_id = make_run(workflow_name="_ic_test", out_dir=None)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="_ic_ok", queue="cpu",
    )
    assert node_executor.execute_node(node_queue.get_node_job(job_id)) == "completed"

    assert seen["cv"] == run_id  # the node body saw the pinned ContextVar
    row = node_queue.get_node_job(job_id)
    assert row["context_delta"]["ok"] is True
    assert row["context_delta"]["_stamped_run"] == run_id  # success finalize ran
    assert events[0] == ("enter", run_id)
    assert events[-1] == ("exit", None)  # teardown ran (ContextVar reset)
    assert _CV.get() is None


def test_tears_down_on_failure_and_no_success_stamp():
    events: list = []
    queue_workflows.set_invoke_context(_ctx_factory(events))

    def run(**_kw):
        assert _CV.get() is not None  # pinned during the run
        raise RuntimeError("boom")

    _install("_ic_boom", run)

    run_id = make_run(workflow_name="_ic_test", out_dir=None)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="_ic_boom", queue="cpu",
    )
    assert node_executor.execute_node(node_queue.get_node_job(job_id)) == "failed"

    assert events[-1] == ("exit", None)  # teardown still ran on the failure path
    assert _CV.get() is None
    row = node_queue.get_node_job(job_id)
    assert "_stamped_run" not in (row["context_delta"] or {})  # finalize is success-only
