"""Late input resolution.

The contract: what the node module sees is the LATEST resolution of its declared
refs, not whatever was snapshotted at enqueue time.
"""

from __future__ import annotations

import sys
import types

import pytest

import queue_workflows
from queue_workflows import dispatcher, node_queue, run_store
from queue_workflows.db import connection
from tests._helpers import make_run


@pytest.fixture(autouse=True)
def _fake_node_pkg():
    queue_workflows.set_node_module_package("qwf_lir_nodes")
    mod = types.ModuleType("qwf_lir_nodes.smoke_heartbeat")
    mod.run = lambda **kw: {"context_delta": {}}
    sys.modules["qwf_lir_nodes.smoke_heartbeat"] = mod
    yield


def _make_run(workflow_name: str = "_late_resolution_test") -> str:
    return make_run(status="queued", workflow_name=workflow_name)


def _set_context_delta(job_id: str, delta: dict) -> None:
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs "
            "SET status = 'completed', context_delta = %s WHERE id = %s",
            (node_queue._as_json(delta), job_id),
        )


# ── resolve_inputs_for_job ────────────────────────────────────────────────


def test_resolve_inputs_uses_latest_sibling_context_delta():
    run_id = _make_run()
    a_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="x", queue="cpu",
    )
    b_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="b", node_module="x", queue="cpu",
        inputs={"k": {"$from": "a.k"}},
    )

    _set_context_delta(a_id, {"k": "v1"})
    assert dispatcher.resolve_inputs_for_job(b_id) == {"k": "v1"}

    _set_context_delta(a_id, {"k": "v2"})
    assert dispatcher.resolve_inputs_for_job(b_id) == {"k": "v2"}


def test_resolve_inputs_uses_run_context_for_run_level_refs():
    run_id = _make_run()
    run_store.update_run(run_id, context={"parcel": {"lat": 50.5, "lon": 22.5}})
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
        inputs={"lat": {"$from": "parcel.lat"}, "lon": {"$from": "parcel.lon"}},
    )
    assert dispatcher.resolve_inputs_for_job(job_id) == {"lat": 50.5, "lon": 22.5}


def test_resolve_inputs_passes_through_literals():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
        inputs={"strength": 0.7, "model": "sdxl"},
    )
    assert dispatcher.resolve_inputs_for_job(job_id) == {"strength": 0.7, "model": "sdxl"}


def test_resolve_inputs_skips_resolution_for_input_node():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="upload", node_module="__input__upload", queue="cpu",
        inputs={"widget": "upload", "target": "image_path"},
    )
    assert dispatcher.resolve_inputs_for_job(job_id) == {
        "widget": "upload", "target": "image_path",
    }


def test_resolve_inputs_falls_back_to_snapshot_on_missing_ref():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
        inputs={"k": {"$from": "absent_sibling.k"}},
    )
    assert dispatcher.resolve_inputs_for_job(job_id) == {"k": {"$from": "absent_sibling.k"}}


# ── Snapshot column behaviour ─────────────────────────────────────────────


def test_resolved_inputs_column_is_a_snapshot():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
        inputs={"strength": 0.7},
    )
    node_queue.set_resolved_inputs(job_id, {"strength": 0.7, "extra": "x"})
    row = node_queue.get_node_job(job_id)
    assert row["resolved_inputs"] == {"strength": 0.7, "extra": "x"}
    assert row["inputs"] == {"strength": 0.7}
    assert row["status"] == "queued"


def test_resolved_inputs_rejects_unserialisable():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    with pytest.raises((TypeError, ValueError)):
        node_queue.set_resolved_inputs(job_id, {"bad": {1, 2}})


# ── End-to-end through execute_node ──────────────────────────────────────


def test_invariant_execute_node_uses_late_resolved_inputs(monkeypatch):
    from queue_workflows import node_executor

    run_id = _make_run()
    a_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="x", queue="cpu",
    )
    b_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="b", node_module="smoke_heartbeat", queue="cpu",
        inputs={"k": {"$from": "a.k"}},
    )
    _set_context_delta(a_id, {"k": "v1"})
    _set_context_delta(a_id, {"k": "v2"})

    captured: dict = {}

    def fake_invoke(*, module_name, inputs, out, handle, **_extra):
        captured.update(inputs)
        return {"context_delta": {}}

    monkeypatch.setattr(node_executor, "_invoke", fake_invoke)
    monkeypatch.setattr(dispatcher, "on_node_completed", lambda *a, **k: 0)

    result = node_executor.execute_node(node_queue.get_node_job(b_id))
    assert result == "completed"
    assert captured == {"k": "v2"}

    row = node_queue.get_node_job(b_id)
    assert row["resolved_inputs"] == {"k": "v2"}
