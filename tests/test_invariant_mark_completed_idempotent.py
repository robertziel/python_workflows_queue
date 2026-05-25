"""``mark_completed`` / ``mark_failed`` idempotency contract.

The WHERE clause narrows to non-terminal states and returns ``None`` when no
rows matched. The contract: "calling ``mark_completed`` against a row that's
already terminal is a no-op and returns None." Same for ``mark_failed``.
"""

from __future__ import annotations

import pytest

from queue_workflows import node_queue
from queue_workflows.db import connection
from tests._helpers import make_run


def _make_run() -> str:
    return make_run(status="queued", workflow_name="_idempotency_test")


def _set_status(job_id: str, status: str) -> None:
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs SET status=%s WHERE id=%s",
            (status, job_id),
        )


# ── mark_completed contract ───────────────────────────────────────────────


def test_invariant_mark_completed_succeeds_on_running_row():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    _set_status(job_id, "running")
    row = node_queue.mark_completed(job_id, context_delta={"k": "v1"}, seconds=1.0)
    assert row is not None
    assert row["status"] == "completed"
    assert row["context_delta"] == {"k": "v1"}


def test_invariant_mark_completed_succeeds_on_queued_row():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    row = node_queue.mark_completed(job_id, context_delta={"k": "v1"}, seconds=0.0)
    assert row is not None
    assert row["status"] == "completed"


def test_invariant_mark_completed_succeeds_on_awaiting_input_row():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="upload", node_module="__input__upload", queue="cpu",
    )
    _set_status(job_id, "awaiting_input")
    row = node_queue.mark_completed(
        job_id, context_delta={"value": "user_input"}, seconds=0.0,
    )
    assert row is not None
    assert row["status"] == "completed"
    assert row["context_delta"] == {"value": "user_input"}


def test_invariant_mark_completed_returns_none_when_already_completed():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    _set_status(job_id, "running")
    first = node_queue.mark_completed(job_id, context_delta={"k": "v1"}, seconds=1.0)
    assert first is not None
    second = node_queue.mark_completed(
        job_id, context_delta={"k": "v2_should_not_appear"}, seconds=0.0,
    )
    assert second is None
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "completed"
    assert row["context_delta"] == {"k": "v1"}


def test_invariant_mark_completed_returns_none_when_failed():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    _set_status(job_id, "running")
    node_queue.mark_failed(job_id, error="boom", seconds=0.5)
    rv = node_queue.mark_completed(
        job_id, context_delta={"oops": "should_not_appear"}, seconds=0.0,
    )
    assert rv is None
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "failed"
    assert row["error"] == "boom"


def test_invariant_mark_completed_returns_none_when_cancelled():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    _set_status(job_id, "cancelled")
    rv = node_queue.mark_completed(
        job_id, context_delta={"k": "should_not_appear"}, seconds=0.0,
    )
    assert rv is None
    assert node_queue.get_node_job(job_id)["status"] == "cancelled"


# ── mark_failed contract ──────────────────────────────────────────────────


def test_invariant_mark_failed_succeeds_on_running_row():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    _set_status(job_id, "running")
    row = node_queue.mark_failed(job_id, error="kaboom", seconds=0.5)
    assert row is not None
    assert row["status"] == "failed"
    assert row["error"] == "kaboom"


def test_invariant_mark_failed_returns_none_when_already_completed():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    _set_status(job_id, "running")
    node_queue.mark_completed(job_id, context_delta={"k": "v1"}, seconds=1.0)
    rv = node_queue.mark_failed(job_id, error="late and wrong", seconds=99.0)
    assert rv is None
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "completed"
    assert row["context_delta"] == {"k": "v1"}
    assert row["error"] is None


def test_invariant_mark_failed_returns_none_when_already_failed():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    _set_status(job_id, "running")
    node_queue.mark_failed(job_id, error="first_error", seconds=0.5)
    rv = node_queue.mark_failed(
        job_id, error="second_error_should_not_appear", seconds=99.0,
    )
    assert rv is None
    row = node_queue.get_node_job(job_id)
    assert row["error"] == "first_error"


# ── Pre-validation: bad context_delta raises before UPDATE ────────────────


def test_invariant_mark_completed_rejects_unserialisable_context_delta():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    _set_status(job_id, "running")

    bad = {"k": {1, 2, 3}}  # set is not JSON-serialisable
    with pytest.raises((TypeError, ValueError)):
        node_queue.mark_completed(job_id, context_delta=bad, seconds=1.0)
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "running"
    assert row["context_delta"] == {}
