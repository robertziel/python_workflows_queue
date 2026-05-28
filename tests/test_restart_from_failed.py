"""Restart-from-failed primitive — keep ``completed`` / ``skipped`` job rows,
delete the rest, so ``dispatcher.start_run`` resumes the run from the failed
branch instead of re-doing the entire DAG.

The engine's ``_find_ready_nodes`` already treats existing ``completed`` /
``skipped`` rows as cursors when re-expanding the DAG (see ``dispatcher.py``):
only nodes WITHOUT a job row whose deps are completed/skipped get enqueued. The
missing primitive was a way to clear the failed branch without nuking the
completed prefix; ``cancel_queued_jobs_for_run`` (the existing helper) flips
queued → cancelled but leaves the rows in place, which would block re-enqueue.

This new helper deletes every non-terminal-complete row and returns the deleted
``node_id``s so the host can cascade into its own artefacts (on-disk dirs,
input submissions). Pure delete-by-status — the dispatcher does the
re-enqueueing on its next tick.
"""

from __future__ import annotations

import pytest

from queue_workflows import node_queue
from queue_workflows.db import connection
from tests._helpers import make_run


def _enqueue_in_status(run_id: str, node_id: str, status: str, *, queue: str = "cpu") -> str:
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id=node_id, node_module="x", queue=queue,
    )
    if status != "queued":
        with connection() as c, c.cursor() as cur:
            cur.execute(
                "UPDATE workflow_node_jobs SET status=%s WHERE id=%s",
                (status, job_id),
            )
    return job_id


def _statuses_by_node(run_id: str) -> dict[str, str]:
    return {j["node_id"]: j["status"]
            for j in node_queue.list_jobs_for_run(run_id)}


# ── delete_non_terminal_jobs_for_run() ──────────────────────────────────────


def test_delete_non_terminal_jobs_drops_failed_cancelled_queued_awaiting():
    """Every status EXCEPT completed / skipped is deleted — those four are the
    statuses a restart must re-do."""
    run_id = make_run()
    _enqueue_in_status(run_id, "n_queued", "queued")
    _enqueue_in_status(run_id, "n_running", "running")  # also non-terminal
    _enqueue_in_status(run_id, "n_failed", "failed")
    _enqueue_in_status(run_id, "n_cancelled", "cancelled")
    _enqueue_in_status(run_id, "n_awaiting", "awaiting_input")

    deleted = node_queue.delete_non_terminal_jobs_for_run(run_id)

    assert set(deleted) == {
        "n_queued", "n_running", "n_failed", "n_cancelled", "n_awaiting",
    }
    assert _statuses_by_node(run_id) == {}


def test_delete_non_terminal_jobs_keeps_completed_and_skipped():
    """Completed + skipped rows are PRESERVED — they're the cursors
    ``_find_ready_nodes`` reads to know "this node is done, move on"."""
    run_id = make_run()
    _enqueue_in_status(run_id, "ok",      "completed")
    _enqueue_in_status(run_id, "skipped", "skipped")
    _enqueue_in_status(run_id, "fail",    "failed")

    deleted = node_queue.delete_non_terminal_jobs_for_run(run_id)

    assert deleted == ["fail"]
    assert _statuses_by_node(run_id) == {"ok": "completed", "skipped": "skipped"}


def test_delete_non_terminal_jobs_scoped_to_one_run():
    """A second run's rows MUST NOT be touched — the helper is per-run."""
    rid_a = make_run()
    rid_b = make_run()
    _enqueue_in_status(rid_a, "a_failed", "failed")
    _enqueue_in_status(rid_b, "b_failed", "failed")
    _enqueue_in_status(rid_b, "b_ok",     "completed")

    deleted = node_queue.delete_non_terminal_jobs_for_run(rid_a)

    assert deleted == ["a_failed"]
    assert _statuses_by_node(rid_a) == {}
    assert _statuses_by_node(rid_b) == {"b_failed": "failed", "b_ok": "completed"}


def test_delete_non_terminal_jobs_is_idempotent():
    """A second call after the first returns ``[]`` and leaves the run alone —
    no rows, nothing to do."""
    run_id = make_run()
    _enqueue_in_status(run_id, "n", "failed")

    first = node_queue.delete_non_terminal_jobs_for_run(run_id)
    second = node_queue.delete_non_terminal_jobs_for_run(run_id)

    assert first == ["n"]
    assert second == []


def test_delete_non_terminal_jobs_returns_empty_when_only_terminal_complete():
    """A run whose every job ran to completion returns ``[]`` — nothing to
    re-do. This is the no-op case (operator clicked retry on an already-
    completed run; the host's controller is responsible for rejecting that,
    but the helper must be safe to call regardless)."""
    run_id = make_run()
    _enqueue_in_status(run_id, "a", "completed")
    _enqueue_in_status(run_id, "b", "skipped")

    deleted = node_queue.delete_non_terminal_jobs_for_run(run_id)

    assert deleted == []
    assert _statuses_by_node(run_id) == {"a": "completed", "b": "skipped"}


# ── End-to-end: start_run resumes from the deleted set ──────────────────────


def test_start_run_after_delete_re_enqueues_only_the_failed_branch():
    """The whole point of the primitive: with completed prefix intact + failed
    rows deleted, ``dispatcher.start_run`` re-enqueues ONLY the failed nodes
    (and their downstream that never got a row in the first place)."""
    import queue_workflows
    from queue_workflows import dispatcher

    # Wire a stub workflow + pipeline so the dispatcher has a DAG to traverse.
    queue_workflows.set_workflow_provider(
        lambda name: {
            "name": name,
            "steps": [{"id": "p", "kind": "pipeline", "pipeline": "_resume_test"}],
        },
        lambda name: {
            "name": name,
            "nodes": [
                {"id": "a", "node": "x"},
                {"id": "b", "node": "x", "depends_on": ["a"]},
                {"id": "c", "node": "x", "depends_on": ["b"]},
            ],
        },
    )

    run_id = make_run(workflow_name="_resume_wf")
    # a ran clean, b ran clean, c failed.
    _enqueue_in_status(run_id, "p/a", "completed")
    _enqueue_in_status(run_id, "p/b", "completed")
    _enqueue_in_status(run_id, "p/c", "failed")

    deleted = node_queue.delete_non_terminal_jobs_for_run(run_id)
    assert deleted == ["p/c"]

    # Resume: start_run should see a, b completed and re-enqueue c only.
    enqueued = dispatcher.start_run(run_id)
    assert enqueued == 1
    jobs = _statuses_by_node(run_id)
    assert jobs == {"p/a": "completed", "p/b": "completed", "p/c": "queued"}
