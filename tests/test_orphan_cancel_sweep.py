"""Orphan-cancel sweep — flips ``queued`` jobs whose parent run is already
``cancelled`` or ``failed`` to ``cancelled``.

The contract this closes: the host's cancel handler typically just flips
``workflow_runs.status='cancelled'`` and stops there (Rails' ``DELETE
/workflow/:id`` does exactly that). The claim SQL refuses such jobs (run-cancel
guard, ``test_invariant_cancel``), so they're never executed — but they linger
in ``queued`` forever, polluting the queue gauges and confusing operators into
thinking workers are stuck.

This sweep is opt-in via ``configure(cancel_orphan_queued_jobs=True)``. Default
is ``False`` so the engine's existing behaviour is preserved byte-for-byte;
hosts that want the cleanup ship the flag in their startup wiring.

Covered:
- ``node_queue.cancel_orphaned_queued_jobs()``
  * flips queued jobs of a cancelled run to cancelled
  * flips queued jobs of a failed run to cancelled
  * leaves jobs of an active (running/queued/awaiting_input) run alone
  * leaves non-queued jobs (running/completed/failed/cancelled) alone
  * is idempotent (a second call after the first returns 0)
  * stamps ``finished_at`` on the rows it flips
- ``configure(cancel_orphan_queued_jobs=...)``
  * defaults to False
  * round-trips True
- ``NodePool._tick`` wiring
  * flag ON ⇒ orphan-cancel sweep runs and flips orphan rows
  * flag OFF (default) ⇒ no flip happens
  * interval-gated (a second tick within the window is a no-op)
"""

from __future__ import annotations

import pytest

import queue_workflows
from queue_workflows import node_pool, node_queue, run_store
from queue_workflows.db import connection
from tests._cancel_helper import cancel_run_via_rails
from tests._helpers import make_run


# ── helpers ──────────────────────────────────────────────────────────────────


def _orphan_pool() -> node_pool.NodePool:
    """A NodePool wired with the orphan-cancel sweep interval forced to 0 so
    a tight test loop isn't suppressed by the gate."""
    pool = node_pool.NodePool(cpu_workers=0, gpu_workers=0, register_builtins=None)
    pool._orphan_cancel_interval_s = 0.0
    return pool


def _job_status(job_id: str) -> str:
    return node_queue.get_node_job(job_id)["status"]


def _job_finished_at(job_id: str):
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT finished_at FROM workflow_node_jobs WHERE id=%s", (job_id,)
        )
        r = cur.fetchone()
        return None if r is None else r["finished_at"]


# ── node_queue.cancel_orphaned_queued_jobs() ────────────────────────────────


def test_cancel_orphaned_queued_jobs_flips_jobs_of_cancelled_run():
    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
    )
    cancel_run_via_rails(run_id)

    n = node_queue.cancel_orphaned_queued_jobs()

    assert n == 1
    assert _job_status(job_id) == "cancelled"
    assert _job_finished_at(job_id) is not None


def test_cancel_orphaned_queued_jobs_flips_jobs_of_failed_run():
    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    run_store.update_run(run_id, status="failed", error="upstream blew up")

    n = node_queue.cancel_orphaned_queued_jobs()

    assert n == 1
    assert _job_status(job_id) == "cancelled"


def test_cancel_orphaned_queued_jobs_leaves_active_runs_alone():
    """A ``queued`` job belonging to a still-active run must NOT be cancelled —
    that's the entire point of the queue."""
    run_id = make_run(status="running")
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    n = node_queue.cancel_orphaned_queued_jobs()

    assert n == 0
    assert _job_status(job_id) == "queued"


def test_cancel_orphaned_queued_jobs_ignores_non_queued_jobs():
    """Only ``status='queued'`` rows are touched: a ``running`` row holding the
    work in flight, or a terminal row, must be left alone. (The claim SQL plus
    the cancel-watcher together handle the ``running`` case cooperatively; we
    must NOT race them here.)"""
    run_id = make_run()
    j_queued = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="x", queue="cpu",
    )
    j_running = node_queue.enqueue_node_job(
        run_id=run_id, node_id="b", node_module="x", queue="cpu",
    )
    j_completed = node_queue.enqueue_node_job(
        run_id=run_id, node_id="c", node_module="x", queue="cpu",
    )
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs SET status='running', started_at=now() "
            "WHERE id=%s", (j_running,),
        )
        cur.execute(
            "UPDATE workflow_node_jobs SET status='completed', finished_at=now() "
            "WHERE id=%s", (j_completed,),
        )

    cancel_run_via_rails(run_id)
    n = node_queue.cancel_orphaned_queued_jobs()

    assert n == 1  # only the queued one
    assert _job_status(j_queued) == "cancelled"
    assert _job_status(j_running) == "running"
    assert _job_status(j_completed) == "completed"


def test_cancel_orphaned_queued_jobs_is_idempotent():
    run_id = make_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    cancel_run_via_rails(run_id)

    first = node_queue.cancel_orphaned_queued_jobs()
    second = node_queue.cancel_orphaned_queued_jobs()

    assert first == 1
    assert second == 0


def test_cancel_orphaned_queued_jobs_sweeps_across_multiple_runs():
    """One sweep call must catch orphans across every cancelled/failed run, not
    just the most recent one — that's the operator-facing fix for legacy zombie
    rows that accumulated over weeks."""
    rid_a = make_run()
    rid_b = make_run()
    rid_c = make_run(status="running")  # NOT orphaned
    ja = node_queue.enqueue_node_job(run_id=rid_a, node_id="x", node_module="m", queue="gpu")
    jb = node_queue.enqueue_node_job(run_id=rid_b, node_id="x", node_module="m", queue="gpu")
    jc = node_queue.enqueue_node_job(run_id=rid_c, node_id="x", node_module="m", queue="gpu")
    cancel_run_via_rails(rid_a)
    run_store.update_run(rid_b, status="failed", error="kaboom")

    n = node_queue.cancel_orphaned_queued_jobs()
    assert n == 2
    assert _job_status(ja) == "cancelled"
    assert _job_status(jb) == "cancelled"
    assert _job_status(jc) == "queued"


# ── configure(cancel_orphan_queued_jobs=...) ─────────────────────────────────


def test_cancel_orphan_queued_jobs_default_is_false():
    cfg = queue_workflows.configure()
    assert cfg.cancel_orphan_queued_jobs is False


def test_configure_cancel_orphan_queued_jobs_roundtrips_true():
    queue_workflows.configure(cancel_orphan_queued_jobs=True)
    assert queue_workflows.get_config().cancel_orphan_queued_jobs is True


# ── NodePool sweep wiring ───────────────────────────────────────────────────


def test_node_pool_tick_sweeps_orphans_when_flag_enabled():
    """With the flag on, NodePool's per-tick sweep flips queued jobs of
    cancelled/failed runs to cancelled — closing the audit gap."""
    queue_workflows.configure(cancel_orphan_queued_jobs=True)
    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
    )
    cancel_run_via_rails(run_id)

    pool = _orphan_pool()
    pool._sweep_orphan_queued_jobs()

    assert _job_status(job_id) == "cancelled"


def test_node_pool_tick_does_not_sweep_when_flag_disabled():
    """Default: flag OFF ⇒ the sweep is a no-op. Existing engine behaviour is
    preserved byte-for-byte (orphans remain queued; the claim SQL still skips
    them via the run-cancel guard)."""
    # Explicit reaffirm of the default — relying on conftest reset is enough
    # but stating it makes the contract obvious to a future reader.
    assert queue_workflows.get_config().cancel_orphan_queued_jobs is False

    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
    )
    cancel_run_via_rails(run_id)

    pool = _orphan_pool()
    pool._sweep_orphan_queued_jobs()

    assert _job_status(job_id) == "queued"


def test_node_pool_orphan_sweep_is_interval_gated():
    """A second sweep within the configured interval must be a no-op even if a
    new orphan landed — protects the 0.5 s dispatch tick from doing the join
    UPDATE every iteration."""
    queue_workflows.configure(cancel_orphan_queued_jobs=True)
    pool = node_pool.NodePool(cpu_workers=0, gpu_workers=0, register_builtins=None)
    pool._orphan_cancel_interval_s = 60.0  # well above the test window

    # First sweep: no orphans, but it stamps ``last_run`` so the gate clamps
    # subsequent calls.
    pool._sweep_orphan_queued_jobs()

    # Now introduce an orphan AFTER the gate has been primed.
    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    cancel_run_via_rails(run_id)

    pool._sweep_orphan_queued_jobs()  # gated — must not fire
    assert _job_status(job_id) == "queued"


def test_node_pool_tick_runs_orphan_sweep():
    """Smoke: _tick() reaches the orphan sweep step. We don't need to drive a
    full DAG here — wire a stub provider so start_run is a no-op, then assert
    the sweep ran."""
    queue_workflows.configure(cancel_orphan_queued_jobs=True)

    # Empty workflow + pipeline so dispatcher.start_run is a no-op.
    queue_workflows.set_workflow_provider(
        lambda name: {"name": name, "steps": []},
        lambda name: {"name": name, "nodes": []},
    )

    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    cancel_run_via_rails(run_id)

    pool = node_pool.NodePool(cpu_workers=0, gpu_workers=0, register_builtins=None)
    pool._orphan_cancel_interval_s = 0.0  # disable gate

    pool._tick()

    assert _job_status(job_id) == "cancelled"
