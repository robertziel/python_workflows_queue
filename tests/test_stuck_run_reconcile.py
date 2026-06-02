"""Stuck-run reconciler — ``dispatcher.reconcile_run`` + the ``NodePool``
periodic sweep that drives it.

The gap this closes: the run-state machine only advances on a node reaching
``completed`` / ``skipped`` (enqueue downstream / finish the run) or ``failed``
(fail the run). A ``cancelled`` node is an unhandled dead-end — it satisfies
neither ``_find_ready_nodes`` nor the all-terminal completion check, and there
is no ``on_node_cancelled``. So once a node is ``cancelled`` while its run is
non-terminal — and ``run_store.reenqueue_running_for_resume`` then blindly
re-queues the run on the next orchestrator restart — the run sits ``queued``
forever with NO live node-job: nothing for a worker to claim, never completing,
never failing. Observed live (runs 3e0e0f61 / bf008700 / 905fe457).

Covered:
- the decision tree of ``reconcile_run`` (pure, monkeypatched): noop when
  terminal / when a live job exists; complete when every node is terminal;
  enqueue a dropped fan-out WITHOUT deleting; requeue a run wedged behind a
  cancelled node by dropping the dead rows + re-expanding; fail a genuinely
  dead-ended run so the status stops lying;
- the NodePool sweep: it finds non-terminal runs with no live node-job and
  reconciles them end-to-end (real DB), skips runs that DO have a live job, is
  interval-gated, fires on the first tick (instant recovery after start), and
  runs inside ``_tick``.
"""

from __future__ import annotations

import time

import pytest

from queue_workflows import dispatcher, node_pool, node_queue, run_store
from queue_workflows.db import connection
from tests._helpers import make_run


# ── reconcile_run decision tree (pure — internals monkeypatched) ───────────


def _wire(monkeypatch, *, run, existing, nodes, process_ready, deleted=()):
    """Stub out every DB-touching helper ``reconcile_run`` calls so the
    decision tree runs in isolation. Returns a capture dict of the mutating
    calls (``update_run`` field-dicts + ``delete_non_terminal`` invocations)."""
    cap: dict = {"updates": [], "deletes": []}
    monkeypatch.setattr(dispatcher.run_store, "get_run", lambda rid: run)
    monkeypatch.setattr(dispatcher, "_jobs_by_node_id", lambda rid: dict(existing))
    monkeypatch.setattr(dispatcher, "_load_workflow", lambda name: {"steps": []})
    monkeypatch.setattr(
        dispatcher, "_nodes_of", lambda wf, run=None: [{"id": n} for n in nodes]
    )
    seq = list(process_ready)
    monkeypatch.setattr(
        dispatcher, "_process_ready",
        lambda rid, wf, r: seq.pop(0) if seq else 0,
    )

    def _del(rid):
        cap["deletes"].append(rid)
        return list(deleted)

    monkeypatch.setattr(
        dispatcher.node_queue, "delete_non_terminal_jobs_for_run", _del
    )
    monkeypatch.setattr(
        dispatcher.run_store, "update_run",
        lambda rid, **f: cap["updates"].append(f) or {**run, **f},
    )
    return cap


def test_reconcile_noop_when_run_already_terminal(monkeypatch):
    cap = _wire(
        monkeypatch,
        run={"id": "r", "status": "cancelled", "workflow_name": "w"},
        existing={}, nodes=["a"], process_ready=[],
    )
    assert dispatcher.reconcile_run("r") == "noop"
    assert cap["updates"] == [] and cap["deletes"] == []


def test_reconcile_noop_when_a_live_job_exists(monkeypatch):
    cap = _wire(
        monkeypatch,
        run={"id": "r", "status": "running", "workflow_name": "w"},
        existing={"a": {"status": "queued"}}, nodes=["a"], process_ready=[],
    )
    assert dispatcher.reconcile_run("r") == "noop"
    assert cap["updates"] == []


def test_reconcile_completes_when_all_nodes_terminal(monkeypatch):
    cap = _wire(
        monkeypatch,
        run={"id": "r", "status": "running", "workflow_name": "w"},
        existing={"a": {"status": "completed"}, "b": {"status": "skipped"}},
        nodes=["a", "b"], process_ready=[],
    )
    assert dispatcher.reconcile_run("r") == "completed"
    assert cap["updates"][-1]["status"] == "completed"
    assert cap["deletes"] == []  # nothing dropped


def test_reconcile_enqueues_dropped_fanout_without_deleting(monkeypatch):
    # 'b' has no row yet and 'a' is completed → _process_ready finds it ready.
    cap = _wire(
        monkeypatch,
        run={"id": "r", "status": "running", "workflow_name": "w"},
        existing={"a": {"status": "completed"}},
        nodes=["a", "b"], process_ready=[1],
    )
    assert dispatcher.reconcile_run("r") == "enqueued"
    assert cap["updates"][-1]["status"] == "running"
    assert cap["deletes"] == []  # recovered non-destructively


def test_reconcile_requeues_run_wedged_behind_cancelled_node(monkeypatch):
    # 'gen' is cancelled (blocks): first _process_ready=0; after the dead row is
    # dropped, the re-expand enqueues it → second _process_ready=1.
    cap = _wire(
        monkeypatch,
        run={"id": "r", "status": "running", "workflow_name": "w"},
        existing={"a": {"status": "completed"}, "gen": {"status": "cancelled"}},
        nodes=["a", "gen"], process_ready=[0, 1], deleted=["gen"],
    )
    assert dispatcher.reconcile_run("r") == "requeued"
    assert cap["deletes"] == ["r"]  # dropped the dead rows
    assert cap["updates"][-1]["status"] == "running"


def test_reconcile_fails_when_nothing_runnable(monkeypatch):
    # Wedged, but dropping the dead rows yields no re-expandable node → honest
    # terminal status instead of a forever-queued lie.
    cap = _wire(
        monkeypatch,
        run={"id": "r", "status": "running", "workflow_name": "w"},
        existing={"a": {"status": "completed"}, "gen": {"status": "cancelled"}},
        nodes=["a", "gen"], process_ready=[0], deleted=[],
    )
    assert dispatcher.reconcile_run("r") == "failed"
    assert cap["updates"][-1]["status"] == "failed"
    assert "error" in cap["updates"][-1]


# ── DB-backed end-to-end (real reconcile + real sweep) ─────────────────────


def _set_job_status(job_id: str, status: str) -> None:
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs SET status=%s, finished_at=now() "
            "WHERE id=%s",
            (status, job_id),
        )


def _wedged_run(monkeypatch) -> str:
    """A real run wedged behind a cancelled terminal node: 'a' completed, 'gen'
    cancelled, run still 'running', no live node-job. Input-only workflow so no
    pipeline-schema expansion is needed."""
    import queue_workflows
    wf = {
        "name": "_stuck_wf", "mode": "node",
        "steps": [
            {"id": "a", "kind": "input", "widget": "confirm"},
            {"id": "gen", "kind": "input", "widget": "confirm",
             "depends_on": ["a"]},
        ],
    }
    queue_workflows.set_workflow_provider(lambda name: wf, lambda name: {})
    run_id = make_run(status="running", workflow_name="_stuck_wf")
    ja = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="__input__confirm", queue="cpu",
    )
    jg = node_queue.enqueue_node_job(
        run_id=run_id, node_id="gen", node_module="__input__confirm", queue="cpu",
    )
    _set_job_status(ja, "completed")
    _set_job_status(jg, "cancelled")
    return run_id


def test_reconcile_puts_wedged_node_back_on_the_queue(monkeypatch):
    run_id = _wedged_run(monkeypatch)

    assert dispatcher.reconcile_run(run_id) == "requeued"

    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    assert jobs["a"]["status"] == "completed"   # completed prefix preserved
    assert jobs["gen"]["status"] == "queued"    # the wedge is back in the queue
    assert run_store.get_run(run_id)["status"] == "running"


def test_sweep_requeues_wedged_run_end_to_end(monkeypatch):
    run_id = _wedged_run(monkeypatch)

    pool = node_pool.NodePool(cpu_workers=0, gpu_workers=0, register_builtins=None)
    pool._stuck_run_interval_s = 0.0
    pool._sweep_stuck_runs()

    jobs = {j["node_id"]: j for j in node_queue.list_jobs_for_run(run_id)}
    assert jobs["gen"]["status"] == "queued"
    assert run_store.get_run(run_id)["status"] == "running"


def test_reconcile_finalises_a_run_whose_terminal_event_was_lost(monkeypatch):
    import queue_workflows
    wf = {
        "name": "_done_wf", "mode": "node",
        "steps": [{"id": "a", "kind": "input", "widget": "confirm"}],
    }
    queue_workflows.set_workflow_provider(lambda name: wf, lambda name: {})
    run_id = make_run(status="running", workflow_name="_done_wf")
    ja = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="__input__confirm", queue="cpu",
    )
    _set_job_status(ja, "completed")

    assert dispatcher.reconcile_run(run_id) == "completed"
    assert run_store.get_run(run_id)["status"] == "completed"


# ── NodePool sweep wiring ──────────────────────────────────────────────────


def _pool() -> node_pool.NodePool:
    pool = node_pool.NodePool(cpu_workers=0, gpu_workers=0, register_builtins=None)
    pool._stuck_run_interval_s = 0.0  # never suppress in a tight test loop
    return pool


def test_sweep_skips_run_that_has_a_live_node_job(monkeypatch):
    run_id = make_run(status="running", workflow_name="_w")
    node_queue.enqueue_node_job(  # a live (queued) job → not stuck
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    seen: list[str] = []
    monkeypatch.setattr(
        dispatcher, "reconcile_run", lambda rid: seen.append(rid) or "noop"
    )
    _pool()._sweep_stuck_runs()
    assert run_id not in seen


def test_sweep_selects_only_non_terminal_runs(monkeypatch):
    stuck = make_run(status="running", workflow_name="_w")      # no jobs → stuck
    done = make_run(status="completed", workflow_name="_w")     # terminal → skip
    seen: list[str] = []
    monkeypatch.setattr(
        dispatcher, "reconcile_run", lambda rid: seen.append(rid) or "noop"
    )
    _pool()._sweep_stuck_runs()
    assert stuck in seen and done not in seen


def test_sweep_is_interval_gated(monkeypatch):
    make_run(status="running", workflow_name="_w")  # one stuck run
    calls: list[str] = []
    monkeypatch.setattr(
        dispatcher, "reconcile_run", lambda rid: calls.append(rid) or "noop"
    )
    pool = _pool()
    pool._stuck_run_interval_s = 5.0
    pool._sweep_stuck_runs()
    pool._sweep_stuck_runs()
    pool._sweep_stuck_runs()
    assert len(calls) == 1  # only the first pass cleared the gate


def test_sweep_fires_on_first_tick_after_start(monkeypatch):
    """``last_run`` starts at 0, so a fresh instance reconciles immediately —
    'run instantly after instance start'."""
    make_run(status="running", workflow_name="_w")
    calls: list[str] = []
    monkeypatch.setattr(
        dispatcher, "reconcile_run", lambda rid: calls.append(rid) or "noop"
    )
    pool = node_pool.NodePool(cpu_workers=0, gpu_workers=0, register_builtins=None)
    assert pool._stuck_run_last_run == 0.0  # default interval (300s), unfired
    pool._sweep_stuck_runs()
    assert len(calls) == 1  # fired despite the 5-min interval, because last_run=0


def test_tick_runs_the_stuck_run_sweep(monkeypatch):
    fired: list[str] = []
    monkeypatch.setattr(node_pool.NodePool, "_drain_dispatch_events", lambda self: None)
    monkeypatch.setattr(node_pool.NodePool, "_sweep_expired_leases", lambda self: None)
    monkeypatch.setattr(node_pool.NodePool, "_sweep_expired_ingest_leases", lambda self: None)
    monkeypatch.setattr(node_pool.NodePool, "_sweep_dead_workers", lambda self: None)
    monkeypatch.setattr(
        node_pool.NodePool, "_sweep_stuck_runs", lambda self: fired.append("stuck")
    )
    pool = node_pool.NodePool(cpu_workers=0, gpu_workers=0, register_builtins=None)
    pool._tick()
    assert fired == ["stuck"]
