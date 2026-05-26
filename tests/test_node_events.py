"""workflow_node_events — durable per-node, per-attempt event history (0011).

Engine half of the node-event-history feature (audit Phase 0):
  * record_node_event_in_txn rides the caller's txn (outbox-atomicity) and
    rolls back with it — the terminal / requeue contract;
  * record_node_event is BEST-EFFORT — a bad write is swallowed (returns None,
    writes nothing) and never propagates, so an event-I/O blip can't fail the
    load-bearing claim / terminal / watchdog path;
  * ON DELETE CASCADE from workflow_runs purges a run's events;
  * prune_node_events drops rows older than the retention window.
"""

from __future__ import annotations

import pytest

from queue_workflows import node_queue
from queue_workflows.db import connection
from tests._helpers import make_run


def _events(run_id: str) -> list[dict]:
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT * FROM workflow_node_events WHERE run_id = %s ORDER BY id",
            (run_id,),
        )
        return list(cur.fetchall())


def _job(run_id: str, node_id: str = "n", queue: str = "gpu") -> str:
    return node_queue.enqueue_node_job(
        run_id=run_id, node_id=node_id, node_module="x", queue=queue,
    )


# ── in-txn writer (terminal / requeue path) ────────────────────────────────


def test_record_in_txn_inserts_full_row():
    run_id = make_run()
    job_id = _job(run_id)
    with connection() as conn, conn.cursor() as cur:
        eid = node_queue.record_node_event_in_txn(
            cur, run_id=run_id, node_id="n", job_id=job_id, attempt=2,
            event_type="gpu_health_trip", host_label="box-a2", queue="gpu",
            model="qwen_image_edit_multi_angles", elapsed_s=12.5,
            error="no GPU activity for 300s; RAM static",
            detail={"max_sm_pct": 0, "ram_anchor_mb": 81000, "ram_now_mb": 81002},
        )
    assert isinstance(eid, int)
    rows = _events(run_id)
    assert len(rows) == 1
    r = rows[0]
    assert r["event_type"] == "gpu_health_trip"
    assert r["attempt"] == 2
    assert r["host_label"] == "box-a2"
    assert r["model"].startswith("qwen")
    assert r["elapsed_s"] == 12.5
    assert r["detail"]["max_sm_pct"] == 0
    assert "static" in r["error"]


def test_record_in_txn_rolls_back_with_caller_txn():
    """Atomic with the state change — if the surrounding txn aborts the event is
    gone (mirrors the dispatch-event atomicity invariant)."""
    run_id = make_run()
    job_id = _job(run_id)
    with pytest.raises(RuntimeError):
        with connection() as conn, conn.cursor() as cur:
            node_queue.record_node_event_in_txn(
                cur, run_id=run_id, node_id="n", job_id=job_id,
                event_type="completed", elapsed_s=1.0,
            )
            raise RuntimeError("boom — abort the txn")
    assert _events(run_id) == []


def test_record_in_txn_rejects_unknown_type():
    run_id = make_run()
    with connection() as conn, conn.cursor() as cur:
        with pytest.raises(ValueError):
            node_queue.record_node_event_in_txn(
                cur, run_id=run_id, node_id="n", event_type="bogus",
            )


# ── best-effort writer (non-terminal sites) ────────────────────────────────


def test_record_best_effort_inserts():
    run_id = make_run()
    eid = node_queue.record_node_event(
        run_id=run_id, node_id="n", event_type="claimed",
        host_label="box-a2", queue="gpu", attempt=0,
    )
    assert isinstance(eid, int)
    assert len(_events(run_id)) == 1


def test_record_best_effort_swallows_bad_type():
    """Unknown type → ValueError inside → swallowed: returns None, writes
    nothing, never propagates to the load-bearing caller."""
    run_id = make_run()
    eid = node_queue.record_node_event(
        run_id=run_id, node_id="n", event_type="definitely_not_valid",
    )
    assert eid is None
    assert _events(run_id) == []


def test_record_best_effort_swallows_fk_violation():
    """An event for a non-existent run (FK violation) is swallowed too."""
    eid = node_queue.record_node_event(
        run_id="00000000-0000-0000-0000-000000000000", node_id="n",
        event_type="claimed",
    )
    assert eid is None


# ── cascade + retention ────────────────────────────────────────────────────


def test_events_cascade_on_run_delete():
    run_id = make_run()
    node_queue.record_node_event(run_id=run_id, node_id="n", event_type="claimed")
    assert len(_events(run_id)) == 1
    with connection() as c, c.cursor() as cur:
        cur.execute("DELETE FROM workflow_runs WHERE id = %s", (run_id,))
    assert _events(run_id) == []


def test_prune_node_events_drops_old_rows():
    run_id = make_run()
    node_queue.record_node_event(run_id=run_id, node_id="n", event_type="claimed")
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO workflow_node_events (run_id, node_id, event_type, created_at) "
            "VALUES (%s, 'n', 'completed', now() - make_interval(days => 40))",
            (run_id,),
        )
    assert len(_events(run_id)) == 2
    deleted = node_queue.prune_node_events(older_than_days=30)
    assert deleted == 1
    rows = _events(run_id)
    assert len(rows) == 1 and rows[0]["event_type"] == "claimed"


# ── emit-site integration (execute_node terminal paths) ────────────────────


def test_execute_node_emits_completed_event(monkeypatch):
    """A real execute_node run leaves a durable ``completed`` node event with a
    populated elapsed_s — the terminal emit is wired (best-effort, post-commit)."""
    from queue_workflows import dispatcher, node_executor

    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    job = node_queue.get_node_job(job_id)
    monkeypatch.setattr(
        node_executor, "_invoke", lambda **kw: {"context_delta": {"ok": True}},
    )
    monkeypatch.setattr(dispatcher, "on_node_completed", lambda *a, **k: 0)

    assert node_executor.execute_node(job) == "completed"
    types = [e["event_type"] for e in _events(run_id)]
    assert "completed" in types
    ce = next(e for e in _events(run_id) if e["event_type"] == "completed")
    assert ce["elapsed_s"] is not None


def test_execute_node_emits_failed_event(monkeypatch):
    """A node body that raises leaves a durable ``failed`` node event carrying
    the error text."""
    from queue_workflows import node_executor

    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    job = node_queue.get_node_job(job_id)

    def boom(**kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(node_executor, "_invoke", boom)

    assert node_executor.execute_node(job) == "failed"
    failed = [e for e in _events(run_id) if e["event_type"] == "failed"]
    assert len(failed) == 1
    assert "kaboom" in (failed[0]["error"] or "")


# ── retention sweep (NodePool, Phase 2) ────────────────────────────────────


def test_nodepool_retention_sweep_prunes_old_events():
    """NodePool._sweep_node_event_retention prunes rows past the window and is
    interval-gated (an immediate second call is a no-op)."""
    from queue_workflows import node_pool

    run_id = make_run()
    node_queue.record_node_event(run_id=run_id, node_id="n", event_type="claimed")
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO workflow_node_events (run_id, node_id, event_type, created_at) "
            "VALUES (%s, 'n', 'completed', now() - make_interval(days => 40))",
            (run_id,),
        )
    assert len(_events(run_id)) == 2

    pool = node_pool.NodePool(register_builtins=None)
    pool._sweep_node_event_retention()  # last_run=0 → runs; prunes the 40-day row
    rows = _events(run_id)
    assert len(rows) == 1 and rows[0]["event_type"] == "claimed"

    # Interval-gated: a second immediate call is a no-op (and must not error).
    pool._sweep_node_event_retention()
    assert len(_events(run_id)) == 1
