"""Lease + wake primitives for the Postgres-as-queue claim path.

Exercises the ``claim_next_*`` path + ``reclaim_expired_leases`` + the
``node_job_ready`` NOTIFY trigger (migration 0006).

Covered:
- claim stamps ``claimed_by`` + a future ``lease_expires_at``;
- the run-cancel guard keeps a queued job whose run is cancelled/failed from
  being claimed;
- warm-model affinity: a warm worker claims the matching-model row over an
  older cold row; a NULL-model (cold) worker claims by priority/FIFO;
- the ``host_priority`` tiebreaker orders deterministically per host;
- ``reclaim_expired_leases`` re-queues an expired-lease ``running`` row and
  leaves a fresh-lease row alone;
- the NOTIFY trigger fires ``node_job_ready`` (carrying the queue) on a queued
  INSERT and on a reclaim UPDATE.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import psycopg

from queue_workflows import db, node_queue
from queue_workflows.db import connection
from tests._helpers import force_lease, make_run, row


# ── Lease stamping ─────────────────────────────────────────────────────────


def test_claim_cpu_stamps_claimed_by_and_future_lease():
    run_id = make_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="x", queue="cpu",
    )
    before = datetime.now(timezone.utc)
    claimed = node_queue.claim_next_cpu_job(0, host="beelink", lease_s=600)
    assert claimed is not None
    assert claimed["status"] == "running"
    assert claimed["claimed_by"] == "beelink"
    assert claimed["lease_expires_at"] is not None
    assert claimed["lease_expires_at"] > before


def test_claim_gpu_stamps_claimed_by_and_future_lease():
    run_id = make_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="x", queue="gpu",
        required_model="sdxl",
    )
    claimed = node_queue.claim_next_gpu_job(
        0, current_model="sdxl", host="spark", lease_s=900,
    )
    assert claimed is not None
    assert claimed["claimed_by"] == "spark"
    delta = claimed["lease_expires_at"] - datetime.now(timezone.utc)
    assert timedelta(seconds=600) < delta < timedelta(seconds=1200)


# ── Run-cancel guard folded into the claim ──────────────────────────────────


def test_claim_skips_job_whose_run_is_cancelled():
    run_id = make_run(status="cancelled")
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="x", queue="cpu",
    )
    assert node_queue.claim_next_cpu_job(0, host="h") is None


def test_claim_skips_job_whose_run_is_failed():
    run_id = make_run(status="failed")
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="x", queue="gpu",
        required_model="sdxl",
    )
    assert node_queue.claim_next_gpu_job(0, current_model="sdxl", host="h") is None


def test_claim_takes_active_run_skips_cancelled_run():
    dead = make_run(status="cancelled")
    live = make_run(status="running")
    node_queue.enqueue_node_job(
        run_id=dead, node_id="dead", node_module="x", queue="cpu", priority=1,
    )
    live_job = node_queue.enqueue_node_job(
        run_id=live, node_id="live", node_module="x", queue="cpu", priority=100,
    )
    claimed = node_queue.claim_next_cpu_job(0, host="h")
    assert claimed is not None
    assert claimed["id"] == live_job


# ── Warm-model affinity ──────────────────────────────────────────────────────


def test_warm_gpu_worker_claims_matching_model_first():
    run_id = make_run()
    node_queue.enqueue_node_job(run_id=run_id, node_id="a", node_module="x",
                                queue="gpu", required_model="sdxl")
    flux = node_queue.enqueue_node_job(run_id=run_id, node_id="b", node_module="x",
                                       queue="gpu", required_model="flux")
    claimed = node_queue.claim_next_gpu_job(0, current_model="flux", host="spark")
    assert claimed["id"] == flux


def test_cold_gpu_worker_claims_by_priority_then_fifo():
    run_id = make_run()
    early = node_queue.enqueue_node_job(run_id=run_id, node_id="a", node_module="x",
                                        queue="gpu", required_model="sdxl",
                                        priority=100)
    node_queue.enqueue_node_job(run_id=run_id, node_id="b", node_module="x",
                                queue="gpu", required_model="flux", priority=100)
    claimed = node_queue.claim_next_gpu_job(0, current_model=None, host="beelink")
    assert claimed["id"] == early


def test_warm_affinity_respects_null_required_model_with_null_current():
    run_id = make_run()
    typed = node_queue.enqueue_node_job(run_id=run_id, node_id="a", node_module="x",
                                        queue="gpu", required_model="sdxl",
                                        priority=100)
    untyped = node_queue.enqueue_node_job(run_id=run_id, node_id="b", node_module="x",
                                          queue="gpu", priority=100)
    claimed = node_queue.claim_next_gpu_job(0, current_model=None, host="spark")
    assert claimed["id"] == untyped
    assert typed


# ── host_priority tiebreaker ────────────────────────────────────────────────


def test_host_priority_high_takes_head_low_takes_tail():
    run_id = make_run()
    first = node_queue.enqueue_node_job(run_id=run_id, node_id="a", node_module="x",
                                        queue="gpu", required_model="sdxl",
                                        priority=100)
    second = node_queue.enqueue_node_job(run_id=run_id, node_id="b", node_module="x",
                                         queue="gpu", required_model="sdxl",
                                         priority=100)
    low = node_queue.claim_next_gpu_job(
        0, current_model="sdxl", host="beelink", host_priority=-1,
    )
    assert low["id"] == second
    high = node_queue.claim_next_gpu_job(
        0, current_model="sdxl", host="spark", host_priority=10,
    )
    assert high["id"] == first


def test_host_priority_default_is_head_first_fifo():
    run_id = make_run()
    first = node_queue.enqueue_node_job(run_id=run_id, node_id="a", node_module="x",
                                        queue="gpu", required_model="sdxl",
                                        priority=100)
    node_queue.enqueue_node_job(run_id=run_id, node_id="b", node_module="x",
                                queue="gpu", required_model="sdxl", priority=100)
    claimed = node_queue.claim_next_gpu_job(0, current_model="sdxl", host="h")
    assert claimed["id"] == first


def test_host_priority_applies_to_cpu_too():
    run_id = make_run()
    first = node_queue.enqueue_node_job(run_id=run_id, node_id="a", node_module="x",
                                        queue="cpu", priority=100)
    second = node_queue.enqueue_node_job(run_id=run_id, node_id="b", node_module="x",
                                         queue="cpu", priority=100)
    low = node_queue.claim_next_cpu_job(0, host="beelink", host_priority=-1)
    assert low["id"] == second
    high = node_queue.claim_next_cpu_job(0, host="spark", host_priority=10)
    assert high["id"] == first


# ── reclaim_expired_leases ───────────────────────────────────────────────────


def test_reclaim_requeues_expired_lease_row():
    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="x", queue="cpu", priority=100,
    )
    force_lease(job_id, expires_in_s=-30)
    reclaimed = node_queue.reclaim_expired_leases()
    ids = [r["id"] for r in reclaimed]
    assert job_id in ids
    r = row(job_id)
    assert r["status"] == "queued"
    assert r["claimed_by"] is None
    assert r["lease_expires_at"] is None
    assert r["started_at"] is None
    assert r["priority"] <= 10


def test_reclaim_returns_run_and_node_ids():
    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="the-node", node_module="x", queue="gpu",
        required_model="sdxl",
    )
    force_lease(job_id, expires_in_s=-5)
    reclaimed = node_queue.reclaim_expired_leases()
    match = next(r for r in reclaimed if r["id"] == job_id)
    assert match["run_id"] == run_id
    assert match["node_id"] == "the-node"


def test_reclaim_leaves_fresh_lease_untouched():
    run_id = make_run()
    fresh = node_queue.enqueue_node_job(
        run_id=run_id, node_id="fresh", node_module="x", queue="cpu",
    )
    expired = node_queue.enqueue_node_job(
        run_id=run_id, node_id="expired", node_module="x", queue="cpu",
    )
    force_lease(fresh, expires_in_s=600)
    force_lease(expired, expires_in_s=-1)
    reclaimed = node_queue.reclaim_expired_leases()
    ids = [r["id"] for r in reclaimed]
    assert expired in ids
    assert fresh not in ids
    assert row(fresh)["status"] == "running"
    assert row(fresh)["claimed_by"] == "host-x"


def test_reclaim_ignores_running_rows_without_a_lease():
    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="legacy", node_module="x", queue="cpu",
    )
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs SET status='running', started_at=now() "
            "WHERE id=%s",
            (job_id,),
        )
    reclaimed = node_queue.reclaim_expired_leases()
    assert job_id not in [r["id"] for r in reclaimed]
    assert row(job_id)["status"] == "running"


# ── NOTIFY trigger ───────────────────────────────────────────────────────────


def _listen_and_capture(action, *, channel="node_job_ready", timeout=3.0):
    payloads: list[str] = []
    with psycopg.connect(db.db_url(), autocommit=True) as listen_conn:
        listen_conn.execute(f"LISTEN {channel}")
        listen_conn.execute("SELECT 1").fetchone()
        action()
        for n in listen_conn.notifies(timeout=timeout, stop_after=1):
            payloads.append(n.payload)
    return payloads


def test_notify_fires_on_queued_insert_with_queue_payload():
    run_id = make_run()

    def _insert():
        node_queue.enqueue_node_job(
            run_id=run_id, node_id="n", node_module="x", queue="gpu",
            required_model="sdxl",
        )

    payloads = _listen_and_capture(_insert)
    assert payloads == ["gpu"]


def test_notify_fires_on_reclaim_requeue():
    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="x", queue="cpu",
    )
    force_lease(job_id, expires_in_s=-10)

    payloads = _listen_and_capture(node_queue.reclaim_expired_leases)
    assert "cpu" in payloads


def test_notify_does_not_fire_on_terminal_transition():
    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="a", node_module="x", queue="cpu",
    )
    node_queue.claim_next_cpu_job(0, host="h")

    def _complete():
        node_queue.mark_completed(job_id, context_delta={}, seconds=0.1)

    payloads = _listen_and_capture(_complete, timeout=1.0)
    assert payloads == []
