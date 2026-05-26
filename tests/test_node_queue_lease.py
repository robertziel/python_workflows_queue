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


# ── requeue_job_for_retry (watchdog re-queue mechanic) ───────────────────────
# The watchdog-trip path re-queues a SINGLE running node-job for a retry on a
# fresh worker — same "running→queued + clear lease + bump priority to front"
# mechanic as reclaim_expired_leases, scoped by id, plus a watchdog_retries++
# counter (migration 0010). CAS-guarded + idempotent like the mark_* ops; writes
# NO dispatch event (the run stays running, only the node re-runs).


def _claimed_running_job(queue="cpu", priority=100):
    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue=queue, priority=priority,
        required_model=("qwen_edit" if queue == "gpu" else None),
    )
    if queue == "gpu":
        node_queue.claim_next_gpu_job(0, host="host-x")
    else:
        node_queue.claim_next_cpu_job(0, host="host-x")
    return run_id, job_id


def test_requeue_for_retry_flips_running_to_queued_and_clears_lease():
    run_id, job_id = _claimed_running_job()
    before = row(job_id)
    assert before["status"] == "running"
    assert before["claimed_by"] == "host-x"
    assert before["lease_expires_at"] is not None

    out = node_queue.requeue_job_for_retry(job_id)
    assert out is not None and out["id"] == job_id
    r = row(job_id)
    assert r["status"] == "queued"
    assert r["claimed_by"] is None
    assert r["lease_expires_at"] is None
    assert r["started_at"] is None


def test_requeue_for_retry_bumps_priority_to_front():
    # Matches reclaim_expired_leases' LEAST(priority, 10): a back-of-queue job
    # (priority 100) jumps to <= 10 so the retry runs promptly.
    run_id, job_id = _claimed_running_job(priority=100)
    node_queue.requeue_job_for_retry(job_id)
    assert row(job_id)["priority"] <= 10


def test_requeue_for_retry_does_not_lower_an_already_high_priority():
    # LEAST(priority, 10) never RAISES the number (lowers urgency): a job already
    # at priority 3 stays 3, not bumped up to 10.
    run_id, job_id = _claimed_running_job(priority=3)
    node_queue.requeue_job_for_retry(job_id)
    assert row(job_id)["priority"] == 3


def test_requeue_for_retry_increments_watchdog_retries():
    run_id, job_id = _claimed_running_job()
    assert row(job_id)["watchdog_retries"] == 0
    node_queue.requeue_job_for_retry(job_id)
    assert row(job_id)["watchdog_retries"] == 1
    # A second cycle (re-claim then re-queue) increments again.
    node_queue.claim_next_cpu_job(0, host="host-x")
    node_queue.requeue_job_for_retry(job_id)
    assert row(job_id)["watchdog_retries"] == 2


def test_requeue_for_retry_returns_none_when_not_running():
    # CAS guard: only a running row is re-queued. A queued row (never claimed) is
    # left untouched and returns None — idempotent like the mark_* transitions.
    run_id = make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    assert node_queue.requeue_job_for_retry(job_id) is None
    r = row(job_id)
    assert r["status"] == "queued"
    assert r["watchdog_retries"] == 0  # not incremented on no-match


def test_requeue_for_retry_idempotent_on_terminal_row():
    run_id, job_id = _claimed_running_job()
    with connection() as c, c.cursor() as cur:
        node_queue.mark_completed_in_txn(cur, job_id, context_delta={}, seconds=1.0)
    # Already completed → no-op, returns None, counter untouched.
    assert node_queue.requeue_job_for_retry(job_id) is None
    r = row(job_id)
    assert r["status"] == "completed"
    assert r["watchdog_retries"] == 0


def test_requeue_for_retry_writes_no_dispatch_event():
    # The run must stay running — only the node re-runs — so NO failed event.
    run_id, job_id = _claimed_running_job()
    node_queue.requeue_job_for_retry(job_id)
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM workflow_dispatch_events WHERE run_id=%s",
            (run_id,),
        )
        assert cur.fetchone()["n"] == 0


def test_requeue_for_retry_fires_node_job_ready_notify():
    # The running→queued flip fires the migration-0006 NOTIFY (carrying the
    # queue) so an idle worker re-claims it at once — same as reclaim.
    run_id, job_id = _claimed_running_job(queue="cpu")
    payloads = _listen_and_capture(lambda: node_queue.requeue_job_for_retry(job_id))
    assert "cpu" in payloads


# ── clear_worker_current_model (Part C — drop the busy ghost on hard-exit) ────
# A watchdog trip hard-exits via os._exit, skipping the worker's finally — so its
# worker_heartbeats row keeps advertising current_model. This helper nulls that
# busy signal + ages last_seen out of the 30 s gauge window so the dead worker
# stops inflating the "N/M GPU busy" gauge. Scoped by (host_label, queue) PK,
# idempotent, returns None on no-match.


def _heartbeat(host, queue, model):
    node_queue.upsert_worker_heartbeat(
        host_label=host, queue=queue, concurrency=1, current_model=model,
    )


def test_clear_current_model_nulls_the_busy_signal_and_ages_last_seen():
    _heartbeat("spark", "gpu", "qwen_edit")
    out = node_queue.clear_worker_current_model("spark", "gpu")
    assert out is not None and out["current_model"] is None
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT current_model, last_seen < now() - interval '30 seconds' AS stale "
            "FROM worker_heartbeats WHERE host_label='spark' AND queue='gpu'"
        )
        r = cur.fetchone()
    assert r["current_model"] is None
    assert r["stale"] is True, "last_seen aged past the gauge window"


def test_clear_current_model_keeps_last_seen_when_mark_stale_false():
    _heartbeat("spark", "gpu", "qwen_edit")
    node_queue.clear_worker_current_model("spark", "gpu", mark_stale=False)
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT current_model, last_seen > now() - interval '30 seconds' AS fresh "
            "FROM worker_heartbeats WHERE host_label='spark' AND queue='gpu'"
        )
        r = cur.fetchone()
    assert r["current_model"] is None
    assert r["fresh"] is True, "mark_stale=False leaves last_seen fresh"


def test_clear_current_model_noop_returns_none_when_row_absent():
    assert node_queue.clear_worker_current_model("ghost", "gpu") is None


def test_clear_current_model_only_touches_the_named_worker():
    _heartbeat("spark", "gpu", "qwen_edit")
    _heartbeat("spark2", "gpu", "wan_i2v")
    node_queue.clear_worker_current_model("spark", "gpu")
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT current_model FROM worker_heartbeats "
            "WHERE host_label='spark2' AND queue='gpu'"
        )
        assert cur.fetchone()["current_model"] == "wan_i2v"


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
