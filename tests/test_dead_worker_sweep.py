"""Dead-worker detector — ``node_queue.flag_stale_workers_holding_running_jobs``
wired into :class:`NodePool` as a periodic sweep (``_sweep_dead_workers``).

The gap this closes: a GPU hardware-hang can wedge the claim-worker PROCESS (a
torch/HIP call blocked in a dead GPU context) so its in-process watchdog can't
act and its ``worker_heartbeats.last_seen`` FREEZES — it silently stops claiming
overflow work — even after the lease-reclaim has re-queued the JOB onto a
healthy host. The orchestrator is a SEPARATE process (GIL-independent of the
wedged worker), so it can observe the frozen heartbeat and FLAG the dead worker
for a host-supervisor to bounce.

Covered:
- a stale heartbeat that still owns a ``running`` job is flagged
  (``last_flagged_dead_at`` stamped) + returned with its running-job count;
- a FRESH heartbeat (even holding a running job) is NOT flagged;
- a stale heartbeat with NO running job is NOT flagged (a worker that died idle
  needs no recovery — there's no stranded work);
- the flag is idempotent: a second sweep within the window does not re-flag, and
  a heartbeat refresh CLEARS the flag so a future hang re-flags cleanly;
- the staleness threshold is honoured (env + arg) — claimed_by joins on the
  worker's host_label, matching the live claim;
- the sweep logs an actionable DEAD WORKER line + is interval-gated + runs in
  ``_tick``; it does NOT touch the job row (the lease-reclaim owns re-queuing).
"""

from __future__ import annotations

import logging
import time

import pytest

from queue_workflows import node_pool, node_queue
from queue_workflows.db import connection
from tests._helpers import make_run


# ── fixtures / helpers ───────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _wipe_heartbeats():
    with connection() as c, c.cursor() as cur:
        cur.execute("DELETE FROM worker_heartbeats")
    yield
    with connection() as c, c.cursor() as cur:
        cur.execute("DELETE FROM worker_heartbeats")


def _dead_worker_pool() -> node_pool.NodePool:
    """A NodePool whose dead-worker sweep we drive directly (no ``start()``),
    interval forced to 0 so the gate never suppresses in a tight loop."""
    pool = node_pool.NodePool(cpu_workers=0, gpu_workers=0, register_builtins=None)
    pool._dead_worker_interval_s = 0.0
    return pool


def _put_heartbeat(host: str, queue: str, *, last_seen_age_s: float) -> None:
    """Insert a ``worker_heartbeats`` row whose ``last_seen`` is
    ``last_seen_age_s`` seconds in the PAST (so the staleness test is
    deterministic without sleeping)."""
    with connection() as c, c.cursor() as cur:
        cur.execute(
            """
            INSERT INTO worker_heartbeats
                (host_label, queue, concurrency, last_seen, known_models)
            VALUES (%s, %s, 1, now() - make_interval(secs => %s), '{}')
            ON CONFLICT (host_label, queue, project) DO UPDATE
                SET last_seen = EXCLUDED.last_seen
            """,
            (host, queue, float(last_seen_age_s)),
        )


def _running_job_owned_by(host: str, *, queue: str = "gpu", model: str | None = None) -> str:
    """A ``running`` ``workflow_node_jobs`` row claimed_by ``host`` (the live
    claim stamps ``claimed_by`` with the worker's host label)."""
    run_id = make_run(workflow_name="_dead_worker_test")
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue=queue,
        required_model=model,
    )
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs "
            "SET status='running', started_at=now(), claimed_by=%s, "
            "    lease_expires_at = now() + interval '600 seconds' "
            "WHERE id=%s",
            (host, job_id),
        )
    return job_id


def _flagged_at(host: str, queue: str):
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT last_flagged_dead_at FROM worker_heartbeats "
            "WHERE host_label=%s AND queue=%s",
            (host, queue),
        )
        r = cur.fetchone()
        return None if r is None else r["last_flagged_dead_at"]


# ── core detection ───────────────────────────────────────────────────────────


def test_flags_stale_worker_holding_running_job():
    """The incident shape: a wan_i2v render wedged the GPU, the worker stopped
    heartbeating (stale ~29 min) but still owned its running job. It must be
    flagged + returned with the running-job count."""
    _running_job_owned_by("host-a", model="wan_i2v")
    _put_heartbeat("host-a", "gpu", last_seen_age_s=120)  # 29-min-style staleness

    flagged = node_queue.flag_stale_workers_holding_running_jobs(stale_after_s=30)

    assert len(flagged) == 1
    row = flagged[0]
    assert row["host_label"] == "host-a"
    assert row["queue"] == "gpu"
    assert row["running_jobs"] == 1
    # The durable, queryable marker is stamped.
    assert _flagged_at("host-a", "gpu") is not None


def test_fresh_heartbeat_holding_running_job_is_not_flagged():
    """A worker that is BEATING normally (last_seen recent) is healthy even
    while it owns a running job — that's the steady state of a long render. Must
    NOT be flagged."""
    _running_job_owned_by("host-a", model="wan_i2v")
    _put_heartbeat("host-a", "gpu", last_seen_age_s=2)  # fresh

    flagged = node_queue.flag_stale_workers_holding_running_jobs(stale_after_s=30)

    assert flagged == []
    assert _flagged_at("host-a", "gpu") is None


def test_stale_worker_with_no_running_job_is_not_flagged():
    """A worker whose heartbeat froze but which owns NO running job needs no
    recovery — there's no stranded work to bounce it for (it'll just fall out of
    the live-capacity gauge). Must NOT be flagged, so the signal stays specific
    to the wedge-holding-work case."""
    _put_heartbeat("idle-host", "cpu", last_seen_age_s=300)  # stale, but idle

    flagged = node_queue.flag_stale_workers_holding_running_jobs(stale_after_s=30)

    assert flagged == []
    assert _flagged_at("idle-host", "cpu") is None


def test_join_is_scoped_to_the_owning_host():
    """A stale worker is flagged ONLY for the running jobs IT owns — a running
    job claimed by a DIFFERENT (healthy) host must not make a stale idle worker
    look like it's holding work."""
    # healthy host owns the only running job; the stale host owns nothing.
    _running_job_owned_by("healthy-host", model="wan_i2v")
    _put_heartbeat("healthy-host", "gpu", last_seen_age_s=2)   # fresh
    _put_heartbeat("stale-idle", "gpu", last_seen_age_s=300)   # stale but jobless

    flagged = node_queue.flag_stale_workers_holding_running_jobs(stale_after_s=30)

    assert flagged == []


def test_attribution_is_scoped_to_the_jobs_queue_family():
    """A host can run SEVERAL workers under one ``host_label`` (host-c runs a
    cpu AND a gpu worker). A gpu job (claimed_by=host-c, queue=gpu) must be
    attributed to the host-c/gpu heartbeat, NOT the host-c/cpu one — so a
    wedged gpu worker flags only its own row, and the healthy cpu worker (fresh
    heartbeat) is left alone even though it shares the host_label."""
    _running_job_owned_by("host-c", queue="gpu", model="wan_i2v")
    _put_heartbeat("host-c", "gpu", last_seen_age_s=120)  # gpu worker wedged
    _put_heartbeat("host-c", "cpu", last_seen_age_s=2)    # cpu worker healthy

    flagged = node_queue.flag_stale_workers_holding_running_jobs(stale_after_s=30)

    assert len(flagged) == 1
    assert (flagged[0]["host_label"], flagged[0]["queue"]) == ("host-c", "gpu")
    assert _flagged_at("host-c", "cpu") is None  # healthy sibling untouched


def test_stale_sibling_without_its_own_queues_job_is_not_flagged():
    """The inverse: if only the CPU worker on a shared host is stale but the
    running job belongs to the GPU queue, the stale cpu row is NOT flagged — the
    gpu job is not the cpu worker's work."""
    _running_job_owned_by("host-c", queue="gpu", model="wan_i2v")
    _put_heartbeat("host-c", "gpu", last_seen_age_s=2)    # gpu worker healthy
    _put_heartbeat("host-c", "cpu", last_seen_age_s=300)  # cpu worker stale but idle

    flagged = node_queue.flag_stale_workers_holding_running_jobs(stale_after_s=30)

    assert flagged == []


def test_counts_all_running_jobs_the_worker_owns():
    """A concurrency>1-style worker (or a worker that grabbed several jobs) gets
    the full running-job count in the returned row."""
    _running_job_owned_by("multi", queue="cpu")
    _running_job_owned_by("multi", queue="cpu")
    _running_job_owned_by("multi", queue="cpu")
    _put_heartbeat("multi", "cpu", last_seen_age_s=120)

    flagged = node_queue.flag_stale_workers_holding_running_jobs(stale_after_s=30)

    assert len(flagged) == 1
    assert flagged[0]["running_jobs"] == 3


# ── idempotency / recovery ───────────────────────────────────────────────────


def test_flag_is_idempotent_within_window():
    """A second sweep inside the staleness window does NOT re-flag (so the 0.5 s
    orchestrator tick doesn't relog every pass)."""
    _running_job_owned_by("host-a", model="wan_i2v")
    _put_heartbeat("host-a", "gpu", last_seen_age_s=120)

    first = node_queue.flag_stale_workers_holding_running_jobs(stale_after_s=30)
    assert len(first) == 1
    # Immediately re-running: already-flagged within the window ⇒ no re-flag.
    second = node_queue.flag_stale_workers_holding_running_jobs(stale_after_s=30)
    assert second == []


def test_heartbeat_refresh_clears_the_flag():
    """After the worker (or its replacement post-bounce) beats again, the flag
    is cleared by ``upsert_worker_heartbeat`` so a FUTURE hang re-flags cleanly
    instead of staying latched from the previous incident."""
    _running_job_owned_by("host-a", model="wan_i2v")
    _put_heartbeat("host-a", "gpu", last_seen_age_s=120)
    node_queue.flag_stale_workers_holding_running_jobs(stale_after_s=30)
    assert _flagged_at("host-a", "gpu") is not None

    # A fresh heartbeat (live worker resumes / replacement starts).
    node_queue.upsert_worker_heartbeat(
        host_label="host-a", queue="gpu", concurrency=1,
    )
    assert _flagged_at("host-a", "gpu") is None


def test_re_flags_after_recovery_then_stale_again():
    """End-to-end of the latch lifecycle: flag → recover (refresh clears it) →
    go stale again ⇒ re-flagged. Proves the detector isn't a one-shot."""
    job = _running_job_owned_by("host-a", model="wan_i2v")
    _put_heartbeat("host-a", "gpu", last_seen_age_s=120)
    assert len(node_queue.flag_stale_workers_holding_running_jobs(stale_after_s=30)) == 1

    # Worker recovers: fresh heartbeat clears the flag.
    node_queue.upsert_worker_heartbeat(host_label="host-a", queue="gpu", concurrency=1)
    assert _flagged_at("host-a", "gpu") is None

    # It still owns the running job (job retained) and goes stale AGAIN.
    assert node_queue.get_node_job(job)["status"] == "running"
    _put_heartbeat("host-a", "gpu", last_seen_age_s=120)
    assert len(node_queue.flag_stale_workers_holding_running_jobs(stale_after_s=30)) == 1


# ── threshold ────────────────────────────────────────────────────────────────


def test_threshold_boundary_arg():
    """Just under the threshold ⇒ not flagged; well over ⇒ flagged. Pins that
    the staleness window is honoured (arg form)."""
    _running_job_owned_by("h", queue="cpu")
    _put_heartbeat("h", "cpu", last_seen_age_s=10)
    assert node_queue.flag_stale_workers_holding_running_jobs(stale_after_s=30) == []

    _put_heartbeat("h", "cpu", last_seen_age_s=40)
    assert len(node_queue.flag_stale_workers_holding_running_jobs(stale_after_s=30)) == 1


def test_threshold_env_override(monkeypatch):
    """``AI_LEADS_STALE_WORKER_AFTER_S`` overrides the 30 s default; junk falls
    back to the default; unset is the default."""
    monkeypatch.delenv("AI_LEADS_STALE_WORKER_AFTER_S", raising=False)
    assert node_queue._stale_worker_after_s() == node_queue.STALE_WORKER_AFTER_S == 30
    monkeypatch.setenv("AI_LEADS_STALE_WORKER_AFTER_S", "5")
    assert node_queue._stale_worker_after_s() == 5
    monkeypatch.setenv("AI_LEADS_STALE_WORKER_AFTER_S", "junk")
    assert node_queue._stale_worker_after_s() == 30


def test_default_threshold_used_when_arg_omitted(monkeypatch):
    """With the default 30 s threshold, a 40 s-stale worker holding work is
    flagged (no explicit arg) — the live call path the sweep uses."""
    monkeypatch.delenv("AI_LEADS_STALE_WORKER_AFTER_S", raising=False)
    _running_job_owned_by("host-a", model="wan_i2v")
    _put_heartbeat("host-a", "gpu", last_seen_age_s=40)
    assert len(node_queue.flag_stale_workers_holding_running_jobs()) == 1


# ── NodePool wiring ──────────────────────────────────────────────────────────


def test_sweep_invokes_detector(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(
        node_queue, "flag_stale_workers_holding_running_jobs",
        lambda **kw: calls.append(time.time()) or [],
    )
    pool = _dead_worker_pool()
    pool._sweep_dead_workers()
    assert len(calls) == 1


def test_sweep_gated_by_interval(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(
        node_queue, "flag_stale_workers_holding_running_jobs",
        lambda **kw: calls.append(time.time()) or [],
    )
    pool = _dead_worker_pool()
    pool._dead_worker_interval_s = 5.0
    pool._sweep_dead_workers()
    pool._sweep_dead_workers()
    pool._sweep_dead_workers()
    assert len(calls) == 1


def test_tick_runs_the_dead_worker_sweep(monkeypatch):
    fired: list[str] = []
    monkeypatch.setattr(node_pool.NodePool, "_drain_dispatch_events", lambda self: None)
    monkeypatch.setattr(node_pool.NodePool, "_sweep_expired_leases", lambda self: None)
    monkeypatch.setattr(node_pool.NodePool, "_sweep_expired_ingest_leases", lambda self: None)
    monkeypatch.setattr(
        node_pool.NodePool, "_sweep_dead_workers",
        lambda self: fired.append("dead-worker"),
    )
    pool = node_pool.NodePool(cpu_workers=0, gpu_workers=0, register_builtins=None)
    pool._tick()
    assert fired == ["dead-worker"]


def test_sweep_logs_actionable_dead_worker_line(caplog):
    _running_job_owned_by("host-a", model="wan_i2v")
    _put_heartbeat("host-a", "gpu", last_seen_age_s=120)

    pool = _dead_worker_pool()
    with caplog.at_level(logging.ERROR):
        pool._sweep_dead_workers()

    msgs = [r.getMessage() for r in caplog.records]
    assert any("DEAD WORKER" in m and "host-a" in m and "wedged" in m for m in msgs), msgs


def test_sweep_does_not_touch_the_job_row():
    """The detector flags the worker but leaves the JOB to the lease-reclaim
    sweep — it must NOT re-queue or otherwise mutate the running job row."""
    job_id = _running_job_owned_by("host-a", model="wan_i2v")
    _put_heartbeat("host-a", "gpu", last_seen_age_s=120)

    pool = _dead_worker_pool()
    pool._sweep_dead_workers()

    row = node_queue.get_node_job(job_id)
    assert row["status"] == "running"        # untouched
    assert row["claimed_by"] == "host-a"      # not cleared


def test_sweep_noop_with_no_workers():
    pool = _dead_worker_pool()
    pool._sweep_dead_workers()  # must not raise
