"""Unit tests for the Postgres-as-queue claim worker (``claim_worker``).

Pins the pieces directly, without a live LISTEN loop:

  * the wall-clock budget table + ``budget_for(job)`` (video budget targets the
    host-configured ``config.video_model_ids``);
  * the watchdog trips at the budget → ``mark_failed`` then hard-exits;
  * lease renewal extends ``lease_expires_at`` while a job runs;
  * ``ClaimWorker.run_once`` claims the next ready row via ``claim_next_*`` and
    runs it through ``execute_node`` to a terminal state;
  * the ingest claim path (fetch/load → ``ingest_executor`` via the registered
    task map);
  * the startup schema-readiness gate (``queue_schema_version`` versions
    cpu/gpu → 6, ingest queues → 8);
  * the worker capacity heartbeat.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import pytest

import queue_workflows
from queue_workflows import claim_worker, node_queue
from queue_workflows.db import connection
from tests._helpers import make_run


@pytest.fixture(autouse=True)
def _video_models_and_node_pkg():
    """Configure the video model set (so the video budget test asserts a real
    value) + a fake node package for the run_once tests."""
    queue_workflows.configure(video_model_ids=frozenset({"wan_i2v", "ltx_flf"}))
    queue_workflows.set_node_module_package("qwf_cw_nodes")
    yield


def _make_run(status: str = "running") -> str:
    return make_run(status=status, workflow_name="_claim_worker_test", out_dir=None)


def _install_fake_node(name: str, run_fn):
    mod = types.ModuleType(f"qwf_cw_nodes.{name}")
    mod.run = run_fn
    sys.modules[f"qwf_cw_nodes.{name}"] = mod


def _lease_expiry(job_id: str):
    return node_queue.get_node_job(job_id)["lease_expires_at"]


def _node_events(job_id: str) -> list[dict]:
    """The append-only forensic event log (migration 0011) for one node-job,
    oldest first. Returned as plain dicts (``detail`` already decoded by psycopg's
    jsonb adapter) so tests can assert on event_type / attempt / detail directly."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT event_type, attempt, detail, error "
            "FROM workflow_node_events WHERE job_id=%s ORDER BY id",
            (job_id,),
        )
        return [dict(r) for r in cur.fetchall()]


# ── budget table ───────────────────────────────────────────────────────────


def test_budget_for_cpu_is_2100s():
    job = {"queue": "cpu", "node_module": "geocode"}
    assert claim_worker.budget_for(job) == 2100


def test_budget_for_input_node_is_120s():
    job = {"queue": "cpu", "node_module": "__input__choose_one"}
    assert claim_worker.budget_for(job) == 120


# ── watchdog ─────────────────────────────────────────────────────────────


def test_watchdog_trips_and_requeues_under_cap():
    """A wall-clock trip on a DAG node-job below the retry cap RE-QUEUES the node
    (running→queued, lease cleared, priority bumped, watchdog_retries++) + writes
    NO failed dispatch event, then hard-exits — uniform with the stall/health
    watchdogs. (Old behaviour was an immediate mark-failed.)"""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu", priority=100,
    )
    node_queue.claim_next_cpu_job(0, host="h")

    exits: list[int] = []
    wd = claim_worker.Watchdog(
        job_id=job_id, budget_s=0.05,
        on_exit=lambda code: exits.append(code), poll_s=0.01,
    )
    wd.start()
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    wd.stop()

    assert exits and exits[0] == 75, "watchdog must hard-exit (code 75) on budget overrun"
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "queued"
    assert row["claimed_by"] is None
    assert row["lease_expires_at"] is None
    assert row["priority"] <= 10
    assert row["watchdog_retries"] == 1
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM workflow_dispatch_events WHERE run_id=%s",
            (run_id,),
        )
        assert cur.fetchone()["n"] == 0, "re-queue must NOT write a failed event"


def test_watchdog_does_not_trip_when_stopped_before_budget():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    node_queue.claim_next_cpu_job(0, host="h")

    exits: list[int] = []
    wd = claim_worker.Watchdog(
        job_id=job_id, budget_s=5.0,
        on_exit=lambda code: exits.append(code), poll_s=0.01,
    )
    wd.start()
    time.sleep(0.1)
    wd.stop()
    time.sleep(0.1)
    assert exits == []
    assert node_queue.get_node_job(job_id)["status"] == "running"


# ── stall (no-progress) watchdog ─────────────────────────────────────────────
# The wall-clock Watchdog only catches a job that runs LONGER than its budget
# (8100 s for a generic GPU job). A node that HANGS — model resident, GPU at
# 0 %, no denoise step in minutes (the Blackwell qwen inference stall) — would
# camp the whole budget before the wall-clock watchdog frees it. The
# StallWatchdog trips on NO PROGRESS: every ``beat()`` (one per diffusion step,
# threaded via ``status_callback``, plus one from the executor right after the
# model load completes) pushes a short deadline out. It stays INERT until the
# FIRST beat so a multi-minute cold model load can't false-trip it; once armed,
# no beat within ``stall_timeout_s`` ⇒ mark failed + hard-exit so the lease
# reclaim re-queues the job onto a healthy host.


# A StallWatchdog whose no-beat timeout fires confirms the physical signal
# before tripping (Part A). These idle/static sampler fakes + a 1-sample, fast
# confirmation window make a confirmed-wedge trip deterministic with no GPU.
def _wedged_stall_watchdog(job_id, *, on_exit, stall_timeout_s=0.05,
                           host_label=None, queue=None):
    return claim_worker.StallWatchdog(
        job_id=job_id, stall_timeout_s=stall_timeout_s,
        on_exit=on_exit, poll_s=0.01,
        gpu_sampler=lambda: 0,       # GPU idle
        ram_sampler=lambda: 2048,    # RAM static
        confirm_samples=1, confirm_poll_s=0.0,
        host_label=host_label, queue=queue,
    )


def test_stall_watchdog_trips_after_first_beat_then_silence_requeues_under_cap():
    """A no-progress trip below the retry cap RE-QUEUES the node for a retry on a
    fresh worker (the user's reconstruct/beat_keyframes case): running→queued,
    lease cleared, priority bumped, watchdog_retries++, NO failed event — then
    hard-exits (code 76). The RUN stays alive. (Old behaviour was mark-failed.)

    The trip is GATED on the physical signal (Part A): the injected samplers read
    GPU idle + RAM static, so the no-beat timeout is CONFIRMED as a true wedge."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
        required_model="qwen_edit", priority=100,
    )
    node_queue.claim_next_gpu_job(0, host="h")

    exits: list[int] = []
    wd = _wedged_stall_watchdog(job_id, on_exit=lambda code: exits.append(code))
    wd.start()
    wd.beat()  # the executor's after-model-load beat arms enforcement
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    wd.stop()

    assert exits and exits[0] == 76, "stall watchdog must hard-exit (code 76)"
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "queued", "stall trip under cap must RE-QUEUE, not fail"
    assert row["claimed_by"] is None
    assert row["lease_expires_at"] is None
    assert row["priority"] <= 10
    assert row["watchdog_retries"] == 1
    # The run stays running; no failed dispatch event was written.
    assert node_queue.get_node_job(job_id) is not None
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM workflow_dispatch_events WHERE run_id=%s",
            (run_id,),
        )
        assert cur.fetchone()["n"] == 0
    from queue_workflows import run_store
    assert run_store.get_run(run_id)["status"] == "running"


# ── Part A — the no-beat trip is GATED on the physical GPU/RAM signal ─────────
# Beat-absence ALONE no longer trips the StallWatchdog. When the no-beat timeout
# fires, the watchdog confirms with the same gpu_health samplers + "GPU idle AND
# RAM static" rule the GpuHealthWatchdog uses. A loading/preparing/slow-but-
# working node (busy GPU OR moving RAM) is NEVER killed; only a genuinely idle +
# static (doing-nothing) worker trips. This fixes the user's false positive:
# "should not kill if the GPU model is being loaded or preparing to start".


def test_stall_no_beat_but_gpu_busy_does_not_trip():
    """No beat for the window, but GPU util is well over idle (a legitimately slow
    diffusion step) ⇒ NOT wedged ⇒ the watchdog resets and does NOT trip."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
        required_model="qwen_edit",
    )
    node_queue.claim_next_gpu_job(0, host="h")

    exits: list[int] = []
    wd = claim_worker.StallWatchdog(
        job_id=job_id, stall_timeout_s=0.05, on_exit=lambda c: exits.append(c),
        poll_s=0.01, confirm_samples=2, confirm_poll_s=0.01,
        gpu_sampler=lambda: 90,      # BUSY (slow step) → not wedged
        ram_sampler=lambda: 2048,    # static
    )
    wd.start()
    wd.beat()                        # arm; then go silent past the timeout
    time.sleep(0.4)                  # many no-beat windows; each confirms "busy"
    wd.stop()
    assert exits == [], "a busy GPU must never be killed by the stall watchdog"
    assert node_queue.get_node_job(job_id)["status"] == "running"
    # The no-kill is correct, but the OPERATOR-facing "why did this look stuck
    # but wasn't" signal must also be emitted (claim_worker._confirm_wedged →
    # stall_suspected). It's the highest-value invisible signal: if the emit
    # were dropped the job would still survive and this test would pass without
    # it — so assert the forensic row carries the busy gpu_sampler value, and
    # that NO real trip event was recorded.
    suspected = [e for e in _node_events(job_id) if e["event_type"] == "stall_suspected"]
    assert suspected, "a confirmed-not-wedged window must record a stall_suspected event"
    assert any(e["detail"].get("max_sm_pct") == 90 for e in suspected), (
        "stall_suspected detail must carry the busy gpu sm%% (90) that spared the job"
    )
    assert not [e for e in _node_events(job_id) if e["event_type"] == "stall_trip"], (
        "a busy GPU must never produce a stall_trip event"
    )


def test_stall_no_beat_but_ram_moving_does_not_trip():
    """No beat for the window AND GPU idle, but container RAM is climbing >
    ram_delta each confirmation window (a model loading weights / preparing) ⇒
    NOT wedged ⇒ the watchdog does NOT trip. This is the core load/prepare
    false-positive fix: a multi-GB load moves RAM far beyond the delta."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
        required_model="qwen_edit",
    )
    node_queue.claim_next_gpu_job(0, host="h")

    exits: list[int] = []
    # RAM grows by 6 GB each read — well over the 5 GB (5120 MB) delta.
    ram_seq = iter(range(0, 1_000_000, 6144))
    wd = claim_worker.StallWatchdog(
        job_id=job_id, stall_timeout_s=0.05, on_exit=lambda c: exits.append(c),
        poll_s=0.01, confirm_samples=2, confirm_poll_s=0.01,
        idle_pct=5, ram_delta_mb=5120,
        gpu_sampler=lambda: 0,                 # idle (load not issuing SM work yet)
        ram_sampler=lambda: next(ram_seq),     # RAM moving (loading)
    )
    wd.start()
    wd.beat()
    time.sleep(0.4)
    wd.stop()
    assert exits == [], "RAM moving > delta (loading/preparing) must not trip"
    assert node_queue.get_node_job(job_id)["status"] == "running"
    # Same operator-facing forensic contract as the busy-GPU case: a suspected-
    # but-not-wedged window (here: GPU idle but RAM climbing ⇒ loading) must
    # still record stall_suspected so the "looked stuck, was loading" story is
    # in-app. GPU was idle here, so the captured max sm%% is 0.
    suspected = [e for e in _node_events(job_id) if e["event_type"] == "stall_suspected"]
    assert suspected, "a moving-RAM (loading) window must record a stall_suspected event"
    assert any(e["detail"].get("max_sm_pct") == 0 for e in suspected), (
        "stall_suspected detail must carry the idle gpu sm%% (0) seen while RAM moved"
    )
    assert not [e for e in _node_events(job_id) if e["event_type"] == "stall_trip"], (
        "a loading (RAM-moving) node must never produce a stall_trip event"
    )


def test_stall_no_beat_and_gpu_idle_and_ram_static_trips():
    """No beat AND GPU idle AND RAM static across the confirmation window ⇒
    genuinely doing nothing ⇒ WEDGED ⇒ trip (and, under the cap, re-queue)."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
        required_model="qwen_edit",
    )
    node_queue.claim_next_gpu_job(0, host="h")

    exits: list[int] = []
    wd = claim_worker.StallWatchdog(
        job_id=job_id, stall_timeout_s=0.05, on_exit=lambda c: exits.append(c),
        poll_s=0.01, confirm_samples=3, confirm_poll_s=0.01,
        gpu_sampler=lambda: 0,       # idle
        ram_sampler=lambda: 2048,    # static
    )
    wd.start()
    wd.beat()
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    wd.stop()
    assert exits and exits[0] == 76, "idle + static ⇒ confirmed wedge ⇒ trip"
    assert node_queue.get_node_job(job_id)["status"] == "queued"


def test_stall_recovers_to_running_when_gpu_resumes_after_a_suspected_stall():
    """A node that LOOKS stalled (no beat) but is busy survives the confirmation
    and keeps running; once it beats again the window is healthy. Proves the
    confirmation RESETS rather than latches — a transient slow patch isn't fatal."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
        required_model="qwen_edit",
    )
    node_queue.claim_next_gpu_job(0, host="h")

    exits: list[int] = []
    wd = claim_worker.StallWatchdog(
        job_id=job_id, stall_timeout_s=0.05, on_exit=lambda c: exits.append(c),
        poll_s=0.01, confirm_samples=2, confirm_poll_s=0.01,
        gpu_sampler=lambda: 80,      # busy throughout
        ram_sampler=lambda: 2048,
    )
    wd.start()
    wd.beat()
    time.sleep(0.2)                  # no-beat window fires, confirms busy, resets
    wd.beat()                        # progress resumes
    time.sleep(0.1)
    wd.stop()
    assert exits == []
    assert node_queue.get_node_job(job_id)["status"] == "running"


# ── Part C — a hard-exiting worker must NOT leave a "busy" current_model ghost ─
# os._exit skips _run_node's finally (mark_idle + the heartbeat never run), so the
# dead worker's worker_heartbeats row keeps advertising current_model and inflates
# the "N/M GPU busy" gauge (the user's "3/2 GPU busy" after a kill). The trip path
# (_requeue_job_and_exit / _fail_job_and_exit) clears current_model + ages
# last_seen out of the gauge window BEFORE the hard-exit.


def _seed_busy_heartbeat(host, queue, model):
    """A GPU worker heartbeat advertising a warm model (the busy signal)."""
    node_queue.upsert_worker_heartbeat(
        host_label=host, queue=queue, concurrency=1, current_model=model,
    )


def test_stall_trip_clears_current_model_busy_ghost():
    """A confirmed stall trip (re-queue path, under cap) clears the worker's
    current_model so the dead worker stops looking busy."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
        required_model="qwen_edit",
    )
    node_queue.claim_next_gpu_job(0, host="host-b")
    _seed_busy_heartbeat("host-b", "gpu", "qwen_edit")

    exits: list[int] = []
    wd = _wedged_stall_watchdog(
        job_id, on_exit=lambda c: exits.append(c),
        host_label="host-b", queue="gpu",
    )
    wd.start(); wd.beat()
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    wd.stop()
    assert exits and exits[0] == 76
    row = _heartbeat_row("host-b", "gpu")
    assert row is not None and row["current_model"] is None, (
        "a hard-exiting worker must clear its current_model busy-ghost"
    )


def test_gpu_health_trip_clears_current_model_busy_ghost():
    """A confirmed GpuHealthWatchdog trip clears current_model too (same Part-C
    fix, both re-queue and fail paths go through _clear_busy_ghost)."""
    run_id, job_id = _running_gpu_job_with_retries(3)  # == cap → fail path
    _seed_busy_heartbeat("host-a", "gpu", "wan_i2v")
    exits: list[int] = []
    wd = claim_worker.GpuHealthWatchdog(
        job_id=job_id, interval_s=0.05, idle_pct=5, ram_delta_mb=5120, poll_s=0.01,
        gpu_sampler=lambda: 0, ram_sampler=lambda: 2048,
        on_exit=lambda c: exits.append(c),
        host_label="host-a", queue="gpu",
    )
    wd.start(); wd.beat()
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    wd.stop()
    assert exits and exits[0] == 78
    row = _heartbeat_row("host-a", "gpu")
    assert row is not None and row["current_model"] is None


def test_busy_ghost_clear_marks_heartbeat_stale_so_gauge_drops_it():
    """Clearing the busy-ghost also ages last_seen past the 30 s gauge window so
    the dead worker drops out of the live-worker count at once (not after 30 s)."""
    _seed_busy_heartbeat("host-b", "gpu", "qwen_edit")
    # Fresh before.
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT last_seen > now() - interval '30 seconds' AS fresh "
            "FROM worker_heartbeats WHERE host_label='host-b' AND queue='gpu'"
        )
        assert cur.fetchone()["fresh"] is True
    claim_worker._clear_busy_ghost("host-b", "gpu")
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT current_model, "
            "       last_seen > now() - interval '30 seconds' AS fresh "
            "FROM worker_heartbeats WHERE host_label='host-b' AND queue='gpu'"
        )
        r = cur.fetchone()
    assert r["current_model"] is None
    assert r["fresh"] is False, "last_seen must be aged out of the gauge window"


def test_busy_ghost_clear_is_noop_without_identity():
    """_clear_busy_ghost is a safe no-op when the worker identity wasn't threaded
    (a unit test constructing a watchdog with just a job_id) — it must never raise
    and never touch unrelated rows."""
    _seed_busy_heartbeat("other", "gpu", "qwen_edit")
    claim_worker._clear_busy_ghost(None, None)
    claim_worker._clear_busy_ghost("", "gpu")
    claim_worker._clear_busy_ghost("other", "")
    # The unrelated row is untouched.
    assert _heartbeat_row("other", "gpu")["current_model"] == "qwen_edit"


def test_stall_watchdog_inert_until_first_beat():
    """Before the first beat the watchdog does NOT enforce — so a long cold
    model load (minutes with no beats) can never false-trip it."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
    )
    node_queue.claim_next_gpu_job(0, host="h")

    exits: list[int] = []
    wd = claim_worker.StallWatchdog(
        job_id=job_id, stall_timeout_s=0.05,
        on_exit=lambda code: exits.append(code), poll_s=0.01,
    )
    wd.start()
    time.sleep(0.2)  # well past the timeout, but no beat has armed it yet
    assert exits == [], "must stay inert until the first progress beat"
    assert node_queue.get_node_job(job_id)["status"] == "running"
    wd.stop()


def test_stall_watchdog_beat_resets_deadline_and_keeps_job_running():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
    )
    node_queue.claim_next_gpu_job(0, host="h")

    exits: list[int] = []
    wd = claim_worker.StallWatchdog(
        job_id=job_id, stall_timeout_s=0.2,
        on_exit=lambda code: exits.append(code), poll_s=0.01,
    )
    wd.start()
    # Beat faster than the timeout for well over one timeout-window total —
    # a deadline that resets on every beat must never trip here.
    for _ in range(8):
        time.sleep(0.05)
        wd.beat()
    assert exits == [], "periodic beats must hold the deadline open"
    assert node_queue.get_node_job(job_id)["status"] == "running"
    wd.stop()


def test_stall_watchdog_does_not_trip_when_stopped_before_timeout():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
    )
    node_queue.claim_next_gpu_job(0, host="h")

    exits: list[int] = []
    wd = claim_worker.StallWatchdog(
        job_id=job_id, stall_timeout_s=5.0,
        on_exit=lambda code: exits.append(code), poll_s=0.01,
    )
    wd.start()
    wd.beat()
    time.sleep(0.1)
    wd.stop()
    time.sleep(0.1)
    assert exits == []
    assert node_queue.get_node_job(job_id)["status"] == "running"


# ── watchdog re-queue-with-cap policy ────────────────────────────────────────
# A watchdog trip RE-QUEUES the node for a retry on a fresh worker (run stays
# alive) while watchdog_retries < AI_LEADS_WATCHDOG_MAX_RETRIES (default 3); once
# the counter has reached the cap it FALLS BACK to the old mark-failed path (+ a
# failed dispatch event → the run fails) so a persistently-wedging node doesn't
# loop forever.


def _running_gpu_job_with_retries(retries: int, *, priority=100):
    """A claimed (running) gpu node-job whose watchdog_retries is preset to
    ``retries`` — the per-job re-queue counter the trip site reads."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
        required_model="qwen_edit", priority=priority,
    )
    node_queue.claim_next_gpu_job(0, host="h")
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs SET watchdog_retries=%s WHERE id=%s",
            (retries, job_id),
        )
    return run_id, job_id


def _assert_failed_with_event(run_id, job_id, *, err_substr):
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "failed", "at/over the cap the trip must FAIL, not re-queue"
    assert err_substr in (row["error"] or "").lower()
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT kind FROM workflow_dispatch_events WHERE run_id=%s",
            (run_id,),
        )
        kinds = [r["kind"] for r in cur.fetchall()]
    assert "failed" in kinds, "the fail path must write a failed dispatch event"


def test_stall_watchdog_fails_at_cap():
    """At the cap (watchdog_retries == default 3) a stall trip FAILS the run
    (mark failed + failed dispatch event), not another re-queue."""
    run_id, job_id = _running_gpu_job_with_retries(3)  # == default cap
    exits: list[int] = []
    wd = _wedged_stall_watchdog(job_id, on_exit=lambda c: exits.append(c))
    wd.start()
    wd.beat()
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    wd.stop()
    assert exits and exits[0] == 76
    _assert_failed_with_event(run_id, job_id, err_substr="no progress")


def test_gpu_health_watchdog_fails_at_cap():
    """At the cap a GpuHealthWatchdog trip FAILS the run, mirroring the stall
    watchdog's fall-back."""
    run_id, job_id = _running_gpu_job_with_retries(3)
    exits: list[int] = []
    wd = claim_worker.GpuHealthWatchdog(
        job_id=job_id, interval_s=0.05, idle_pct=5, ram_delta_mb=5120, poll_s=0.01,
        gpu_sampler=lambda: 0, ram_sampler=lambda: 2048,
        on_exit=lambda c: exits.append(c),
    )
    wd.start()
    wd.beat()
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    wd.stop()
    assert exits and exits[0] == 78
    _assert_failed_with_event(run_id, job_id, err_substr="no gpu activity")


def test_watchdog_cap_boundary_last_retry_requeues_then_next_fails():
    """The boundary is exact: with cap=3, a trip at retries=2 (< 3) still
    RE-QUEUES (the 3rd attempt), incrementing to 3; a trip at retries=3 (== cap)
    FAILS. Proven by driving the SAME job through both."""
    run_id, job_id = _running_gpu_job_with_retries(2)  # 2 < cap(3) → re-queue
    exits: list[int] = []
    wd = _wedged_stall_watchdog(job_id, on_exit=lambda c: exits.append(c))
    wd.start(); wd.beat()
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    wd.stop()
    assert exits and exits[0] == 76
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "queued", "retries=2 < cap=3 ⇒ re-queue"
    assert row["watchdog_retries"] == 3, "the re-queue incremented to the cap"

    # Re-claim + trip again: now retries == cap ⇒ FAIL.
    node_queue.claim_next_gpu_job(0, host="h")
    exits2: list[int] = []
    wd2 = _wedged_stall_watchdog(job_id, on_exit=lambda c: exits2.append(c))
    wd2.start(); wd2.beat()
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits2:
        time.sleep(0.02)
    wd2.stop()
    assert exits2 and exits2[0] == 76
    _assert_failed_with_event(run_id, job_id, err_substr="no progress")


def test_watchdog_max_retries_env_override(monkeypatch):
    """AI_LEADS_WATCHDOG_MAX_RETRIES overrides the cap, read at trip-time. With
    the cap set to 1, a job already at retries=1 FAILS instead of re-queuing —
    even though the module default (3) would have re-queued."""
    monkeypatch.setenv("AI_LEADS_WATCHDOG_MAX_RETRIES", "1")
    assert claim_worker._watchdog_max_retries() == 1
    run_id, job_id = _running_gpu_job_with_retries(1)  # == overridden cap
    exits: list[int] = []
    wd = _wedged_stall_watchdog(job_id, on_exit=lambda c: exits.append(c))
    wd.start(); wd.beat()
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    wd.stop()
    assert exits and exits[0] == 76
    _assert_failed_with_event(run_id, job_id, err_substr="no progress")


def test_watchdog_max_retries_default_and_env_parse(monkeypatch):
    """The cap default is 3; the accessor reads the env override / falls back on
    junk (never crashes a worker on a bad env)."""
    assert claim_worker.WATCHDOG_MAX_RETRIES == 3
    monkeypatch.delenv("AI_LEADS_WATCHDOG_MAX_RETRIES", raising=False)
    assert claim_worker._watchdog_max_retries() == 3
    monkeypatch.setenv("AI_LEADS_WATCHDOG_MAX_RETRIES", "5")
    assert claim_worker._watchdog_max_retries() == 5
    monkeypatch.setenv("AI_LEADS_WATCHDOG_MAX_RETRIES", "junk")
    assert claim_worker._watchdog_max_retries() == 3


def test_ingest_watchdog_trip_marks_failed_not_requeue():
    """An ingest job (ingest_jobs table — no DAG run, no watchdog_retries column)
    keeps the OLD mark-failed behaviour on a wall-clock trip: there's no run to
    keep alive, and the ingest lease-reclaim re-queues separately."""
    queue_workflows.register_ingest_task("run_fetch_all", lambda reason: {})
    job_id = node_queue.enqueue_ingest_job(task_name="run_fetch_all", queue="fetch")
    node_queue.claim_next_ingest_job("fetch", host="h")
    exits: list[int] = []
    wd = claim_worker.Watchdog(
        job_id=job_id, budget_s=0.05, table="ingest_jobs",
        on_exit=lambda c: exits.append(c), poll_s=0.01,
    )
    wd.start()
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    wd.stop()
    assert exits and exits[0] == 75
    assert node_queue.get_ingest_job(job_id)["status"] == "failed"


# ── job-status watcher (abandon a job taken from us) ─────────────────────────
# When a job is re-queued / reassigned out from under a worker (restart-resume,
# lease reclaim), the worker must hard-exit so it doesn't double-run the row a
# fresh claimant is now executing. Scoped to claimed_by, NOT bare status, so the
# worker's OWN mark_completed/mark_failed (which keep claimed_by) don't trip it.


def test_job_status_watcher_kills_when_job_requeued():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
    )
    node_queue.claim_next_gpu_job(0, host="host-a")  # running, claimed_by=host-a

    exits: list[int] = []
    w = claim_worker.JobStatusWatcher(
        job_id=job_id, claimed_by="host-a",
        on_exit=lambda c: exits.append(c), poll_s=0.02,
    )
    w.start()
    node_queue.reclaim_all_running_for_resume()  # external re-queue clears claimed_by
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    w.stop()
    assert exits, "must hard-exit when the job is re-queued out from under it"


def test_job_status_watcher_quiet_while_job_ours_and_running():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
    )
    node_queue.claim_next_gpu_job(0, host="host-a")

    exits: list[int] = []
    w = claim_worker.JobStatusWatcher(
        job_id=job_id, claimed_by="host-a",
        on_exit=lambda c: exits.append(c), poll_s=0.02,
    )
    w.start()
    time.sleep(0.15)
    w.stop()
    assert exits == []


def test_job_status_watcher_ignores_own_completion():
    """A normal completion sets status but KEEPS claimed_by, so the watcher must
    NOT fire — only an external hand-off (claimed_by cleared/changed) kills."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
    )
    node_queue.claim_next_gpu_job(0, host="host-a")

    exits: list[int] = []
    w = claim_worker.JobStatusWatcher(
        job_id=job_id, claimed_by="host-a",
        on_exit=lambda c: exits.append(c), poll_s=0.02,
    )
    w.start()
    with connection() as conn, conn.cursor() as cur:
        node_queue.mark_completed_in_txn(
            cur, job_id, context_delta={}, seconds=1.0, vm_rss_mb_peak=None,
        )
    time.sleep(0.15)
    w.stop()
    assert exits == [], "own completion keeps claimed_by → must not self-kill"


# ── lease renewal ──────────────────────────────────────────────────────────


def test_lease_renewer_extends_lease():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    claimed = node_queue.claim_next_cpu_job(0, host="host-x", lease_s=600)
    assert claimed["claimed_by"] == "host-x"
    before = _lease_expiry(job_id)

    renewer = claim_worker.LeaseRenewer(
        job_id=job_id, claimed_by="host-x", lease_s=600, interval_s=0.05,
    )
    renewer.start()
    time.sleep(0.2)
    renewer.stop()
    after = _lease_expiry(job_id)
    assert after > before


def test_lease_renewer_only_touches_its_own_claim():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    node_queue.claim_next_cpu_job(0, host="real-owner", lease_s=600)
    before = _lease_expiry(job_id)

    renewer = claim_worker.LeaseRenewer(
        job_id=job_id, claimed_by="impostor", lease_s=600, interval_s=0.05,
    )
    renewer.start()
    time.sleep(0.2)
    renewer.stop()
    after = _lease_expiry(job_id)
    assert after == before


# ── claim + execute one job ─────────────────────────────────────────────────


def test_run_once_claims_and_executes_cpu_job():
    run_id = _make_run()
    captured: dict = {}

    def run(*, inputs=None, out=None, model_handle=None, status_callback=None,
            cancel_event=None):
        captured["ran"] = True
        return {"context_delta": {"done": True}}

    _install_fake_node("_cw_cpu", run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="_cw_cpu", queue="cpu",
    )

    worker = claim_worker.ClaimWorker(queue="cpu", host="test-host")
    assert worker.run_once() is True
    assert captured.get("ran") is True
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "completed"
    assert row["context_delta"] == {"done": True}
    assert row["claimed_by"] == "test-host"


def test_run_once_returns_false_when_nothing_to_claim():
    worker = claim_worker.ClaimWorker(queue="cpu", host="test-host")
    assert worker.run_once() is False


def test_run_once_parks_input_node_via_outbox_not_module_import():
    """Regression: an ``__input__`` job must be PARKED by the worker — marked
    awaiting_input + an ``awaiting_input`` outbox event for NodePool to drain —
    NOT handed to execute_node, which would ``import`` the sentinel module and
    die with ModuleNotFoundError. This is the claim→park contract the
    queue_workflows extraction dropped."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="pick_perspective",
        node_module="__input__pick_perspective", queue="cpu",
        inputs={"widget": "choose_one", "target": "perspective"},
    )
    worker = claim_worker.ClaimWorker(queue="cpu", host="test-host")
    # Must NOT raise ModuleNotFoundError; the job is handled (claimed + parked).
    assert worker.run_once() is True
    assert node_queue.get_node_job(job_id)["status"] == "awaiting_input"
    # And it emitted the outbox event NodePool drains → on_node_awaiting_input.
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT kind FROM workflow_dispatch_events "
            "WHERE run_id = %s AND node_id = %s",
            (run_id, "pick_perspective"),
        )
        kinds = [r["kind"] for r in cur.fetchall()]
    assert "awaiting_input" in kinds


def test_gpu_claim_skips_job_whose_model_worker_cannot_serve():
    """Capability gate (regression for the gate dropped in the Phase-5
    extraction): a gpu worker must NOT claim a job whose required_model isn't in
    its known_models — a capable peer gets it instead."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="big", node_module="render",
        queue="gpu", required_model="model_B",
    )
    # This worker only serves model_A → must not claim model_B's job.
    assert node_queue.claim_next_gpu_job(0, host="h1", known_models=["model_A"]) is None
    assert node_queue.get_node_job(job_id)["status"] == "queued"
    # A worker that DOES serve model_B claims it.
    claimed = node_queue.claim_next_gpu_job(
        0, host="h2", known_models=["model_A", "model_B"],
    )
    assert claimed is not None and claimed["id"] == job_id


def test_gpu_claim_falls_back_to_claim_any_without_known_models():
    """No advertised capability set (worker hasn't heartbeated its registry yet)
    → claim-any, so a cold/unconfigured worker can't wedge the queue."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="render",
        queue="gpu", required_model="model_X",
    )
    claimed = node_queue.claim_next_gpu_job(0, host="h", known_models=None)
    assert claimed is not None and claimed["id"] == job_id


def test_gpu_claim_serves_modelless_job_regardless_of_capability():
    """A job with no required_model is claimable by any gpu worker."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="x", queue="gpu",
    )
    claimed = node_queue.claim_next_gpu_job(0, host="h", known_models=["model_A"])
    assert claimed is not None and claimed["id"] == job_id


def test_node_reports_progress_gate_keys_on_status_callback_param():
    """A node opts into no-progress policing purely by declaring a
    ``status_callback`` param — that's the gate the worker checks before arming
    a StallWatchdog."""
    def with_cb(*, out=None, status_callback=None):
        return {}

    def without_cb(*, out=None, inputs=None):
        return {}

    _install_fake_node("_cw_with_cb", with_cb)
    _install_fake_node("_cw_without_cb", without_cb)
    assert claim_worker.ClaimWorker._node_reports_progress("_cw_with_cb") is True
    assert claim_worker.ClaimWorker._node_reports_progress("_cw_without_cb") is False


def test_run_node_threads_a_callable_status_callback_to_gpu_node():
    """End-to-end wiring: a gpu node that declares ``status_callback`` receives
    a CALLABLE (the StallWatchdog beat), not ``None`` — so each reported step
    pushes the no-progress deadline out. (Pre-fix the executor hard-wired
    ``status_callback=None``, so no node could ever beat.)

    Uses a MODEL-BACKED job: the inline lane (``run_once`` → ``_claim`` with
    ``require_model=True``) is the warm-model diffusion lane; no-model GPU jobs
    now go to the pool lane instead."""
    run_id = _make_run()
    seen: dict = {}

    def run(*, out=None, model_handle=None, status_callback=None, cancel_event=None):
        seen["is_callable"] = callable(status_callback)
        if status_callback:
            status_callback(7)  # a per-step beat, with an arg
        return {"context_delta": {"ok": True}}

    _install_fake_node("_cw_gpu_progress", run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="_cw_gpu_progress", queue="gpu",
        required_model="sdxl",
    )

    class _Cache:
        current_model = "sdxl"

        def require_model(self, model_id):
            return object()

        def mark_busy(self): ...
        def mark_idle(self): ...

    worker = claim_worker.ClaimWorker(queue="gpu", host="host-b", model_cache=_Cache())
    assert worker.run_once() is True
    assert seen.get("is_callable") is True
    assert node_queue.get_node_job(job_id)["status"] == "completed"


def test_stall_watchdog_not_armed_for_video_models(monkeypatch):
    """A video-model gpu job is NOT stall-policed (the tight 120 s StallWatchdog
    would false-trip a slow video backend), but it IS guarded by the
    HEALTH-driven GpuHealthWatchdog (video is the whole point of the health
    guard). So a video render gets a CALLABLE status_callback that fans out to
    the health watchdog ONLY — no StallWatchdog is constructed. Regression
    guard: a live fence_render_narrative video render was hard-stopped at 120 s
    by an over-eager StallWatchdog arm."""
    run_id = _make_run()
    seen: dict = {}

    constructed: list[str] = []
    real_stall = claim_worker.StallWatchdog
    real_health = claim_worker.GpuHealthWatchdog

    def stall_spy(**kw):
        constructed.append("stall")
        return real_stall(**kw)

    def health_spy(**kw):
        constructed.append("health")
        # inert sampler so the watchdog never trips during the test
        kw.setdefault("gpu_sampler", lambda: 100)
        kw.setdefault("ram_sampler", lambda: 0)
        return real_health(**kw)

    monkeypatch.setattr(claim_worker, "StallWatchdog", stall_spy)
    monkeypatch.setattr(claim_worker, "GpuHealthWatchdog", health_spy)

    def run(*, out=None, model_handle=None, status_callback=None, cancel_event=None):
        seen["status_callback"] = status_callback
        return {"context_delta": {"ok": True}}

    _install_fake_node("_cw_video", run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="v", node_module="_cw_video", queue="gpu",
        required_model="wan_i2v",  # in the autouse fixture's video_model_ids
    )

    class _Cache:
        current_model = "wan_i2v"

        def require_model(self, model_id):
            return object()

        def mark_busy(self): ...
        def mark_idle(self): ...

    worker = claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=_Cache())
    assert worker.run_once() is True
    # Health watchdog armed (video is the point); StallWatchdog NOT constructed.
    assert "health" in constructed, "video model must get the GpuHealthWatchdog"
    assert "stall" not in constructed, "video model must NOT get the tight StallWatchdog"
    # The node still receives a CALLABLE callback (the health watchdog's beat
    # fan-out), not None — so its per-step beats reset the health window.
    assert callable(seen.get("status_callback")), "video render gets the health-beat callback"
    assert node_queue.get_node_job(job_id)["status"] == "completed"


def test_gpu_job_gets_health_watchdog_and_no_wallclock_watchdog(monkeypatch):
    """Wiring: a GPU job is policed by the HEALTH-driven GpuHealthWatchdog and
    gets NO wall-clock Watchdog (no fixed time cap — the user's spec). A CPU job
    is the inverse: a wall-clock Watchdog, no health watchdog."""
    constructed: list[str] = []
    real_wd = claim_worker.Watchdog
    real_health = claim_worker.GpuHealthWatchdog

    def wd_spy(**kw):
        constructed.append("wallclock")
        return real_wd(**kw)

    def health_spy(**kw):
        constructed.append("health")
        kw.setdefault("gpu_sampler", lambda: 100)  # busy ⇒ never trips
        kw.setdefault("ram_sampler", lambda: 0)
        return real_health(**kw)

    monkeypatch.setattr(claim_worker, "Watchdog", wd_spy)
    monkeypatch.setattr(claim_worker, "GpuHealthWatchdog", health_spy)

    def run(*, out=None, model_handle=None, status_callback=None, cancel_event=None):
        return {"context_delta": {"ok": True}}

    _install_fake_node("_cw_gpu_guard", run)

    class _Cache:
        current_model = "sdxl"
        def require_model(self, model_id): return object()
        def mark_busy(self): ...
        def mark_idle(self): ...

    # Model-backed GPU job → health watchdog, no wall-clock watchdog. (The
    # inline lane is the warm-model diffusion lane — require_model=True.)
    run_id = _make_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="_cw_gpu_guard", queue="gpu",
        required_model="sdxl",
    )
    assert claim_worker.ClaimWorker(
        queue="gpu", host="host-a", model_cache=_Cache(),
    ).run_once() is True
    assert "health" in constructed
    assert "wallclock" not in constructed, "GPU job must NOT get a fixed wall-clock cap"

    # CPU job → wall-clock watchdog, no health watchdog.
    constructed.clear()
    run_id = _make_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="c", node_module="_cw_gpu_guard", queue="cpu",
    )
    assert claim_worker.ClaimWorker(queue="cpu", host="host-c").run_once() is True
    assert "wallclock" in constructed
    assert "health" not in constructed, "CPU job must NOT get a health watchdog"


def test_gpu_model_backed_no_progress_node_still_gets_started_health_watchdog(monkeypatch):
    """GAP #2 regression (diffusion lane): a MODEL-BACKED GPU node with no
    ``status_callback`` param never beats — under the old inert-until-beat arming
    the health watchdog was never armed and the node had NO time bound (the
    wall-clock cap was removed). The fix arms it AT start, so it must still be
    CONSTRUCTED and STARTED. No StallWatchdog (the node doesn't report progress),
    no wall-clock Watchdog (GPU).

    Now scoped to the INLINE (warm-model diffusion) lane — that's the lane the
    GpuHealthWatchdog guards. (A NO-model GPU node is handled by the pool lane,
    which deliberately does NOT arm the health watchdog — see the pool-lane
    tests; its bound is the lease + JobStatusWatcher + reclaim, not GPU health,
    because a VLM job is HTTP-bound.)"""
    started: list[str] = []
    real_health = claim_worker.GpuHealthWatchdog

    def health_spy(**kw):
        kw.setdefault("gpu_sampler", lambda: 100)   # busy ⇒ never trips in-test
        kw.setdefault("ram_sampler", lambda: 0)
        wd = real_health(**kw)
        real_start = wd.start
        def traced_start():
            started.append("health")
            return real_start()
        wd.start = traced_start
        return wd

    monkeypatch.setattr(claim_worker, "GpuHealthWatchdog", health_spy)

    # No status_callback param ⇒ _node_reports_progress is False ⇒ no StallWatchdog.
    def run(*, out=None, model_handle=None, cancel_event=None):
        return {"context_delta": {"ok": True}}

    _install_fake_node("_cw_model_no_progress", run)

    class _Cache:
        current_model = "sdxl"
        def require_model(self, model_id): return object()
        def mark_busy(self): ...
        def mark_idle(self): ...

    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="_cw_model_no_progress",
        queue="gpu", required_model="sdxl",
    )
    assert claim_worker.ClaimWorker(
        queue="gpu", host="host-a", model_cache=_Cache(),
    ).run_once() is True
    assert started == ["health"], (
        "a model-backed/no-progress GPU node must still get a STARTED health "
        "watchdog (armed at start) so it is bounded"
    )
    assert node_queue.get_node_job(job_id)["status"] == "completed"


def test_inline_lane_does_not_claim_no_model_gpu_job():
    """The two-lane split: the inline lane (``run_once`` → ``_claim`` with
    ``require_model=True``) does NOT claim a no-model GPU job — that row belongs
    to the pool lane. ``run_once`` returns False (nothing claimable by this lane)
    and the row stays ``queued`` for the pool feeder to take."""
    run_id = _make_run()

    def run(*, out=None, cancel_event=None):
        return {"context_delta": {"ok": True}}

    _install_fake_node("_cw_inline_skip_nomodel", run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="_cw_inline_skip_nomodel",
        queue="gpu",  # no required_model ⇒ pool lane's, not inline's
    )

    class _Cache:
        current_model = None
        def require_model(self, model_id): return object()
        def mark_busy(self): ...
        def mark_idle(self): ...

    worker = claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=_Cache())
    assert worker.run_once() is False, "inline lane must not claim a no-model GPU job"
    assert node_queue.get_node_job(job_id)["status"] == "queued"


def test_run_once_skips_job_under_cancelled_run():
    run_id = _make_run(status="cancelled")
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    worker = claim_worker.ClaimWorker(queue="cpu", host="test-host")
    assert worker.run_once() is False
    assert node_queue.get_node_job(job_id)["status"] == "queued"


# ── GPU health watchdog (HEALTH-driven, replaces the wall-clock cap) ─────────
# The GpuHealthWatchdog never kills a GPU job for elapsed time. Every window it
# checks two per-container signals and TRIPS only when the worker is truly
# wedged: GPU util stayed idle (<= idle_pct) AND container RAM was static
# (|Δ| <= ram_delta_mb) across the whole window. A busy GPU OR a > ram_delta RAM
# move ⇒ healthy ⇒ window resets, job runs on. It arms AT start with a generous
# load_grace_s FIRST window (so a load-phase hang AND a GPU node that never beats
# are both bounded); the first beat (executor post-load) collapses the window to
# the normal interval_s cadence. Samplers are injected so tests never shell out
# to nvidia-smi.


def _gpu_health_job():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="x", queue="gpu",
        required_model="qwen_edit",
    )
    node_queue.claim_next_gpu_job(0, host="h")
    return job_id


def _drain(exits, timeout_s=3.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline and not exits:
        time.sleep(0.01)


def test_gpu_health_does_not_trip_while_gpu_busy():
    """(a) GPU busy (util well over idle_pct) ⇒ never trips, even with static
    RAM and many checkpoints elapsed."""
    job_id = _gpu_health_job()
    exits: list[int] = []
    wd = claim_worker.GpuHealthWatchdog(
        job_id=job_id, interval_s=0.05, idle_pct=5, ram_delta_mb=5120,
        poll_s=0.01,
        gpu_sampler=lambda: 90,          # busy
        ram_sampler=lambda: 1000,        # static
        on_exit=lambda c: exits.append(c),
    )
    wd.start()
    wd.beat()  # post-load beat (already armed at start); collapse to interval_s
    time.sleep(0.4)  # many interval_s windows
    wd.stop()
    assert exits == [], "a busy GPU must never be killed"
    assert node_queue.get_node_job(job_id)["status"] == "running"


def test_gpu_health_does_not_trip_while_ram_moves_even_if_gpu_idle():
    """(b) GPU idle but RAM climbs > ram_delta_mb each window ⇒ healthy work
    (staging / decode) ⇒ never trips."""
    job_id = _gpu_health_job()
    exits: list[int] = []
    # RAM grows by 6 GB every read — well over the 5 GB delta.
    ram_seq = iter(range(0, 1_000_000, 6144))
    wd = claim_worker.GpuHealthWatchdog(
        job_id=job_id, interval_s=0.05, idle_pct=5, ram_delta_mb=5120,
        poll_s=0.01,
        gpu_sampler=lambda: 0,           # idle
        ram_sampler=lambda: next(ram_seq),
        on_exit=lambda c: exits.append(c),
    )
    wd.start()
    wd.beat()  # post-load beat (already armed at start); collapse to interval_s
    time.sleep(0.4)
    wd.stop()
    assert exits == [], "RAM moving > delta must keep the job alive even at 0% GPU"
    assert node_queue.get_node_job(job_id)["status"] == "running"


def test_gpu_health_trips_when_gpu_idle_and_ram_static_requeues_under_cap():
    """(c) GPU idle AND RAM static across a window ⇒ wedged ⇒ below the retry cap
    RE-QUEUES the node for a retry (running→queued, lease cleared, priority
    bumped, watchdog_retries++, NO failed event) + hard-exits (code 78). (Old
    behaviour was mark-failed.)"""
    job_id = _gpu_health_job()
    exits: list[int] = []
    wd = claim_worker.GpuHealthWatchdog(
        job_id=job_id, interval_s=0.05, idle_pct=5, ram_delta_mb=5120,
        poll_s=0.01,
        gpu_sampler=lambda: 0,           # idle
        ram_sampler=lambda: 2048,        # static
        on_exit=lambda c: exits.append(c),
    )
    wd.start()
    wd.beat()  # post-load beat → window=interval_s; then no movement ⇒ trip at checkpoint
    _drain(exits)
    wd.stop()
    assert exits and exits[0] == 78, "wedged worker must hard-exit (code 78)"
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "queued", "health trip under cap must RE-QUEUE, not fail"
    assert row["claimed_by"] is None
    assert row["lease_expires_at"] is None
    assert row["priority"] <= 10
    assert row["watchdog_retries"] == 1
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM workflow_dispatch_events WHERE run_id=%s",
            (row["run_id"],),
        )
        assert cur.fetchone()["n"] == 0


def test_gpu_health_armed_at_start_no_beat_survives_load_grace_then_trips():
    """(d) NEW contract: armed AT start (no beat needed). With GPU idle AND RAM
    static it does NOT trip before ``load_grace_s`` (the cold-load grace), and
    DOES trip once the grace elapses — closing the unbounded-load-phase gap and
    the no-model/no-progress GPU node gap. No beat is ever delivered here."""
    job_id = _gpu_health_job()
    exits: list[int] = []
    wd = claim_worker.GpuHealthWatchdog(
        job_id=job_id, interval_s=5.0,   # large: prove the first window is load_grace_s
        load_grace_s=0.2, idle_pct=5, ram_delta_mb=5120,
        poll_s=0.01,
        gpu_sampler=lambda: 0,           # idle
        ram_sampler=lambda: 2048,        # static
        on_exit=lambda c: exits.append(c),
    )
    wd.start()
    # Before the load grace elapses: armed, but the first checkpoint is at
    # now + load_grace_s, so no trip yet (a healthy multi-minute load is safe).
    time.sleep(0.08)
    assert exits == [], "must not trip before load_grace_s even though armed"
    assert node_queue.get_node_job(job_id)["status"] == "running"
    # After the grace, GPU still idle + RAM still static ⇒ genuinely hung ⇒ trip.
    _drain(exits)
    wd.stop()
    assert exits and exits[0] == 78, "hung load must trip after load_grace_s (code 78)"
    # Under the retry cap the trip RE-QUEUES (running→queued), not fail.
    assert node_queue.get_node_job(job_id)["status"] == "queued"


def test_gpu_health_loading_job_does_not_trip_during_grace_even_though_armed():
    """A genuinely-LOADING job during the grace window moves RAM (weights, GBs)
    far more than the delta, so even though the watchdog is armed at start it
    never trips: the "GPU idle AND RAM static" rule sees RAM moving ⇒ healthy.
    This is exactly why arming-at-start is safe for the cold-load phase."""
    job_id = _gpu_health_job()
    exits: list[int] = []
    # RAM climbs by 6 GB each read (> the 5 GB delta) — a model loading weights.
    ram_seq = iter(range(0, 1_000_000, 6144))
    wd = claim_worker.GpuHealthWatchdog(
        job_id=job_id, interval_s=0.05, load_grace_s=0.05,
        idle_pct=5, ram_delta_mb=5120, poll_s=0.01,
        gpu_sampler=lambda: 0,           # idle (load not yet issuing SM work)
        ram_sampler=lambda: next(ram_seq),
        on_exit=lambda c: exits.append(c),
    )
    wd.start()                           # armed at start; NO beat (still loading)
    time.sleep(0.4)                      # many load_grace_s/interval_s windows
    wd.stop()
    assert exits == [], "a loading job (RAM moving > delta) must not trip while armed"
    assert node_queue.get_node_job(job_id)["status"] == "running"


def test_gpu_health_first_beat_collapses_window_to_interval_s():
    """After the arming-at-start load-grace window, the first beat (executor
    post-load) collapses the next window to interval_s, NOT load_grace_s — so
    once the model is warm the cadence is the tight 5-min (here tiny) interval.
    Proven by: a long load_grace_s that would mask a trip, a beat, then a trip
    that fires on the short interval_s instead of waiting out the grace."""
    job_id = _gpu_health_job()
    exits: list[int] = []
    wd = claim_worker.GpuHealthWatchdog(
        job_id=job_id, interval_s=0.1, load_grace_s=30.0,  # grace ≫ interval
        idle_pct=5, ram_delta_mb=5120, poll_s=0.01,
        gpu_sampler=lambda: 0,           # idle
        ram_sampler=lambda: 2048,        # static
        on_exit=lambda c: exits.append(c),
    )
    wd.start()
    wd.beat()                            # post-load: collapse window to interval_s
    # If beat() (wrongly) re-used load_grace_s, this 1 s wait would see no trip.
    # It must trip on the 0.1 s interval_s instead.
    _drain(exits, timeout_s=1.0)
    wd.stop()
    assert exits and exits[0] == 78, "after a beat the window is interval_s, not load_grace_s"
    # Under the retry cap the trip RE-QUEUES (running→queued), not fail.
    assert node_queue.get_node_job(job_id)["status"] == "queued"


def test_gpu_health_beat_resets_window():
    """(e) A progress beat re-anchors the window: an idle-GPU + static-RAM job
    that keeps beating faster than interval_s never trips (the beat is the extra
    liveness signal threaded as the node status_callback)."""
    job_id = _gpu_health_job()
    exits: list[int] = []
    wd = claim_worker.GpuHealthWatchdog(
        job_id=job_id, interval_s=0.1, idle_pct=5, ram_delta_mb=5120,
        poll_s=0.01,
        gpu_sampler=lambda: 0,           # idle
        ram_sampler=lambda: 2048,        # static
        on_exit=lambda c: exits.append(c),
    )
    wd.start()
    wd.beat()  # post-load beat → window=interval_s (collapses the load grace)
    # Beat faster than interval_s for well over one window total — each beat
    # resets the checkpoint so the trip condition is never reached.
    for _ in range(10):
        time.sleep(0.03)
        wd.beat()
    assert exits == [], "beats faster than interval_s must hold the window open"
    assert node_queue.get_node_job(job_id)["status"] == "running"
    wd.stop()


def test_gpu_health_does_not_trip_when_stopped_before_checkpoint():
    job_id = _gpu_health_job()
    exits: list[int] = []
    wd = claim_worker.GpuHealthWatchdog(
        job_id=job_id, interval_s=5.0, idle_pct=5, ram_delta_mb=5120,
        poll_s=0.01,
        gpu_sampler=lambda: 0, ram_sampler=lambda: 2048,
        on_exit=lambda c: exits.append(c),
    )
    wd.start()
    wd.beat()
    time.sleep(0.1)
    wd.stop()
    time.sleep(0.05)
    assert exits == []
    assert node_queue.get_node_job(job_id)["status"] == "running"


def test_gpu_health_threshold_defaults_are_documented():
    """The module-level health thresholds carry the documented defaults
    (300 s interval / 1200 s load-grace / 5 % / 5120 MB = 5 GiB)."""
    assert claim_worker.GPU_HEALTH_INTERVAL_S == 300.0
    assert claim_worker.GPU_HEALTH_LOAD_GRACE_S == 1200.0
    assert claim_worker.GPU_IDLE_PCT == 5
    assert claim_worker.GPU_HEALTH_RAM_DELTA_MB == 5120


def test_gpu_health_load_grace_default_when_unset(monkeypatch):
    """The load-grace window is env-overridable via AI_LEADS_GPU_HEALTH_LOAD_GRACE_S
    through the same _env_float helper; unset/junk → 1200 s, set → override."""
    monkeypatch.delenv("AI_LEADS_GPU_HEALTH_LOAD_GRACE_S", raising=False)
    assert claim_worker._env_float("AI_LEADS_GPU_HEALTH_LOAD_GRACE_S", 1200.0) == 1200.0
    monkeypatch.setenv("AI_LEADS_GPU_HEALTH_LOAD_GRACE_S", "600")
    assert claim_worker._env_float("AI_LEADS_GPU_HEALTH_LOAD_GRACE_S", 1200.0) == 600.0
    monkeypatch.setenv("AI_LEADS_GPU_HEALTH_LOAD_GRACE_S", "junk")
    assert claim_worker._env_float("AI_LEADS_GPU_HEALTH_LOAD_GRACE_S", 1200.0) == 1200.0


def test_gpu_health_env_parsers_override_and_fall_back(monkeypatch):
    """The env-parsing helpers read the override when set + valid, fall back to
    the default when unset or junk — so every threshold is env-overridable."""
    # unset → default
    monkeypatch.delenv("AI_LEADS_GPU_IDLE_PCT", raising=False)
    assert claim_worker._env_int("AI_LEADS_GPU_IDLE_PCT", 5) == 5
    monkeypatch.delenv("AI_LEADS_GPU_HEALTH_INTERVAL_S", raising=False)
    assert claim_worker._env_float("AI_LEADS_GPU_HEALTH_INTERVAL_S", 300.0) == 300.0
    # set + valid → override
    monkeypatch.setenv("AI_LEADS_GPU_IDLE_PCT", "9")
    assert claim_worker._env_int("AI_LEADS_GPU_IDLE_PCT", 5) == 9
    monkeypatch.setenv("AI_LEADS_GPU_HEALTH_INTERVAL_S", "42.5")
    assert claim_worker._env_float("AI_LEADS_GPU_HEALTH_INTERVAL_S", 300.0) == 42.5
    monkeypatch.setenv("AI_LEADS_GPU_HEALTH_RAM_DELTA_MB", "1024")
    assert claim_worker._env_int("AI_LEADS_GPU_HEALTH_RAM_DELTA_MB", 5120) == 1024
    # set + junk → default (never crash a worker on a bad env)
    monkeypatch.setenv("AI_LEADS_GPU_IDLE_PCT", "not-a-number")
    assert claim_worker._env_int("AI_LEADS_GPU_IDLE_PCT", 5) == 5


# ── ingest (fetch/load) claim + execute ──────────────────────────────────────


def test_claim_worker_accepts_fetch_and_load_queues():
    for q in ("fetch", "load"):
        w = claim_worker.ClaimWorker(queue=q, host="host-c")
        assert w.queue == q


def test_claim_worker_rejects_unknown_queue():
    with pytest.raises(ValueError):
        claim_worker.ClaimWorker(queue="nonsense", host="h")


def test_run_once_claims_and_executes_ingest_job():
    """A fetch claim worker claims the next queued ingest row, runs it via the
    registered ingest task (no DAG run, no cancel-watcher), marks it
    completed."""
    ran: list[str] = []
    queue_workflows.register_ingest_task(
        "run_fetch_all", lambda reason: ran.append(reason) or {"ok": 1},
    )
    job_id = node_queue.enqueue_ingest_job(
        task_name="run_fetch_all", queue="fetch", reason="tick",
    )
    worker = claim_worker.ClaimWorker(queue="fetch", host="host-c")
    assert worker.run_once() is True
    assert ran == ["tick"]
    row = node_queue.get_ingest_job(job_id)
    assert row["status"] == "completed"
    assert row["claimed_by"] == "host-c"


def test_run_once_ingest_returns_false_when_empty():
    worker = claim_worker.ClaimWorker(queue="load", host="host-c")
    assert worker.run_once() is False


def test_budget_for_fetch_and_load():
    assert claim_worker.budget_for({"queue": "fetch"}) == claim_worker.FETCH_BUDGET_S
    assert claim_worker.budget_for({"queue": "load"}) == claim_worker.LOAD_BUDGET_S


# ── host-configurable ingest queue names (G1) ────────────────────────────────


def test_claim_worker_accepts_host_configured_ingest_queue():
    queue_workflows.configure(ingest_queues=frozenset({"hydro", "hydraulic", "corrdiff"}))
    w = claim_worker.ClaimWorker(queue="hydraulic", host="host-c")
    assert w.queue == "hydraulic"
    assert w._is_ingest is True
    assert w._wake_channel == "ingest_job_ready"


def test_claim_worker_rejects_queue_not_in_configured_set():
    queue_workflows.configure(ingest_queues=frozenset({"hydraulic"}))
    with pytest.raises(ValueError):
        claim_worker.ClaimWorker(queue="fetch", host="h")  # narrowed out


def test_configure_rejects_reserved_queue_names():
    with pytest.raises(ValueError):
        queue_workflows.configure(ingest_queues=frozenset({"hydro", "gpu"}))


def test_budget_for_custom_ingest_queue_uses_config_default():
    queue_workflows.configure(ingest_default_budget_s=1234)
    assert claim_worker.budget_for({"queue": "hydraulic"}) == 1234


def test_await_schema_custom_ingest_queue_requires_v8(monkeypatch):
    from queue_workflows import db
    queue_workflows.configure(ingest_queues=frozenset({"hydro"}))
    seen: dict = {}
    monkeypatch.setattr(
        db, "wait_for_schema", lambda mv, **kw: seen.update(min_version=mv) or mv,
    )
    claim_worker.ClaimWorker(queue="hydro", host="h").await_schema()
    assert seen["min_version"] == 8


def test_ingest_lease_renewer_extends_lease():
    queue_workflows.register_ingest_task("run_fetch_all", lambda reason: {})
    job_id = node_queue.enqueue_ingest_job(task_name="run_fetch_all", queue="fetch")
    node_queue.claim_next_ingest_job("fetch", host="host-x", lease_s=600)
    before = node_queue.get_ingest_job(job_id)["lease_expires_at"]

    renewer = claim_worker.LeaseRenewer(
        job_id=job_id, claimed_by="host-x", lease_s=600, interval_s=0.05,
        table="ingest_jobs",
    )
    renewer.start()
    time.sleep(0.2)
    renewer.stop()
    after = node_queue.get_ingest_job(job_id)["lease_expires_at"]
    assert after > before


def test_run_once_gpu_passes_current_model_for_affinity(monkeypatch):
    run_id = _make_run()
    node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="x", queue="gpu",
        required_model="sdxl",
    )

    seen: dict = {}
    real_claim = node_queue.claim_next_gpu_job

    def spy_claim(worker_lane=0, current_model=None, **kw):
        seen["current_model"] = current_model
        return real_claim(worker_lane, current_model, **kw)

    monkeypatch.setattr(node_queue, "claim_next_gpu_job", spy_claim)

    class _Cache:
        current_model = "sdxl"

        def require_model(self, model_id):
            return object()

    worker = claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=_Cache())
    worker.run_once()
    assert seen["current_model"] == "sdxl"


# ── startup schema-readiness gate ────────────────────────────────────────────


def test_await_schema_waits_for_required_version(monkeypatch):
    """``await_schema`` delegates to ``db.wait_for_schema(min_version)`` with the
    queue's required engine version (cpu/gpu → 6, ingest queues → 8)."""
    from queue_workflows import db

    seen: dict = {}

    def fake_wait(min_version, **kw):
        seen["min_version"] = min_version
        return min_version

    monkeypatch.setattr(db, "wait_for_schema", fake_wait)

    claim_worker.ClaimWorker(queue="cpu", host="h").await_schema()
    assert seen["min_version"] == 6
    claim_worker.ClaimWorker(queue="gpu", host="h").await_schema()
    assert seen["min_version"] == 6
    claim_worker.ClaimWorker(queue="fetch", host="h").await_schema()
    assert seen["min_version"] == 8
    claim_worker.ClaimWorker(queue="load", host="h").await_schema()
    assert seen["min_version"] == 8


@pytest.mark.pg_only
def test_run_forever_awaits_schema_before_listening(monkeypatch):
    order: list[str] = []

    worker = claim_worker.ClaimWorker(queue="cpu", host="h")

    # The schema gate runs first; do NOT stop here — stopping during await_schema
    # would short-circuit ``db.listen_with_reconnect`` (which checks the stop flag
    # BEFORE issuing LISTEN), so LISTEN would never fire. Instead stop on the first
    # drain, AFTER the LISTEN has been issued, so run_forever issues LISTEN exactly
    # once and then tears down. (This is what the old test got wrong once LISTEN was
    # refactored into listen_with_reconnect.)
    monkeypatch.setattr(worker, "await_schema", lambda: order.append("await_schema"))
    monkeypatch.setattr(worker, "_park_until_enabled", lambda: True)
    monkeypatch.setattr(
        worker, "run_once",
        lambda: (worker.stop(), order.append("run_once"), False)[-1],
    )

    import psycopg

    class _FakeListen:
        def execute(self, *a, **kw): order.append("listen")
        def notifies(self, *a, **kw): return iter(())
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(psycopg, "connect", lambda *a, **kw: _FakeListen())

    worker.run_forever()
    assert order[0] == "await_schema"
    assert "listen" in order
    assert order.index("await_schema") < order.index("listen")
    # LISTEN happens before the queue is drained (run_once).
    assert order.index("listen") < order.index("run_once")


def test_run_forever_gpu_starts_and_stops_llm_backend_factory(monkeypatch):
    """The minimal LLM-factory hook: a GPU worker arms the backend factory's
    config-change LISTEN invalidator after its heartbeat and stops it on teardown.
    Driven through run_forever with a fake LISTEN + immediate stop, asserting the
    gpu-gated factory.start()/stop() both fire (spied on the factory module)."""
    from queue_workflows.llm_backends import factory as llm_factory

    calls: list[str] = []
    monkeypatch.setattr(llm_factory, "start", lambda: calls.append("start"))
    monkeypatch.setattr(llm_factory, "stop", lambda: calls.append("stop"))

    worker = claim_worker.ClaimWorker(queue="gpu", host="host-a")
    # Skip the real schema gate, the hw sampler, and the boot park-gate; stop the
    # loop on the first claim so run_forever drains once then tears down.
    monkeypatch.setattr(worker, "await_schema", lambda: None)
    monkeypatch.setattr(worker, "_park_until_enabled", lambda: True)
    monkeypatch.setattr(worker, "run_once", lambda: worker.stop() or False)

    import queue_workflows.hw_metrics as hw_metrics

    class _Sampler:
        def stop(self): ...
        def join(self, timeout=None): ...

    monkeypatch.setattr(
        hw_metrics, "start_hw_metrics_sampler_flocked", lambda: _Sampler(),
    )

    import psycopg

    class _FakeListen:
        def execute(self, *a, **kw): ...
        def notifies(self, *a, **kw): return iter(())
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(psycopg, "connect", lambda *a, **kw: _FakeListen())

    worker.run_forever()
    assert calls == ["start", "stop"], (
        "a gpu worker must arm the LLM factory after the heartbeat and stop it on "
        "teardown"
    )


def test_run_forever_cpu_does_not_touch_llm_backend_factory(monkeypatch):
    """The hook is GPU-only: a CPU worker must never start/stop the LLM factory
    (no co-tenant VLM on a CPU worker)."""
    from queue_workflows.llm_backends import factory as llm_factory

    calls: list[str] = []
    monkeypatch.setattr(llm_factory, "start", lambda: calls.append("start"))
    monkeypatch.setattr(llm_factory, "stop", lambda: calls.append("stop"))

    worker = claim_worker.ClaimWorker(queue="cpu", host="host-c")
    monkeypatch.setattr(worker, "await_schema", lambda: None)
    monkeypatch.setattr(worker, "_park_until_enabled", lambda: True)
    monkeypatch.setattr(worker, "run_once", lambda: worker.stop() or False)

    import psycopg

    class _FakeListen:
        def execute(self, *a, **kw): ...
        def notifies(self, *a, **kw): return iter(())
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(psycopg, "connect", lambda *a, **kw: _FakeListen())

    worker.run_forever()
    assert calls == [], "a CPU worker must not touch the LLM backend factory"


def test_wait_for_schema_returns_when_version_present(monkeypatch):
    from queue_workflows import db

    monkeypatch.setattr(db, "current_schema_version", lambda **kw: 7)
    slept: list[float] = []
    got = db.wait_for_schema(6, sleep_fn=lambda s: slept.append(s))
    assert got == 7
    assert slept == []


def test_wait_for_schema_polls_until_ready(monkeypatch):
    from queue_workflows import db

    versions = iter([0, 0, 6])
    monkeypatch.setattr(db, "current_schema_version", lambda **kw: next(versions))
    slept: list[float] = []
    got = db.wait_for_schema(6, poll_s=0.01, sleep_fn=lambda s: slept.append(s))
    assert got == 6
    assert len(slept) == 2


def test_wait_for_schema_times_out(monkeypatch):
    from queue_workflows import db

    monkeypatch.setattr(db, "current_schema_version", lambda **kw: 0)
    with pytest.raises(TimeoutError):
        db.wait_for_schema(6, timeout_s=0.0, sleep_fn=lambda s: None)


# ── worker capacity heartbeat ────────────────────────────────────────────────


def _heartbeat_row(host: str, queue: str):
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT host_label, queue, concurrency, current_model, known_models "
            "FROM worker_heartbeats WHERE host_label=%s AND queue=%s",
            (host, queue),
        )
        return cur.fetchone()


@pytest.fixture(autouse=True)
def _wipe_heartbeats():
    with connection() as c, c.cursor() as cur:
        cur.execute("DELETE FROM worker_heartbeats")
    yield
    with connection() as c, c.cursor() as cur:
        cur.execute("DELETE FROM worker_heartbeats")


@pytest.fixture
def _heartbeat_enabled(monkeypatch):
    monkeypatch.delenv("AI_LEADS_DISABLE_WORKER_HEARTBEAT", raising=False)


def test_heartbeat_emit_once_upserts_cpu_row(_heartbeat_enabled):
    emitter = claim_worker.HeartbeatEmitter(queue="cpu", host_label="host-c")
    emitter.emit_once()
    row = _heartbeat_row("host-c", "cpu")
    assert row is not None
    assert row["concurrency"] == 1
    assert row["current_model"] is None


def test_heartbeat_emit_once_gpu_reports_current_model(_heartbeat_enabled):
    class _Cache:
        current_model = None

    cache = _Cache()
    emitter = claim_worker.HeartbeatEmitter(
        queue="gpu", host_label="host-a", model_cache=cache,
    )

    emitter.emit_once()
    assert _heartbeat_row("host-a", "gpu")["current_model"] is None

    cache.current_model = "qwen_edit"
    emitter.emit_once()
    assert _heartbeat_row("host-a", "gpu")["current_model"] == "qwen_edit"

    cache.current_model = None
    emitter.emit_once()
    assert _heartbeat_row("host-a", "gpu")["current_model"] is None


@pytest.mark.parametrize("queue", ["fetch", "load", "hydro"])
def test_heartbeat_emits_for_ingest_queues(queue, _heartbeat_enabled):
    # G5 + migration 0008: ingest-family workers heartbeat too (the cpu/gpu-only
    # CHECK is gone), with current_model NULL, so a host's queue gauge sees them.
    emitter = claim_worker.HeartbeatEmitter(queue=queue, host_label="host-c")
    assert emitter._enabled is True
    emitter.emit_once()
    row = _heartbeat_row("host-c", queue)
    assert row is not None
    assert row["concurrency"] == 1
    assert row["current_model"] is None


def test_heartbeat_disabled_by_env(monkeypatch):
    monkeypatch.setenv("AI_LEADS_DISABLE_WORKER_HEARTBEAT", "1")
    emitter = claim_worker.HeartbeatEmitter(queue="cpu", host_label="host-c")
    assert emitter._enabled is False
    emitter.start()
    assert emitter._thread is None
    assert _heartbeat_row("host-c", "cpu") is None


def test_heartbeat_thread_refreshes_then_stops(_heartbeat_enabled):
    emitter = claim_worker.HeartbeatEmitter(
        queue="cpu", host_label="host-c", interval_s=0.02,
    )
    emitter.start()
    first = _heartbeat_row("host-c", "cpu")
    assert first is not None

    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT last_seen FROM worker_heartbeats "
            "WHERE host_label='host-c' AND queue='cpu'"
        )
        before = cur.fetchone()["last_seen"]

    # Poll until the daemon's next upsert advances last_seen. A fixed sleep
    # flakes under load: the 0.02s-interval thread may not get scheduled (or two
    # upserts collide on the same now()) inside a tight fixed window.
    after = before
    deadline = time.time() + 3.0
    while time.time() < deadline:
        time.sleep(0.05)
        with connection() as c, c.cursor() as cur:
            cur.execute(
                "SELECT last_seen FROM worker_heartbeats "
                "WHERE host_label='host-c' AND queue='cpu'"
            )
            after = cur.fetchone()["last_seen"]
        if after > before:
            break
    assert after > before

    emitter.stop()
    assert emitter._thread is None


def test_claim_worker_wires_heartbeat_emitter():
    class _Cache:
        current_model = "sdxl"

    w = claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=_Cache())
    assert w.heartbeat._queue == "gpu"
    assert w.heartbeat._host_label == "host-a"
    assert w.heartbeat._current_model() == "sdxl"

    cpu = claim_worker.ClaimWorker(queue="cpu", host="host-c")
    assert cpu.heartbeat._current_model() is None


# ── GPU VLM pool lane (PAR-sized no-model concurrency) ──────────────────────
#
# The GPU claim worker runs TWO lanes: the existing concurrency-1 INLINE lane
# (warm-model diffusion, require_model=True) and a NEW PAR-sized POOL lane for
# no-model VLM jobs (require_model=False — HTTP to a per-host vLLM server that
# batches PAR requests on the GPU). These tests pin the pool lane + the
# disjoint-claim contract WITHOUT spinning the live run_forever LISTEN loop.


class _PoolCache:
    """A GPU model cache spy: records every mark_busy/mark_idle/require_model so a
    test can assert the POOL lane never touches it (a VLM job loads no warm
    model)."""

    def __init__(self, current_model=None):
        self.current_model = current_model
        self.busy_calls = 0
        self.idle_calls = 0
        self.require_calls: list[str] = []

    def require_model(self, model_id):
        self.require_calls.append(model_id)
        return object()

    def mark_busy(self):
        self.busy_calls += 1

    def mark_idle(self):
        self.idle_calls += 1


def test_inline_claim_passes_require_model_true(monkeypatch):
    """The inline lane's ``_claim`` (gpu) narrows to model-backed jobs
    (require_model=True) so it never grabs a no-model VLM row — the disjoint-lane
    contract. (CPU still claims via claim_next_cpu_job; ingest via ingest.)"""
    seen = {}

    def spy(worker_lane, current_model=None, **kw):
        seen.update(kw)
        seen["current_model"] = current_model
        return None

    monkeypatch.setattr(node_queue, "claim_next_gpu_job", spy)
    w = claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=_PoolCache())
    assert w._claim() is None
    assert seen["require_model"] is True


def test_pool_claim_passes_require_model_false(monkeypatch):
    """The pool lane's ``_claim_pool`` narrows to no-model jobs
    (require_model=False, current_model=None) — the VLM lane."""
    seen = {}

    def spy(worker_lane, current_model=None, **kw):
        seen.update(kw)
        seen["current_model"] = current_model
        return None

    monkeypatch.setattr(node_queue, "claim_next_gpu_job", spy)
    w = claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=_PoolCache())
    assert w._claim_pool() is None
    assert seen["require_model"] is False
    assert seen["current_model"] is None


def test_pool_parallelism_reads_llm_config_and_clamps(monkeypatch):
    """PAR for the pool = llm_config_for(host, 'gpu').parallelism, clamped >= 1.
    Read live so a UI change takes effect without a restart."""
    from queue_workflows import worker_control

    w = claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=_PoolCache())

    monkeypatch.setattr(
        worker_control, "llm_config_for",
        lambda h, q: worker_control.LLMConfig(parallelism=8),
    )
    assert w._pool_parallelism() == 8

    # Clamp: a bogus sub-1 value floors at 1 (never a zero-width pool).
    monkeypatch.setattr(
        worker_control, "llm_config_for",
        lambda h, q: worker_control.LLMConfig(parallelism=0),
    )
    assert w._pool_parallelism() == 1

    # A read failure also floors at 1 (never crash the feeder / heartbeat).
    def boom(h, q):
        raise RuntimeError("db blip")

    monkeypatch.setattr(worker_control, "llm_config_for", boom)
    assert w._pool_parallelism() == 1


def test_pool_budget_reserves_one_slot_for_inline_diffusion():
    """A GPU machine's capacity is PAR TOTAL node-jobs. The VLM pool's per-cycle
    budget is the full PAR while the inline diffusion lane is idle, and PAR - 1
    while a diffusion runs inline — so total (1 inline diffusion + pool VLM) never
    exceeds PAR. PAR=1 + a diffusion ⇒ 0 VLM slots (the one slot IS the diffusion).
    Clamped ≥ 0."""
    w = claim_worker.ClaimWorker(queue="gpu", host="host-budget", model_cache=_PoolCache())

    w._inline_running = False
    assert w._pool_budget(4) == 4
    assert w._pool_budget(1) == 1

    w._inline_running = True
    assert w._pool_budget(4) == 3   # diffusion takes one of the 4 → 1 + 3 = 4 total
    assert w._pool_budget(1) == 0   # the single slot IS the diffusion → 0 VLM


def test_run_pool_node_executes_no_model_job_and_does_not_touch_cache():
    """``_run_pool_node`` runs a no-model GPU job to completion WITHOUT touching
    the model cache (no mark_busy/idle, no require_model) — a VLM job loads no
    warm model in-worker."""
    run_id = _make_run()
    ran = {}

    def run(*, out=None, inputs=None, cancel_event=None):
        ran["did"] = True
        return {"context_delta": {"ok": True}}

    _install_fake_node("_cw_pool_vlm", run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="v", node_module="_cw_pool_vlm", queue="gpu",
    )
    job = node_queue.claim_next_gpu_job(0, None, host="host-a", require_model=False)
    assert job is not None and job["id"] == job_id

    cache = _PoolCache()
    w = claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=cache)
    assert w._run_pool_node(job) is True
    assert ran.get("did") is True
    assert node_queue.get_node_job(job_id)["status"] == "completed"
    # The pool lane NEVER touches the warm-model cache.
    assert cache.busy_calls == 0
    assert cache.idle_calls == 0
    assert cache.require_calls == []


def test_run_pool_node_does_not_arm_gpu_or_stall_watchdogs(monkeypatch):
    """``_run_pool_node`` must NOT construct a GpuHealthWatchdog or StallWatchdog
    (those police an in-worker diffusion hang; a VLM job's GPU work is in the
    server, HTTP-bound here). It DOES renew the lease (LeaseRenewer constructed)."""
    constructed: list[str] = []
    for cls in ("GpuHealthWatchdog", "StallWatchdog", "Watchdog", "LeaseRenewer"):
        real = getattr(claim_worker, cls)

        def make_spy(name, real_cls):
            def spy(**kw):
                constructed.append(name)
                return real_cls(**kw)
            return spy

        monkeypatch.setattr(claim_worker, cls, make_spy(cls, real))

    run_id = _make_run()

    def run(*, out=None, inputs=None, cancel_event=None):
        return {"context_delta": {"ok": True}}

    _install_fake_node("_cw_pool_noguard", run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="v", node_module="_cw_pool_noguard", queue="gpu",
    )
    job = node_queue.claim_next_gpu_job(0, None, host="host-a", require_model=False)

    w = claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=_PoolCache())
    assert w._run_pool_node(job) is True
    assert "GpuHealthWatchdog" not in constructed, "pool lane must NOT arm the GPU health watchdog"
    assert "StallWatchdog" not in constructed, "pool lane must NOT arm the stall watchdog"
    assert "Watchdog" not in constructed, "pool lane must NOT arm a wall-clock watchdog"
    assert "LeaseRenewer" in constructed, "pool lane MUST renew the lease"
    assert node_queue.get_node_job(job_id)["status"] == "completed"


def test_run_pool_node_renews_lease_while_running():
    """The pool lane keeps its lease fresh: a long-running no-model job has its
    ``lease_expires_at`` pushed out by the per-job LeaseRenewer (a fast-interval
    renewer so the test doesn't wait the production 10s)."""
    run_id = _make_run()
    started = threading.Event()
    release = threading.Event()

    def run(*, out=None, inputs=None, cancel_event=None):
        started.set()
        release.wait(timeout=5.0)
        return {"context_delta": {}}

    _install_fake_node("_cw_pool_lease", run)
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="v", node_module="_cw_pool_lease", queue="gpu",
    )
    job = node_queue.claim_next_gpu_job(0, None, host="host-a", require_model=False)
    before = _lease_expiry(job_id)

    w = claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=_PoolCache())
    real_renewer = claim_worker.LeaseRenewer

    def fast_renewer(**kw):
        kw["interval_s"] = 0.05
        return real_renewer(**kw)

    claim_worker.LeaseRenewer = fast_renewer
    try:
        t = threading.Thread(target=lambda: w._run_pool_node(job), daemon=True)
        t.start()
        assert started.wait(timeout=5.0)
        deadline = time.time() + 3.0
        after = before
        while time.time() < deadline:
            time.sleep(0.05)
            after = _lease_expiry(job_id)
            if after > before:
                break
        release.set()
        t.join(timeout=5.0)
    finally:
        claim_worker.LeaseRenewer = real_renewer
        release.set()
    assert after > before, "pool lane must renew the lease while the job runs"
    assert node_queue.get_node_job(job_id)["status"] == "completed"


def test_run_pool_node_parks_input_node_via_outbox():
    """An ``__input__`` sentinel in the pool lane parks via the durable outbox
    (mark awaiting_input + dispatch event), exactly like the inline lane — never
    handed to execute_node (which would ModuleNotFoundError)."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="i", node_module="__input__choose_one", queue="gpu",
    )
    job = node_queue.claim_next_gpu_job(0, None, host="host-a", require_model=False)
    assert job is not None

    w = claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=_PoolCache())
    assert w._run_pool_node(job) is True
    assert node_queue.get_node_job(job_id)["status"] == "awaiting_input"


def test_pool_feeder_runs_up_to_par_concurrently_never_over(monkeypatch):
    """The feeder keeps up to PAR no-model jobs IN FLIGHT at once — and NEVER
    more than PAR submitted. A fake node blocks on an event; we assert exactly
    PAR are running concurrently while (PAR + extra) jobs are queued, then
    release them and confirm all complete."""
    from queue_workflows import worker_control

    PAR = 3
    EXTRA = 2
    monkeypatch.setattr(
        worker_control, "llm_config_for",
        lambda h, q: worker_control.LLMConfig(parallelism=PAR),
    )

    inflight = {"n": 0, "max": 0}
    inflight_lock = threading.Lock()
    release = threading.Event()

    def run(*, out=None, inputs=None, cancel_event=None):
        with inflight_lock:
            inflight["n"] += 1
            inflight["max"] = max(inflight["max"], inflight["n"])
        release.wait(timeout=10.0)
        with inflight_lock:
            inflight["n"] -= 1
        return {"context_delta": {}}

    _install_fake_node("_cw_pool_block", run)
    run_id = _make_run()
    ids = [
        node_queue.enqueue_node_job(
            run_id=run_id, node_id=f"v{i}", node_module="_cw_pool_block",
            queue="gpu",
        )
        for i in range(PAR + EXTRA)
    ]

    w = claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=_PoolCache())
    w._start_pool_lane()
    try:
        # Wait until PAR jobs are concurrently in flight.
        deadline = time.time() + 8.0
        while time.time() < deadline:
            with inflight_lock:
                if inflight["n"] >= PAR:
                    break
            time.sleep(0.02)
        with inflight_lock:
            assert inflight["n"] == PAR, f"expected {PAR} concurrent, saw {inflight['n']}"
        # The worker's accounting must never exceed PAR submitted.
        assert w._pool_inflight <= PAR
        # The EXTRA jobs are still queued (not over-claimed).
        queued = [i for i in ids if node_queue.get_node_job(i)["status"] == "queued"]
        assert len(queued) == EXTRA, "feeder over-claimed past PAR"
        release.set()
        # All jobs eventually complete as slots free up.
        deadline = time.time() + 10.0
        while time.time() < deadline:
            done = [i for i in ids if node_queue.get_node_job(i)["status"] == "completed"]
            if len(done) == len(ids):
                break
            time.sleep(0.05)
    finally:
        release.set()
        w.stop()
        w._stop_pool_lane()
    assert inflight["max"] == PAR, f"max concurrency was {inflight['max']}, expected {PAR}"
    done = [i for i in ids if node_queue.get_node_job(i)["status"] == "completed"]
    assert len(done) == len(ids)


def test_pool_feeder_does_not_claim_model_backed_jobs(monkeypatch):
    """Lane isolation: the pool feeder claims ONLY no-model jobs. A model-backed
    (diffusion) job queued alongside is left untouched for the inline lane."""
    from queue_workflows import worker_control
    monkeypatch.setattr(
        worker_control, "llm_config_for",
        lambda h, q: worker_control.LLMConfig(parallelism=2),
    )

    run_id = _make_run()
    release = threading.Event()

    def run(*, out=None, inputs=None, cancel_event=None):
        release.wait(timeout=5.0)
        return {"context_delta": {}}

    _install_fake_node("_cw_pool_only", run)
    vlm_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="vlm", node_module="_cw_pool_only", queue="gpu",
    )
    diffusion_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="diff", node_module="_cw_pool_only", queue="gpu",
        required_model="sdxl",
    )

    w = claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=_PoolCache())
    w._start_pool_lane()
    try:
        deadline = time.time() + 6.0
        while time.time() < deadline:
            if node_queue.get_node_job(vlm_id)["status"] in ("running", "completed"):
                break
            time.sleep(0.02)
        # VLM job claimed by the pool; the diffusion job is NOT (stays queued —
        # only the inline lane, which we did not start, would take it).
        assert node_queue.get_node_job(vlm_id)["status"] in ("running", "completed")
        assert node_queue.get_node_job(diffusion_id)["status"] == "queued", (
            "pool feeder must NOT claim a model-backed diffusion job"
        )
        release.set()
    finally:
        release.set()
        w.stop()
        w._stop_pool_lane()


# ── heartbeat advertises max(1, PAR) for GPU (= the machine's GPU concurrency) ─


def test_gpu_heartbeat_concurrency_is_parallelism(_heartbeat_enabled, monkeypatch):
    """The GPU heartbeat advertises ``concurrency = max(1, PAR)`` — the machine's
    GPU job concurrency, i.e. its VLM pool size (worker_controls.llm_parallelism).
    So a PAR-4 vLLM machine shows GPU x/4 in the consumer pill; an ollama machine
    (PAR 1) shows x/1. NOT 1+PAR — the inline diffusion job is one of those slots,
    not an extra one."""
    from queue_workflows import worker_control
    monkeypatch.setattr(
        worker_control, "llm_config_for",
        lambda h, q: worker_control.LLMConfig(parallelism=4),
    )

    w = claim_worker.ClaimWorker(queue="gpu", host="host-hb", model_cache=_PoolCache())
    w.heartbeat.emit_once()
    row = _heartbeat_row("host-hb", "gpu")
    assert row is not None
    assert row["concurrency"] == 4


def test_cpu_heartbeat_concurrency_stays_one(_heartbeat_enabled):
    """CPU (and ingest) heartbeats advertise concurrency=1 — the same single
    structural slot GPU now advertises (the VLM pool is no longer folded in)."""
    w = claim_worker.ClaimWorker(queue="cpu", host="host-hb2")
    w.heartbeat.emit_once()
    row = _heartbeat_row("host-hb2", "cpu")
    assert row is not None
    assert row["concurrency"] == 1


# ── watchdog-trip forensic node-events (migration 0011), from the WATCHDOG side ─
# Every watchdog trip emits TWO forensic node-events in one logical trip: the
# cause event (stall_trip / gpu_health_trip / budget_trip) recorded by
# _watchdog_trip BEFORE the requeue-vs-fail decision, then the OUTCOME event
# (requeued under cap, or failed at cap) ridden in by _requeue_job_and_exit /
# _fail_job_and_exit. test_node_events.py covers only the node_queue layer + the
# execute_node twin; nothing asserts these from the live watchdog path. Without
# these tests a deleted/mislabeled _trip_event_type, a dropped 'requeued' emit,
# or a wrong attempt would pass every existing test.


def test_watchdog_trip_records_forensic_and_requeued_node_events():
    """A StallWatchdog trip under the cap appends BOTH a 'stall_trip' cause event
    (attempt == the pre-requeue watchdog_retries, 0) and a 'requeued' outcome
    event (attempt == bumped retries 1, detail.retry == '1', detail.cap == the
    configured cap) — and NO 'failed' event, because the run stays alive."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
        required_model="qwen_edit", priority=100,
    )
    node_queue.claim_next_gpu_job(0, host="h")

    exits: list[int] = []
    wd = _wedged_stall_watchdog(job_id, on_exit=lambda c: exits.append(c))
    wd.start()
    wd.beat()
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    wd.stop()
    assert exits and exits[0] == 76

    events = _node_events(job_id)
    types = [e["event_type"] for e in events]
    assert "stall_trip" in types, "the cause event must be recorded from the trip site"
    assert "requeued" in types, "the under-cap outcome event must be recorded"
    assert "failed" not in types, "an under-cap trip must NOT record a 'failed' event"

    trip = next(e for e in events if e["event_type"] == "stall_trip")
    assert trip["attempt"] == 0, "the cause event ties to the attempt that wedged (0)"

    requeued = next(e for e in events if e["event_type"] == "requeued")
    assert requeued["attempt"] == 1, "the requeued event carries the bumped attempt"
    cap = claim_worker._watchdog_max_retries()
    assert requeued["detail"].get("retry") == 1
    assert requeued["detail"].get("cap") == cap, (
        "the requeued event must record the retry cap so an operator sees N/cap"
    )


def test_gpu_health_trip_records_gpu_health_trip_node_event():
    """A GpuHealthWatchdog trip labels its cause event 'gpu_health_trip' (not the
    stall/budget label) — guards _trip_event_type's gpu branch."""
    job_id = _gpu_health_job()
    exits: list[int] = []
    wd = claim_worker.GpuHealthWatchdog(
        job_id=job_id, interval_s=0.05, idle_pct=5, ram_delta_mb=5120, poll_s=0.01,
        gpu_sampler=lambda: 0, ram_sampler=lambda: 2048,
        on_exit=lambda c: exits.append(c),
    )
    wd.start()
    wd.beat()
    _drain(exits)
    wd.stop()
    assert exits and exits[0] == 78

    types = [e["event_type"] for e in _node_events(job_id)]
    assert "gpu_health_trip" in types, "a health trip must record a 'gpu_health_trip' cause"
    assert "requeued" in types, "under the cap it also records the requeued outcome"
    assert "stall_trip" not in types and "budget_trip" not in types, (
        "the health trip must not be mislabeled as stall/budget"
    )


def test_budget_watchdog_trip_records_budget_trip_node_event():
    """A wall-clock Watchdog trip labels its cause event 'budget_trip' — guards
    the _trip_event_type default (neither 'stall' nor 'gpu' in the label)."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu", priority=100,
    )
    node_queue.claim_next_cpu_job(0, host="h")

    exits: list[int] = []
    wd = claim_worker.Watchdog(
        job_id=job_id, budget_s=0.05,
        on_exit=lambda code: exits.append(code), poll_s=0.01,
    )
    wd.start()
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    wd.stop()
    assert exits and exits[0] == 75

    types = [e["event_type"] for e in _node_events(job_id)]
    assert "budget_trip" in types, "a wall-clock trip must record a 'budget_trip' cause"
    assert "requeued" in types
    assert "stall_trip" not in types and "gpu_health_trip" not in types


def test_watchdog_trip_at_cap_records_cause_and_failed_node_events():
    """At the cap the trip path FAILS the run, so the forensic log carries the
    cause event AND a 'failed' event (NOT 'requeued'); the failed event ties to
    the final attempt (== the cap, 3)."""
    run_id, job_id = _running_gpu_job_with_retries(3)  # == default cap
    exits: list[int] = []
    wd = claim_worker.GpuHealthWatchdog(
        job_id=job_id, interval_s=0.05, idle_pct=5, ram_delta_mb=5120, poll_s=0.01,
        gpu_sampler=lambda: 0, ram_sampler=lambda: 2048,
        on_exit=lambda c: exits.append(c),
    )
    wd.start()
    wd.beat()
    _drain(exits)
    wd.stop()
    assert exits and exits[0] == 78

    events = _node_events(job_id)
    types = [e["event_type"] for e in events]
    assert "gpu_health_trip" in types, "the cause event is recorded even at the cap"
    assert "failed" in types, "at the cap the outcome event must be 'failed'"
    assert "requeued" not in types, "at the cap there is no retry, so no 'requeued'"

    failed = next(e for e in events if e["event_type"] == "failed")
    assert failed["attempt"] == 3, "the failed event ties to the final (capped) attempt"


# ── flaky-probe robustness: a missing RAM reading must NOT save a wedged worker ─
# Source documents (claim_worker.py: _evaluate_window_locked / _confirm_wedged)
# that a None RAM reading is treated as "no movement", so a flaky cgroup probe
# can't keep a GPU-idle + wedged worker alive forever. Every other gpu-health /
# stall trip test feeds a concrete integer ram_sampler, so the None branch is
# untested: a refactor that defaulted ram_moved=True on None would never kill an
# idle worker whose RAM file is unreadable.


def test_gpu_health_trips_when_gpu_idle_and_ram_probe_unreadable():
    """GPU idle AND the RAM probe returns None (unreadable cgroup file) ⇒ a None
    reading is 'no movement', so the worker is still confirmed wedged and the
    health watchdog trips (exit 78 + under-cap re-queue). A None must NOT be
    read as RAM-moving (which would spare a genuinely hung worker)."""
    job_id = _gpu_health_job()
    exits: list[int] = []
    wd = claim_worker.GpuHealthWatchdog(
        job_id=job_id, interval_s=0.05, idle_pct=5, ram_delta_mb=5120, poll_s=0.01,
        gpu_sampler=lambda: 0,
        ram_sampler=lambda: None,        # probe unreadable every read
        on_exit=lambda c: exits.append(c),
    )
    wd.start()
    wd.beat()
    _drain(exits)
    wd.stop()
    assert exits and exits[0] == 78, (
        "idle GPU + no RAM signal must still be treated as wedged (trip 78)"
    )
    assert node_queue.get_node_job(job_id)["status"] == "queued", (
        "under the cap the unreadable-RAM trip RE-QUEUES the node"
    )


def test_stall_confirm_wedged_true_when_gpu_idle_and_ram_probe_unreadable():
    """The StallWatchdog twin: GPU idle + ram_sampler returns None ⇒ ram_moved is
    False (ram_anchor None), so _confirm_wedged returns True and the no-beat
    timeout TRIPS (exit 76). A flaky RAM probe can't keep a stalled worker alive."""
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
        required_model="qwen_edit",
    )
    node_queue.claim_next_gpu_job(0, host="h")

    exits: list[int] = []
    wd = claim_worker.StallWatchdog(
        job_id=job_id, stall_timeout_s=0.05, on_exit=lambda c: exits.append(c),
        poll_s=0.01, confirm_samples=2, confirm_poll_s=0.0,
        idle_pct=5, ram_delta_mb=5120,
        gpu_sampler=lambda: 0,
        ram_sampler=lambda: None,        # probe unreadable
    )
    # Direct predicate check: idle GPU + None RAM ⇒ confirmed wedged.
    assert wd._confirm_wedged() is True, (
        "GPU idle + unreadable RAM (None) must confirm as wedged, not spared"
    )
    # And end-to-end the no-beat timeout trips through to a hard exit.
    wd.start()
    wd.beat()
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    wd.stop()
    assert exits and exits[0] == 76
    assert node_queue.get_node_job(job_id)["status"] == "queued"
