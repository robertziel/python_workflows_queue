"""Worker ON/OFF control — the worker-side enforcement.

Covers (the worker-read side of the feature):
- WorkerControlWatcher.check_once: OFF ⇒ re-queue in-flight + on_exit(79); ON /
  absent ⇒ no-op; an OFF row with an unimplemented stop_policy falls back to hard;
- the watcher THREAD trips on an OFF flip (driven by the NOTIFY), with an injected
  on_exit so the test process survives;
- AI_LEADS_DISABLE_WORKER_CONTROL keeps the watcher inert (tests);
- ClaimWorker.requeue_inflight_for_control re-queues this worker's rows and clears
  its busy-ghost (resume-style, no retry bump);
- ClaimWorker._park_until_enabled: returns at once when ON, and resumes when an
  OFF worker is turned back ON (NOTIFY-driven), without claiming meanwhile.

All ClaimWorkers here use queue='cpu' so no GPU model cache (no torch) is built;
the control machinery is queue-agnostic.
"""

from __future__ import annotations

import threading
import time

from queue_workflows import claim_worker, node_queue, worker_control
from queue_workflows.db import connection
from tests._helpers import make_run


# ── helpers ────────────────────────────────────────────────────────────────


def _running_node_job(host: str, *, queue: str = "cpu") -> str:
    run_id = make_run(workflow_name="_wcw_test")
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue=queue,
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


def _worker(host: str, *, queue: str = "cpu") -> claim_worker.ClaimWorker:
    return claim_worker.ClaimWorker(queue=queue, host=host)


# ── WorkerControlWatcher.check_once ────────────────────────────────────────────


def test_check_once_hard_stops_when_off():
    worker = _worker("host-a")
    job_id = _running_node_job("host-a", queue="cpu")
    worker_control.disable_worker("host-a", "cpu")  # OFF, hard

    exits: list[int] = []
    w = worker_control.WorkerControlWatcher(
        worker=worker, on_exit=lambda code: exits.append(code),
    )
    assert w.check_once() is True
    assert exits == [worker_control.EXIT_CONTROL_HARD_STOP]

    # in-flight job released back to the queue (resume-style)
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "queued"
    assert row["claimed_by"] is None
    assert (row.get("watchdog_retries") or 0) == 0


def test_check_once_noop_when_on_or_absent():
    worker = _worker("host-a")
    exits: list[int] = []
    w = worker_control.WorkerControlWatcher(
        worker=worker, on_exit=lambda code: exits.append(code),
    )
    assert w.check_once() is False             # no row ⇒ ON
    worker_control.enable_worker("host-a", "cpu")
    assert w.check_once() is False             # explicit ON
    assert exits == []


def test_check_once_falls_back_to_hard_for_unknown_policy():
    """An OFF row whose stop_policy isn't (yet) implemented must still stop the
    worker — fall back to hard rather than ignore the operator. The bad policy is
    written directly, bypassing set_worker_control's guard."""
    worker = _worker("fb")
    _running_node_job("fb", queue="cpu")
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO worker_controls (host_label, queue, desired_state, stop_policy) "
            "VALUES ('fb', 'cpu', 'off', 'drain')"
        )
    exits: list[int] = []
    w = worker_control.WorkerControlWatcher(
        worker=worker, on_exit=lambda code: exits.append(code),
    )
    assert w.check_once() is True
    assert exits == [worker_control.EXIT_CONTROL_HARD_STOP]


def test_check_once_scoped_to_this_workers_queue():
    """A cpu worker must not stop because the GPU worker on the same host was
    turned off."""
    worker = _worker("host-c", queue="cpu")
    worker_control.disable_worker("host-c", "gpu")  # gpu off, cpu still on
    exits: list[int] = []
    w = worker_control.WorkerControlWatcher(
        worker=worker, on_exit=lambda code: exits.append(code),
    )
    assert w.check_once() is False
    assert exits == []


# ── WorkerControlWatcher thread ────────────────────────────────────────────────


def test_watcher_thread_trips_on_off_flip(monkeypatch):
    monkeypatch.setenv("AI_LEADS_WORKER_CONTROL_POLL_S", "0.2")
    worker = _worker("box-t")
    _running_node_job("box-t", queue="cpu")

    fired = threading.Event()
    codes: list[int] = []

    def _fake_exit(code: int) -> None:
        codes.append(code)
        fired.set()

    w = worker_control.WorkerControlWatcher(
        worker=worker, on_exit=_fake_exit, poll_s=0.2,
    )
    w.start()
    try:
        worker_control.disable_worker("box-t", "cpu")  # fires the NOTIFY
        assert fired.wait(timeout=5.0), "watcher did not trip on OFF"
        assert codes == [worker_control.EXIT_CONTROL_HARD_STOP]
    finally:
        w.stop()


def test_watcher_disabled_via_env(monkeypatch):
    monkeypatch.setenv("AI_LEADS_DISABLE_WORKER_CONTROL", "1")
    worker = _worker("dis")
    w = worker_control.WorkerControlWatcher(worker=worker, on_exit=lambda c: None)
    w.start()
    assert w._thread is None  # disabled ⇒ no daemon started
    w.stop()


# ── ClaimWorker.requeue_inflight_for_control ───────────────────────────────────


def test_requeue_inflight_requeues_and_returns_count():
    worker = _worker("rq")
    _running_node_job("rq", queue="cpu")
    _running_node_job("rq", queue="cpu")
    assert worker.requeue_inflight_for_control() == 2


def test_requeue_inflight_clears_busy_ghost(monkeypatch):
    """It nulls this worker's current_model (the GPU-busy gauge) before the exit —
    same pre-exit bookkeeping a watchdog trip does (os._exit skips the finally)."""
    worker = _worker("rq2")
    _running_node_job("rq2", queue="cpu")
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        node_queue, "clear_worker_current_model",
        lambda h, q, **k: calls.append((h, q)),
    )
    worker.requeue_inflight_for_control()
    assert calls == [("rq2", "cpu")]


# ── ClaimWorker._park_until_enabled (boot gate) ────────────────────────────────


def test_park_gate_returns_immediately_when_on():
    worker = _worker("parkhost")
    # No control row ⇒ ON ⇒ returns at once (must not block / open a LISTEN).
    worker._park_until_enabled()


def test_park_gate_resumes_when_enabled(monkeypatch):
    """An OFF worker parks (no claim) and resumes the instant it's turned back ON
    — the full park→NOTIFY→resume path."""
    monkeypatch.setenv("AI_LEADS_WORKER_CONTROL_POLL_S", "0.2")
    worker = _worker("parkhost2")
    worker_control.disable_worker("parkhost2", "cpu")  # OFF ⇒ would park

    done = threading.Event()
    th = threading.Thread(
        target=lambda: (worker._park_until_enabled(), done.set()), daemon=True,
    )
    th.start()
    # Still parked while OFF.
    assert not done.wait(timeout=0.5)
    # Turn it back ON — the NOTIFY (or the 0.2 s safety poll) wakes the gate.
    worker_control.enable_worker("parkhost2", "cpu")
    assert done.wait(timeout=5.0), "park gate did not resume after enable"
