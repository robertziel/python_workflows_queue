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


# ── budget table ───────────────────────────────────────────────────────────


def test_budget_for_video_model_is_1800s():
    job = {"queue": "gpu", "required_model": "wan_i2v"}
    assert claim_worker.budget_for(job) == 1800


def test_budget_for_generic_gpu_is_8100s():
    job = {"queue": "gpu", "required_model": "qwen_edit"}
    assert claim_worker.budget_for(job) == 8100


def test_budget_for_gpu_without_model_is_8100s():
    job = {"queue": "gpu", "required_model": None}
    assert claim_worker.budget_for(job) == 8100


def test_budget_for_cpu_is_2100s():
    job = {"queue": "cpu", "node_module": "geocode"}
    assert claim_worker.budget_for(job) == 2100


def test_budget_for_input_node_is_120s():
    job = {"queue": "cpu", "node_module": "__input__choose_one"}
    assert claim_worker.budget_for(job) == 120


# ── watchdog ─────────────────────────────────────────────────────────────


def test_watchdog_trips_and_marks_failed_at_budget():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
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

    assert exits, "watchdog must have hard-exited on budget overrun"
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "failed"
    assert "wall-clock budget" in (row["error"] or "")


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


def test_stall_watchdog_trips_after_first_beat_then_silence():
    run_id = _make_run()
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="gpu",
        required_model="qwen_edit",
    )
    node_queue.claim_next_gpu_job(0, host="h")

    exits: list[int] = []
    wd = claim_worker.StallWatchdog(
        job_id=job_id, stall_timeout_s=0.05,
        on_exit=lambda code: exits.append(code), poll_s=0.01,
    )
    wd.start()
    wd.beat()  # the executor's after-model-load beat arms enforcement
    deadline = time.time() + 3.0
    while time.time() < deadline and not exits:
        time.sleep(0.02)
    wd.stop()

    assert exits, "stall watchdog must hard-exit when progress stops after arming"
    row = node_queue.get_node_job(job_id)
    assert row["status"] == "failed"
    assert "no progress" in (row["error"] or "").lower()


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
    ``status_callback=None``, so no node could ever beat.)"""
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
    )

    class _Cache:
        current_model = None

        def require_model(self, model_id):
            return object()

        def mark_busy(self): ...
        def mark_idle(self): ...

    worker = claim_worker.ClaimWorker(queue="gpu", host="box-a2", model_cache=_Cache())
    assert worker.run_once() is True
    assert seen.get("is_callable") is True
    assert node_queue.get_node_job(job_id)["status"] == "completed"


def test_stall_watchdog_not_armed_for_video_models():
    """A video-model gpu job is NOT stall-policed: video backends step slowly
    and report progress only per beat-segment (minutes apart), which would
    false-trip the 120 s window. Their backstop is the 1800 s wall-clock budget.
    So a progress-reporting node on a video model receives status_callback=None
    (watchdog not armed). Regression guard: a live fence_render_narrative video
    render was hard-stopped at 120 s by an over-eager arm."""
    run_id = _make_run()
    seen: dict = {}

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

    worker = claim_worker.ClaimWorker(queue="gpu", host="box-a", model_cache=_Cache())
    assert worker.run_once() is True
    assert seen.get("status_callback") is None, "video model must NOT arm the stall watchdog"
    assert node_queue.get_node_job(job_id)["status"] == "completed"


def test_run_once_skips_job_under_cancelled_run():
    run_id = _make_run(status="cancelled")
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id="n", node_module="x", queue="cpu",
    )
    worker = claim_worker.ClaimWorker(queue="cpu", host="test-host")
    assert worker.run_once() is False
    assert node_queue.get_node_job(job_id)["status"] == "queued"


# ── ingest (fetch/load) claim + execute ──────────────────────────────────────


def test_claim_worker_accepts_fetch_and_load_queues():
    for q in ("fetch", "load"):
        w = claim_worker.ClaimWorker(queue=q, host="box-b")
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
    worker = claim_worker.ClaimWorker(queue="fetch", host="box-b")
    assert worker.run_once() is True
    assert ran == ["tick"]
    row = node_queue.get_ingest_job(job_id)
    assert row["status"] == "completed"
    assert row["claimed_by"] == "box-b"


def test_run_once_ingest_returns_false_when_empty():
    worker = claim_worker.ClaimWorker(queue="load", host="box-b")
    assert worker.run_once() is False


def test_budget_for_fetch_and_load():
    assert claim_worker.budget_for({"queue": "fetch"}) == claim_worker.FETCH_BUDGET_S
    assert claim_worker.budget_for({"queue": "load"}) == claim_worker.LOAD_BUDGET_S


# ── host-configurable ingest queue names (G1) ────────────────────────────────


def test_claim_worker_accepts_host_configured_ingest_queue():
    queue_workflows.configure(ingest_queues=frozenset({"hydro", "hydraulic", "corrdiff"}))
    w = claim_worker.ClaimWorker(queue="hydraulic", host="box-b")
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

    worker = claim_worker.ClaimWorker(queue="gpu", host="box-a", model_cache=_Cache())
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


def test_run_forever_awaits_schema_before_listening(monkeypatch):
    order: list[str] = []

    worker = claim_worker.ClaimWorker(queue="cpu", host="h")

    def fake_await():
        order.append("await_schema")
        worker.stop()

    monkeypatch.setattr(worker, "await_schema", fake_await)
    monkeypatch.setattr(worker, "run_once", lambda: order.append("run_once") or False)

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
    emitter = claim_worker.HeartbeatEmitter(queue="cpu", host_label="box-b")
    emitter.emit_once()
    row = _heartbeat_row("box-b", "cpu")
    assert row is not None
    assert row["concurrency"] == 1
    assert row["current_model"] is None


def test_heartbeat_emit_once_gpu_reports_current_model(_heartbeat_enabled):
    class _Cache:
        current_model = None

    cache = _Cache()
    emitter = claim_worker.HeartbeatEmitter(
        queue="gpu", host_label="box-a", model_cache=cache,
    )

    emitter.emit_once()
    assert _heartbeat_row("box-a", "gpu")["current_model"] is None

    cache.current_model = "qwen_edit"
    emitter.emit_once()
    assert _heartbeat_row("box-a", "gpu")["current_model"] == "qwen_edit"

    cache.current_model = None
    emitter.emit_once()
    assert _heartbeat_row("box-a", "gpu")["current_model"] is None


@pytest.mark.parametrize("queue", ["fetch", "load", "hydro"])
def test_heartbeat_emits_for_ingest_queues(queue, _heartbeat_enabled):
    # G5 + migration 0008: ingest-family workers heartbeat too (the cpu/gpu-only
    # CHECK is gone), with current_model NULL, so a host's queue gauge sees them.
    emitter = claim_worker.HeartbeatEmitter(queue=queue, host_label="box-b")
    assert emitter._enabled is True
    emitter.emit_once()
    row = _heartbeat_row("box-b", queue)
    assert row is not None
    assert row["concurrency"] == 1
    assert row["current_model"] is None


def test_heartbeat_disabled_by_env(monkeypatch):
    monkeypatch.setenv("AI_LEADS_DISABLE_WORKER_HEARTBEAT", "1")
    emitter = claim_worker.HeartbeatEmitter(queue="cpu", host_label="box-b")
    assert emitter._enabled is False
    emitter.start()
    assert emitter._thread is None
    assert _heartbeat_row("box-b", "cpu") is None


def test_heartbeat_thread_refreshes_then_stops(_heartbeat_enabled):
    emitter = claim_worker.HeartbeatEmitter(
        queue="cpu", host_label="box-b", interval_s=0.02,
    )
    emitter.start()
    first = _heartbeat_row("box-b", "cpu")
    assert first is not None

    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT last_seen FROM worker_heartbeats "
            "WHERE host_label='box-b' AND queue='cpu'"
        )
        before = cur.fetchone()["last_seen"]

    time.sleep(0.1)
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT last_seen FROM worker_heartbeats "
            "WHERE host_label='box-b' AND queue='cpu'"
        )
        after = cur.fetchone()["last_seen"]
    assert after > before

    emitter.stop()
    assert emitter._thread is None


def test_claim_worker_wires_heartbeat_emitter():
    class _Cache:
        current_model = "sdxl"

    w = claim_worker.ClaimWorker(queue="gpu", host="box-a", model_cache=_Cache())
    assert w.heartbeat._queue == "gpu"
    assert w.heartbeat._host_label == "box-a"
    assert w.heartbeat._current_model() == "sdxl"

    cpu = claim_worker.ClaimWorker(queue="cpu", host="box-b")
    assert cpu.heartbeat._current_model() is None
