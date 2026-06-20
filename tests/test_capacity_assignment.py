"""Capacity-aware GPU model assignment + unassignable red-flag (migration 0015).

Covers the four seams:
  * model_registry.fits_within — which models fit a given VRAM.
  * hw_metrics.total_vram_mb — the machine's total VRAM (max single device).
  * upsert_worker_heartbeat — persists vram_total_mb + fits_models.
  * node_queue.flag_unassignable_gpu_jobs — fleet sweep: red-flag a queued GPU
    model-job no live machine can hold, emit once, clear when assignable / gone.
  * ClaimWorker._claim — the per-machine capacity gate (won't claim a model it
    can't hold; won't fall through to claim-any when capacity is known).
"""

from __future__ import annotations

import uuid

import pytest

from queue_workflows import claim_worker, hw_metrics, model_registry, node_queue
from queue_workflows.db import connection
from queue_workflows.model_registry import ModelSpec
from tests._helpers import make_run


# ── model_registry.fits_within ────────────────────────────────────────────


@pytest.fixture
def _registry():
    """A clean registry with three sized models, restored after the test."""
    saved = dict(model_registry.MODELS)
    model_registry.clear_for_tests()
    model_registry.register(ModelSpec(id="small", loader=lambda: None, est_vram_gb=4.0))
    model_registry.register(ModelSpec(id="mid", loader=lambda: None, est_vram_gb=12.0))
    model_registry.register(ModelSpec(id="huge", loader=lambda: None, est_vram_gb=42.0))
    model_registry.register(ModelSpec(id="free", loader=lambda: None, est_vram_gb=0.0))
    yield
    model_registry.clear_for_tests()
    for spec in saved.values():
        model_registry.register(spec)


def test_fits_within_unknown_capacity_returns_all(_registry):
    # Capacity unknown ⇒ advertise everything (claim-any-capable, no wedge).
    assert model_registry.fits_within(None) == ["free", "huge", "mid", "small"]


def test_fits_within_filters_by_vram(_registry):
    # 16 GB machine: small(4) + mid(12) fit, huge(42) does not; free(0) always.
    assert model_registry.fits_within(16 * 1024) == ["free", "mid", "small"]


def test_fits_within_tiny_machine_only_zero_cost_models(_registry):
    # 1 GB: nothing with a real estimate fits; the est<=0 "free" model still does.
    assert model_registry.fits_within(1 * 1024) == ["free"]


def test_fits_within_headroom_reserves_margin(_registry):
    # mid=12GB needs 12*1024=12288MB raw; with headroom 1.5 it needs 18432MB.
    assert "mid" in model_registry.fits_within(13 * 1024, headroom=1.0)
    assert "mid" not in model_registry.fits_within(13 * 1024, headroom=1.5)


# ── hw_metrics.total_vram_mb ───────────────────────────────────────────────


def test_total_vram_mb_takes_max_single_device(monkeypatch):
    monkeypatch.setattr(
        hw_metrics, "_gpu_probe",
        lambda: [{"vram_total_mb": 24000}, {"vram_total_mb": 81920}],
    )
    assert hw_metrics.total_vram_mb() == 81920


def test_total_vram_mb_none_when_no_gpu(monkeypatch):
    monkeypatch.setattr(hw_metrics, "_gpu_probe", lambda: [])
    assert hw_metrics.total_vram_mb() is None


def test_total_vram_mb_implausibly_small_is_unknown(monkeypatch):
    # An AMD APU's rocm-smi carveout (observed 512 MB) must read as UNKNOWN
    # (None ⇒ fail-open / claim-any), NOT as a 512 MB cap that fits nothing.
    monkeypatch.delenv("AI_LEADS_GPU_VRAM_TOTAL_MB", raising=False)
    monkeypatch.setattr(hw_metrics, "_gpu_probe", lambda: [{"vram_total_mb": 512}])
    assert hw_metrics.total_vram_mb() is None


def test_total_vram_mb_env_override_wins(monkeypatch):
    # Operator-declared capacity is the reliable source on unified-memory GPUs;
    # it overrides whatever the (bogus) probe says.
    monkeypatch.setenv("AI_LEADS_GPU_VRAM_TOTAL_MB", "131072")
    monkeypatch.setattr(hw_metrics, "_gpu_probe", lambda: [{"vram_total_mb": 512}])
    assert hw_metrics.total_vram_mb() == 131072


def test_total_vram_mb_env_override_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("AI_LEADS_GPU_VRAM_TOTAL_MB", "not-a-number")
    monkeypatch.setattr(hw_metrics, "_gpu_probe", lambda: [{"vram_total_mb": 81920}])
    assert hw_metrics.total_vram_mb() == 81920


def test_total_vram_mb_swallows_probe_error(monkeypatch):
    def _boom():
        raise RuntimeError("nvidia-smi gone")
    monkeypatch.setattr(hw_metrics, "_gpu_probe", _boom)
    assert hw_metrics.total_vram_mb() is None


# ── heartbeat persistence ──────────────────────────────────────────────────


def _heartbeat_row(host: str, queue: str = "gpu") -> dict:
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT * FROM worker_heartbeats WHERE host_label=%s AND queue=%s",
            (host, queue),
        )
        return cur.fetchone()


def test_upsert_persists_vram_and_fits():
    node_queue.upsert_worker_heartbeat(
        host_label="cap-host", queue="gpu", concurrency=1,
        vram_total_mb=16384, fits_models=["small", "mid"],
    )
    row = _heartbeat_row("cap-host")
    assert row["vram_total_mb"] == 16384
    assert sorted(row["fits_models"]) == ["mid", "small"]


def test_upsert_defaults_fits_empty_when_omitted():
    node_queue.upsert_worker_heartbeat(
        host_label="cpu-host", queue="cpu", concurrency=1,
    )
    row = _heartbeat_row("cpu-host", "cpu")
    assert row["vram_total_mb"] is None
    assert row["fits_models"] == []


# ── fleet unassignable sweep ───────────────────────────────────────────────


def _queued_gpu_job(model: str) -> tuple[str, str]:
    run_id = make_run(workflow_name="_cap_test")
    job_id = node_queue.enqueue_node_job(
        run_id=run_id, node_id=f"n-{uuid.uuid4().hex[:8]}",
        node_module="some_node", queue="gpu", required_model=model,
    )
    return run_id, job_id


def _fresh_gpu_worker(host: str, fits: list[str], vram_mb: int = 32768) -> None:
    node_queue.upsert_worker_heartbeat(
        host_label=host, queue="gpu", concurrency=1,
        vram_total_mb=vram_mb, fits_models=fits,
    )


def _age_heartbeat(host: str, seconds: int) -> None:
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE worker_heartbeats SET last_seen = now() - make_interval(secs=>%s) "
            "WHERE host_label=%s",
            (seconds, host),
        )


def test_unassignable_flagged_when_no_machine_fits():
    _fresh_gpu_worker("box-a", fits=["small", "mid"], vram_mb=16384)
    _run, job_id = _queued_gpu_job("huge")

    flagged = node_queue.flag_unassignable_gpu_jobs()

    ids = [r["id"] for r in flagged]
    assert job_id in ids
    row = node_queue.get_node_job(job_id)
    assert row["unassignable_at"] is not None
    assert "huge" in row["unassignable_reason"]
    assert "16384" in row["unassignable_reason"]


def test_assignable_job_not_flagged():
    _fresh_gpu_worker("box-b", fits=["small", "mid", "huge"], vram_mb=81920)
    _run, job_id = _queued_gpu_job("huge")

    flagged = node_queue.flag_unassignable_gpu_jobs()

    assert job_id not in [r["id"] for r in flagged]
    assert node_queue.get_node_job(job_id)["unassignable_at"] is None


def test_no_fresh_gpu_worker_is_noop_not_flag():
    # Liveness guard: whole GPU fleet down (stale heartbeat) ⇒ NOT a capacity
    # verdict ⇒ no flag (that's the dead-worker sweep's job).
    _fresh_gpu_worker("box-c", fits=["small"], vram_mb=8192)
    _age_heartbeat("box-c", 120)  # older than the 30 s window
    _run, job_id = _queued_gpu_job("huge")

    flagged = node_queue.flag_unassignable_gpu_jobs()

    assert flagged == []
    assert node_queue.get_node_job(job_id)["unassignable_at"] is None


def test_flag_is_idempotent_returns_only_new():
    _fresh_gpu_worker("box-d", fits=["small"], vram_mb=8192)
    _run, job_id = _queued_gpu_job("huge")

    first = node_queue.flag_unassignable_gpu_jobs()
    assert job_id in [r["id"] for r in first]
    # Second sweep: already flagged ⇒ NOT returned again (no duplicate event).
    second = node_queue.flag_unassignable_gpu_jobs()
    assert job_id not in [r["id"] for r in second]


def test_flag_clears_when_capable_machine_appears():
    _fresh_gpu_worker("box-e", fits=["small"], vram_mb=8192)
    _run, job_id = _queued_gpu_job("huge")
    node_queue.flag_unassignable_gpu_jobs()
    assert node_queue.get_node_job(job_id)["unassignable_at"] is not None

    # A big machine comes online that CAN hold "huge".
    _fresh_gpu_worker("box-big", fits=["small", "mid", "huge"], vram_mb=81920)
    node_queue.flag_unassignable_gpu_jobs()
    assert node_queue.get_node_job(job_id)["unassignable_at"] is None


def test_flag_clears_when_job_leaves_queue():
    _fresh_gpu_worker("box-f", fits=["small"], vram_mb=8192)
    _run, job_id = _queued_gpu_job("huge")
    node_queue.flag_unassignable_gpu_jobs()
    assert node_queue.get_node_job(job_id)["unassignable_at"] is not None

    with connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE workflow_node_jobs SET status='cancelled' WHERE id=%s", (job_id,)
        )
    node_queue.flag_unassignable_gpu_jobs()
    assert node_queue.get_node_job(job_id)["unassignable_at"] is None


# ── per-machine claim gate (ClaimWorker._claim) ────────────────────────────


class _FakeCache:
    current_model = None


def _gpu_worker_with_vram(monkeypatch, vram_mb):
    monkeypatch.setattr(hw_metrics, "total_vram_mb", lambda: vram_mb)
    w = claim_worker.ClaimWorker(queue="gpu", host="claim-host", model_cache=_FakeCache())
    return w


def test_claim_gate_skips_model_too_big(monkeypatch, _registry):
    # A 16 GB machine must NOT claim a "huge" (42 GB) model-job.
    _run, job_id = _queued_gpu_job("huge")
    w = _gpu_worker_with_vram(monkeypatch, 16384)
    claimed = w._claim()
    assert claimed is None
    assert node_queue.get_node_job(job_id)["status"] == "queued"


def test_claim_gate_takes_model_that_fits(monkeypatch, _registry):
    # An 80 GB machine CAN claim "huge".
    _run, job_id = _queued_gpu_job("huge")
    w = _gpu_worker_with_vram(monkeypatch, 81920)
    claimed = w._claim()
    assert claimed is not None and claimed["id"] == job_id
    assert node_queue.get_node_job(job_id)["status"] == "running"


def test_claim_gate_unknown_vram_falls_back_to_claim_any(monkeypatch, _registry):
    # VRAM unknown (no probe) ⇒ fits=all ⇒ claim-any-capable (no wedge).
    _run, job_id = _queued_gpu_job("huge")
    w = _gpu_worker_with_vram(monkeypatch, None)
    claimed = w._claim()
    assert claimed is not None and claimed["id"] == job_id


def test_claim_gate_cold_worker_known_vram_falls_back_to_claim_any(monkeypatch):
    """A COLD worker (registry not yet populated) on a KNOWN-small box must
    still claim a model-job — claim-any-capable — so it never wedges the queue.

    The gate distinguishes a *too-small* machine (registry populated, nothing
    fits ⇒ ``has_models`` True ⇒ claim nothing) from a *cold* worker (registry
    empty ⇒ ``has_models`` False ⇒ fall back to claim-any). Both produce
    ``fits == []`` for different reasons; only ``has_models`` tells them apart.
    Every other claim-gate test runs under the ``_registry`` fixture, so
    ``has_models`` is always True there — this is the ONLY case exercising the
    empty-registry leg. Simplifying the guard to ``if vram is not None and not
    fits: return None`` (dropping the ``has_models`` term) would make this cold
    worker return None forever and wedge the gpu queue; this test catches that.
    """
    # Deliberately do NOT use the _registry fixture: registry must be EMPTY so
    # known_ids()==[] ⇒ has_models is False. Save/restore to stay a good citizen.
    saved = dict(model_registry.MODELS)
    model_registry.clear_for_tests()
    try:
        assert model_registry.known_ids() == []  # precondition: cold/empty
        _run, job_id = _queued_gpu_job("huge")
        # Known SMALL VRAM (16 GB) — a real int, not None. Were the gate only
        # `vram is not None and not fits`, this would return None.
        w = _gpu_worker_with_vram(monkeypatch, 16384)
        claimed = w._claim()
        assert claimed is not None
        assert claimed["id"] == job_id
        assert node_queue.get_node_job(job_id)["status"] == "running"
    finally:
        model_registry.clear_for_tests()
        for spec in saved.values():
            model_registry.register(spec)


# ── NodePool wiring: _sweep_unassignable_jobs (interval gate + event emit) ──


def test_node_pool_unassignable_sweep_emits_event_and_is_gated(monkeypatch):
    """The NodePool layer that drives ``flag_unassignable_gpu_jobs`` must emit
    exactly one ``unassignable`` node event per newly-flagged row AND honour its
    interval gate.

    ``flag_unassignable_gpu_jobs`` is tested directly above, but the wiring in
    ``NodePool._sweep_unassignable_jobs`` — call the flag fn, log, and emit one
    ``record_node_event(event_type='unassignable', queue='gpu', ...)`` per row,
    interval-gated — has no coverage. A regression dropping the event emit, or
    breaking the interval gate (re-running the flag fn every tick), would go
    unnoticed. We monkeypatch the flag fn (so no DB rows are needed) and spy the
    event writer.
    """
    from queue_workflows import node_pool

    fake_row = {
        "id": "j1", "run_id": "r1", "node_id": "n1",
        "required_model": "huge", "unassignable_reason": "no machine fits 'huge'",
    }
    flag_calls = {"n": 0}

    def _fake_flag(*args, **kwargs):
        flag_calls["n"] += 1
        return [fake_row]

    events: list[dict] = []

    def _spy_event(**kwargs):
        events.append(kwargs)
        return 1

    monkeypatch.setattr(node_queue, "flag_unassignable_gpu_jobs", _fake_flag)
    monkeypatch.setattr(node_queue, "record_node_event", _spy_event)

    pool = node_pool.NodePool(cpu_workers=0, gpu_workers=0, register_builtins=None)

    # First sweep: interval 0 ⇒ always runs. One flagged row ⇒ exactly one event.
    pool._unassignable_interval_s = 0.0
    pool._sweep_unassignable_jobs()

    assert flag_calls["n"] == 1
    assert len(events) == 1
    ev = events[0]
    assert ev["event_type"] == "unassignable"
    assert ev["queue"] == "gpu"
    assert ev["run_id"] == "r1"
    assert ev["node_id"] == "n1"
    assert ev["model"] == "huge"

    # Now gate hard: interval 60 s, with last_run just stamped by the call above.
    # Two more sweeps must both be suppressed (the flag fn is NOT re-run).
    pool._unassignable_interval_s = 60.0
    flag_calls["n"] = 0
    events.clear()
    pool._sweep_unassignable_jobs()
    pool._sweep_unassignable_jobs()

    assert flag_calls["n"] <= 1  # interval gate suppresses the repeat sweeps
    # Gated sweeps do no work ⇒ no events leak through.
    assert events == []
