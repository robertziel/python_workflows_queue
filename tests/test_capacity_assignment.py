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
