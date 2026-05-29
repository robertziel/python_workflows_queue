"""Flock-guarded hw_metrics sampler start — one sampler per host.

The flock (``start_hw_metrics_sampler_flocked``) is a cheap SECONDARY guard
against an accidental double-start within a host; the primary one-per-host
guarantee comes from the single GPU container per host (only the gpu claim
worker calls the starter). These drive the starter directly + pin the
claim-worker wiring (gpu starts it; cpu/fetch/load do not).
"""

from __future__ import annotations

import os

import pytest

from queue_workflows import claim_worker, hw_metrics


class _FakeSampler:
    instances: list = []

    def __init__(self, *a, **kw):
        self.started = False
        self.stopped = False
        _FakeSampler.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout=None) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_flock_state(monkeypatch, tmp_path):
    if hw_metrics._hw_metrics_lock_fd is not None:
        try:
            os.close(hw_metrics._hw_metrics_lock_fd)
        except OSError:
            pass
    hw_metrics._hw_metrics_lock_fd = None
    hw_metrics._hw_metrics_thread = None
    _FakeSampler.instances = []
    monkeypatch.setattr(hw_metrics, "HwMetricsSampler", _FakeSampler)
    monkeypatch.setattr(
        hw_metrics, "_HW_METRICS_LOCK_PATH", str(tmp_path / "hw.lock"),
    )
    monkeypatch.delenv("AI_LEADS_DISABLE_HW_METRICS", raising=False)
    yield
    if hw_metrics._hw_metrics_lock_fd is not None:
        try:
            os.close(hw_metrics._hw_metrics_lock_fd)
        except OSError:
            pass
    hw_metrics._hw_metrics_lock_fd = None
    hw_metrics._hw_metrics_thread = None


# ── the shared flock-guarded starter ────────────────────────────────────────


def test_first_caller_wins_flock_and_starts_sampler():
    sampler = hw_metrics.start_hw_metrics_sampler_flocked()
    assert sampler is not None
    assert sampler.started is True
    assert hw_metrics._hw_metrics_lock_fd is not None


def test_second_call_in_process_is_idempotent_not_a_second_sampler():
    first = hw_metrics.start_hw_metrics_sampler_flocked()
    assert first is not None and first.started

    second = hw_metrics.start_hw_metrics_sampler_flocked()
    assert second is first
    assert len(_FakeSampler.instances) == 1


def test_loses_flock_when_another_process_holds_it(monkeypatch):
    import fcntl

    def _contended(fd, op):
        raise BlockingIOError("Resource temporarily unavailable")

    monkeypatch.setattr(fcntl, "flock", _contended)
    sampler = hw_metrics.start_hw_metrics_sampler_flocked()
    assert sampler is None
    assert hw_metrics._hw_metrics_lock_fd is None
    assert _FakeSampler.instances == []


def test_disabled_by_env_starts_nothing(monkeypatch):
    monkeypatch.setenv("AI_LEADS_DISABLE_HW_METRICS", "1")
    sampler = hw_metrics.start_hw_metrics_sampler_flocked()
    assert sampler is None
    assert hw_metrics._hw_metrics_lock_fd is None
    assert _FakeSampler.instances == []


def test_starter_swallows_sampler_construction_error(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("sampler init blew up")

    monkeypatch.setattr(hw_metrics, "HwMetricsSampler", _boom)
    sampler = hw_metrics.start_hw_metrics_sampler_flocked()
    assert sampler is None


# ── claim-worker wiring ──────────────────────────────────────────────────────


def _drive_run_forever(worker, monkeypatch):
    monkeypatch.setattr(worker, "await_schema", lambda: None)
    monkeypatch.setattr(worker, "run_once", lambda: False)

    import psycopg

    class _FakeListen:
        def execute(self, *a, **kw): worker.stop()
        def notifies(self, *a, **kw): return iter(())
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(psycopg, "connect", lambda *a, **kw: _FakeListen())
    worker.run_forever()


def test_claim_worker_gpu_starts_flocked_sampler(monkeypatch):
    started: list = []
    monkeypatch.setattr(
        hw_metrics, "start_hw_metrics_sampler_flocked",
        lambda: started.append("gpu") or _FakeSampler(),
    )

    worker = claim_worker.ClaimWorker(queue="gpu", host="host-a")
    _drive_run_forever(worker, monkeypatch)

    assert started == ["gpu"]


@pytest.mark.parametrize("queue", ["cpu", "fetch", "load"])
def test_claim_worker_non_gpu_does_not_start_sampler(queue, monkeypatch):
    started: list = []
    monkeypatch.setattr(
        hw_metrics, "start_hw_metrics_sampler_flocked",
        lambda: started.append(queue) or _FakeSampler(),
    )

    worker = claim_worker.ClaimWorker(queue=queue, host="host-a")
    _drive_run_forever(worker, monkeypatch)

    assert started == []
    assert worker._hw_sampler is None


def test_claim_worker_stops_sampler_on_exit(monkeypatch):
    sampler = _FakeSampler()
    monkeypatch.setattr(
        hw_metrics, "start_hw_metrics_sampler_flocked", lambda: sampler,
    )
    worker = claim_worker.ClaimWorker(queue="gpu", host="host-a")
    _drive_run_forever(worker, monkeypatch)
    assert sampler.stopped is True


def test_claim_worker_gpu_sampler_loss_is_non_fatal(monkeypatch):
    monkeypatch.setattr(
        hw_metrics, "start_hw_metrics_sampler_flocked", lambda: None,
    )
    worker = claim_worker.ClaimWorker(queue="gpu", host="host-c")
    monkeypatch.setattr(worker, "await_schema", lambda: None)
    ran: list = []
    monkeypatch.setattr(worker, "run_once", lambda: ran.append(1) and False)

    import psycopg

    class _FakeListen:
        def execute(self, *a, **kw): worker.stop()
        def notifies(self, *a, **kw): return iter(())
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(psycopg, "connect", lambda *a, **kw: _FakeListen())
    worker.run_forever()
