"""GPU-blind self-park.

A GPU worker whose container can't use a CUDA device (``torch.cuda.is_available()``
False — the box-c post-recreate device-cgroup race: HOST GPU healthy, container
blind) must REFUSE to claim. Otherwise it fast-fails every GPU job ~1 s after
claiming (``cudaErrorNoDevice``) — a silent black hole that poisons every run a
lane fans onto it. See ``worklog/gpu-blind-worker-self-exclude.md``.

The probe is TRI-STATE so the gate fires ONLY on the real blind condition (torch
present but no device), not when torch is merely absent (a non-GPU / test env) —
otherwise it would false-park and break every torch-less GPU unit test. The test
env has no torch, so the torch-present cases inject a fake module.
"""
from __future__ import annotations

import logging
import sys
import types

from queue_workflows import claim_worker


def _fake_torch(is_available):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=is_available)
    )


# ── _probe_gpu_usable (tri-state) ────────────────────────────────────────────


def test_probe_gpu_usable_true_when_cuda_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(lambda: True))
    assert claim_worker._probe_gpu_usable() is True


def test_probe_gpu_usable_false_when_torch_present_but_no_device(monkeypatch):
    # The box-c-blind condition: torch imports, is_available() returns False.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(lambda: False))
    assert claim_worker._probe_gpu_usable() is False


def test_probe_gpu_usable_false_when_probe_raises(monkeypatch):
    # NVML init failure surfaces as a raise — still a definitively unusable device.
    def _boom():
        raise RuntimeError("Can't initialize NVML")

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(_boom))
    assert claim_worker._probe_gpu_usable() is False


def test_probe_gpu_usable_none_when_torch_absent(monkeypatch):
    # sys.modules[name] = None makes `import name` raise ⇒ "can't judge".
    monkeypatch.setitem(sys.modules, "torch", None)
    assert claim_worker._probe_gpu_usable() is None


# ── _refuse_blind_gpu (the gate) ─────────────────────────────────────────────


class _FakeCache:
    current_model = None


def _gpu_worker(host="h"):
    return claim_worker.ClaimWorker(queue="gpu", host=host, model_cache=_FakeCache())


def test_refuse_blind_gpu_true_and_loud_when_device_unusable(monkeypatch, caplog):
    monkeypatch.setattr(claim_worker, "_probe_gpu_usable", lambda: False)
    with caplog.at_level(logging.ERROR):
        assert _gpu_worker()._refuse_blind_gpu() is True
    # The refusal is the operator's only signal — it must be loud.
    assert any("REFUSING" in r.getMessage() for r in caplog.records)


def test_refuse_blind_gpu_false_when_gpu_usable(monkeypatch):
    monkeypatch.setattr(claim_worker, "_probe_gpu_usable", lambda: True)
    assert _gpu_worker()._refuse_blind_gpu() is False


def test_refuse_blind_gpu_false_when_cannot_judge(monkeypatch):
    # None (no torch) ⇒ NOT the blind condition ⇒ do NOT refuse (keeps a non-GPU
    # / test environment claiming exactly as before).
    monkeypatch.setattr(claim_worker, "_probe_gpu_usable", lambda: None)
    assert _gpu_worker()._refuse_blind_gpu() is False


def test_refuse_blind_gpu_false_for_cpu_and_never_probes(monkeypatch):
    calls = {"n": 0}

    def _probe():
        calls["n"] += 1
        return False

    monkeypatch.setattr(claim_worker, "_probe_gpu_usable", _probe)
    w = claim_worker.ClaimWorker(queue="cpu", host="cpu-host")
    assert w._refuse_blind_gpu() is False
    # A cpu/ingest worker must short-circuit before probing CUDA at all.
    assert calls["n"] == 0
