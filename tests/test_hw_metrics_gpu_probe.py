"""GPU probe selection in ``hw_metrics``.

Two box families feed the queue indicator: ROCm/rocm-smi and NVIDIA/nvidia-smi.
The sampler picks the right CLI at process start and pins the choice — these
cover the parser shapes and the dispatcher's selection logic.
"""

from __future__ import annotations

import subprocess

from queue_workflows import hw_metrics as hm


def _reset_probe():
    hm._GPU_PROBE = None


# ── nvidia-smi parser ────────────────────────────────────────────────────


def test_nvidia_smi_parses_csv(monkeypatch):
    sample = b"0, 42, 1234, 8192\n1, 0, 5, 8192\n"
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: sample)
    out = hm._nvidia_smi()
    assert out == [
        {"id": 0, "use_pct": 42, "vram_used_mb": 1234, "vram_total_mb": 8192},
        {"id": 1, "use_pct": 0, "vram_used_mb": 5, "vram_total_mb": 8192},
    ]


def test_nvidia_smi_empty_on_failure(monkeypatch):
    def boom(*a, **kw):
        raise subprocess.CalledProcessError(1, "nvidia-smi")
    monkeypatch.setattr(subprocess, "check_output", boom)
    assert hm._nvidia_smi() == []


def test_nvidia_smi_skips_malformed_rows(monkeypatch):
    sample = b"0, 10, 100, 1000\nthis is junk\n1, 20, 200, 2000\n"
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: sample)
    out = hm._nvidia_smi()
    assert [g["id"] for g in out] == [0, 1]


def test_nvidia_smi_handles_unified_memory_na(monkeypatch):
    """GB10 (Grace Blackwell) reports ``[N/A]`` for VRAM (shared system
    memory). Utilisation still parses; don't drop the row."""
    sample = b"0, 35, [N/A], [N/A]\n"
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: sample)
    out = hm._nvidia_smi()
    assert out == [
        {"id": 0, "use_pct": 35, "vram_used_mb": 0, "vram_total_mb": 0},
    ]


# ── Vendor selection at probe init ───────────────────────────────────────


def test_select_gpu_probe_prefers_nvidia_when_available(monkeypatch):
    _reset_probe()
    monkeypatch.setattr(hm, "_which", lambda c: c == "nvidia-smi")
    assert hm._select_gpu_probe() is hm._nvidia_smi


def test_select_gpu_probe_falls_back_to_rocm(monkeypatch):
    _reset_probe()
    monkeypatch.setattr(hm, "_which", lambda c: c == "rocm-smi")
    assert hm._select_gpu_probe() is hm._rocm_smi


def test_select_gpu_probe_returns_empty_when_no_cli(monkeypatch):
    _reset_probe()
    monkeypatch.setattr(hm, "_which", lambda c: False)
    probe = hm._select_gpu_probe()
    assert probe() == []


def test_gpu_probe_is_cached(monkeypatch):
    _reset_probe()
    calls: list[str] = []

    def counting_which(cmd: str) -> bool:
        calls.append(cmd)
        return cmd == "nvidia-smi"

    monkeypatch.setattr(hm, "_which", counting_which)
    monkeypatch.setattr(
        hm, "_nvidia_smi",
        lambda: [{"id": 0, "use_pct": 0, "vram_used_mb": 0, "vram_total_mb": 0}],
    )

    hm._gpu_probe()
    hm._gpu_probe()
    hm._gpu_probe()
    assert calls.count("nvidia-smi") == 1
