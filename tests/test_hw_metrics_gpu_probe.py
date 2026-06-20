"""GPU probe selection in ``hw_metrics``.

Two box families feed the queue indicator: ROCm/rocm-smi and NVIDIA/nvidia-smi.
The sampler picks the right CLI at process start and pins the choice — these
cover the parser shapes and the dispatcher's selection logic.
"""

from __future__ import annotations

import json
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
    """box-c (Grace Blackwell) reports ``[N/A]`` for VRAM (shared system
    memory). Utilisation still parses; don't drop the row."""
    sample = b"0, 35, [N/A], [N/A]\n"
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: sample)
    out = hm._nvidia_smi()
    assert out == [
        {"id": 0, "use_pct": 35, "vram_used_mb": 0, "vram_total_mb": 0},
    ]


# ── rocm-smi parser ──────────────────────────────────────────────────────
#
# The engine explicitly targets BOTH nvidia and rocm box families, but the AMD
# parser had no direct coverage. A regression in the JSON key names, the
# byte→MB math, the ``cardN`` regex, or the fail-to-``[]`` behaviour would
# silently zero out all AMD GPU telemetry and — via ``total_vram_mb()``'s
# ``max()`` — wreck capacity-aware claim gating on the live AMD fleet.


_GB = 1024 ** 3


def test_rocm_smi_parses_json_and_handles_failures(monkeypatch):
    """Happy path: byte→MB math, ``system`` key skipped, sorted by card index.

    ``VRAM Total Used Memory (B)`` is in BYTES; the probe must divide by
    1024*1024 to publish MB consistent with the nvidia parser. The non-``cardN``
    ``system`` key must be ignored (it carries driver metadata, not a GPU)."""
    payload = json.dumps({
        "card0": {
            "GPU use (%)": "77",
            "VRAM Total Used Memory (B)": str(3 * _GB),
            "VRAM Total Memory (B)": str(8 * _GB),
        },
        "card1": {
            "GPU use (%)": 0,
            "VRAM Total Used Memory (B)": 0,
            "VRAM Total Memory (B)": str(8 * _GB),
        },
        "system": {"driver": "x"},
    }).encode()
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: payload)
    assert hm._rocm_smi() == [
        {"id": 0, "use_pct": 77, "vram_used_mb": 3072, "vram_total_mb": 8192},
        {"id": 1, "use_pct": 0, "vram_used_mb": 0, "vram_total_mb": 8192},
    ]


def test_rocm_smi_empty_on_failure(monkeypatch):
    """rocm-smi missing/erroring ⇒ ``[]`` (never raises into the sampler loop)."""
    def boom(*a, **kw):
        raise subprocess.CalledProcessError(1, "rocm-smi")
    monkeypatch.setattr(subprocess, "check_output", boom)
    assert hm._rocm_smi() == []


def test_rocm_smi_empty_on_non_json(monkeypatch):
    """Garbage output (e.g. a warning banner) ⇒ ``[]``, not a JSON crash."""
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: b"not json")
    assert hm._rocm_smi() == []


def test_rocm_smi_missing_use_key_defaults_to_zero(monkeypatch):
    """A card dict missing ``GPU use (%)`` keeps the row with ``use_pct`` 0
    rather than dropping the GPU — VRAM totals still feed capacity gating."""
    payload = json.dumps({
        "card0": {
            "VRAM Total Used Memory (B)": str(2 * _GB),
            "VRAM Total Memory (B)": str(8 * _GB),
        },
    }).encode()
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: payload)
    assert hm._rocm_smi() == [
        {"id": 0, "use_pct": 0, "vram_used_mb": 2048, "vram_total_mb": 8192},
    ]


def test_as_int_edge_cases():
    """``_as_int`` underpins the rocm parser: ``None`` propagates (so the caller
    can default), percent suffixes are stripped, garbage is ``None``, ints pass
    through unchanged."""
    assert hm._as_int(None) is None
    assert hm._as_int("55%") == 55
    assert hm._as_int("garbage") is None
    assert hm._as_int(7) == 7


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
