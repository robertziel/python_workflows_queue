"""Unit tests for the per-container GPU/RAM health samplers (``gpu_health``).

These pin the box-c-specific signal logic WITHOUT shelling out to nvidia-smi:

  * ``_parse_pmon_sm_pct`` — MAX per-process ``sm%`` from ``nvidia-smi pmon
    -s u`` text; ``-`` → 0; an all-dash / header-only block → None (so the
    caller falls back to the box-level probe).
  * ``gpu_util_pct`` — prefers the per-container pmon path; falls back to the
    box-level probe when pmon yields None.
  * ``container_ram_mb`` — reads the container's own cgroup memory.current.
"""

from __future__ import annotations

import builtins
import io

from queue_workflows import gpu_health


# ── pmon sm% parser ─────────────────────────────────────────────────────────


# A real box-c sample (box-a2): the render PID busy at sm=95, the idle gnome/Xorg
# G-type rows at "-", taken from `nvidia-smi pmon -c 1 -s u` inside a container.
_PMON_BUSY = """\
# gpu        pid  type    sm   mem   enc   dec   command
# Idx          #   C/G     %     %     %     %    name
    0       2470     G      -     -     -     -    Xorg
    0       2839     G      -     -     -     -    gnome-shell
    0       3762     C     95     0     -     -    python
"""

# Inside the render container the driver shows ONLY this container's PID (1),
# namespaced — the ollama sidecar is invisible. Busy.
_PMON_CONTAINER_BUSY = """\
# gpu        pid  type    sm   mem   enc   dec   command
# Idx          #   C/G     %     %     %     %    name
    0          1     C     95     0     -     -    python
"""

# Our container's process present but idle (sm = "-"): the common box-c hang
# shape. This is a REAL process row, so it parses to 0 (not None) — a genuine
# "no measurable GPU work" reading the watchdog can act on.
_PMON_CONTAINER_IDLE = """\
# gpu        pid  type    sm   mem   enc   dec   command
# Idx          #   C/G     %     %     %     %    name
    0          1     C      -     0     -     -    python
"""

# Empty namespace placeholder row (no process at all): all dashes incl. pid.
_PMON_EMPTY = """\
# gpu        pid  type    sm   mem   enc   dec   command
# Idx          #   C/G     %     %     %     %    name
    0          -     -      -     -     -     -    -
"""

# Header only (pmon printed nothing for this namespace).
_PMON_HEADER_ONLY = """\
# gpu        pid  type    sm   mem   enc   dec   command
# Idx          #   C/G     %     %     %     %    name
"""


def test_parse_pmon_takes_max_sm_across_rows():
    assert gpu_health._parse_pmon_sm_pct(_PMON_BUSY) == 95


def test_parse_pmon_container_busy():
    assert gpu_health._parse_pmon_sm_pct(_PMON_CONTAINER_BUSY) == 95


def test_parse_pmon_real_idle_process_is_zero_not_none():
    """A real process row whose sm% is '-' parses to 0 (measurable idle), so the
    watchdog reads it as 'no GPU work' rather than falling back to box level."""
    assert gpu_health._parse_pmon_sm_pct(_PMON_CONTAINER_IDLE) == 0


def test_parse_pmon_empty_namespace_returns_none():
    """No process rows ⇒ None ⇒ caller falls back to the box-level probe rather
    than reading a misleading hard 0."""
    assert gpu_health._parse_pmon_sm_pct(_PMON_EMPTY) is None


def test_parse_pmon_header_only_returns_none():
    assert gpu_health._parse_pmon_sm_pct(_PMON_HEADER_ONLY) is None


# ── gpu_util_pct dispatch (pmon preferred, box fallback) ─────────────────────


def test_gpu_util_prefers_container_pmon(monkeypatch):
    monkeypatch.setattr(gpu_health, "_nvidia_pmon_container_sm_pct", lambda: 42)
    # box probe must NOT be consulted when pmon gives a value
    monkeypatch.setattr(gpu_health, "_box_gpu_util_pct", lambda: 999)
    assert gpu_health.gpu_util_pct() == 42


def test_gpu_util_falls_back_to_box_when_pmon_none(monkeypatch):
    monkeypatch.setattr(gpu_health, "_nvidia_pmon_container_sm_pct", lambda: None)
    monkeypatch.setattr(gpu_health, "_box_gpu_util_pct", lambda: 7)
    assert gpu_health.gpu_util_pct() == 7


def test_box_gpu_util_takes_max_over_gpus(monkeypatch):
    from queue_workflows import hw_metrics
    monkeypatch.setattr(
        hw_metrics, "_gpu_probe",
        lambda: [{"use_pct": 3}, {"use_pct": 88}, {"use_pct": 0}],
    )
    assert gpu_health._box_gpu_util_pct() == 88


def test_box_gpu_util_zero_when_no_gpus(monkeypatch):
    from queue_workflows import hw_metrics
    monkeypatch.setattr(hw_metrics, "_gpu_probe", lambda: [])
    assert gpu_health._box_gpu_util_pct() == 0


# ── container_ram_mb (cgroup memory.current → MB, /proc fallback) ────────────


def test_container_ram_reads_cgroup_memory_current(monkeypatch):
    real_open = builtins.open

    def fake_open(path, *a, **kw):
        if str(path) == gpu_health._CGROUP_MEMORY_CURRENT:
            return io.StringIO("6442450944\n")  # 6 GiB in bytes
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)
    assert gpu_health.container_ram_mb() == 6144  # 6 GiB → 6144 MiB


def test_container_ram_falls_back_to_proc_rss(monkeypatch):
    real_open = builtins.open

    def fake_open(path, *a, **kw):
        if str(path) == gpu_health._CGROUP_MEMORY_CURRENT:
            raise FileNotFoundError(path)
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(gpu_health, "_proc_self_rss_mb", lambda: 321)
    assert gpu_health.container_ram_mb() == 321
