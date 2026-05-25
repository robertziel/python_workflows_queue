"""Unit tests for the cgroup-based per-container attribution sampler.

Pure filesystem reads + a docker-SDK call, both faked. Tests focus on:
* delta calculation (no baseline first call; busy/idle CPU%);
* RAM summed across containers;
* missing files / missing socket short-circuit;
* docker-name filter ignores foreign containers (the configurable prefix).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from queue_workflows.cgroup_attribution import CgroupAttribution


class _FakeContainer:
    def __init__(self, cid: str, name: str) -> None:
        self.id = cid
        self.name = name


class _FakeContainers:
    def __init__(self, containers: list[_FakeContainer]) -> None:
        self._all = containers

    def list(self, filters=None, ignore_removed=True):
        prefix = (filters or {}).get("name", "")
        return [c for c in self._all if prefix in c.name]


class _FakeDocker:
    def __init__(self, containers: list[_FakeContainer]) -> None:
        self.containers = _FakeContainers(containers)

    def ping(self):
        return True


def _write_scope(root: Path, cid: str, usage_usec: int, mem_bytes: int) -> None:
    d = root / "system.slice" / f"docker-{cid}.scope"
    d.mkdir(parents=True, exist_ok=True)
    (d / "cpu.stat").write_text(
        f"usage_usec {usage_usec}\nuser_usec 0\nsystem_usec 0\n"
    )
    (d / "memory.current").write_text(f"{mem_bytes}\n")


def _make(tmp: Path, *, ncpu: int = 4) -> CgroupAttribution:
    return CgroupAttribution(cgroup_root=str(tmp), name_prefix="ai_leads-", ncpu=ncpu)


def test_returns_none_when_cgroup_root_missing(tmp_path):
    a = _make(tmp_path / "does-not-exist")
    assert a.sample() is None


def test_first_call_has_ram_but_no_cpu_baseline(tmp_path):
    _write_scope(tmp_path, "abc123", usage_usec=1_000_000, mem_bytes=128 * 1024 * 1024)
    a = _make(tmp_path)
    a._docker = _FakeDocker([_FakeContainer("abc123", "ai_leads-workers-1")])

    out = a.sample()
    assert out is not None
    assert out["cpu_percent"] is None
    assert out["ram_used_mb"] == 128


def test_second_call_computes_cpu_delta(tmp_path, monkeypatch):
    _write_scope(tmp_path, "abc123", usage_usec=0, mem_bytes=64 * 1024 * 1024)
    a = _make(tmp_path, ncpu=4)
    a._docker = _FakeDocker([_FakeContainer("abc123", "ai_leads-workers-1")])

    import queue_workflows.cgroup_attribution as mod
    times = iter([100.0, 101.0])
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(times))

    a.sample()
    _write_scope(tmp_path, "abc123", usage_usec=2_000_000, mem_bytes=64 * 1024 * 1024)
    out = a.sample()
    assert out is not None
    assert out["cpu_percent"] == pytest.approx(50.0, abs=0.5)


def test_idle_container_reads_as_zero_percent(tmp_path, monkeypatch):
    _write_scope(tmp_path, "abc123", usage_usec=42, mem_bytes=8 * 1024 * 1024)
    a = _make(tmp_path, ncpu=4)
    a._docker = _FakeDocker([_FakeContainer("abc123", "ai_leads-workers-1")])

    import queue_workflows.cgroup_attribution as mod
    times = iter([100.0, 101.0])
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(times))

    a.sample()
    out = a.sample()
    assert out is not None
    assert out["cpu_percent"] == pytest.approx(0.0, abs=0.001)


def test_ram_sums_across_multiple_containers(tmp_path):
    _write_scope(tmp_path, "aaa", usage_usec=0, mem_bytes=100 * 1024 * 1024)
    _write_scope(tmp_path, "bbb", usage_usec=0, mem_bytes=250 * 1024 * 1024)
    a = _make(tmp_path)
    a._docker = _FakeDocker([
        _FakeContainer("aaa", "ai_leads-workers-1"),
        _FakeContainer("bbb", "ai_leads-rails-1"),
    ])
    out = a.sample()
    assert out is not None
    assert out["ram_used_mb"] == 350


def test_foreign_containers_excluded_by_prefix(tmp_path):
    """Containers from other compose projects on the same host must NOT count
    toward our slice — the module re-validates the prefix."""
    _write_scope(tmp_path, "ours", usage_usec=0, mem_bytes=10 * 1024 * 1024)
    _write_scope(tmp_path, "theirs", usage_usec=0, mem_bytes=2_000 * 1024 * 1024)
    a = _make(tmp_path)
    a._docker = _FakeDocker([
        _FakeContainer("ours", "ai_leads-workers-1"),
        _FakeContainer("theirs", "icon-pk-prep"),
    ])
    out = a.sample()
    assert out is not None
    assert out["ram_used_mb"] == 10


def test_missing_scope_dir_does_not_crash(tmp_path):
    a = _make(tmp_path)
    a._docker = _FakeDocker([_FakeContainer("ghost", "ai_leads-workers-1")])
    out = a.sample()
    assert out is not None
    assert out["ram_used_mb"] == 0
    assert out["cpu_percent"] is None


def test_returns_none_when_docker_socket_unavailable(tmp_path, monkeypatch):
    a = _make(tmp_path)
    a._docker_unavailable_logged = True
    a._docker = None
    monkeypatch.setattr(a, "_get_docker", lambda: None)
    assert a.sample() is None


def test_default_prefix_comes_from_config(monkeypatch):
    """When no explicit name_prefix is passed, the default reads
    config.container_prefix (the configurable seam, plan §1f/§2c)."""
    import queue_workflows
    queue_workflows.configure(container_prefix="myproj-")
    a = CgroupAttribution(cgroup_root="/nonexistent")
    assert a._prefix == "myproj-"
