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


class _RaisingContainers:
    """A docker client whose ``containers.list`` blows up — models a daemon
    hiccup or a socket that vanished mid-run."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    @property
    def containers(self):  # accessed as ``client.containers.list``
        outer = self

        class _C:
            def list(self, filters=None, ignore_removed=True):
                raise outer._exc

        return _C()

    def ping(self):
        return True


def test_docker_list_exception_is_swallowed_not_propagated(tmp_path):
    """A ``containers.list`` failure (daemon hiccup / socket gone mid-run) must
    be caught and surface as *no attribution this tick* — never propagate, or
    it would crash the single hw_metrics sampler thread and kill ALL host
    telemetry. Drives the real ``_our_container_ids`` except-clause (not a
    stubbed ``_get_docker``)."""
    _write_scope(tmp_path, "abc123", usage_usec=0, mem_bytes=64 * 1024 * 1024)
    a = _make(tmp_path)
    a._docker = _RaisingContainers(RuntimeError("daemon gone"))

    # The real fallback path: list() raises -> _our_container_ids() is None.
    assert a._our_container_ids() is None
    # ...and sample() degrades to None rather than raising.
    assert a.sample() is None


def test_get_docker_returns_none_and_logs_once_on_ping_failure(tmp_path, monkeypatch):
    """``from_env()`` succeeds but ``ping()`` raises (socket file present but
    daemon dead): ``_get_docker`` must return None, leave ``_docker`` None, and
    set the log-once flag so the warning isn't spammed every tick."""
    import sys
    import types

    pinged = {"n": 0}

    class _DeadClient:
        def ping(self):
            pinged["n"] += 1
            raise OSError("connection refused")

    fake_docker = types.ModuleType("docker")
    fake_docker.from_env = lambda: _DeadClient()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "docker", fake_docker)

    a = _make(tmp_path)
    a._docker = None
    assert a._docker_unavailable_logged is False

    assert a._get_docker() is None
    assert a._docker is None
    assert a._docker_unavailable_logged is True
    assert pinged["n"] == 1

    # Second call: flag already set, so no re-log and (since _docker stays None)
    # it retries from_env/ping but must still return None without raising.
    assert a._get_docker() is None
    assert a._docker_unavailable_logged is True


def test_get_docker_returns_none_on_import_error(tmp_path, monkeypatch):
    """When the docker SDK isn't installed, ``import docker`` raises
    ImportError inside ``_get_docker`` — it must be caught, return None, and
    flip the log-once flag (so a pg-only/SDK-less deploy degrades gracefully)."""
    import sys

    # A None entry in sys.modules makes ``import docker`` raise ImportError.
    monkeypatch.setitem(sys.modules, "docker", None)

    a = _make(tmp_path)
    a._docker = None
    assert a._docker_unavailable_logged is False

    assert a._get_docker() is None
    assert a._docker_unavailable_logged is True


def test_returns_none_when_docker_socket_unavailable(tmp_path, monkeypatch):
    """Top-level contract: when no docker client is obtainable, ``sample()``
    returns None so the caller emits only system-wide totals."""
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
