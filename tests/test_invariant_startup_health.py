"""Startup health gate.

``_await_recovery`` reclaims every expired lease in a bounded retry loop that
raises if recovery never succeeds. Three invariants:
1. Recovery raises persistently → ``start()`` raises + dispatch thread never
   starts.
2. Recovery raises once then succeeds → ``start()`` returns normally.
3. No background work runs when the health gate failed.
"""

from __future__ import annotations

import threading

import pytest

from queue_workflows import node_pool, node_queue


# ── 1. Persistent failure → start() raises ────────────────────────────────


def test_startup_raises_when_recovery_fails(monkeypatch):
    def boom() -> int:
        raise RuntimeError("simulated DB unreachable")

    monkeypatch.setattr(node_queue, "reclaim_expired_leases", boom)

    pool = node_pool.NodePool(register_builtins=None)
    with pytest.raises(RuntimeError, match="recovery"):
        pool._await_recovery(max_attempts=2, backoff_s=0.0)


def test_start_propagates_recovery_failure(monkeypatch):
    def boom() -> int:
        raise RuntimeError("simulated DB unreachable")

    monkeypatch.setattr(node_queue, "reclaim_expired_leases", boom)
    monkeypatch.setenv("AI_LEADS_NODE_POOL_RECOVERY_RETRIES", "2")
    monkeypatch.setenv("AI_LEADS_NODE_POOL_RECOVERY_BACKOFF_S", "0.0")

    pool = node_pool.NodePool(register_builtins=None)
    with pytest.raises(RuntimeError):
        pool.start()


# ── 2. Transient failure → eventual success ───────────────────────────────


def test_startup_succeeds_after_transient_recovery_failure(monkeypatch):
    calls = {"n": 0}
    real = node_queue.reclaim_expired_leases

    def flaky() -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first attempt fails")
        return real()

    monkeypatch.setattr(node_queue, "reclaim_expired_leases", flaky)

    pool = node_pool.NodePool(register_builtins=None)
    pool._await_recovery(max_attempts=3, backoff_s=0.0)
    assert calls["n"] == 2


def test_startup_caps_retries_at_max_attempts(monkeypatch):
    calls = {"n": 0}

    def always_fails() -> int:
        calls["n"] += 1
        raise RuntimeError("permanent")

    monkeypatch.setattr(node_queue, "reclaim_expired_leases", always_fails)

    pool = node_pool.NodePool(register_builtins=None)
    with pytest.raises(RuntimeError):
        pool._await_recovery(max_attempts=4, backoff_s=0.0)
    assert calls["n"] == 4


# ── 3. No dispatch when health gate fails ────────────────────────────────


def test_invariant_no_dispatch_thread_when_health_fails(monkeypatch):
    def boom() -> int:
        raise RuntimeError("simulated DB unreachable")

    monkeypatch.setattr(node_queue, "reclaim_expired_leases", boom)
    monkeypatch.setenv("AI_LEADS_NODE_POOL_RECOVERY_RETRIES", "1")
    monkeypatch.setenv("AI_LEADS_NODE_POOL_RECOVERY_BACKOFF_S", "0.0")

    pool = node_pool.NodePool(register_builtins=None)
    with pytest.raises(RuntimeError):
        pool.start()

    assert pool._dispatch_thread is None
    assert pool._input_listener is None


def test_node_pool_start_does_not_start_hw_metrics_sampler(monkeypatch):
    """The orchestrator must NOT start the hw_metrics sampler — it's owned by
    the gpu claim worker. Spy on the shared starter and assert it's never
    called from ``NodePool.start``."""
    from queue_workflows import hw_metrics

    called: list = []
    monkeypatch.setattr(
        hw_metrics, "start_hw_metrics_sampler_flocked", lambda: called.append(1),
    )
    if hasattr(node_pool, "start_hw_metrics_sampler_flocked"):
        monkeypatch.setattr(
            node_pool, "start_hw_metrics_sampler_flocked", lambda: called.append(1),
        )

    pool = node_pool.NodePool(register_builtins=None)
    try:
        pool.start()
        assert called == []
    finally:
        pool.stop()


def test_invariant_recovery_runs_before_dispatch_thread_starts(monkeypatch):
    events: list[str] = []
    real = node_queue.reclaim_expired_leases

    def recorder() -> int:
        events.append("recovery")
        return real()

    monkeypatch.setattr(node_queue, "reclaim_expired_leases", recorder)

    real_thread_start = threading.Thread.start

    def recorder_start(self):
        if self.name == "node-pool-dispatch":
            events.append("dispatch_thread_start")
        return real_thread_start(self)

    monkeypatch.setattr(threading.Thread, "start", recorder_start)

    pool = node_pool.NodePool(register_builtins=None)
    try:
        pool.start()
        assert events.index("recovery") < events.index("dispatch_thread_start")
    finally:
        pool.stop()
