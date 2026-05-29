"""LLM supervisor — idle-VRAM reclaim for a vllm sidecar.

vLLM (unlike ollama) has NO built-in idle unload: it holds the model in VRAM
for its whole lifetime. So the engine runs a SUPERVISOR daemon that watches
LLM-request activity and, after an idle window, SIGTERMs the vllm sidecar
(docker ``restart: unless-stopped`` brings it back on the next request). For
ollama the supervisor is INERT — its daemon self-manages idle.

These tests mirror ``test_gpu_idle_unload.py``: the supervisor *thread* is
disabled under tests (conftest sets ``AI_LEADS_DISABLE_LLM_SUPERVISOR``), so we
drive the logic directly via the pure decision helper ``vllm_should_stop`` and a
single ``reap_idle_once`` tick on an :class:`LLMSupervisor` wrapping an in-file
``_FakeBackend``. The supervisor is generic over any backend exposing the
duck-typed surface (server_type / inflight / idle_seconds / idle_ttl_s /
is_running / stop_server) — the real ollama/vllm backends are NOT imported here,
keeping this test decoupled from the concurrently-authored backend modules.
"""

from __future__ import annotations

import threading

import pytest

from queue_workflows.llm_backends.supervisor import (
    LLMSupervisor,
    vllm_should_stop,
)


# ── duck-typed fake backend ────────────────────────────────────────────────


class _FakeBackend:
    """A settable stand-in for a real LLM backend, exposing exactly the surface
    the supervisor depends on. ``stop_server`` records the call + returns a
    caller-supplied flag so a test can assert the kill happened (and how often).
    Set ``stop_raises=True`` to exercise the best-effort swallow path."""

    def __init__(
        self,
        *,
        server_type: str = "vllm",
        inflight: int = 0,
        idle_s: float = 9999.0,
        idle_ttl_s: float = 60.0,
        running: bool = True,
        stop_result: bool = True,
        stop_raises: bool = False,
    ) -> None:
        self.server_type = server_type
        self.inflight = inflight
        self._idle_s = idle_s
        self.idle_ttl_s = idle_ttl_s
        self.running = running
        self._stop_result = stop_result
        self._stop_raises = stop_raises
        self.stop_calls = 0

    def idle_seconds(self) -> float:
        return self._idle_s

    def is_running(self) -> bool:
        return self.running

    def stop_server(self) -> bool:
        self.stop_calls += 1
        if self._stop_raises:
            raise RuntimeError("boom stopping the sidecar")
        # A real backend flips running off once the process is gone.
        if self._stop_result:
            self.running = False
        return self._stop_result


@pytest.fixture(autouse=True)
def _enable_supervisor(monkeypatch):
    """conftest sets AI_LEADS_DISABLE_LLM_SUPERVISOR=1 for the whole session;
    clear it by default here so ``_enabled`` reflects the backend, not the global
    test gate. The one env-gate test re-sets it explicitly."""
    monkeypatch.delenv("AI_LEADS_DISABLE_LLM_SUPERVISOR", raising=False)
    yield


# ── pure decision: vllm_should_stop ─────────────────────────────────────────


def test_should_stop_decision_matrix():
    ttl = 60.0
    # running + idle past TTL + nothing in flight ⇒ stop.
    assert vllm_should_stop(True, 0, 70, ttl) is True
    # exactly at the TTL boundary ⇒ stop (>=).
    assert vllm_should_stop(True, 0, ttl, ttl) is True
    # in-flight requests ⇒ never stop (busy).
    assert vllm_should_stop(True, 1, 70, ttl) is False
    # not idle long enough ⇒ keep.
    assert vllm_should_stop(True, 0, 30, ttl) is False
    # already stopped ⇒ nothing to do.
    assert vllm_should_stop(False, 0, 70, ttl) is False
    # ttl <= 0 disables the supervisor entirely.
    assert vllm_should_stop(True, 0, 70, 0) is False
    assert vllm_should_stop(True, 0, 70, -1) is False


# ── reap_idle_once: the single tick ─────────────────────────────────────────


def test_reaps_idle_vllm_sidecar():
    backend = _FakeBackend(server_type="vllm", inflight=0, idle_s=9999, running=True)
    sup = LLMSupervisor(backend=backend)
    assert sup.reap_idle_once() is True
    assert backend.stop_calls == 1


def test_keeps_sidecar_while_requests_in_flight():
    backend = _FakeBackend(server_type="vllm", inflight=2, idle_s=9999, running=True)
    sup = LLMSupervisor(backend=backend)
    assert sup.reap_idle_once() is False
    assert backend.stop_calls == 0


def test_keeps_recently_used_sidecar():
    backend = _FakeBackend(
        server_type="vllm", inflight=0, idle_s=5.0, idle_ttl_s=60.0, running=True,
    )
    sup = LLMSupervisor(backend=backend)
    assert sup.reap_idle_once() is False
    assert backend.stop_calls == 0


def test_keeps_already_stopped_sidecar():
    backend = _FakeBackend(server_type="vllm", inflight=0, idle_s=9999, running=False)
    sup = LLMSupervisor(backend=backend)
    assert sup.reap_idle_once() is False
    assert backend.stop_calls == 0


def test_ollama_backend_is_inert():
    """Ollama self-manages idle — the supervisor must NEVER stop it, even when it
    looks idle past the TTL."""
    backend = _FakeBackend(server_type="ollama", inflight=0, idle_s=9999, running=True)
    sup = LLMSupervisor(backend=backend)
    assert sup.reap_idle_once() is False
    assert backend.stop_calls == 0


def test_reap_swallows_backend_errors():
    """A backend whose stop_server() raises must NOT propagate — the reaper is
    best-effort and returns False so the daemon loop keeps ticking."""
    backend = _FakeBackend(
        server_type="vllm", inflight=0, idle_s=9999, running=True, stop_raises=True,
    )
    sup = LLMSupervisor(backend=backend)
    assert sup.reap_idle_once() is False        # swallowed, no exception
    assert backend.stop_calls == 1              # it did attempt the stop


# ── _enabled gate ───────────────────────────────────────────────────────────


def test_enabled_true_for_vllm_with_positive_ttl():
    sup = LLMSupervisor(backend=_FakeBackend(server_type="vllm", idle_ttl_s=60.0))
    assert sup._enabled is True


def test_disabled_via_env(monkeypatch):
    """The session-wide test gate (or an ops kill-switch): supervisor stays inert
    and start() spawns no thread."""
    monkeypatch.setenv("AI_LEADS_DISABLE_LLM_SUPERVISOR", "1")
    backend = _FakeBackend(server_type="vllm", idle_ttl_s=60.0)
    sup = LLMSupervisor(backend=backend)
    assert sup._enabled is False
    sup.start()
    assert sup._thread is None                  # disabled ⇒ no daemon started
    sup.stop()


def test_disabled_for_ollama_backend():
    sup = LLMSupervisor(backend=_FakeBackend(server_type="ollama", idle_ttl_s=60.0))
    assert sup._enabled is False


def test_disabled_when_ttl_zero_or_negative():
    assert LLMSupervisor(
        backend=_FakeBackend(server_type="vllm", idle_ttl_s=0.0)
    )._enabled is False
    assert LLMSupervisor(
        backend=_FakeBackend(server_type="vllm", idle_ttl_s=-1.0)
    )._enabled is False


def test_disabled_ollama_start_spawns_no_thread():
    """An inert (ollama) supervisor's start() is a no-op — no daemon thread."""
    sup = LLMSupervisor(backend=_FakeBackend(server_type="ollama", idle_ttl_s=60.0))
    sup.start()
    assert sup._thread is None
    sup.stop()


# ── _poll_interval math (same shape as ModelCache) ───────────────────────────


def test_poll_interval_math():
    # ttl/5 in the normal band.
    assert LLMSupervisor(
        backend=_FakeBackend(idle_ttl_s=60.0)
    )._poll_interval() == 12.0
    # clamped UP to the 5 s floor.
    assert LLMSupervisor(
        backend=_FakeBackend(idle_ttl_s=10.0)
    )._poll_interval() == 5.0
    # clamped DOWN to the 60 s ceiling.
    assert LLMSupervisor(
        backend=_FakeBackend(idle_ttl_s=600.0)
    )._poll_interval() == 60.0


# ── one real-thread smoke test ───────────────────────────────────────────────


def test_thread_runs_one_tick_then_exits_cleanly():
    """With a fake sleep_fn that flips the stop event after the first call, the
    loop ticks reap_idle_once exactly once (stopping the idle sidecar) then exits
    via stop(). Proves start()/stop() wire the daemon + event correctly without a
    real wall-clock wait."""
    backend = _FakeBackend(server_type="vllm", inflight=0, idle_s=9999, running=True)

    ticked = threading.Event()

    def _fake_sleep(_secs: float) -> None:
        # After the first sleep, ask the loop to stop; the loop still runs one
        # reap before re-checking the event at the top.
        sup._stop.set()

    sup = LLMSupervisor(backend=backend, sleep_fn=_fake_sleep)
    # Wrap reap so the test observes the tick fired.
    _orig = sup.reap_idle_once

    def _spy() -> bool:
        ticked.set()
        return _orig()

    sup.reap_idle_once = _spy  # type: ignore[method-assign]

    sup.start()
    assert sup._thread is not None
    assert ticked.wait(timeout=5.0), "supervisor thread never ticked"
    sup.stop()
    assert sup._thread is None
    assert backend.stop_calls == 1              # the one tick stopped the sidecar
