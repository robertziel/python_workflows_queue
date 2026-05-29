"""vLLM-specific behaviour beyond the shared :class:`LLMBackend` contract.

WHY THIS TEST EXISTS — and why it touches no real I/O. The shared request
accounting + config echo are already pinned by ``test_llm_backend_contract.py``;
this module pins the bits that are vLLM's OWN: its endpoint paths, its real
start/stop *state machine*, and the sticky Sleep-Mode-L2 fallback. vLLM is a
docker sidecar holding ONE model in VRAM for its whole lifetime — no built-in
idle unload, no runtime model swap — so the backend must (a) SIGTERM the sidecar
to free VRAM (the supervisor's lever; docker ``restart: unless-stopped`` respawns
it), and (b) treat a *model switch* as a slow stop→bring-up cycle, because the
fast path (vLLM Sleep-Mode L2 + engine reload) is stubbed-unimplemented and, once
attempted, is remembered as UNSUPPORTED so it's never retried.

To stay DB-free and wall-clock-free, every I/O seam is injected. A single in-file
``_FakeServer`` models the sidecar (``up`` / ``served``); the four seam fns close
over it (``kill_fn`` flips it down, ``ensure_up_fn`` flips it up serving a model,
``health_fn`` reports liveness, ``served_model_fn`` reports the served id). A
mutable list-cell clock + a ``sleep_fn`` that advances it give a virtual clock so
the cold-start poll + timeout are deterministic with no real sleeping — exactly
the library's "pure logic with injectable now_fn/sleep_fn seams" philosophy.
"""

from __future__ import annotations

import pytest

from queue_workflows.llm_backends.vllm import VLLMBackend, VLLMState


# ── in-file fake sidecar + seam fns ─────────────────────────────────────────


class _FakeServer:
    """A settable stand-in for the vLLM docker sidecar. ``up`` is liveness;
    ``served`` is the model id currently in VRAM (None when down). The seam fns
    below close over an instance so a test can drive the whole state machine
    without a real process, socket, or docker."""

    def __init__(self, *, up: bool = False, served: str | None = None) -> None:
        self.up = up
        self.served = served
        self.kill_calls = 0
        self.ensure_up_calls: list[str] = []


def _seams(server: _FakeServer):
    """Build the four backend I/O seams closing over ``server`` (plus a small
    spy on each so tests can assert call counts / args)."""

    def kill_fn() -> bool:
        # SIGTERM the sidecar: it stops, VRAM freed, served id forgotten. Always
        # "signalled one" here (the server was up when we reached for kill).
        server.kill_calls += 1
        was_up = server.up
        server.up = False
        server.served = None
        return was_up

    def ensure_up_fn(model_id: str) -> None:
        # The Phase-6 "make it come up serving model_id" seam — in the fake this
        # is synchronous (docker restart:unless-stopped + entrypoint, collapsed).
        server.ensure_up_calls.append(model_id)
        server.up = True
        server.served = model_id

    def health_fn() -> bool:
        return server.up

    def served_model_fn() -> str | None:
        return server.served

    return kill_fn, ensure_up_fn, health_fn, served_model_fn


def _clock():
    """A mutable monotonic clock: ``cell[0]`` is now; ``now`` reads it. Bump the
    cell (directly or via the advancing sleep_fn) to move time without sleeping."""
    cell = [0.0]
    return cell, (lambda: cell[0])


def _make(
    server: _FakeServer,
    *,
    base_url: str = "http://h:8000",
    parallelism: int = 1,
    idle_ttl_s: float = 60.0,
    served_model: str | None = None,
    now_fn=None,
    sleep_fn=lambda _s: None,
) -> VLLMBackend:
    """Construct a backend wired to ``server``'s seams. ``sleep_fn`` defaults to a
    no-op (instant poll); cold-start tests pass an advancing one."""
    kill_fn, ensure_up_fn, health_fn, served_model_fn = _seams(server)
    kw = {}
    if now_fn is not None:
        kw["now_fn"] = now_fn
    return VLLMBackend(
        base_url=base_url,
        parallelism=parallelism,
        idle_ttl_s=idle_ttl_s,
        served_model=served_model,
        sleep_fn=sleep_fn,
        kill_fn=kill_fn,
        ensure_up_fn=ensure_up_fn,
        health_fn=health_fn,
        served_model_fn=served_model_fn,
        **kw,
    )


# ── endpoints / identity ────────────────────────────────────────────────────


def test_server_type_is_vllm():
    assert _make(_FakeServer()).server_type == "vllm"


def test_chat_url_is_v1_chat_completions():
    b = _make(_FakeServer())
    assert b.chat_url == "http://h:8000/v1/chat/completions"
    assert b.chat_url.endswith("/v1/chat/completions")


def test_health_url_is_root_health_not_v1():
    b = _make(_FakeServer())
    # vLLM's health probe is at the server ROOT /health, NOT /v1/health.
    assert b.health_url == "http://h:8000/health"
    assert b.health_url.endswith("/health")
    assert "/v1/health" not in b.health_url


def test_base_url_trailing_slash_handling():
    """A trailing slash on base_url must not double up in the endpoints (the base
    ctor strips it)."""
    b = _make(_FakeServer(), base_url="http://h:8000/")
    assert b.base_url == "http://h:8000"
    assert b.chat_url == "http://h:8000/v1/chat/completions"
    assert b.health_url == "http://h:8000/health"


# ── ensure_ready: warm hit ───────────────────────────────────────────────────


def test_ensure_ready_warm_hit_does_nothing():
    """Server already up AND serving the wanted model ⇒ no kill, no bring-up, just
    SERVING. The known served model comes from the constructor's served_model."""
    server = _FakeServer(up=True, served="X")
    b = _make(server, served_model="X")
    b.ensure_ready("X")
    assert b.state == VLLMState.SERVING
    assert b._served_model == "X"
    assert server.kill_calls == 0
    assert server.ensure_up_calls == []


# ── ensure_ready: cold start ─────────────────────────────────────────────────


def test_ensure_ready_cold_start_brings_up_and_serves():
    """Server DEAD ⇒ LOADING → ensure_up(model) → health polled → SERVING, with
    current served reflecting the model."""
    server = _FakeServer(up=False, served=None)
    b = _make(server)
    assert b.state == VLLMState.DEAD
    b.ensure_ready("X")
    assert b.state == VLLMState.SERVING
    assert server.ensure_up_calls == ["X"]
    assert server.kill_calls == 0
    assert b._served_model == "X"


def test_ensure_ready_cold_start_optimistic_served_when_models_endpoint_blank():
    """If served_model_fn returns None (the /v1/models probe came back empty/early)
    the backend optimistically trusts the model it just asked to bring up."""
    server = _FakeServer(up=False, served=None)
    kill_fn, ensure_up_fn, health_fn, _served = _seams(server)

    def _blank_served() -> str | None:
        return None  # endpoint not ready / returned no models

    b = VLLMBackend(
        base_url="http://h:8000",
        idle_ttl_s=60.0,
        sleep_fn=lambda _s: None,
        kill_fn=kill_fn,
        ensure_up_fn=ensure_up_fn,
        health_fn=health_fn,
        served_model_fn=_blank_served,
    )
    b.ensure_ready("X")
    assert b.state == VLLMState.SERVING
    assert b._served_model == "X"  # optimistic fallback to the requested id


def test_ensure_ready_cold_start_times_out():
    """health_fn never True ⇒ ensure_ready raises TimeoutError. Driven by a fake
    clock that the sleep_fn advances, so no real waiting happens."""
    cell, now = _clock()

    server = _FakeServer(up=False, served=None)

    def kill_fn() -> bool:
        return False

    def ensure_up_fn(_m: str) -> None:
        # Simulate a sidecar that never becomes healthy (entrypoint wedged).
        server.ensure_up_calls.append(_m)
        # deliberately do NOT set server.up

    def health_fn() -> bool:
        return server.up  # stays False forever

    def served_model_fn() -> str | None:
        return None

    def advancing_sleep(secs: float) -> None:
        cell[0] += secs  # virtual time passes without a real sleep

    b = VLLMBackend(
        base_url="http://h:8000",
        idle_ttl_s=60.0,
        now_fn=now,
        sleep_fn=advancing_sleep,
        kill_fn=kill_fn,
        ensure_up_fn=ensure_up_fn,
        health_fn=health_fn,
        served_model_fn=served_model_fn,
    )
    with pytest.raises(TimeoutError):
        b.ensure_ready("X", timeout_s=5.0)
    # It DID attempt the bring-up before giving up.
    assert server.ensure_up_calls == ["X"]


# ── ensure_ready: model switch + sleep-L2 stub + sticky ──────────────────────


def test_ensure_ready_switch_attempts_sleep_l2_once_then_slow_path():
    """Up serving "A", asked for "B": Sleep-L2 is attempted exactly once (and
    raises NotImplementedError, the stub), then the slow path runs — kill the
    server, bring it up serving "B" — landing SERVING "B"."""
    server = _FakeServer(up=True, served="A")
    b = _make(server, served_model="A")

    sleep_l2_calls = {"n": 0}
    _orig = b._sleep_l2_reload

    def _spy(model_id: str):
        sleep_l2_calls["n"] += 1
        return _orig(model_id)  # still raises NotImplementedError

    b._sleep_l2_reload = _spy  # type: ignore[method-assign]

    b.ensure_ready("B")
    assert sleep_l2_calls["n"] == 1            # attempted exactly once
    assert server.kill_calls == 1             # slow path killed the old server
    assert server.ensure_up_calls == ["B"]    # and brought it up serving B
    assert b.state == VLLMState.SERVING
    assert b._served_model == "B"
    assert b.sleep_unsupported is True        # sticky flag now set


def test_sleep_l2_is_sticky_second_switch_skips_it():
    """After the first failed Sleep-L2, a SECOND switch (B→C) must NOT re-attempt
    Sleep-L2 — it goes straight to the slow path, and the UNSUPPORTED_SLEEP flag
    stays set. Proves the sticky-disable, not a per-call retry."""
    server = _FakeServer(up=True, served="A")
    b = _make(server, served_model="A")

    sleep_l2_calls = {"n": 0}
    _orig = b._sleep_l2_reload

    def _spy(model_id: str):
        sleep_l2_calls["n"] += 1
        return _orig(model_id)

    b._sleep_l2_reload = _spy  # type: ignore[method-assign]

    b.ensure_ready("B")          # first switch: attempts sleep-L2 once
    assert sleep_l2_calls["n"] == 1
    assert b.sleep_unsupported is True

    b.ensure_ready("C")          # second switch: must skip sleep-L2 entirely
    assert sleep_l2_calls["n"] == 1            # NOT re-attempted
    assert b.state == VLLMState.SERVING
    assert b._served_model == "C"
    assert server.ensure_up_calls == ["B", "C"]
    assert b.sleep_unsupported is True         # still sticky


def test_unsupported_sleep_state_reflected_after_attempt():
    """After a failed Sleep-L2 attempt, the UNSUPPORTED_SLEEP fact is observable as
    a property (so a test / log can see the sticky disable)."""
    server = _FakeServer(up=True, served="A")
    b = _make(server, served_model="A")
    assert b.sleep_unsupported is False
    b.ensure_ready("B")
    assert b.sleep_unsupported is True


# ── stop_server ──────────────────────────────────────────────────────────────


def test_stop_server_kills_running_then_idempotent():
    """A running server: stop_server kills once + returns True + state DEAD. A
    SECOND call returns False WITHOUT re-killing (idempotent — nothing to stop)."""
    server = _FakeServer(up=True, served="X")
    b = _make(server, served_model="X")
    # Make the backend believe it's serving (so stop has something to stop).
    b.ensure_ready("X")
    assert b.state == VLLMState.SERVING

    assert b.stop_server() is True
    assert server.kill_calls == 1
    assert b.state == VLLMState.DEAD
    assert b._served_model is None

    # Already dead ⇒ no-op.
    assert b.stop_server() is False
    assert server.kill_calls == 1             # NOT re-killed


def test_stop_server_on_never_started_is_a_noop():
    """A backend whose server was never up: stop_server is a truthful no-op (False,
    no kill) — the supervisor logs a non-reclaim."""
    server = _FakeServer(up=False, served=None)
    b = _make(server)
    assert b.state == VLLMState.DEAD
    assert b.stop_server() is False
    assert server.kill_calls == 0


# ── is_running ───────────────────────────────────────────────────────────────


def test_is_running_reflects_health_fn():
    server = _FakeServer(up=True, served="X")
    b = _make(server)
    assert b.is_running() is True
    server.up = False
    assert b.is_running() is False


def test_is_running_swallows_health_fn_errors():
    """A health probe that raises (network blip) must NOT propagate — is_running
    returns False so the supervisor / caller treats it as down."""

    def _boom() -> bool:
        raise RuntimeError("connection refused")

    b = VLLMBackend(
        base_url="http://h:8000",
        idle_ttl_s=60.0,
        health_fn=_boom,
        kill_fn=lambda: True,
        ensure_up_fn=lambda _m: None,
        served_model_fn=lambda: None,
        sleep_fn=lambda _s: None,
    )
    assert b.is_running() is False


# ── supervisor integration smoke ─────────────────────────────────────────────


def test_supervisor_reaps_idle_vllm_backend(monkeypatch):
    """A real LLMSupervisor wrapping THIS backend: with the backend idle (no
    inflight, clock advanced past idle_ttl_s), reap_idle_once must fire
    stop_server (kill called). Proves the backend satisfies the supervisor's
    duck-typed surface (server_type / inflight / idle_seconds / idle_ttl_s /
    is_running / stop_server)."""
    # conftest sets AI_LEADS_DISABLE_LLM_SUPERVISOR session-wide; clear it so the
    # backend's own server_type/ttl decide _enabled (we only call reap directly).
    monkeypatch.delenv("AI_LEADS_DISABLE_LLM_SUPERVISOR", raising=False)
    from queue_workflows.llm_backends.supervisor import LLMSupervisor

    cell, now = _clock()
    server = _FakeServer(up=True, served="X")
    b = _make(server, idle_ttl_s=60.0, served_model="X", now_fn=now)
    b.ensure_ready("X")
    assert b.state == VLLMState.SERVING

    # Advance the idle clock past the TTL (no request ever marked => idle since
    # construction; bump well beyond 60s).
    cell[0] = 1000.0
    assert b.idle_seconds() >= 60.0

    sup = LLMSupervisor(backend=b, poll_s=0.01)
    assert sup._enabled is True
    assert sup.reap_idle_once() is True       # decided to stop + stop returned True
    assert server.kill_calls == 1
    assert b.state == VLLMState.DEAD


# ── inherited request accounting wiring (super().__init__ sanity) ────────────


def test_request_accounting_inherited():
    """One quick assert that super().__init__ wired the base's request accounting:
    mark_request_start bumps inflight and a busy backend is never idle."""
    cell, now = _clock()
    b = _make(_FakeServer(), now_fn=now)
    assert b.inflight == 0
    b.mark_request_start("m")
    assert b.inflight == 1
    cell[0] = 1000.0
    assert b.idle_seconds() == 0.0            # busy ⇒ never idle
    assert b.current_model == "m"
    b.mark_request_end()
    assert b.inflight == 0
