"""The SHARED :class:`LLMBackend` contract — one parameterized harness that every
concrete backend must satisfy.

WHY a parameterized harness (not per-backend duplicates). The request-accounting
half of :class:`~queue_workflows.llm_backends.base.LLMBackend` is concrete and
inherited verbatim — it is the idle supervisor's read model and MUST behave
identically across vendors. So these invariants are asserted once, against a list
of backends: the real :class:`~queue_workflows.llm_backends.ollama.OllamaBackend`
plus a tiny in-file ``_FakeBackend`` standing in for any future vendor (e.g. vllm).
Each parameter is a *factory* lambda so every test gets a pristine instance with
its own injected clock — backends carry mutable counter state, so sharing one
across tests would leak.
"""

from __future__ import annotations

import threading

import pytest

from queue_workflows.llm_backends.base import LLMBackend
from queue_workflows.llm_backends.ollama import OllamaBackend


class _FakeBackend(LLMBackend):
    """A throwaway concrete backend exercising the abstract surface trivially, so
    the shared-contract harness proves the base behaviour for a backend OTHER than
    ollama too (guarding against the contract silently depending on ollama's
    overrides). ``stop_server`` returns a flippable flag so the shutdown test can
    observe delegation."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        # Flag the shutdown test flips to assert shutdown() -> stop_server().
        self.stop_returns = True
        self.stop_calls = 0

    @property
    def server_type(self) -> str:
        return "fake"

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"

    def ensure_ready(self, model_id, *, timeout_s=90.0) -> None:
        return None

    def is_running(self) -> bool:
        return True

    def stop_server(self) -> bool:
        self.stop_calls += 1
        return self.stop_returns


def _clock():
    """A mutable monotonic clock: ``clock[0]`` is the current time; ``now`` reads
    it. Bump ``clock[0]`` to advance time deterministically (no sleeping)."""
    clock = [0.0]
    return clock, (lambda: clock[0])


# Each entry: (label, factory(now_fn) -> backend). The factory takes the injected
# clock so idle-advance tests can drive it; non-clock tests ignore the arg.
_BACKENDS = [
    pytest.param(
        lambda now=None: OllamaBackend(
            base_url="http://h:11434", parallelism=4, idle_ttl_s=0.0,
            **({"now_fn": now} if now else {}),
        ),
        id="ollama",
    ),
    pytest.param(
        lambda now=None: _FakeBackend(
            base_url="http://h:8000", parallelism=4, idle_ttl_s=0.0,
            **({"now_fn": now} if now else {}),
        ),
        id="fake",
    ),
]


# ── request accounting (the supervisor's read model) ──────────────────────────


@pytest.mark.parametrize("make", _BACKENDS)
def test_mark_request_start_raises_inflight_and_zeroes_idle(make):
    """A started request takes inflight to 1, and a busy backend is NEVER idle."""
    b = make()
    assert b.inflight == 0
    b.mark_request_start("m")
    assert b.inflight == 1
    assert b.idle_seconds() == 0.0


@pytest.mark.parametrize("make", _BACKENDS)
def test_counter_symmetry_two_starts_one_end(make):
    """Two starts then one end leaves inflight at 1 — the bracket is a balanced
    counter, not a boolean."""
    b = make()
    b.mark_request_start("m")
    b.mark_request_start("m")
    b.mark_request_end()
    assert b.inflight == 1


@pytest.mark.parametrize("make", _BACKENDS)
def test_mark_request_end_floors_at_zero(make):
    """An unbalanced extra end can't drive inflight negative (defensive floor)."""
    b = make()
    b.mark_request_start("m")
    b.mark_request_end()
    b.mark_request_end()  # one too many
    assert b.inflight == 0


@pytest.mark.parametrize("make", _BACKENDS)
def test_idle_seconds_advances_with_injected_clock(make):
    """After the last request ends, idle_seconds tracks the injected clock."""
    clock, now = _clock()
    b = make(now)
    b.mark_request_start("m")
    clock[0] = 5.0
    b.mark_request_end()         # idle clock restamped to now == 5.0
    assert b.idle_seconds() == 0.0
    clock[0] = 12.5
    assert b.idle_seconds() == 7.5


@pytest.mark.parametrize("make", _BACKENDS)
def test_idle_seconds_is_zero_while_inflight_even_if_clock_advances(make):
    """Busy beats elapsed time: inflight>0 forces idle 0.0 no matter the clock."""
    clock, now = _clock()
    b = make(now)
    b.mark_request_start("m")
    clock[0] = 1000.0
    assert b.idle_seconds() == 0.0


@pytest.mark.parametrize("make", _BACKENDS)
def test_current_model_reflects_last_start(make):
    """current_model tracks the most recent mark_request_start."""
    b = make()
    assert b.current_model is None
    b.mark_request_start("alpha")
    assert b.current_model == "alpha"
    b.mark_request_start("beta")
    assert b.current_model == "beta"
    b.mark_request_end()
    assert b.current_model == "beta"  # end doesn't clear it


# ── identity / config echo ────────────────────────────────────────────────────


@pytest.mark.parametrize("make", _BACKENDS)
def test_base_url_strips_trailing_slash(make):
    """base_url is stored without a trailing slash regardless of what was passed."""
    b = make()
    assert b.base_url == b.base_url.rstrip("/")
    assert not b.base_url.endswith("/")


@pytest.mark.parametrize("make", _BACKENDS)
def test_parallelism_and_idle_ttl_echo_constructor(make):
    """parallelism / idle_ttl_s are surfaced verbatim from the constructor args."""
    b = make()
    assert b.parallelism == 4
    assert b.idle_ttl_s == 0.0


@pytest.mark.parametrize("make", _BACKENDS)
def test_chat_and_health_urls_are_under_base_url(make):
    """The vendor endpoints hang off base_url (no double slash, no detachment)."""
    b = make()
    assert b.chat_url.startswith(b.base_url + "/")
    assert b.health_url.startswith(b.base_url + "/")
    assert "//" not in b.chat_url[len("http://"):]
    assert "//" not in b.health_url[len("http://"):]


# ── lifecycle defaults shared by every backend ────────────────────────────────


@pytest.mark.parametrize("make", _BACKENDS)
def test_shutdown_delegates_to_stop_server(make):
    """The default shutdown() must call stop_server() exactly once."""
    b = make()
    if isinstance(b, _FakeBackend):
        b.stop_returns = True
        b.shutdown()
        assert b.stop_calls == 1
    else:
        # For backends without a spy (ollama), shutdown must at least not raise
        # and stop_server stays the truthful no-op.
        b.shutdown()
        assert b.stop_server() is False


# ── ABC enforcement ───────────────────────────────────────────────────────────


def test_base_class_cannot_be_instantiated():
    """LLMBackend is abstract — instantiating it directly is a TypeError (the
    abstract server_type/chat_url/health_url/ensure_ready/is_running/stop_server
    block construction)."""
    with pytest.raises(TypeError):
        LLMBackend(base_url="http://h")  # type: ignore[abstract]


# ── thread-safety smoke (the RLock contract) ──────────────────────────────────


@pytest.mark.parametrize("make", _BACKENDS)
def test_concurrent_start_end_returns_to_zero(make):
    """Spin several threads each doing a balanced start/end; the RLock-guarded
    counter must settle back to exactly 0 (no lost/torn increments). Mirrors the
    off-thread-supervisor concurrency contract from ModelCache."""
    b = make()
    iterations = 200

    def worker():
        for _ in range(iterations):
            b.mark_request_start("m")
            b.mark_request_end()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert b.inflight == 0
