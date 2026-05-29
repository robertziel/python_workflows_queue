"""The per-``(host, queue)`` LLM-backend FACTORY — the live owner of the concrete
``LLMBackend`` + its ``LLMSupervisor``, kept in sync with the DB config (0013).

WHAT THIS PINS — and why it touches no network. The backend modules + supervisor
are already DB-free and unit-tested; the factory's NEW job is the read-through
cache and its identity contract: hand back the SAME backend instance while the
config snapshot ``(server_type, parallelism, vllm_idle_ttl_s)`` is unchanged (so the
request counters + the vllm state machine survive), rebuild ONLY on an actual
change, and refresh from the DB on a TTL / a NOTIFY invalidation. To keep that pure,
every side effect is injected: a ``build_backend_fn`` returns in-file FAKE backends
(recordable ``shutdown``, settable ``server_type``); a spy ``supervisor_factory``
records construction + start/stop; the TTL clock is a mutable cell. The DB read
(``worker_control.llm_config_for``) IS real — conftest provides the ``_test`` DB and
truncates ``worker_controls`` between tests; we call ``reset_default_for_tests`` in a
fixture so the process singleton's state never leaks.
"""

from __future__ import annotations

import threading

import pytest

from queue_workflows import worker_control
from queue_workflows.db import db_url
from queue_workflows.llm_backends import factory as factory_mod
from queue_workflows.llm_backends.factory import BackendFactory


# ── in-file fakes (no network, recordable) ──────────────────────────────────


class _FakeBackend:
    """A settable stand-in for a real LLM backend, capturing exactly what the
    factory hands it + recording ``shutdown`` so a rebuild/teardown is observable."""

    def __init__(
        self, *, server_type: str, base_url: str, parallelism: int, idle_ttl_s: float,
    ) -> None:
        self.server_type = server_type
        self.base_url = base_url
        self.parallelism = parallelism
        self.idle_ttl_s = idle_ttl_s
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _SpySupervisor:
    """A spy ``LLMSupervisor`` recording construction + start/stop on the backend
    it wraps, so a test asserts a vllm backend got a supervisor (and an ollama one
    did not), and that a rebuild stopped the old one."""

    def __init__(self, backend: _FakeBackend) -> None:
        self.backend = backend
        self.start_calls = 0
        self.stop_calls = 0
        backend._supervisor = self  # type: ignore[attr-defined]

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def _clock():
    """A mutable monotonic clock: ``cell[0]`` is now; the returned fn reads it.
    Bump the cell to cross the TTL without sleeping."""
    cell = [0.0]
    return cell, (lambda: cell[0])


def _make_factory(*, ttl_s=10.0, now_fn=None, ollama_url=None, vllm_url=None):
    """A factory wired to the in-file fakes: ``build_backend_fn`` returns a
    ``_FakeBackend`` carrying the args the factory passed; ``supervisor_factory``
    returns a ``_SpySupervisor``. The built backends are tracked on the returned
    ``built`` list, the supervisors on ``supervisors``."""
    built: list[_FakeBackend] = []
    supervisors: list[_SpySupervisor] = []

    def build(server_type, base_url, parallelism, idle_ttl_s):
        b = _FakeBackend(
            server_type=server_type, base_url=base_url,
            parallelism=parallelism, idle_ttl_s=idle_ttl_s,
        )
        built.append(b)
        return b

    def make_sup(backend):
        s = _SpySupervisor(backend)
        supervisors.append(s)
        return s

    f = BackendFactory(
        ttl_s=ttl_s,
        now_fn=now_fn or (lambda: 0.0),
        build_backend_fn=build,
        supervisor_factory=make_sup,
        ollama_url=ollama_url,
        vllm_url=vllm_url,
    )
    return f, built, supervisors


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the process-wide _DEFAULT factory between tests so a test that touched
    the module singleton (start/get/stop) doesn't leak its cache / running thread."""
    factory_mod.reset_default_for_tests()
    yield
    factory_mod.reset_default_for_tests()


# ── DB read-through: vllm + ollama ───────────────────────────────────────────


def test_reads_vllm_config_from_db_and_builds_vllm_backend():
    worker_control.set_llm_config(
        "h", "gpu", server_type="vllm", parallelism=128, vllm_idle_ttl_s=30,
    )
    f, built, supervisors = _make_factory(vllm_url="http://h:8000/v1")
    backend = f.get_backend("h", "gpu")
    assert backend.server_type == "vllm"
    assert backend.base_url == "http://h:8000"      # /v1 stripped to root
    assert backend.parallelism == 128
    assert backend.idle_ttl_s == 30                 # vllm gets the configured TTL
    assert len(built) == 1
    # vllm ⇒ a supervisor was constructed + armed.
    assert len(supervisors) == 1
    assert supervisors[0].backend is backend
    assert supervisors[0].start_calls == 1


def test_defaults_to_ollama_with_zero_ttl_and_no_supervisor_when_no_row():
    f, built, supervisors = _make_factory(ollama_url="http://ol:11434")
    backend = f.get_backend("nobody", "gpu")
    assert backend.server_type == "ollama"
    assert backend.base_url == "http://ol:11434"
    assert backend.parallelism == 1                 # DEFAULT_LLM_PARALLELISM
    assert backend.idle_ttl_s == 0                  # ollama self-manages idle
    assert len(built) == 1
    assert supervisors == []                         # ollama ⇒ NO supervisor


# ── identity preservation + TTL throttle ─────────────────────────────────────


def test_identity_preserved_within_ttl_even_after_a_db_change():
    """Two get_backend calls inside the TTL return the SAME instance with NO DB
    re-read — so even a config change written between them is NOT observed yet
    (the TTL throttle). This proves the request loop isn't SELECTing every call."""
    cell, now = _clock()
    worker_control.set_llm_config("h", "gpu", server_type="vllm", parallelism=8)
    f, built, _ = _make_factory(ttl_s=10.0, now_fn=now)

    first = f.get_backend("h", "gpu")
    # Change the DB, but stay inside the TTL window (clock not advanced).
    worker_control.set_llm_config("h", "gpu", parallelism=999)
    second = f.get_backend("h", "gpu")

    assert second is first, "within TTL ⇒ same instance, no DB read"
    assert len(built) == 1
    assert first.parallelism == 8, "the throttled call did not pick up the change"


def test_no_rebuild_after_ttl_when_snapshot_unchanged():
    """Past the TTL the factory DOES re-read the DB, but an UNCHANGED snapshot keeps
    the same instance — a re-read is not a rebuild."""
    cell, now = _clock()
    worker_control.set_llm_config("h", "gpu", server_type="vllm", parallelism=8)
    f, built, _ = _make_factory(ttl_s=10.0, now_fn=now)

    first = f.get_backend("h", "gpu")
    cell[0] += 999.0                                 # cross the TTL
    second = f.get_backend("h", "gpu")

    assert second is first, "unchanged snapshot ⇒ keep the instance"
    assert len(built) == 1


def test_rebuild_after_ttl_when_config_changed_shuts_down_old():
    """Past the TTL WITH a snapshot change (parallelism 128→256) ⇒ a NEW instance,
    and the OLD one's shutdown() ran (frees a vllm sidecar)."""
    cell, now = _clock()
    worker_control.set_llm_config(
        "h", "gpu", server_type="vllm", parallelism=128, vllm_idle_ttl_s=30,
    )
    f, built, supervisors = _make_factory(ttl_s=10.0, now_fn=now)

    old = f.get_backend("h", "gpu")
    worker_control.set_llm_config("h", "gpu", parallelism=256)
    cell[0] += 999.0                                 # cross the TTL → re-read
    new = f.get_backend("h", "gpu")

    assert new is not old, "changed snapshot ⇒ rebuild"
    assert new.parallelism == 256
    assert old.shutdown_calls == 1, "the replaced backend was shut down"
    # vllv→vllv rebuild: old supervisor stopped, a fresh one started.
    assert len(supervisors) == 2
    assert supervisors[0].stop_calls == 1
    assert supervisors[1].start_calls == 1


def test_invalidate_forces_rebuild_even_within_ttl():
    """invalidate(h,'gpu') (what the NOTIFY listener calls) forces the next call to
    re-read despite the TTL — so an operator's edit takes effect at once."""
    cell, now = _clock()
    worker_control.set_llm_config("h", "gpu", server_type="vllm", parallelism=8)
    f, built, _ = _make_factory(ttl_s=10.0, now_fn=now)

    first = f.get_backend("h", "gpu")
    worker_control.set_llm_config("h", "gpu", parallelism=64)
    f.invalidate("h", "gpu")                         # NOTIFY-driven re-read
    second = f.get_backend("h", "gpu")               # clock NOT advanced

    assert second is not first, "invalidate bypasses the TTL throttle"
    assert second.parallelism == 64
    assert first.shutdown_calls == 1


# ── server-type switch ↔ supervisor lifecycle ────────────────────────────────


def test_switch_ollama_to_vllm_builds_a_supervisor():
    cell, now = _clock()
    worker_control.set_llm_config("h", "gpu", server_type="ollama")
    f, built, supervisors = _make_factory(ttl_s=10.0, now_fn=now)

    ollama = f.get_backend("h", "gpu")
    assert ollama.server_type == "ollama"
    assert supervisors == [], "ollama ⇒ no supervisor"

    worker_control.set_llm_config("h", "gpu", server_type="vllm")
    cell[0] += 999.0
    vllm = f.get_backend("h", "gpu")
    assert vllm.server_type == "vllm"
    assert vllm is not ollama
    assert len(supervisors) == 1, "the switch to vllm built + armed a supervisor"
    assert supervisors[0].start_calls == 1


def test_switch_vllm_to_ollama_stops_old_supervisor_and_starts_none():
    cell, now = _clock()
    worker_control.set_llm_config("h", "gpu", server_type="vllm", parallelism=8)
    f, built, supervisors = _make_factory(ttl_s=10.0, now_fn=now)

    vllm = f.get_backend("h", "gpu")
    assert len(supervisors) == 1

    worker_control.set_llm_config("h", "gpu", server_type="ollama")
    cell[0] += 999.0
    ollama = f.get_backend("h", "gpu")
    assert ollama.server_type == "ollama"
    assert ollama is not vllm
    assert vllm.shutdown_calls == 1
    assert supervisors[0].stop_calls == 1, "the old vllm supervisor was stopped"
    assert len(supervisors) == 1, "ollama gets NO new supervisor"


# ── URL normalization ─────────────────────────────────────────────────────────


def test_vllm_url_normalized_to_root():
    """The deployed ai_leads env value is the OpenAI base (…/8000/v1/); the backend
    appends /v1/chat/completions itself, so the factory strips a trailing /v1 (+
    slash) to the server root."""
    worker_control.set_llm_config("h", "gpu", server_type="vllm")
    f, _, _ = _make_factory(vllm_url="http://h:8000/v1/")
    backend = f.get_backend("h", "gpu")
    assert backend.base_url == "http://h:8000"


def test_vllm_url_already_root_is_unchanged():
    worker_control.set_llm_config("h", "gpu", server_type="vllm")
    f, _, _ = _make_factory(vllm_url="http://h:8000/")
    assert f.get_backend("h", "gpu").base_url == "http://h:8000"


def test_vllm_url_resolves_from_config_env(monkeypatch):
    """No override ⇒ read the VALUE off the env NAME on the config
    (ollama_url_env / vllm_url_env), with the same /v1-stripping normalization."""
    from queue_workflows.config import get_config

    monkeypatch.setenv(get_config().vllm_url_env, "http://envhost:8000/v1")
    worker_control.set_llm_config("h", "gpu", server_type="vllm")
    f, _, _ = _make_factory()                        # no url overrides
    assert f.get_backend("h", "gpu").base_url == "http://envhost:8000"


def test_ollama_url_default_when_env_unset(monkeypatch):
    from queue_workflows.config import get_config

    monkeypatch.delenv(get_config().ollama_url_env, raising=False)
    f, _, _ = _make_factory()                        # no override, no env
    assert f.get_backend("nobody", "gpu").base_url == factory_mod.DEFAULT_OLLAMA_URL


# ── stop() releases everything ─────────────────────────────────────────────────


def test_stop_shuts_down_all_cached_backends_and_supervisors():
    worker_control.set_llm_config("h1", "gpu", server_type="vllm", parallelism=8)
    worker_control.set_llm_config("h2", "gpu", server_type="ollama")
    f, built, supervisors = _make_factory()
    b1 = f.get_backend("h1", "gpu")                  # vllm + supervisor
    b2 = f.get_backend("h2", "gpu")                  # ollama, no supervisor
    assert len(built) == 2 and len(supervisors) == 1

    f.stop()
    assert b1.shutdown_calls == 1
    assert b2.shutdown_calls == 1
    assert supervisors[0].stop_calls == 1
    # The cache is cleared — a get after stop rebuilds.
    b1b = f.get_backend("h1", "gpu")
    assert b1b is not b1


# ── listener gate ──────────────────────────────────────────────────────────────


def test_listener_disabled_by_env_spawns_no_thread():
    """With the session-wide test gate set (conftest default), start() spawns no
    daemon — tests never open a LISTEN connection."""
    # conftest sets AI_LEADS_DISABLE_LLM_CONFIG_LISTENER=1; assert the gate holds.
    f, _, _ = _make_factory()
    assert f._listener_enabled is False
    f.start()
    assert f._listener_thread is None
    f.stop()                                          # safe when never started


def test_module_functions_delegate_to_default_singleton():
    """The thin module functions drive the process-wide _DEFAULT. With the listener
    gated off, start()/stop() are safe no-ops on the thread; get_backend reads the
    DB through the real (default-seam) factory."""
    worker_control.set_llm_config("host-c", "gpu", server_type="ollama")
    factory_mod.start()                               # gated ⇒ no thread
    backend = factory_mod.get_backend("host-c", "gpu")
    assert backend.server_type == "ollama"
    factory_mod.invalidate("host-c", "gpu")          # no raise
    factory_mod.stop()


# ── real-listener smoke (gate cleared) ─────────────────────────────────────────


def test_real_listener_invalidates_on_notify(monkeypatch):
    """End-to-end: with the gate CLEARED, start() spawns the LISTEN daemon; a
    set_llm_config fires the 0013 NOTIFY on worker_llm_config_changed; the daemon
    invalidates the keyed entry so the next get_backend re-reads. Kept fast +
    robust: a short poll loop, and the entry is pre-built so 'invalidated' is the
    only thing under test."""
    monkeypatch.delenv("AI_LEADS_DISABLE_LLM_CONFIG_LISTENER", raising=False)
    worker_control.set_llm_config("smoke", "gpu", server_type="ollama")
    f, built, _ = _make_factory(ttl_s=999.0)          # huge TTL: only NOTIFY can refresh
    assert f._listener_enabled is True

    entry_key = ("smoke", "gpu")
    f.get_backend("smoke", "gpu")                     # build + cache the entry
    f.start()
    try:
        # Make sure the daemon's LISTEN is established before we fire the NOTIFY
        # (a NOTIFY delivered before LISTEN is missed). Poll for the thread + a
        # tiny settle via a dedicated LISTEN of our own as a barrier is overkill;
        # instead fire repeatedly until the entry flips invalid (idempotent set).
        import time as _time

        # ALTERNATE the value every iteration: the 0013 trigger NOTIFYs only when
        # an LLM column actually changes, so repeating the SAME parallelism would
        # fire just one NOTIFY (on the first change) — and if the daemon's LISTEN
        # wasn't registered yet for that single event, the test could never
        # recover (the LISTEN-before-NOTIFY race). Toggling guarantees a fresh
        # NOTIFY on every loop, so the test is robust once LISTEN is established.
        deadline = _time.time() + 8.0
        seen_invalid = False
        i = 0
        while _time.time() < deadline:
            i += 1
            worker_control.set_llm_config("smoke", "gpu", parallelism=7 + (i % 2))
            _time.sleep(0.05)
            with f._lock:
                ent = f._cache.get(entry_key)
            if ent is not None and ent.invalid:
                seen_invalid = True
                break
        assert seen_invalid, "the NOTIFY-driven listener must invalidate the entry"
    finally:
        f.stop()


# ── vllm lifecycle hooks (set_vllm_lifecycle → the default-built backend) ──────


def test_default_build_threads_vllm_lifecycle_hooks():
    """The DEFAULT build wires ``config.vllm_stop_fn`` / ``vllm_start_fn`` into the
    real VLLMBackend's kill / ensure_up seams, so a host that called
    ``set_vllm_lifecycle`` (ai_leads → docker-over-UDS) controls the SIBLING
    sidecar. Uses the real ``_default_build_backend`` (not a fake)."""
    import queue_workflows

    stopped = []
    started = []
    queue_workflows.set_vllm_lifecycle(
        lambda: (stopped.append(True), True)[1],
        lambda model: started.append(model),
    )
    backend = factory_mod._default_build_backend(
        worker_control.SERVER_TYPE_VLLM, "http://h:8000", 128, 60.0,
    )
    # The configured stop_fn must BE the backend's kill seam (what stop_server
    # delegates to on idle) — not the pkill default. Exercise it directly: a
    # freshly-built backend is DEAD, so stop_server() short-circuits without
    # calling kill, which is why we assert the wired seam itself.
    assert backend._kill_fn() is True
    assert stopped == [True]
    # ensure_up (the cold-start LOADING transition) must route to start_fn.
    backend._ensure_up_fn("Qwen/Qwen2.5-VL")
    assert started == ["Qwen/Qwen2.5-VL"]


def test_default_build_vllm_without_hooks_uses_builtin_defaults():
    """No host hook ⇒ the backend keeps its built-in seams (None → default), so an
    unconfigured deployment is unchanged (the conftest reset clears any prior hook)."""
    backend = factory_mod._default_build_backend(
        worker_control.SERVER_TYPE_VLLM, "http://h:8000", 8, 60.0,
    )
    # The default kill_fn is the module's pkill helper, NOT None.
    from queue_workflows.llm_backends import vllm as vllm_mod
    assert backend._kill_fn is vllm_mod._default_kill_fn
