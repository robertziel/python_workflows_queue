"""Unit tests for the warm-model ``ModelCache``.

  * one load per model across repeated ``require_model`` calls (warm hit);
  * a swap drops the old handle and loads the new one;
  * the publish callback fires ``None`` mid-swap then the new id, and is
    skipped on a warm hit;
  * the empty-registry re-registration fallback (constructor-injected
    ``register_builtins``);
  * the idle-reaper decision matrix + a single ``reap_idle_once`` tick.
"""

from __future__ import annotations

import time

from queue_workflows import model_registry
from queue_workflows.model_cache import ModelCache, gpu_should_unload


# ── decision matrix (pure) ────────────────────────────────────────────────


def test_should_unload_decision_matrix():
    ttl = 600.0
    assert gpu_should_unload(True, 0, 700, ttl) is True
    assert gpu_should_unload(True, 1, 700, ttl) is False
    assert gpu_should_unload(True, 0, 300, ttl) is False
    assert gpu_should_unload(False, 0, 700, ttl) is False
    assert gpu_should_unload(True, 0, 700, 0) is False
    assert gpu_should_unload(True, 0, ttl, ttl) is True


# ── require_model ──────────────────────────────────────────────────────────


def test_require_model_one_load_per_model():
    model_registry.clear_for_tests()
    loads: list[str] = []

    def _loader():
        loads.append("load")
        return object()

    model_registry.register(model_registry.ModelSpec(
        id="alpha", loader=_loader, est_vram_gb=1.0,
    ))
    try:
        cache = ModelCache(publish_current_model=lambda _m: None, idle_ttl_s=0.0)
        h1 = cache.require_model("alpha")
        h2 = cache.require_model("alpha")
        h3 = cache.require_model("alpha")
        assert h1 is h2 is h3
        assert len(loads) == 1
        assert cache.current_model == "alpha"
        model_registry.register(model_registry.ModelSpec(
            id="beta", loader=_loader, est_vram_gb=1.0,
        ))
        cache.require_model("beta")
        assert len(loads) == 2
        assert cache.current_model == "beta"
    finally:
        model_registry.clear_for_tests()


def test_require_model_publish_sequence_on_swap():
    model_registry.clear_for_tests()
    upserts: list = []
    model_registry.register(model_registry.ModelSpec(id="model_a", loader=lambda: object()))
    model_registry.register(model_registry.ModelSpec(id="model_b", loader=lambda: object()))
    try:
        cache = ModelCache(
            publish_current_model=lambda m: upserts.append(m), idle_ttl_s=0.0,
        )
        cache.require_model("model_a")
        cache.require_model("model_b")
        assert upserts == [None, "model_a", None, "model_b"]
    finally:
        model_registry.clear_for_tests()


def test_require_model_skips_publish_on_warm_hit():
    model_registry.clear_for_tests()
    calls: list = []
    model_registry.register(model_registry.ModelSpec(id="model_a", loader=lambda: object()))
    try:
        cache = ModelCache(
            publish_current_model=lambda m: calls.append(m), idle_ttl_s=0.0,
        )
        cache.require_model("model_a")
        cache.require_model("model_a")
        cache.require_model("model_a")
        assert calls == [None, "model_a"]
    finally:
        model_registry.clear_for_tests()


def test_require_model_reregisters_on_empty_registry():
    """Belt-and-braces: if the registry is empty when require_model runs (fork
    race in prod), the cache re-runs the idempotent builtin registration and
    retries rather than KeyError-ing. Here the registration is constructor-
    injected (the host's hook)."""
    model_registry.clear_for_tests()
    reg_calls = {"n": 0}

    def _fake_register():
        reg_calls["n"] += 1
        model_registry.register(model_registry.ModelSpec(
            id="late_model", loader=lambda: object(),
        ))

    cache = ModelCache(
        publish_current_model=lambda _m: None,
        idle_ttl_s=0.0,
        register_builtins=_fake_register,
    )
    try:
        handle = cache.require_model("late_model")
        assert handle is not None
        assert reg_calls["n"] == 1
        assert cache.current_model == "late_model"
    finally:
        model_registry.clear_for_tests()


def test_require_model_reregisters_via_config_registrar():
    """When NO constructor registrar is injected, the empty-registry fallback
    uses the engine's configured ``builtin_model_registrar`` (the plan §2b-3
    inversion — never an import of a host's builtin_models)."""
    import queue_workflows
    model_registry.clear_for_tests()
    reg_calls = {"n": 0}

    def _registrar():
        reg_calls["n"] += 1
        model_registry.register(model_registry.ModelSpec(
            id="cfg_model", loader=lambda: object(),
        ))

    queue_workflows.set_builtin_model_registrar(_registrar)
    try:
        cache = ModelCache(publish_current_model=lambda _m: None, idle_ttl_s=0.0)
        handle = cache.require_model("cfg_model")
        assert handle is not None
        assert reg_calls["n"] == 1
    finally:
        model_registry.clear_for_tests()


# ── idle reaper ────────────────────────────────────────────────────────────


def _warm_cache() -> ModelCache:
    cache = ModelCache(publish_current_model=lambda _m: None, idle_ttl_s=600.0)
    cache._current_model = "smoke_model"
    cache._current_handle = object()
    return cache


def test_reaps_idle_model():
    cache = _warm_cache()
    cache._active = 0
    cache._last_used = time.monotonic() - 9999
    assert cache.reap_idle_once() is True
    assert cache.current_handle is None
    assert cache.current_model is None


def test_keeps_model_while_a_task_is_running():
    cache = _warm_cache()
    cache._active = 1
    cache._last_used = time.monotonic() - 9999
    assert cache.reap_idle_once() is False
    assert cache.current_handle is not None


def test_keeps_recently_used_model():
    cache = _warm_cache()
    cache._active = 0
    cache._last_used = time.monotonic()
    assert cache.reap_idle_once() is False
    assert cache.current_handle is not None


def test_disabled_when_ttl_zero():
    cache = _warm_cache()
    cache._idle_ttl_s = 0.0
    cache._active = 0
    cache._last_used = time.monotonic() - 9999
    assert cache.reap_idle_once() is False
    assert cache.current_handle is not None


def test_mark_idle_resets_clock_and_busy():
    cache = _warm_cache()
    cache._active = 1
    cache._last_used = time.monotonic() - 9999
    cache.mark_idle()
    assert cache.active == 0
    assert cache.reap_idle_once() is False
    assert cache.current_handle is not None


def test_mark_busy_increments_active():
    cache = _warm_cache()
    cache._active = 0
    cache.mark_busy()
    assert cache.active == 1
