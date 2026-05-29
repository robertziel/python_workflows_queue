"""GPU idle-unload: a warm model is dropped after the idle TTL of no traffic,
freeing VRAM — but never while a GPU job is running.

The reaper *thread* is disabled under tests (conftest sets
AI_LEADS_DISABLE_GPU_IDLE_REAPER); we drive its logic directly via the pure
decision helper and a single ``reap_idle_once`` tick on a real ModelCache.
"""

from __future__ import annotations

import sys
import time
import types
import uuid

import pytest

import queue_workflows
from queue_workflows import (
    claim_worker as _claim_worker,
    gpu_model_cache,
    model_cache as _model_cache,
    node_queue as _node_queue,
)
from queue_workflows.model_cache import ModelCache, gpu_should_unload
from tests._helpers import make_run


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Reset the process-wide ModelCache so each test sees a clean warm slot."""
    gpu_model_cache._reset_gpu_model_cache_for_tests()
    yield
    gpu_model_cache._reset_gpu_model_cache_for_tests()


def test_should_unload_decision_matrix():
    ttl = 600.0
    assert gpu_should_unload(True, 0, 700, ttl) is True
    assert gpu_should_unload(True, 1, 700, ttl) is False
    assert gpu_should_unload(True, 0, 300, ttl) is False
    assert gpu_should_unload(False, 0, 700, ttl) is False
    assert gpu_should_unload(True, 0, 700, 0) is False
    assert gpu_should_unload(True, 0, ttl, ttl) is True


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


def test_keeps_model_while_a_job_is_running():
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


def test_mark_idle_resets_idle_clock_and_busy():
    cache = _warm_cache()
    cache._active = 1
    cache._last_used = time.monotonic() - 9999
    cache.mark_idle()
    assert cache.active == 0
    assert cache.reap_idle_once() is False
    assert cache.current_handle is not None


# ── claim-worker path: the busy-bracket that protects mid-inference ──────────


@pytest.fixture(autouse=True)
def _fake_node_pkg():
    """Resolve fake node modules under a test package."""
    queue_workflows.set_node_module_package("qwf_idle_nodes")
    yield


def _make_gpu_run() -> str:
    return make_run(workflow_name="_idle_unload_test", out_dir="/tmp/out")


def _install_node(name: str, run_fn):
    mod = types.ModuleType(f"qwf_idle_nodes.{name}")
    mod.run = run_fn
    sys.modules[f"qwf_idle_nodes.{name}"] = mod


def _warm_real_cache() -> _model_cache.ModelCache:
    cache = _model_cache.ModelCache(idle_ttl_s=600.0)
    cache._current_model = "qwen_edit"
    cache._current_handle = object()
    cache._last_used = time.monotonic() - 9999
    cache.require_model = lambda model_id: cache._current_handle  # type: ignore[assignment]
    return cache


def test_run_node_holds_active_during_inference_so_reaper_keeps_model():
    run_id = _make_gpu_run()
    cache = _warm_real_cache()
    observed: dict = {}

    def run(*, inputs=None, out=None, model_handle=None, status_callback=None,
            cancel_event=None, model_load_seconds=None):
        observed["active_during"] = cache.active
        observed["reaped_mid_job"] = cache.reap_idle_once()
        observed["handle_after_reap"] = cache.current_handle
        return {"context_delta": {"ok": True}}

    _install_node("_idle_gpu_busy", run)
    _node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="_idle_gpu_busy", queue="gpu",
        required_model="qwen_edit",
    )

    worker = _claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=cache)
    assert worker.run_once() is True

    assert observed["active_during"] >= 1
    assert observed["reaped_mid_job"] is False
    assert observed["handle_after_reap"] is not None


def test_run_node_releases_active_after_job_so_reaper_can_unload():
    run_id = _make_gpu_run()
    cache = _warm_real_cache()

    def run(*, inputs=None, out=None, model_handle=None, status_callback=None,
            cancel_event=None, model_load_seconds=None):
        return {"context_delta": {"ok": True}}

    _install_node("_idle_gpu_done", run)
    _node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="_idle_gpu_done", queue="gpu",
        required_model="qwen_edit",
    )

    worker = _claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=cache)
    worker.run_once()

    assert cache.active == 0
    cache._last_used = time.monotonic() - 9999
    assert cache.reap_idle_once() is True
    assert cache.current_handle is None


def test_run_node_releases_active_even_when_node_raises():
    run_id = _make_gpu_run()
    cache = _warm_real_cache()

    def run(*, inputs=None, out=None, model_handle=None, status_callback=None,
            cancel_event=None, model_load_seconds=None):
        raise RuntimeError("boom mid-inference")

    _install_node("_idle_gpu_raise", run)
    job_id = _node_queue.enqueue_node_job(
        run_id=run_id, node_id="g", node_module="_idle_gpu_raise", queue="gpu",
        required_model="qwen_edit",
    )

    worker = _claim_worker.ClaimWorker(queue="gpu", host="host-a", model_cache=cache)
    assert worker.run_once() is True
    assert _node_queue.get_node_job(job_id)["status"] == "failed"
    assert cache.active == 0


def test_run_node_cpu_does_not_touch_busy_bracket():
    run_id = _make_gpu_run()

    def run(*, inputs=None, out=None, model_handle=None, status_callback=None,
            cancel_event=None):
        return {"context_delta": {"ok": True}}

    _install_node("_idle_cpu_node", run)
    job_id = _node_queue.enqueue_node_job(
        run_id=run_id, node_id="c", node_module="_idle_cpu_node", queue="cpu",
    )
    worker = _claim_worker.ClaimWorker(queue="cpu", host="host-c")
    assert worker.run_once() is True
    assert _node_queue.get_node_job(job_id)["status"] == "completed"
