"""``ModelCache.require_model`` publishes ``current_model`` to
``worker_heartbeats`` so the dispatcher's affinity routing + the queue gauge
know which box has which model warm.

Three states matter:
  * ``current_model=NULL`` mid-swap, BEFORE the loader runs.
  * ``current_model=<id>`` after the loader returns successfully.
  * No-op when the same model is requested twice (warm cache hit).

The advertise lives in ``gpu_model_cache._publish_current_model`` (wired into
the process-wide cache's ``publish_current_model`` callback); these drive it
through a real :class:`ModelCache` exactly as the GPU claim worker does.
"""

from __future__ import annotations

import pytest

from queue_workflows import gpu_model_cache, model_registry
from queue_workflows.db import connection
from queue_workflows.model_cache import ModelCache


@pytest.fixture(autouse=True)
def _wipe_heartbeats(monkeypatch):
    # Allow current_model upserts in this test file (the suite-wide
    # AI_LEADS_DISABLE_WORKER_HEARTBEAT skip is for the daemon thread, but
    # _publish_current_model honours the same flag — override here). The
    # host-label env defaults to AI_LEADS_HOST_LABEL (config.host_label_env).
    monkeypatch.delenv("AI_LEADS_DISABLE_WORKER_HEARTBEAT", raising=False)
    monkeypatch.setenv("AI_LEADS_HOST_LABEL", "test-host")
    model_registry.clear_for_tests()
    gpu_model_cache._reset_gpu_model_cache_for_tests()
    with connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM worker_heartbeats")
    yield
    gpu_model_cache._reset_gpu_model_cache_for_tests()
    model_registry.clear_for_tests()


def _cache() -> ModelCache:
    return ModelCache(
        publish_current_model=lambda m: gpu_model_cache._publish_current_model(m),
    )


def _row(host: str = "test-host"):
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT host_label, queue, current_model FROM worker_heartbeats "
            "WHERE host_label=%s AND queue='gpu'",
            (host,),
        )
        return cur.fetchone()


def test_require_model_publishes_loaded_id():
    model_registry.register(model_registry.ModelSpec(
        id="smoke_test_model", loader=lambda: object(),
    ))

    cache = _cache()
    handle = cache.require_model("smoke_test_model")
    assert handle is not None

    row = _row()
    assert row is not None
    assert row["current_model"] == "smoke_test_model"


def test_require_model_clears_then_publishes_on_swap(monkeypatch):
    upserts: list = []
    real = gpu_model_cache._publish_current_model

    def spy(model_id):
        upserts.append(model_id)
        real(model_id)
    monkeypatch.setattr(gpu_model_cache, "_publish_current_model", spy)

    model_registry.register(model_registry.ModelSpec(id="model_a", loader=lambda: object()))
    model_registry.register(model_registry.ModelSpec(id="model_b", loader=lambda: object()))

    cache = _cache()
    cache.require_model("model_a")
    cache.require_model("model_b")

    assert upserts == [None, "model_a", None, "model_b"]


def test_require_model_skips_publish_on_warm_hit(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        gpu_model_cache, "_publish_current_model", lambda m: calls.append(m),
    )
    model_registry.register(model_registry.ModelSpec(id="model_a", loader=lambda: object()))

    cache = _cache()
    cache.require_model("model_a")
    cache.require_model("model_a")
    cache.require_model("model_a")

    assert calls == [None, "model_a"]
