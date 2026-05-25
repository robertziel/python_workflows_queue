"""Process-wide GPU warm-model cache + its ``current_model`` advertise.

The live PG claim worker owns the GPU warm slot here. A GPU worker is one
process holding one model (concurrency-1 by contract), so one cache per
process is right.

The cache logic itself lives in :mod:`model_cache` (decoupled from the DB).
This module wires the single process-wide instance and injects the "publish my
current_model to ``worker_heartbeats``" side effect — the gauge's GPU busy
signal + the dispatcher's affinity-routing input. The injection stays a
late-bound shim so a test that monkeypatches :func:`_publish_current_model` is
still observed by the cache.
"""

from __future__ import annotations

import logging
import os
import socket

from queue_workflows.model_cache import ModelCache

log = logging.getLogger(__name__)


# Lazily-constructed so importing this module doesn't read the idle-TTL env
# before tests / compose can set it. The singleton is the GPU worker
# process's single warm slot.
_GPU_MODEL_CACHE: ModelCache | None = None


def gpu_model_cache() -> ModelCache:
    """Return the process-wide warm-model cache, constructing it on first
    use. The advertise-side effect is wired via a late-bound shim that
    re-reads :func:`_publish_current_model` off this module on every call —
    so a test that monkeypatches it (incl. the mid-load
    ``current_model=NULL`` publish) is still observed by the cache. The
    cache itself never imports psycopg."""
    global _GPU_MODEL_CACHE
    if _GPU_MODEL_CACHE is None:
        _GPU_MODEL_CACHE = ModelCache(
            publish_current_model=lambda m: _publish_current_model(m),
        )
    return _GPU_MODEL_CACHE


def _reset_gpu_model_cache_for_tests() -> None:
    """TEST-ONLY. Drop the process-wide cache so the next
    :func:`gpu_model_cache` builds a fresh one — keeps the warm-slot state
    (current_model, active count, idle TTL) from leaking across tests."""
    global _GPU_MODEL_CACHE
    _GPU_MODEL_CACHE = None


def _publish_current_model(model_id: str | None) -> None:
    """Update ``worker_heartbeats.current_model`` for THIS GPU worker so
    the dispatcher's affinity routing + the queue gauge can see what's
    loaded. Called by ``ModelCache.require_model`` — once with ``None``
    mid-swap, then with the new model_id after the loader returns.

    No-op when ``AI_LEADS_DISABLE_WORKER_HEARTBEAT`` is set (tests).
    ``current_model`` is GPU-only by design, and the GPU cache is only ever
    constructed by a gpu-queue worker, so this always upserts the ``gpu``
    row. Failures are swallowed: a transient DB blip should not crash a
    worker that already has the model loaded successfully.
    """
    if os.environ.get("AI_LEADS_DISABLE_WORKER_HEARTBEAT"):
        return
    from queue_workflows.config import get_config
    host = (
        os.environ.get(get_config().host_label_env, "").strip()
        or socket.gethostname()
    )
    try:
        from queue_workflows import model_registry, node_queue
        node_queue.upsert_worker_heartbeat(
            host_label=host, queue="gpu",
            concurrency=1,
            current_model=model_id,
            known_models=model_registry.known_ids(),
        )
    except Exception:
        log.exception(
            "[worker_heartbeat] current_model upsert failed (%s)", model_id,
        )
