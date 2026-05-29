"""Warm-model cache for the GPU claim worker.

A :class:`ModelCache` keeps a single loaded model handle across
consecutive jobs that want the same model — the core win over the
legacy step queue where each pipeline function re-loaded its weights
every call. ``require_model(id)`` returns the live handle, swapping
(drop + load) only on a model change. An optional idle-unload reaper
drops the warm handle after a quiet period to free VRAM for the
co-tenant vLLM / ollama server.

This module is deliberately decoupled from the worker-heartbeat
machinery and the DB:

  * the "publish my current_model" side effect is injected as a
    callback (``publish_current_model``) — the process-wide instance in
    ``gpu_model_cache`` wires it to the ``worker_heartbeats`` upsert; a
    unit test passes a spy or a no-op. The cache itself never imports
    psycopg.
  * model resolution goes through :mod:`model_registry`, including the
    empty-registry re-registration fallback. That fallback is the injected
    ``register_builtins`` hook — the engine default is the
    ``config.builtin_model_registrar`` (a no-op unless a host wires it),
    NOT an import of any host's ``builtin_models`` (the plan §2b-3 inversion).

The cache is concurrency-1 by contract (the GPU worker runs one job at
a time), but its handle + counters are guarded by an ``RLock`` because
the idle reaper runs on its own daemon thread.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable

log = logging.getLogger(__name__)


def gpu_should_unload(
    handle_present: bool, active: int, idle_s: float, ttl_s: float,
) -> bool:
    """Pure decision for the idle reaper: unload only when a model is
    loaded, no GPU task is running, and it's been idle at least
    ``ttl_s``. ``ttl_s <= 0`` disables idle unload entirely."""
    if ttl_s <= 0:
        return False
    return handle_present and active <= 0 and idle_s >= ttl_s


# Default idle TTL (s); <= 0 disables. Default 60s (1 min) — offload a
# warm model quickly so the shared GPU (the vLLM / ollama co-tenant on
# host-a) frees VRAM between bursts. Override per-host via the env var.
DEFAULT_IDLE_TTL_S = float(os.environ.get("AI_LEADS_GPU_MODEL_IDLE_TTL_S", "60"))


class ModelCache:
    """A single-slot warm-model cache.

    Attributes:

      * ``current_model`` / ``current_handle`` — the loaded slot.
      * ``active`` — GPU tasks executing now (0 or 1 under concurrency-1).
      * ``last_used`` — ``time.monotonic()`` of the last touch (used by
        the idle reaper).
      * ``idle_ttl_s`` — idle-unload TTL; <= 0 disables.
    """

    def __init__(
        self,
        *,
        publish_current_model: Callable[[str | None], None] | None = None,
        idle_ttl_s: float | None = None,
        register_builtins: Callable[[], None] | None = None,
    ) -> None:
        # Injected "advertise current_model" side effect. Default no-op so
        # a plain ModelCache has zero external dependencies.
        self._publish = publish_current_model or (lambda _m: None)
        # Injected idempotent builtin-model registration — the
        # empty-registry fallback. Default (None) falls through to the
        # engine's configured registrar (config.builtin_model_registrar),
        # which is a no-op unless a host wired it.
        self._register_builtins = register_builtins
        self._idle_ttl_s: float = (
            DEFAULT_IDLE_TTL_S if idle_ttl_s is None else float(idle_ttl_s)
        )

        self._current_model: str | None = None
        self._current_handle: Any = None
        self._cache_lock = threading.RLock()
        self._last_used: float = 0.0
        self._active: int = 0
        self._reaper_started: bool = False

    # ── read-only views (the shim proxies these) ─────────────────────────

    @property
    def current_model(self) -> str | None:
        return self._current_model

    @property
    def current_handle(self) -> Any:
        return self._current_handle

    @property
    def active(self) -> int:
        return self._active

    @property
    def last_used(self) -> float:
        return self._last_used

    @property
    def idle_ttl_s(self) -> float:
        return self._idle_ttl_s

    @property
    def lock(self) -> threading.RLock:
        return self._cache_lock

    # ── core ─────────────────────────────────────────────────────────────

    def require_model(self, model_id: str) -> Any:
        """Return the loaded handle for ``model_id``, swapping if the
        cache currently holds a different model. Never moves weights to
        CPU on switch — just drops the ref + forces gc + empty_cache +
        malloc_trim (see :meth:`drop_cache`)."""
        with self._cache_lock:
            # Touch the idle clock + arm the unload reaper (idempotent) so
            # a long-warm model is freed once traffic stops.
            self._last_used = time.monotonic()
            self.ensure_idle_reaper()
            if self._current_model == model_id and self._current_handle is not None:
                return self._current_handle
            self.drop_cache()
            # Mid-swap: publish current_model=NULL so the dispatcher's
            # affinity routing doesn't try to pin a same-model job here
            # while we're loading. Re-publish the new model_id after the
            # loader returns successfully.
            self._publish(None)
            from queue_workflows import model_registry
            try:
                spec = model_registry.get(model_id)
            except KeyError:
                # Belt-and-braces: the prefork worker_process_init signal
                # can race with the first task on a freshly-forked child.
                # If we land here, the registry is empty; re-run the
                # idempotent registration and retry.
                log.warning(
                    "[ModelCache] registry empty for %r — re-registering builtins",
                    model_id,
                )
                self._do_register_builtins()
                spec = model_registry.get(model_id)
            log.info("[ModelCache] loading %s", model_id)
            self._current_handle = spec.loader()
            self._current_model = model_id
            self._publish(model_id)
            return self._current_handle

    def _do_register_builtins(self) -> None:
        if self._register_builtins is not None:
            self._register_builtins()
            return
        # No constructor-injected registrar — fall through to the engine's
        # configured builtin-model registrar (a no-op unless a host wired one
        # via queue_workflows.set_builtin_model_registrar). NB this is the
        # plan §2b-3 inversion: the engine NEVER imports a host's
        # ``builtin_models`` here.
        from queue_workflows.config import get_config
        get_config().builtin_model_registrar()

    def drop_cache(self) -> None:
        """Drop the warm handle + force gc + empty_cache + malloc_trim."""
        with self._cache_lock:
            if self._current_handle is None:
                return
            import ctypes
            import gc
            log.info("[ModelCache] dropping cache %s", self._current_model)
            self._current_handle = None
            self._current_model = None
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

    # ── busy tracking (the prerun/postrun bracket) ───────────────────────

    def mark_busy(self) -> None:
        """A task started — keep the reaper from unloading mid-inference."""
        with self._cache_lock:
            self._active += 1

    def mark_idle(self) -> None:
        """A task ended — drop the busy count and restart the idle clock so
        the TTL counts from the END of the last task."""
        with self._cache_lock:
            self._active = max(0, self._active - 1)
            self._last_used = time.monotonic()

    # ── idle model unload ────────────────────────────────────────────────

    def ensure_idle_reaper(self) -> None:
        """Start the idle-unload daemon once. No-op when disabled
        (TTL<=0) or under tests (``AI_LEADS_DISABLE_GPU_IDLE_REAPER``)."""
        if (self._reaper_started
                or self._idle_ttl_s <= 0
                or os.environ.get("AI_LEADS_DISABLE_GPU_IDLE_REAPER")):
            return
        self._reaper_started = True
        threading.Thread(
            target=self._idle_reaper_loop,
            name="gpu-model-idle-reaper",
            daemon=True,
        ).start()
        log.info("[ModelCache] idle-unload reaper armed (TTL=%.0fs)", self._idle_ttl_s)

    def _idle_reaper_loop(self) -> None:
        poll = max(5.0, min(60.0, self._idle_ttl_s / 5.0))
        while True:
            time.sleep(poll)
            try:
                self.reap_idle_once()
            except Exception:
                log.exception("[ModelCache] idle reaper tick failed")

    def reap_idle_once(self) -> bool:
        """Drop the warm model iff one is loaded, nothing is running, and
        it's been idle ≥ TTL. Returns True if it unloaded."""
        with self._cache_lock:
            idle_s = time.monotonic() - self._last_used
            if not gpu_should_unload(
                self._current_handle is not None,
                self._active,
                idle_s,
                self._idle_ttl_s,
            ):
                return False
            log.info(
                "[ModelCache] %s idle %.0fs ≥ TTL %.0fs — unloading to free VRAM",
                self._current_model, idle_s, self._idle_ttl_s,
            )
            self.drop_cache()
            self._publish(None)
            return True


__all__ = ["ModelCache", "gpu_should_unload", "DEFAULT_IDLE_TTL_S"]
