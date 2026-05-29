"""LLM idle-VRAM supervisor — frees a vllm sidecar's VRAM after a quiet window.

WHY THIS EXISTS. ollama and vLLM occupy the same shared GPU as the warm-model
cache (the ``ModelCache`` co-tenant on host-a / host-b). ollama unloads an idle
model on its own (its ``keep_alive`` / OLLAMA_KEEP_ALIVE timer). vLLM does NOT:
it pins the model in VRAM for the entire lifetime of the server process, with no
built-in idle eviction. So when a host is configured for the vllm backend, the
engine runs a SUPERVISOR daemon that watches LLM-request activity and, after the
configured idle window, SIGTERMs the vllm sidecar to give that VRAM back to the
GPU worker between bursts.

This class only decides WHEN to stop — the actual kill is
``backend.stop_server()`` (the vLLM backend SIGTERMs its docker sidecar; docker's
``restart: unless-stopped`` brings it back on the next request, exactly the way
worker_control's HARD stop relies on the container supervisor for the restart
leg). For an ollama backend the supervisor is INERT: ollama's own daemon
self-manages idle, so stopping it here would only fight that mechanism.

DECOUPLING. The supervisor is GENERIC over any backend object exposing the
duck-typed :class:`_Backend` surface below — it imports NOTHING from the concrete
``base`` / ``ollama`` / ``vllm`` backend modules (those are authored separately;
keeping this module free of them avoids an import cycle and lets a test drive a
plain fake). It depends only on the stdlib.

MIRRORS ``model_cache.py``. The shape is intentionally identical to that
module's GPU idle reaper so the two read the same:

  * the pure decision ``vllm_should_stop`` is the analog of ``gpu_should_unload``
    (running + nothing in flight + idle ≥ ttl; ttl<=0 disables);
  * :meth:`LLMSupervisor.reap_idle_once` is one tick of the reaper;
  * :meth:`_loop` polls on the same ``max(5, min(60, ttl/5))`` cadence;
  * :meth:`start` gates on :attr:`_enabled` (the ``AI_LEADS_DISABLE_LLM_SUPERVISOR``
    env kill-switch, plus the vllm-only / positive-ttl conditions).

The daemon-thread plumbing (``start`` / ``stop`` / ``threading.Event`` /
injectable ``sleep_fn`` / ``_enabled`` env gate) follows worker_control's
:class:`WorkerControlWatcher`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable

log = logging.getLogger(__name__)


# ── poll-cadence bounds (mirror ModelCache._idle_reaper_loop's clamp) ───────────

#: Never poll faster than this (a tiny TTL shouldn't busy-spin the daemon).
DEFAULT_SUPERVISOR_POLL_FLOOR_S = 5.0
#: Never poll slower than this (a huge TTL still re-checks at least this often).
DEFAULT_SUPERVISOR_POLL_CEIL_S = 60.0

#: The one backend type the supervisor acts on. ollama self-manages idle, so for
#: every other type the supervisor is inert (see :attr:`LLMSupervisor._enabled`).
_VLLM = "vllm"

#: Env kill-switch (matches the engine's other ``AI_LEADS_DISABLE_*`` gates,
#: byte-compatible with ai_leads' conftest). Set ⇒ the supervisor stays inert and
#: ``start()`` spawns no thread, regardless of backend.
DISABLE_ENV = "AI_LEADS_DISABLE_LLM_SUPERVISOR"


def vllm_should_stop(
    running: bool, inflight: int, idle_s: float, ttl_s: float,
) -> bool:
    """Pure decision for the idle supervisor (mirrors
    :func:`queue_workflows.model_cache.gpu_should_unload`): stop the server only
    when it is running, nothing is in flight, and it has been idle at least
    ``ttl_s``. ``ttl_s <= 0`` disables idle stop entirely."""
    if ttl_s <= 0:
        return False
    return running and inflight <= 0 and idle_s >= ttl_s


class LLMSupervisor:
    """A daemon thread that frees a vllm sidecar's VRAM after an idle window.

    INERT for ollama (its own daemon self-manages idle) and when the configured
    idle TTL is non-positive or the :data:`DISABLE_ENV` kill-switch is set. The
    actual kill mechanism is ``backend.stop_server()``; this class only decides
    WHEN. Modelled on :class:`queue_workflows.model_cache.ModelCache`'s idle
    reaper and on :class:`queue_workflows.worker_control.WorkerControlWatcher`'s
    daemon-thread plumbing.

    The ``backend`` is any object exposing the duck-typed surface:

      * ``server_type``  -> str   ('ollama' | 'vllm')
      * ``inflight``     -> int   (in-flight LLM requests)
      * ``idle_seconds()`` -> float (0.0 while inflight>0, else seconds idle)
      * ``idle_ttl_s``   -> float (configured idle window; 0 disables)
      * ``is_running()`` -> bool  (is the server process up)
      * ``stop_server()`` -> bool (SIGTERM the sidecar; True iff it stopped one)
    """

    def __init__(
        self,
        *,
        backend: Any,
        poll_s: float | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._backend = backend
        # An explicit poll_s overrides the TTL-derived cadence (tests use a tiny
        # value); None ⇒ compute from the backend's idle_ttl_s in _poll_interval.
        self._poll_s = None if poll_s is None else float(poll_s)
        # Injectable so a test can flip the stop event instead of sleeping; the
        # default is the real ``time.sleep``.
        self._sleep_fn = sleep_fn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def _enabled(self) -> bool:
        """True only when the supervisor should actually run: the
        :data:`DISABLE_ENV` kill-switch is unset, the backend is vllm (ollama
        self-manages idle), and the configured idle TTL is positive
        (``<=0`` disables, mirroring ``vllm_should_stop`` / ``gpu_should_unload``)."""
        if os.environ.get(DISABLE_ENV):
            return False
        if getattr(self._backend, "server_type", None) != _VLLM:
            return False
        return float(self._backend.idle_ttl_s) > 0

    def reap_idle_once(self) -> bool:
        """One supervisor tick: stop the vllm sidecar iff it is running, idle past
        its TTL, and nothing is in flight. Returns ``True`` iff it stopped the
        server. INERT for non-vllm backends (returns ``False`` without touching
        the backend's kill lever). Best-effort: any backend error is swallowed +
        logged and treated as "didn't stop", so the daemon loop keeps ticking."""
        backend = self._backend
        # Ollama (or any non-vllm) self-manages idle — never reach for its kill.
        if getattr(backend, "server_type", None) != _VLLM:
            return False
        try:
            if not vllm_should_stop(
                backend.is_running(),
                backend.inflight,
                backend.idle_seconds(),
                float(backend.idle_ttl_s),
            ):
                return False
            log.info(
                "[LLMSupervisor] vllm idle ≥ TTL %.0fs — stopping sidecar to free VRAM",
                float(backend.idle_ttl_s),
            )
            return bool(backend.stop_server())
        except Exception:
            # A flaky docker call / introspection error must not crash the daemon
            # or surface to the caller — the next tick retries (worker_control's
            # check_once uses the same best-effort posture).
            log.exception("[LLMSupervisor] idle reap tick failed; will retry")
            return False

    def _poll_interval(self) -> float:
        """Poll cadence, clamped to ``[FLOOR, CEIL]`` around ``idle_ttl_s/5`` —
        the same shape as ``ModelCache._idle_reaper_loop``. An explicit
        constructor ``poll_s`` overrides it (tests inject a tiny value)."""
        if self._poll_s is not None:
            return self._poll_s
        return max(
            DEFAULT_SUPERVISOR_POLL_FLOOR_S,
            min(DEFAULT_SUPERVISOR_POLL_CEIL_S, float(self._backend.idle_ttl_s) / 5.0),
        )

    def start(self) -> None:
        """Spawn the idle-supervisor daemon. No-op when :attr:`_enabled` is False
        (kill-switch set, ollama backend, or non-positive TTL) — exactly like
        ``ModelCache.ensure_idle_reaper`` / ``WorkerControlWatcher.start``."""
        if not self._enabled:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="llm-idle-supervisor",
        )
        self._thread.start()
        log.info(
            "[LLMSupervisor] idle supervisor armed (vllm, TTL=%.0fs, poll=%.0fs)",
            float(self._backend.idle_ttl_s), self._poll_interval(),
        )

    def _loop(self) -> None:
        """Sleep one poll interval, then reap once, until ``stop()`` is called.
        Sleep-then-reap (not reap-then-sleep) so a freshly-started sidecar gets at
        least one idle window before the first stop check — matching
        ``ModelCache._idle_reaper_loop``. Each tick's exceptions are swallowed in
        :meth:`reap_idle_once`, so the loop itself can't die on a transient error."""
        poll = self._poll_interval()
        while not self._stop.is_set():
            self._sleep_fn(poll)
            self.reap_idle_once()

    def stop(self) -> None:
        """Signal the loop to exit and join the daemon (bounded, mirrors
        ``WorkerControlWatcher.stop``). Idempotent + safe when never started."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


__all__ = [
    "LLMSupervisor",
    "vllm_should_stop",
    "DEFAULT_SUPERVISOR_POLL_FLOOR_S",
    "DEFAULT_SUPERVISOR_POLL_CEIL_S",
]
