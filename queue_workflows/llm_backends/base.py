"""The :class:`LLMBackend` port — a host-agnostic handle to a per-machine LLM
server (an ollama daemon or a vllm sidecar), plus the request accounting the
idle supervisor reads to decide when to free VRAM.

WHY THIS EXISTS. The fleet runs a co-tenant LLM server next to the GPU claim
worker (the VLM that captions / reasons over images). Which *kind* of server a
machine runs — ollama (an externally-managed long-lived daemon) or vllm (a
docker sidecar the library SIGTERMs when idle to reclaim VRAM) — is per-machine,
operator-set state living on ``worker_controls`` (migration 0013). This port is
the polymorphic seam: the worker reads its config, a factory (built later, NOT
here) constructs the matching concrete backend, and the rest of the worker drives
it through this one surface — never branching on the server type itself.

DESIGN — what the backend owns, and what it does NOT.

  * It owns **request accounting**, nothing more on the data-plane. ``inflight``
    is the live concurrent-request count; ``idle_seconds()`` is how long the
    server has been quiet. The idle supervisor (a separate daemon, built by
    another agent against THIS surface) reads those two to decide whether to call
    :meth:`stop_server`. This accounting is the direct analog of
    :class:`~queue_workflows.model_cache.ModelCache`'s ``active`` busy-counter and
    its ``last_used`` idle clock, and it shares that class's concurrency
    discipline — see the lock note below.
  * It does **NOT make the HTTP call**. The host node owns the chat POST; the
    backend only hands out :attr:`chat_url` (the endpoint to POST to) and brackets
    the call with :meth:`mark_request_start` / :meth:`mark_request_end`. Keeping
    the transport out of the backend is what lets this module stay free of any
    HTTP client dependency and import nothing but stdlib + ``queue_workflows.*``
    (the Phase-6 inversion guard).
  * It is deliberately **OpenAI-chat-shape biased and vendor-coupled** to exactly
    ollama + vllm. This is not a general LLM abstraction; both backends speak the
    same ``POST /…/chat`` request shape, so the host node is written once. A third
    vendor would mean a new request shape, not just a new subclass — and that is
    explicitly out of scope.

CONCURRENCY. The claim worker is concurrency-1 by contract (one job at a time),
so the data-plane bracket is never racing itself. But the idle supervisor reads
:attr:`inflight` / :meth:`idle_seconds` from its **own daemon thread**, exactly as
``ModelCache``'s idle reaper reads ``active`` / ``last_used`` from a separate
thread. So the counters + the idle clock are guarded by an :class:`~threading.RLock`
— a reentrant lock so a method holding it may call another locked accessor without
self-deadlock. ``now_fn`` is injected (default :func:`time.monotonic`) so tests can
drive the idle clock deterministically without sleeping.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Callable


class LLMBackend(ABC):
    """Abstract per-machine LLM-server handle. See the module docstring for the
    contract; concrete subclasses are
    :class:`~queue_workflows.llm_backends.ollama.OllamaBackend` (no-op lifecycle,
    counters only) and the vllm backend (real start/stop), added in a later phase.

    The request-accounting half (:meth:`mark_request_start` /
    :meth:`mark_request_end` / :attr:`inflight` / :meth:`idle_seconds`) is
    **concrete and shared** by every backend — it is the supervisor's read model
    and must behave identically regardless of vendor. Only the identity URLs and
    the lifecycle (:meth:`ensure_ready` / :meth:`is_running` / :meth:`stop_server`)
    are abstract, because those are the only things the two vendors do differently.
    """

    def __init__(
        self,
        *,
        base_url: str,
        parallelism: int = 1,
        idle_ttl_s: float = 0.0,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        # base_url is stored right-stripped of a trailing '/' so subclasses can
        # build endpoints by plain f-string concatenation (``f"{base_url}/api/chat"``)
        # without doubling the slash — the factory passes whatever the operator
        # typed, which may or may not end in '/'.
        self._base_url = base_url.rstrip("/")
        # parallelism = the SIDECAR's concurrent-request capacity
        # (ollama OLLAMA_NUM_PARALLEL / vllm --max-num-seqs). NOT the claim
        # concurrency, which is 1 by contract; this is surfaced in the queue UI
        # and may legitimately be large for vllm. Echoed back via the property.
        self._parallelism = parallelism
        # idle_ttl_s + now_fn feed idle_seconds() for the supervisor. now_fn is
        # injectable so tests advance a fake clock instead of sleeping.
        self._idle_ttl_s = float(idle_ttl_s)
        self._now_fn = now_fn

        # The request-accounting state, mirroring ModelCache's (active, last_used)
        # pair. Guarded by an RLock because the supervisor reads it off-thread.
        self._lock = threading.RLock()
        self._inflight = 0
        self._last_used = now_fn()
        self._current_model: str | None = None

    # ── identity / config (mostly concrete; the URLs are vendor-specific) ─────

    @property
    @abstractmethod
    def server_type(self) -> str:
        """The registry name of this backend: ``"ollama"`` or ``"vllm"``. Used by
        the factory + the queue UI; never branched on by the worker data-plane."""

    @property
    def base_url(self) -> str:
        """The server root, trailing slash stripped (e.g. ``http://host:11434``)."""
        return self._base_url

    @property
    def parallelism(self) -> int:
        """The sidecar's concurrent-request capacity (NOT claim concurrency)."""
        return self._parallelism

    @property
    def idle_ttl_s(self) -> float:
        """Seconds of zero requests before the supervisor frees VRAM; 0 disables
        (ollama always passes 0 — its own KEEP_ALIVE owns idle)."""
        return self._idle_ttl_s

    @property
    def current_model(self) -> str | None:
        """The model id of the most recent :meth:`mark_request_start`, or ``None``
        before the first request. RLock-read so the supervisor sees a torn-free
        value."""
        with self._lock:
            return self._current_model

    @property
    @abstractmethod
    def chat_url(self) -> str:
        """The full chat-completions endpoint the HOST node POSTs to. The backend
        never calls it itself — it only owns the URL + the request bracket."""

    @property
    @abstractmethod
    def health_url(self) -> str:
        """The full endpoint a readiness probe GETs to confirm the server is up."""

    # ── request accounting (CONCRETE — the supervisor's read model) ───────────

    def mark_request_start(self, model_id: str) -> None:
        """Open the request bracket: bump :attr:`inflight`, stamp the idle clock
        to *now*, and record ``model_id`` as the :attr:`current_model`. The host
        node calls this immediately before its chat POST. Mirrors
        ``ModelCache.mark_busy`` (busy is never idle)."""
        with self._lock:
            self._inflight += 1
            self._last_used = self._now_fn()
            self._current_model = model_id

    def mark_request_end(self) -> None:
        """Close the request bracket: drop :attr:`inflight` (floored at 0 so an
        unbalanced extra end can never go negative) and restart the idle clock so
        the TTL counts from the END of the last request. Mirrors
        ``ModelCache.mark_idle``. The host node calls this in a ``finally`` so a
        failed POST still balances the counter."""
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self._last_used = self._now_fn()

    @property
    def inflight(self) -> int:
        """Live concurrent-request count (RLock-read)."""
        with self._lock:
            return self._inflight

    def idle_seconds(self) -> float:
        """How long the server has been quiet, for the supervisor's idle decision.
        Always ``0.0`` while a request is in flight — a busy server is never idle,
        exactly as ``gpu_should_unload`` requires ``active <= 0``. Otherwise the
        monotonic gap since the last bracket edge, floored at 0 (a non-monotonic
        injected clock can't yield a negative idle)."""
        with self._lock:
            if self._inflight > 0:
                return 0.0
            return max(0.0, self._now_fn() - self._last_used)

    # ── lifecycle (ABSTRACT — ollama no-ops these, vllm implements for real) ──

    @abstractmethod
    def ensure_ready(self, model_id: str, *, timeout_s: float = 90.0) -> None:
        """Block until the server can serve ``model_id`` (start it / load weights
        if needed), or raise on ``timeout_s``. A no-op for ollama (the daemon is
        externally managed and hot-swaps models itself); the real cold-start for
        vllm."""

    @abstractmethod
    def is_running(self) -> bool:
        """Whether the server process is currently up. ``True`` always for ollama
        (an external daemon assumed reachable); a real liveness check for vllm."""

    @abstractmethod
    def stop_server(self) -> bool:
        """Free this server's VRAM by stopping it. Returns ``True`` iff it actually
        stopped something (so the supervisor can log a real reclaim vs a no-op).
        Always ``False`` for ollama (never library-managed); SIGTERMs the sidecar
        for vllm."""

    def shutdown(self) -> None:
        """Release the server on worker teardown. Default delegates to
        :meth:`stop_server` — correct for both vendors today (ollama's is a no-op,
        vllm's frees the sidecar). Concrete so subclasses inherit teardown for
        free."""
        self.stop_server()


__all__ = ["LLMBackend"]
