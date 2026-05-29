"""The :class:`OllamaBackend` — the trivial concrete LLM backend.

WHY ALL THE LIFECYCLE METHODS ARE NO-OPS. Ollama is an externally-managed,
long-lived daemon (a host systemd unit / its own container), NOT a process this
library starts or stops. The daemon owns its own idle behaviour via
``OLLAMA_KEEP_ALIVE`` — it offloads a model from VRAM after that env-set quiet
window without anyone telling it to — so the engine has nothing to do on the
lifecycle axis:

  * :meth:`ensure_ready` is a no-op: the daemon is assumed already up, and it
    cold-loads (and hot-swaps between) models on demand when the host node's
    first chat request names one. There is no engine-driven warm-up step.
  * :meth:`is_running` returns ``True`` unconditionally: we assume the operator-run
    daemon is reachable. (A real readiness probe would GET :attr:`health_url`, but
    that's the host node's call to make — the backend deliberately performs no I/O.)
  * :meth:`stop_server` returns ``False``: the engine never stops the daemon, so it
    never reclaims VRAM this way, so the idle supervisor's stop attempt is always a
    truthful no-op. (This is why the factory sets ``idle_ttl_s=0`` for ollama —
    KEEP_ALIVE already covers idle; the supervisor needn't fire at all.)

What the backend DOES carry is the inherited request accounting
(:meth:`mark_request_start` / :meth:`mark_request_end` / :attr:`inflight` /
:meth:`idle_seconds`) — kept for symmetry with the vllm backend and so the queue
UI shows the same live-request / idle stats regardless of which server a machine
runs. Only the two endpoint URLs are ollama-specific (``/api/chat`` for
completions, ``/api/tags`` as the cheap liveness GET).
"""

from __future__ import annotations

from queue_workflows.llm_backends.base import LLMBackend


class OllamaBackend(LLMBackend):
    """An externally-managed ollama daemon. Counters only; no lifecycle ownership.

    See the module docstring for why every lifecycle method is a no-op. All
    constructor args + request accounting are inherited unchanged from
    :class:`~queue_workflows.llm_backends.base.LLMBackend`.
    """

    @property
    def server_type(self) -> str:
        return "ollama"

    @property
    def chat_url(self) -> str:
        # ollama's native chat-completions endpoint. base_url is already
        # trailing-slash-stripped by the base ctor, so this can't double up.
        return f"{self.base_url}/api/chat"

    @property
    def health_url(self) -> str:
        # /api/tags lists installed models — a cheap, side-effect-free GET that
        # doubles as the daemon's liveness probe.
        return f"{self.base_url}/api/tags"

    def ensure_ready(self, model_id: str, *, timeout_s: float = 90.0) -> None:
        # No-op: the daemon is externally managed and loads/swaps models itself on
        # the first request. We never warm it up. (Signature mirrors the base for
        # the introspecting host wiring; args intentionally unused.)
        return None

    def is_running(self) -> bool:
        # The external daemon is assumed reachable; we perform no I/O to confirm it.
        return True

    def stop_server(self) -> bool:
        # The engine never manages the ollama daemon, so there is nothing to stop
        # and no VRAM to reclaim here — always a truthful no-op for the supervisor.
        return False


__all__ = ["OllamaBackend"]
