"""The :class:`VLLMBackend` — the real start/stop LLM backend.

WHY THIS IS THE NON-TRIVIAL BACKEND (contrast :class:`OllamaBackend`). vLLM is a
docker *sidecar* the library actually manages, and it differs from ollama on
every lifecycle axis:

  * **No built-in idle unload.** vLLM pins its model in VRAM for the entire
    lifetime of the server process — there is no ``OLLAMA_KEEP_ALIVE`` analog. So
    the engine's :class:`~queue_workflows.llm_backends.supervisor.LLMSupervisor`
    is the idle mechanism: after a quiet window it calls :meth:`stop_server`, which
    SIGTERMs the sidecar. Docker's ``restart: unless-stopped`` then respawns it on
    the next request (the same "kill, let the container supervisor restart it" leg
    worker_control's HARD stop relies on). Freeing VRAM == killing the process;
    there is no lighter lever.

  * **No runtime model swap.** A vLLM engine serves ONE model id for its lifetime.
    Switching models therefore can't be a hot reload through the running engine.
    The PRIMARY intended fast path is vLLM **Sleep-Mode L2** (offload weights to
    host RAM, free VRAM, then reload the *new* model's weights into the same
    process) — but that wiring is a later phase, so :meth:`_sleep_l2_reload` is a
    deliberate **stub that raises ``NotImplementedError``**. :meth:`ensure_ready`
    catches that, sets a STICKY ``_sleep_unsupported`` flag (state
    :data:`VLLMState.UNSUPPORTED_SLEEP`), and falls back to the SLOW path: stop the
    server, bring it back up serving the new model, wait for health. Once the
    sticky flag is set, subsequent switches skip the Sleep-L2 attempt entirely and
    go straight to the slow path — we never re-probe a capability we've already
    learned this build lacks.

DB-FREE + I/O-INJECTED (the library TDD philosophy). The backend performs NO I/O
at module import and owns NO transport client. Every side-effecting operation is
an injected seam with a real default:

  * ``kill_fn() -> bool`` — SIGTERM the sidecar process; True iff it signalled one.
    Default ``pkill -f`` the vLLM api_server pattern via :mod:`subprocess`.
  * ``ensure_up_fn(model_id) -> None`` — make the sidecar come up serving
    ``model_id``. Default is a NO-OP: today the bring-up is owned by docker
    ``restart: unless-stopped`` + a (Phase-6) entrypoint that reads the desired
    model from config. The seam exists so that entrypoint, once built, plugs in
    here without touching the state machine.
  * ``health_fn() -> bool`` — GET :attr:`health_url`, True iff 200. Default uses
    httpx, **lazily imported inside the default** so importing this module needs no
    httpx (the Phase-6 host-import guard + a bare interpreter stay green).
  * ``served_model_fn() -> str | None`` — GET ``{base_url}/v1/models`` and return
    the first served id. Default httpx, lazy-imported the same way.

The default seams are the ONLY place httpx / subprocess appear, and only inside
the function bodies — the module top imports nothing but stdlib + the base.

CONCURRENCY. The state (:attr:`state` / ``_served_model`` / the sticky flag) is
guarded by the base's reentrant ``self._lock`` — the SAME RLock the inherited
request accounting uses — because the supervisor calls :meth:`is_running` /
:meth:`stop_server` from its own daemon thread (see ``supervisor.py``). We hold
the lock across the read-modify of the state machine but RELEASE it around the
health poll's ``sleep_fn`` — never sleep holding a lock — so a concurrent
supervisor ``is_running`` probe isn't blocked behind a 90 s cold-start wait.
"""

from __future__ import annotations

import enum
import logging
import shutil
import subprocess
import time
from typing import Callable

from queue_workflows.llm_backends.base import LLMBackend

log = logging.getLogger(__name__)


# ── poll cadence for the cold-start health wait ─────────────────────────────

#: How often :meth:`VLLMBackend.ensure_ready` re-probes ``health_fn`` while
#: waiting for a freshly-(re)started sidecar to answer. Short so a fast boot is
#: noticed promptly; the overall wait is bounded by ``timeout_s``.
_HEALTH_POLL_INTERVAL_S = 1.0

#: The process command pattern the default ``kill_fn`` targets — vLLM's
#: OpenAI-compatible server is launched as ``... -m vllm.entrypoints.openai.api_server``.
#: ``pkill -f`` matches it anywhere in the full command line.
_VLLM_PROC_PATTERN = "vllm.entrypoints.openai.api_server"


class VLLMState(str, enum.Enum):
    """The vLLM sidecar lifecycle, tracked on :attr:`VLLMBackend.state`.

    A ``str`` enum so it logs / serialises as its plain value (``"serving"``)
    while still being a singleton for identity comparison in tests.

      * ``DEAD``               — no sidecar process up (initial, or post-stop).
      * ``LOADING``            — bring-up issued; waiting for health to go green.
      * ``SERVING``            — up and serving ``_served_model``.
      * ``SLEEPING_L2``        — (reserved) mid Sleep-Mode-L2 offload; today
                                 transient/unused because the L2 path is stubbed.
      * ``RELOADING``          — handling a model switch (slow stop→bring-up).
      * ``UNSUPPORTED_SLEEP``  — a Sleep-L2 attempt failed; the sticky flag is set
                                 and this build will never retry Sleep-L2. (The
                                 backend still transitions on to a real serving
                                 state; this value is surfaced via
                                 :attr:`VLLMBackend.sleep_unsupported` so the fact
                                 is observable independent of the live state.)
    """

    DEAD = "dead"
    LOADING = "loading"
    SERVING = "serving"
    SLEEPING_L2 = "sleeping_l2"
    RELOADING = "reloading"
    UNSUPPORTED_SLEEP = "unsupported_sleep"


# ── default I/O seams (the ONLY place httpx / subprocess appear) ────────────


def _default_kill_fn() -> bool:
    """SIGTERM the vLLM api_server process via ``pkill -f``. Returns True iff
    pkill reports it signalled at least one process (exit 0); False if none matched
    (exit 1) or pkill is unavailable. Best-effort — any OS error ⇒ False, never
    raises, because the supervisor swallows but shouldn't have to."""
    pkill = shutil.which("pkill")
    if pkill is None:
        log.warning("[VLLMBackend] pkill not found; cannot SIGTERM the sidecar")
        return False
    try:
        # -f matches the full command line; default signal is SIGTERM. Exit 0 ⇒
        # at least one matched, 1 ⇒ none matched.
        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [pkill, "-f", _VLLM_PROC_PATTERN],
            capture_output=True,
        )
        return proc.returncode == 0
    except OSError:
        log.exception("[VLLMBackend] pkill invocation failed")
        return False


def _default_ensure_up_fn(model_id: str) -> None:  # noqa: ARG001 — Phase-6 seam
    """Default bring-up: a NO-OP. Today the sidecar's respawn is owned by docker
    ``restart: unless-stopped`` and a (Phase-6) entrypoint that reads the desired
    model from config; this library hook is the place that entrypoint plugs into.
    Documented as a seam so the state machine's health-poll loop already accounts
    for a restart it doesn't itself trigger."""
    return None


def _default_health_fn(health_url: str) -> bool:
    """GET ``health_url``; True iff HTTP 200. httpx is imported HERE (lazily) so the
    module top stays httpx-free. Any error ⇒ False (down)."""
    try:
        import httpx  # lazy: keep module import httpx-free

        resp = httpx.get(health_url, timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def _default_served_model_fn(base_url: str) -> str | None:
    """GET ``{base_url}/v1/models`` and return the first served model id, or None
    if the endpoint is unreachable / empty. httpx imported lazily, as above."""
    try:
        import httpx  # lazy

        resp = httpx.get(f"{base_url}/v1/models", timeout=5.0)
        resp.raise_for_status()
        data = resp.json().get("data") or []
        if data:
            return data[0].get("id")
        return None
    except Exception:
        return None


class VLLMBackend(LLMBackend):
    """A library-managed vLLM docker sidecar (one model, whole-lifetime VRAM).

    See the module docstring for WHY the lifecycle is non-trivial. The request
    accounting + config echo are inherited verbatim from
    :class:`~queue_workflows.llm_backends.base.LLMBackend`; this class implements
    the real :meth:`ensure_ready` / :meth:`is_running` / :meth:`stop_server` plus
    the sticky Sleep-Mode-L2 fallback.
    """

    def __init__(
        self,
        *,
        base_url: str,
        parallelism: int = 1,
        idle_ttl_s: float = 60.0,
        served_model: str | None = None,
        now_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        kill_fn: Callable[[], bool] | None = None,
        ensure_up_fn: Callable[[str], None] | None = None,
        health_fn: Callable[[], bool] | None = None,
        served_model_fn: Callable[[], str | None] | None = None,
    ) -> None:
        # The base owns base_url stripping + the request-accounting RLock/counters.
        super().__init__(
            base_url=base_url,
            parallelism=parallelism,
            idle_ttl_s=idle_ttl_s,
            now_fn=now_fn,
        )
        # sleep_fn is the cold-start poll's wait; injectable so tests drive a
        # virtual clock (an advancing sleep_fn) without real waiting.
        self._sleep_fn = sleep_fn

        # I/O seams. Each default closes over self's URLs/root, captured here so the
        # stored callable matches the spec's parameter-less / (model_id) signatures
        # and httpx stays inside the default bodies (never at module import).
        self._kill_fn = kill_fn or _default_kill_fn
        self._ensure_up_fn = ensure_up_fn or _default_ensure_up_fn
        self._health_fn = health_fn or (lambda: _default_health_fn(self.health_url))
        self._served_model_fn = served_model_fn or (
            lambda: _default_served_model_fn(self.base_url)
        )

        # State machine (RLock-guarded via the inherited self._lock). If the
        # operator told us a model is already served at construction, start in
        # SERVING; otherwise DEAD. _served_model is the engine's belief about what
        # the running sidecar serves (reconciled against served_model_fn on bring-up).
        self._served_model: str | None = served_model
        self._state: VLLMState = (
            VLLMState.SERVING if served_model is not None else VLLMState.DEAD
        )
        # Sticky: set True the first time a Sleep-L2 reload attempt fails. Once set,
        # ensure_ready never re-attempts Sleep-L2 (goes straight to the slow path).
        self._sleep_unsupported: bool = False

    # ── identity / endpoints (vLLM-specific) ─────────────────────────────────

    @property
    def server_type(self) -> str:
        return "vllm"

    @property
    def chat_url(self) -> str:
        # vLLM's OpenAI-compatible chat completions live under /v1. base_url is
        # already trailing-slash-stripped by the base ctor.
        return f"{self.base_url}/v1/chat/completions"

    @property
    def health_url(self) -> str:
        # vLLM exposes liveness at the server ROOT /health — NOT /v1/health.
        return f"{self.base_url}/health"

    # ── observable state (RLock-read so the supervisor sees no torn value) ────

    @property
    def state(self) -> VLLMState:
        """The current lifecycle state (for tests + logging)."""
        with self._lock:
            return self._state

    @property
    def sleep_unsupported(self) -> bool:
        """Whether the sticky UNSUPPORTED_SLEEP flag is set — i.e. a Sleep-Mode-L2
        reload was attempted and failed, so this build will never retry it."""
        with self._lock:
            return self._sleep_unsupported

    # ── lifecycle ────────────────────────────────────────────────────────────

    def ensure_ready(self, model_id: str, *, timeout_s: float = 90.0) -> None:
        """Block until the sidecar is up and serving ``model_id`` (start / switch as
        needed), or raise ``TimeoutError`` after ``timeout_s``. Idempotent +
        thread-safe.

        Three cases (see the module docstring for the WHY of the switch path):

          1. **Warm hit** — already up serving ``model_id`` ⇒ SERVING, return.
          2. **Cold start** — not running ⇒ LOADING → ``ensure_up_fn`` → poll
             ``health_fn`` to True (or TimeoutError) → reconcile served id → SERVING.
          3. **Switch** — up serving a DIFFERENT model ⇒ RELOADING → try Sleep-L2
             once (unless already known unsupported), catch its NotImplementedError
             and set the sticky flag, then the slow path (stop → cold-start the new
             model).

        After the bring-up + health wait, the served id is RECONCILED against the
        ``/v1/models`` probe and a MISMATCH raises ``RuntimeError`` (FAIL LOUD).
        The default ``ensure_up_fn`` is a no-op + docker ``restart: unless-stopped``
        brings the sidecar back up on its BAKED ``--model`` flag, NOT the requested
        one — so a cross-model switch can come up serving the wrong model. Failing
        here lets the consumer soft-degrade (e.g. fall back to ollama) instead of
        silently POSTing prompts to the wrong model. (A blank probe — endpoint
        empty/early — still optimistically trusts the requested id; only a probe
        that names a DIFFERENT served id is treated as a mismatch.)
        """
        # Snapshot the decision under the lock; the health poll itself runs without
        # the lock held (we must not sleep holding it).
        with self._lock:
            running = self.is_running()
            if running and self._served_model == model_id:
                # Warm hit — reaffirm SERVING and we're done.
                self._state = VLLMState.SERVING
                return
            if running and self._served_model != model_id:
                # Model switch: attempt the (stubbed) Sleep-L2 fast path once,
                # unless we've already learned it's unsupported.
                self._state = VLLMState.RELOADING
                if not self._sleep_unsupported:
                    try:
                        self._sleep_l2_reload(model_id)
                        # If a real implementation ever succeeds, it has reloaded
                        # the engine in place: mark served + SERVING and return.
                        self._served_model = model_id
                        self._state = VLLMState.SERVING
                        return
                    except NotImplementedError:
                        # Stub (or a build genuinely lacking Sleep-Mode): remember
                        # it permanently so we never re-probe, and fall through to
                        # the slow path below.
                        log.info(
                            "[VLLMBackend] Sleep-Mode-L2 reload unsupported; "
                            "falling back to stop+restart and disabling future "
                            "L2 attempts"
                        )
                        self._sleep_unsupported = True
                        self._state = VLLMState.UNSUPPORTED_SLEEP
                # Slow path: stop the running server (held lock is fine — kill is
                # not a sleep), then fall through to the cold-start bring-up below.
                self._do_stop_locked()
            # Cold-start bring-up (also the tail of the slow switch path): issue
            # the bring-up, then leave the lock for the health poll.
            self._state = VLLMState.LOADING
        # ── lock released — issue the bring-up + poll health without holding it ──
        self._ensure_up_fn(model_id)
        self._await_health(timeout_s=timeout_s)
        # Reconcile the served id, then commit SERVING under the lock. If the
        # /v1/models probe named a model, trust it; else (blank/unreachable probe)
        # optimistically assume the model we just asked to bring up.
        probed = None
        try:
            probed = self._served_model_fn()
        except Exception:
            probed = None
        observed = probed or model_id
        with self._lock:
            self._served_model = observed
            self._state = VLLMState.SERVING
        # FAIL LOUD on a cross-model mismatch: the default bring-up (docker
        # restart:unless-stopped + baked --model) can resurrect the sidecar
        # serving its baked model, not the one we asked for. Raise so the
        # consumer soft-degrades instead of POSTing to the wrong model. A blank
        # probe (observed fell back to model_id) never trips this.
        if observed != model_id:
            raise RuntimeError(
                f"vLLM came up serving {observed!r}, not {model_id!r} "
                "(baked --model; cross-model switch needs container recreate)"
            )

    def _await_health(self, *, timeout_s: float) -> None:
        """Poll ``health_fn`` every :data:`_HEALTH_POLL_INTERVAL_S` until it returns
        True or the injected clock has advanced ``timeout_s`` since entry. Raises
        ``TimeoutError`` on expiry. Runs WITHOUT the lock held (it sleeps), so a
        concurrent supervisor probe isn't blocked behind the cold-start wait."""
        deadline = self._now_fn() + timeout_s
        while True:
            try:
                if self._health_fn():
                    return
            except Exception:
                # A flaky probe mid-boot is just "not ready yet"; keep polling.
                pass
            if self._now_fn() >= deadline:
                raise TimeoutError(
                    f"vLLM sidecar at {self.health_url} not healthy after "
                    f"{timeout_s:.0f}s"
                )
            self._sleep_fn(_HEALTH_POLL_INTERVAL_S)

    def _sleep_l2_reload(self, model_id: str) -> None:  # noqa: ARG002 — stub
        """PRIMARY model-switch fast path: vLLM Sleep-Mode L2 (offload current
        weights to host RAM, free VRAM, reload ``model_id`` into the same engine).

        STUBBED — raises ``NotImplementedError``. The real wiring (a control call
        to the sidecar's sleep/wake + load endpoints) is a later phase.
        :meth:`ensure_ready` catches this, sets the sticky ``_sleep_unsupported``
        flag, and falls back to the slow stop→restart path. Kept as a named method
        (not inlined) so a test can spy that it's attempted exactly once, then never
        again once the sticky flag is set."""
        raise NotImplementedError(
            "vLLM Sleep-Mode-L2 in-place reload is not wired yet; "
            "falling back to stop+restart"
        )

    def is_running(self) -> bool:
        """A real liveness check: return ``health_fn()`` (HTTP 200 at
        :attr:`health_url`). Must NOT raise — any probe error is swallowed to
        ``False`` (down), because the supervisor and ensure_ready both call this and
        a network blip must not crash either."""
        try:
            return bool(self._health_fn())
        except Exception:
            return False

    def stop_server(self) -> bool:
        """Free this sidecar's VRAM by SIGTERMing it (docker respawns it on the next
        request). Returns ``True`` iff it actually stopped something. A no-op (False,
        no ``kill_fn`` call) when there is nothing to stop — i.e. the server is
        already down. Thread-safe: the supervisor calls this from its own thread."""
        with self._lock:
            # "Nothing to stop" == we believe it's dead AND a live probe agrees.
            # Checking state first avoids an HTTP probe on the common already-dead
            # path; if state says serving/loading we still confirm liveness so a
            # crashed-but-state-stale sidecar isn't "killed" with a truthful False.
            if self._state == VLLMState.DEAD and not self.is_running():
                return False
            return self._do_stop_locked()

    def _do_stop_locked(self) -> bool:
        """Kill the sidecar + reset state to DEAD. MUST be called with ``self._lock``
        held (both :meth:`stop_server` and the slow switch path in
        :meth:`ensure_ready` invoke it under the lock). Returns the ``kill_fn``
        result. ``kill_fn`` is a signal, not a sleep, so holding the lock is fine."""
        result = bool(self._kill_fn())
        self._state = VLLMState.DEAD
        self._served_model = None
        return result


__all__ = ["VLLMBackend", "VLLMState"]
