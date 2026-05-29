"""The per-``(host_label, queue)`` LLM-backend FACTORY — the live owner of the
concrete :class:`LLMBackend` singleton AND its :class:`LLMSupervisor`, kept in
sync with the operator's DB config.

WHY THIS EXISTS. The backend modules (``base`` / ``ollama`` / ``vllm``) and the
idle :class:`~queue_workflows.llm_backends.supervisor.LLMSupervisor` are
deliberately DB-free, I/O-injected, unit-testable parts. SOMETHING has to read a
machine's ``worker_controls`` LLM config (migration 0013, via
:func:`queue_workflows.worker_control.llm_config_for`), pick the matching concrete
backend, hand it the right base URL, and — for vllm — arm a supervisor to free its
VRAM after an idle window. That is THIS factory. The (later-built) VLM node calls
:func:`get_backend(host, queue)` at request time and brackets its HTTP POST on the
returned backend; the supervisor reads that SAME instance off-thread to decide when
to SIGTERM the sidecar. So the factory's contract is two-fold:

  * **Identity preservation** — within one ``(host, queue)``, hand back the SAME
    backend instance across calls as long as the config SNAPSHOT
    ``(server_type, parallelism, vllm_idle_ttl_s)`` is unchanged. The backend's
    request counters (``inflight`` / ``idle_seconds``) and the vllm state machine
    (``_served_model`` / the sticky Sleep-L2 flag) live ON that instance; rebuild it
    spuriously and you reset all of that, so the supervisor and the request bracket
    would lose their read model. Only an ACTUAL config change rebuilds.
  * **Freshness** — when the operator flips a machine ollama↔vllm or retunes it, the
    next request must observe the change. A 0013 NOTIFY on
    :data:`~queue_workflows.worker_control.LLM_CONFIG_NOTIFY_CHANNEL` invalidates the
    cache instantly (the LISTEN invalidator thread below); a TTL re-read is the
    dropped-NOTIFY fallback.

DESIGN — a process-singleton with injected side effects, the same idiom as
:mod:`queue_workflows.model_cache` / :mod:`queue_workflows.gpu_model_cache`:

  * ``build_backend_fn`` constructs the concrete backend — default builds a real
    :class:`~queue_workflows.llm_backends.ollama.OllamaBackend` /
    :class:`~queue_workflows.llm_backends.vllm.VLLMBackend` with their default I/O
    seams (httpx lazy, never called at construction — the factory does no health
    probe on build). A test injects a fake.
  * ``supervisor_factory`` builds the :class:`LLMSupervisor` for a vllm backend —
    default ``LLMSupervisor(backend=backend)``; ``.start()`` is idempotent and
    env-gated (``AI_LEADS_DISABLE_LLM_SUPERVISOR``), so it is INERT in tests.
  * the URL per server type is resolved from the env NAMES on
    :class:`~queue_workflows.config.EngineConfig` (``ollama_url_env`` /
    ``vllm_url_env``) — the DB owns WHICH server + tunables, the URL is deployment
    topology and stays in env. Constructor ``ollama_url`` / ``vllm_url`` kwargs
    override env (tests).

CONCURRENCY. ``get_backend`` runs on the node thread; the LISTEN invalidator calls
``invalidate`` from its own daemon thread; a rebuild calls ``shutdown`` / supervisor
``stop`` which may block briefly. The cache dict + per-key flags are guarded by a
single reentrant :class:`~threading.RLock` (reentrant so ``get_backend`` may call a
locked helper without self-deadlock). We hold the lock across the read-modify of a
cache entry; the DB read + backend build run UNDER the lock too — they are cheap
(one SELECT, no I/O on construction), and holding the lock keeps two racing
``get_backend`` calls from each building a backend for the same key.

The daemon-thread plumbing (``start`` / ``stop`` / ``threading.Event`` /
autocommit-connect LISTEN loop / ``_enabled`` env gate) MIRRORS
:class:`queue_workflows.worker_control.WorkerControlWatcher`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from queue_workflows import worker_control
from queue_workflows.config import get_config
from queue_workflows.db import db_url
from queue_workflows.llm_backends.base import LLMBackend
from queue_workflows.llm_backends.ollama import OllamaBackend
from queue_workflows.llm_backends.supervisor import LLMSupervisor
from queue_workflows.llm_backends.vllm import VLLMBackend

log = logging.getLogger(__name__)


# ── env defaults + the listener kill-switch ─────────────────────────────────

#: Fallback base URLs when the env NAME on the config (ollama_url_env /
#: vllm_url_env) is unset — the localhost/docker-host defaults documented on
#: ``EngineConfig``.
DEFAULT_OLLAMA_URL = "http://host.docker.internal:11434"
DEFAULT_VLLM_URL = "http://host.docker.internal:8000"

#: Env kill-switch for the LISTEN invalidator thread. DISTINCT from the
#: supervisor's ``AI_LEADS_DISABLE_LLM_SUPERVISOR`` — a host may want the idle
#: supervisor inert (e.g. a test) while still keeping config-change LISTEN, or
#: vice-versa, so the two gates are independent. Added to the engine conftest's
#: ``setdefault`` block so tests don't spawn the thread.
DISABLE_LISTENER_ENV = "AI_LEADS_DISABLE_LLM_CONFIG_LISTENER"

#: Default TTL (s) between DB re-reads of a key's config when no NOTIFY fired —
#: the dropped-NOTIFY safety net behind the instant LISTEN invalidation.
DEFAULT_CONFIG_TTL_S = 10.0


@dataclass(frozen=True)
class _Snapshot:
    """The config fields that DECIDE the backend's identity. Two reads with the
    same snapshot ⇒ the SAME backend is kept (no rebuild). The base URL is NOT in
    the snapshot: it is env-derived deployment topology, fixed for a process'
    lifetime, so a change to it isn't a runtime reconfiguration the factory tracks
    (a redeploy restarts the process)."""

    server_type: str
    parallelism: int
    vllm_idle_ttl_s: int

    @classmethod
    def from_config(cls, cfg: worker_control.LLMConfig) -> "_Snapshot":
        return cls(
            server_type=cfg.server_type,
            parallelism=int(cfg.parallelism),
            vllm_idle_ttl_s=int(cfg.vllm_idle_ttl_s),
        )


@dataclass
class _Entry:
    """One cached ``(host, queue)`` slot: the live snapshot it was built for, the
    backend instance whose identity we preserve, its supervisor (vllm only; ``None``
    for ollama), the monotonic time of the last DB check (for the TTL throttle), and
    a force-rebuild flag the NOTIFY invalidator sets."""

    snapshot: _Snapshot
    backend: LLMBackend
    supervisor: Any  # LLMSupervisor | None
    last_check: float
    invalid: bool = False


def _default_build_backend(
    server_type: str, base_url: str, parallelism: int, idle_ttl_s: float,
) -> LLMBackend:
    """Build the real concrete backend for ``server_type`` with its DEFAULT I/O
    seams. ``VLLMBackend``'s httpx-backed health/served seams are lazy (imported
    inside their bodies and only CALLED on ``ensure_ready`` / ``is_running``), and
    the factory never calls those at construction — so building one here does no
    network I/O and keeps the host-import guard green."""
    if server_type == worker_control.SERVER_TYPE_VLLM:
        # Thread the host's vllm lifecycle hooks (ai_leads wires docker-over-UDS
        # stop/start so the in-worker supervisor can control the SIBLING sidecar
        # container). Each is None unless a host wired it via set_vllm_lifecycle;
        # the backend treats None as "use my built-in default" (``kill_fn or
        # _default_kill_fn``), so an unconfigured deployment is unchanged.
        cfg = get_config()
        return VLLMBackend(
            base_url=base_url, parallelism=parallelism, idle_ttl_s=idle_ttl_s,
            kill_fn=cfg.vllm_stop_fn,
            ensure_up_fn=cfg.vllm_start_fn,
        )
    # Anything else resolves to ollama (the default-safe server type). ollama's
    # idle is owned by its own daemon's KEEP_ALIVE, so idle_ttl_s is forced to 0
    # by the caller and the supervisor stays inert regardless.
    return OllamaBackend(
        base_url=base_url, parallelism=parallelism, idle_ttl_s=idle_ttl_s,
    )


class BackendFactory:
    """The per-``(host, queue)`` owner of the live :class:`LLMBackend` + its
    :class:`LLMSupervisor`, kept in sync with the DB config (migration 0013).

    See the module docstring for the identity-preservation + freshness contract.
    All side effects are injected so a test drives fakes with no network / no DB
    threads:

      * ``build_backend_fn(server_type, base_url, parallelism, idle_ttl_s) ->
        LLMBackend`` — construct the concrete backend (default: real ollama/vllm).
      * ``supervisor_factory(backend) -> LLMSupervisor`` — build the idle supervisor
        for a vllm backend (default: ``LLMSupervisor(backend=backend)``).
      * ``now_fn() -> float`` — the TTL clock (default ``time.monotonic``); a test
        feeds a mutable cell to cross the TTL deterministically.
      * ``ollama_url`` / ``vllm_url`` — override the env URL resolution (tests).
    """

    def __init__(
        self,
        *,
        ttl_s: float = DEFAULT_CONFIG_TTL_S,
        now_fn: Callable[[], float] = time.monotonic,
        build_backend_fn: Callable[..., LLMBackend] | None = None,
        supervisor_factory: Callable[[Any], Any] | None = None,
        ollama_url: str | None = None,
        vllm_url: str | None = None,
    ) -> None:
        self._ttl_s = float(ttl_s)
        self._now_fn = now_fn
        self._build_backend = build_backend_fn or _default_build_backend
        self._supervisor_factory = supervisor_factory or (
            lambda backend: LLMSupervisor(backend=backend)
        )
        # None ⇒ resolve from env at use-time; a non-None override is used verbatim
        # (after the same normalization the env path gets).
        self._ollama_url_override = ollama_url
        self._vllm_url_override = vllm_url

        self._lock = threading.RLock()
        self._cache: dict[tuple[str, str], _Entry] = {}

        # LISTEN invalidator plumbing (mirrors WorkerControlWatcher).
        self._listener_stop = threading.Event()
        self._listener_thread: threading.Thread | None = None

    # ── URL resolution ───────────────────────────────────────────────────────

    def _ollama_base_url(self) -> str:
        if self._ollama_url_override is not None:
            return self._normalize_ollama(self._ollama_url_override)
        raw = os.environ.get(get_config().ollama_url_env, "").strip()
        return self._normalize_ollama(raw or DEFAULT_OLLAMA_URL)

    def _vllm_base_url(self) -> str:
        if self._vllm_url_override is not None:
            return self._normalize_vllm(self._vllm_url_override)
        raw = os.environ.get(get_config().vllm_url_env, "").strip()
        return self._normalize_vllm(raw or DEFAULT_VLLM_URL)

    @staticmethod
    def _normalize_ollama(url: str) -> str:
        """Ollama's chat endpoint is built off the ROOT (``/api/chat``), so we only
        strip a trailing slash; nothing else to peel."""
        return url.rstrip("/")

    @staticmethod
    def _normalize_vllm(url: str) -> str:
        """Normalize the vllm URL to the server ROOT. ``VLLMBackend`` appends
        ``/v1/chat/completions`` itself, but the deployed ai_leads env value is the
        OpenAI base (``http://host:8000/v1``) — so strip a trailing ``/v1`` (after
        any trailing slash) to avoid a doubled ``/v1/v1``. Idempotent for a value
        already at the root."""
        u = url.rstrip("/")
        if u.endswith("/v1"):
            u = u[: -len("/v1")]
        return u.rstrip("/")

    def _base_url_for(self, server_type: str) -> str:
        if server_type == worker_control.SERVER_TYPE_VLLM:
            return self._vllm_base_url()
        return self._ollama_base_url()

    # ── the read-through cache ─────────────────────────────────────────────────

    def get_backend(self, host_label: str, queue: str) -> LLMBackend:
        """Return the live backend for ``(host_label, queue)``, building/rebuilding
        it only when the DB config snapshot actually changed.

        Fast path: a cached entry whose ``last_check`` is within ``ttl_s`` and not
        flagged invalid is returned WITHOUT touching the DB — the TTL throttle that
        keeps a hot request loop from SELECTing every call. Otherwise re-read
        ``llm_config_for``, refresh ``last_check``, clear the invalid flag, and:
        same snapshot ⇒ return the cached instance unchanged (identity preserved);
        changed snapshot ⇒ rebuild (shutdown the old + stop its supervisor, build
        the new + arm a supervisor for vllm)."""
        key = (host_label, queue)
        with self._lock:
            entry = self._cache.get(key)
            now = self._now_fn()
            if (
                entry is not None
                and not entry.invalid
                and (now - entry.last_check) < self._ttl_s
            ):
                # Within the TTL and not invalidated — the throttle hit; no DB read.
                return entry.backend

            cfg = worker_control.llm_config_for(host_label, queue)
            snapshot = _Snapshot.from_config(cfg)

            if entry is not None and entry.snapshot == snapshot:
                # Re-read confirmed nothing changed — refresh the check clock + drop
                # the invalid flag, but KEEP the same instance (its counters / vllm
                # state survive). This is the identity-preservation core.
                entry.last_check = now
                entry.invalid = False
                return entry.backend

            # Snapshot changed (or first build for this key) — rebuild.
            if entry is not None:
                self._teardown_entry(entry)
            new_entry = self._build_entry(snapshot, now)
            self._cache[key] = new_entry
            return new_entry.backend

    def invalidate(self, host_label: str, queue: str) -> None:
        """Force the next :meth:`get_backend` for this key to re-read the DB,
        regardless of the TTL. Set by the LISTEN invalidator on a NOTIFY so an
        operator's config edit takes effect at once (the TTL is only the fallback).
        A no-op for a key we've never built — there's nothing cached to refresh,
        and the first ``get_backend`` will read the DB anyway."""
        key = (host_label, queue)
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                entry.invalid = True

    def _build_entry(self, snapshot: _Snapshot, now: float) -> _Entry:
        """Construct a backend (+ a vllm supervisor) for ``snapshot``. Caller holds
        the lock. idle_ttl mapping: ollama → 0 (its daemon self-manages idle, and
        the supervisor is inert for it anyway); vllm → the configured
        ``vllm_idle_ttl_s``. Both get the configured ``parallelism``."""
        base_url = self._base_url_for(snapshot.server_type)
        is_vllm = snapshot.server_type == worker_control.SERVER_TYPE_VLLM
        idle_ttl_s = float(snapshot.vllm_idle_ttl_s) if is_vllm else 0.0
        backend = self._build_backend(
            snapshot.server_type, base_url, snapshot.parallelism, idle_ttl_s,
        )
        supervisor = None
        if is_vllm:
            # Arm the idle supervisor for the vllm sidecar. ``start()`` is idempotent
            # + env-gated (AI_LEADS_DISABLE_LLM_SUPERVISOR) so it's inert in tests
            # and a no-op for a non-positive TTL. ollama gets no supervisor.
            supervisor = self._supervisor_factory(backend)
            try:
                supervisor.start()
            except Exception:
                # Arming the daemon must never break handing back a usable backend.
                log.exception(
                    "[BackendFactory] supervisor start failed for vllm %s "
                    "(idle reclaim disabled for this backend)", base_url,
                )
        return _Entry(
            snapshot=snapshot, backend=backend, supervisor=supervisor,
            last_check=now, invalid=False,
        )

    def _teardown_entry(self, entry: _Entry) -> None:
        """Release a replaced/dropped entry: stop its supervisor (if any), then
        ``shutdown`` the backend (frees a vllm sidecar's VRAM; a no-op for ollama).
        Best-effort — a flaky stop/shutdown must not block the rebuild or the
        process teardown. Caller holds the lock; supervisor.stop() joins its own
        daemon (bounded 2s), which doesn't re-enter the factory, so no deadlock."""
        if entry.supervisor is not None:
            try:
                entry.supervisor.stop()
            except Exception:
                log.exception("[BackendFactory] supervisor stop failed (ignored)")
        try:
            entry.backend.shutdown()
        except Exception:
            log.exception("[BackendFactory] backend shutdown failed (ignored)")

    # ── LISTEN invalidator (mirrors WorkerControlWatcher) ──────────────────────

    @property
    def _listener_enabled(self) -> bool:
        """The LISTEN invalidator runs unless its dedicated kill-switch is set
        (tests set it in conftest). DISTINCT from the supervisor's gate."""
        return not bool(os.environ.get(DISABLE_LISTENER_ENV))

    def start(self) -> None:
        """Spawn the daemon that LISTENs
        :data:`~queue_workflows.worker_control.LLM_CONFIG_NOTIFY_CHANNEL` and, on each
        ``"host|queue"`` payload, invalidates that key so the next request re-reads.
        Idempotent (a second call while running is a no-op) and env-gated — a no-op
        when :attr:`_listener_enabled` is False, exactly like
        ``WorkerControlWatcher.start`` / ``LLMSupervisor.start``."""
        if not self._listener_enabled or self._listener_thread is not None:
            return
        self._listener_stop.clear()
        self._listener_thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="llm-config-invalidator",
        )
        self._listener_thread.start()
        log.info("[BackendFactory] LLM-config invalidator armed")

    def _listen_loop(self) -> None:
        """Autocommit-connect, ``LISTEN`` the config channel, and invalidate the
        keyed entry on each NOTIFY. MIRRORS ``WorkerControlWatcher._loop``: a
        ``notifies(timeout=…, stop_after=1)`` wait so ``stop()`` is observed within
        one poll; the whole loop swallows + logs on crash (a dropped LISTEN must not
        take down the worker — the TTL re-read still keeps configs fresh)."""
        import psycopg

        try:
            with psycopg.connect(db_url(), autocommit=True) as listen_conn:
                listen_conn.execute(
                    f"LISTEN {worker_control.LLM_CONFIG_NOTIFY_CHANNEL}"
                )
                while not self._listener_stop.is_set():
                    for notify in listen_conn.notifies(
                        timeout=self._ttl_s, stop_after=1,
                    ):
                        self._handle_notify(notify.payload)
                    if self._listener_stop.is_set():
                        return
        except Exception:
            log.exception("[BackendFactory] config-invalidator loop crashed")

    def _handle_notify(self, payload: str | None) -> None:
        """Parse a ``"host|queue"`` NOTIFY payload (the 0013 trigger's format) and
        invalidate that key. A malformed/empty payload is ignored — the TTL re-read
        is the backstop, so a stray NOTIFY can't crash the daemon."""
        if not payload or "|" not in payload:
            return
        host_label, _, queue = payload.partition("|")
        if host_label and queue:
            self.invalidate(host_label, queue)

    def stop(self) -> None:
        """Stop the invalidator daemon AND release every cached backend (shutdown +
        supervisor stop). Called on worker teardown. Idempotent + safe when never
        started; joins the daemon bounded (mirrors ``WorkerControlWatcher.stop``)."""
        self._listener_stop.set()
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=2.0)
            self._listener_thread = None
        with self._lock:
            for entry in self._cache.values():
                self._teardown_entry(entry)
            self._cache.clear()


# ── module singleton + thin functions (model_cache idiom) ────────────────────

_DEFAULT = BackendFactory()


def get_backend(host_label: str, queue: str) -> LLMBackend:
    """Process-wide :func:`BackendFactory.get_backend` — the node's request-time
    entry point."""
    return _DEFAULT.get_backend(host_label, queue)


def invalidate(host_label: str, queue: str) -> None:
    """Process-wide :func:`BackendFactory.invalidate`."""
    _DEFAULT.invalidate(host_label, queue)


def start() -> None:
    """Start the process-wide factory's LISTEN invalidator (env-gated, idempotent).
    The gpu claim worker calls this once its heartbeat is up."""
    _DEFAULT.start()


def stop() -> None:
    """Stop the process-wide factory's invalidator + release all cached backends.
    The gpu claim worker calls this on teardown."""
    _DEFAULT.stop()


def reset_default_for_tests() -> None:
    """TEST-ONLY. Stop + replace the process-wide :data:`_DEFAULT` so a test's
    singleton state (cached backends, the running invalidator) doesn't leak into the
    next — the analog of ``gpu_model_cache._reset_gpu_model_cache_for_tests``."""
    global _DEFAULT
    try:
        _DEFAULT.stop()
    except Exception:
        log.exception("[BackendFactory] reset_default_for_tests: stop failed")
    _DEFAULT = BackendFactory()


__all__ = ["BackendFactory", "get_backend"]
