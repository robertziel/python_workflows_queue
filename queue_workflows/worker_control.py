"""Worker ON/OFF control — durable desired-state per ``(host_label, queue)`` plus
the worker-side watcher that enforces it.

An operator (or a host UI such as ai_leads' Rails, which shares the same Postgres)
turns a machine's cpu / gpu / ingest worker ON or OFF by writing a ``worker_controls``
row (migration 0012). This module owns:

  * the **state accessors** — :func:`set_worker_control` / :func:`get_worker_control`
    / :func:`enable_worker` / :func:`disable_worker` (a thin INSERT … ON CONFLICT,
    so a non-Python consumer can write the same row directly + let the row trigger
    fire the wake NOTIFY);
  * the **stop-policy registry** :data:`STOP_POLICIES` — the extensibility seam.
    Only ``"hard"`` is implemented today (kill in-flight work + free RAM now);
    ``"drain"`` (finish current task, then stop) and ``"pause"`` (stop claiming,
    keep the model warm) are reserved names that slot in later as additional
    handlers with NO schema/API change;
  * the **worker-side watcher** :class:`WorkerControlWatcher` — a daemon thread,
    modelled on :class:`queue_workflows.claim_worker.JobStatusWatcher`, that
    LISTENs ``worker_control`` (+ a safety poll) and, on observing OFF, dispatches
    the row's ``stop_policy``.

WHY HARD STOP IS A PROCESS EXIT. A claim worker runs the node body INLINE on its
main thread (no thread/subprocess wraps it), so a watcher thread cannot preempt
in-flight work — and a wedged CUDA kernel won't cooperatively cancel. Terminating
the PROCESS is the only thing that reliably stops the work and reclaims RAM/VRAM
(the OS tears down the CUDA context on exit); every in-engine watchdog uses the
same lever (``os._exit``). The supervisor (docker ``restart: on-failure``) brings
the container back; on boot the claim worker re-reads ``worker_controls`` and
PARKS instead of claiming while still OFF (see ``claim_worker.run_forever``).

This module imports NOTHING from ``claim_worker`` (claim_worker imports IT), so a
stop-policy handler takes the worker as a duck-typed object exposing ``host`` /
``queue`` / ``requeue_inflight_for_control()``.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from dataclasses import dataclass
from typing import Any, Callable

from queue_workflows import node_queue
from queue_workflows.config import get_config
from queue_workflows.db import connection, db_url

log = logging.getLogger(__name__)


# ── constants ───────────────────────────────────────────────────────────────

#: Process exit code a control HARD stop uses. Distinct from the watchdog codes
#: (75 budget / 76 stall / 77 reassigned / 78 gpu-health) so an operator can tell
#: a control-driven stop apart in the logs. NON-ZERO so docker's
#: ``restart: on-failure`` brings the container back — it re-reads worker_controls
#: on boot and PARKS while still OFF, so it doesn't immediately re-claim.
EXIT_CONTROL_HARD_STOP = 79

STATE_ON = "on"
STATE_OFF = "off"
_VALID_STATES = frozenset({STATE_ON, STATE_OFF})

#: pg_notify channel the row trigger (migration 0012) fires on every write.
NOTIFY_CHANNEL = "worker_control"

#: Safety-poll cadence behind the LISTEN wake (mirrors the cancel-watcher's 5 s).
#: Catches a dropped NOTIFY and a control row written before this worker booted.
#: Env-overridable for ops/tests (matches node_queue.STALE_WORKER_AFTER_S shape).
WORKER_CONTROL_POLL_S = 5.0


# ── per-machine LLM server config (migration 0013) ─────────────────────────────

#: pg_notify channel the 0013 trigger fires when an LLM-config column changes.
#: DELIBERATELY separate from NOTIFY_CHANNEL ('worker_control') so a config edit
#: does NOT look like an ON/OFF change to the WorkerControlWatcher — the backend
#: factory LISTENs this one for instant refresh (10 s TTL is the fallback).
LLM_CONFIG_NOTIFY_CHANNEL = "worker_llm_config_changed"

SERVER_TYPE_OLLAMA = "ollama"
SERVER_TYPE_VLLM = "vllm"
#: The valid LLM server types — kept in lockstep with the 0013 column CHECK.
VALID_SERVER_TYPES = frozenset({SERVER_TYPE_OLLAMA, SERVER_TYPE_VLLM})

#: Defaults mirroring the 0013 column DEFAULTs (single source of truth for the
#: insert-COALESCE path and the default-safe llm_config_for fallback).
DEFAULT_LLM_SERVER_TYPE = SERVER_TYPE_OLLAMA
DEFAULT_LLM_PARALLELISM = 1
DEFAULT_VLLM_IDLE_TTL_S = 60


@dataclass(frozen=True)
class LLMConfig:
    """The per-``(host_label, queue)`` LLM server config a worker reads to decide
    which backend to drive. ``parallelism`` is the SIDECAR's concurrent-request
    capacity (ollama OLLAMA_NUM_PARALLEL / vllm --max-num-seqs), NOT the
    claim-worker concurrency (1 by contract). ``vllm_idle_ttl_s`` is the idle
    window before the supervisor SIGTERMs the vllm sidecar (ignored for ollama)."""

    server_type: str = DEFAULT_LLM_SERVER_TYPE
    parallelism: int = DEFAULT_LLM_PARALLELISM
    vllm_idle_ttl_s: int = DEFAULT_VLLM_IDLE_TTL_S


def _worker_control_poll_s() -> float:
    raw = (os.environ.get("AI_LEADS_WORKER_CONTROL_POLL_S", "") or "").strip()
    if not raw:
        return WORKER_CONTROL_POLL_S
    try:
        return max(0.1, float(raw))
    except (TypeError, ValueError):
        return WORKER_CONTROL_POLL_S


def _default_host() -> str:
    """This host's label — same derivation the claim worker uses
    (``config.host_label_env`` env, else the OS hostname)."""
    return os.environ.get(get_config().host_label_env, "").strip() or socket.gethostname()


# ── state accessors ───────────────────────────────────────────────────────────


def set_worker_control(
    host_label: str,
    queue: str,
    *,
    desired_state: str,
    stop_policy: str = "hard",
    requested_by: str | None = None,
    conn: Any = None,
) -> None:
    """Upsert the desired control state for a ``(host_label, queue)`` worker.

    Validates ``desired_state`` ∈ {on, off} and ``stop_policy`` against the
    in-code :data:`STOP_POLICIES` registry BEFORE touching the DB (fail-before-
    write, matching :func:`node_queue.enqueue_ingest_job`). The migration-0012
    trigger fires the ``worker_control`` NOTIFY so the worker wakes immediately.

    ``conn`` is an optional host psycopg connection: when given the INSERT runs on
    it so the caller controls the transaction (the row + the wake NOTIFY commit
    with the caller's own work); when ``None`` a pooled connection autocommits."""
    if desired_state not in _VALID_STATES:
        raise ValueError(
            f"desired_state must be in {sorted(_VALID_STATES)}, got {desired_state!r}"
        )
    if stop_policy not in STOP_POLICIES:
        raise ValueError(
            f"stop_policy must be a registered policy {sorted(STOP_POLICIES)}, got "
            f"{stop_policy!r} (register a handler in worker_control.STOP_POLICIES)"
        )
    sql = """
        INSERT INTO worker_controls
            (host_label, queue, desired_state, stop_policy, requested_by, updated_at)
        VALUES (%s, %s, %s, %s, %s, now())
        ON CONFLICT (host_label, queue) DO UPDATE
            SET desired_state = EXCLUDED.desired_state,
                stop_policy   = EXCLUDED.stop_policy,
                requested_by  = EXCLUDED.requested_by,
                updated_at    = now()
    """
    params = (host_label, queue, desired_state, stop_policy, requested_by)
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(sql, params)
    else:
        with connection() as own, own.cursor() as cur:
            cur.execute(sql, params)


def get_worker_control(host_label: str, queue: str) -> dict[str, Any] | None:
    """Return the control row for a ``(host_label, queue)`` worker, or ``None``
    when no row exists OR the ``worker_controls`` table hasn't been migrated yet.

    Never raises ``UndefinedTable``: a consumer DB that predates migration 0012
    simply has no control state ⇒ the worker treats it as ON (default-on, zero
    behaviour change). This is what lets the engine run unchanged before 0012 is
    applied — the claim worker's ``await_schema`` gate only waits for the lease
    migrations (6/8), not 12."""
    import psycopg

    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT host_label, queue, desired_state, stop_policy, "
                "       requested_by, updated_at "
                "FROM worker_controls WHERE host_label = %s AND queue = %s",
                (host_label, queue),
            )
            return cur.fetchone()
    except psycopg.errors.UndefinedTable:
        return None


def desired_state_for(host_label: str, queue: str) -> str:
    """The effective desired state for a worker: ``'off'`` ONLY when an explicit
    OFF row exists, else ``'on'`` (no row / table absent / on). The single
    decision point both the claim worker's park-gate and the watcher consult."""
    row = get_worker_control(host_label, queue)
    if row and row.get("desired_state") == STATE_OFF:
        return STATE_OFF
    return STATE_ON


def disable_worker(
    host_label: str, queue: str, *, stop_policy: str = "hard",
    requested_by: str | None = None,
) -> None:
    """Turn a worker OFF (hard-stop by default). Convenience wrapper over
    :func:`set_worker_control`."""
    set_worker_control(
        host_label, queue, desired_state=STATE_OFF,
        stop_policy=stop_policy, requested_by=requested_by,
    )


def enable_worker(
    host_label: str, queue: str, *, requested_by: str | None = None,
) -> None:
    """Turn a worker back ON. Convenience wrapper over
    :func:`set_worker_control`."""
    set_worker_control(
        host_label, queue, desired_state=STATE_ON, requested_by=requested_by,
    )


# ── per-machine LLM server config accessors (migration 0013) ───────────────────


def set_llm_config(
    host_label: str,
    queue: str,
    *,
    server_type: str | None = None,
    parallelism: int | None = None,
    vllm_idle_ttl_s: int | None = None,
    conn: Any = None,
) -> None:
    """Upsert the LLM server config for a ``(host_label, queue)`` worker.

    This is a SOFT config change, NOT the ON/OFF switch: it writes ONLY the LLM
    columns and leaves ``desired_state`` / ``stop_policy`` untouched (a config
    edit must never stop a running worker). The 0013 trigger fires the dedicated
    :data:`LLM_CONFIG_NOTIFY_CHANNEL` so the worker's backend factory refreshes.

    PARTIAL by design — pass only the field(s) you're changing. A ``None`` field
    keeps the existing value on an UPDATE (``COALESCE(EXCLUDED, existing)``) and
    falls back to the module default on the INSERT of a brand-new row (the
    columns are NOT NULL, so we can't write NULL). Validates each given value
    BEFORE the write (fail-before-write, matching :func:`set_worker_control`).

    ``conn`` threads an optional host connection so the row + wake NOTIFY commit
    inside the caller's txn; ``None`` autocommits on a pooled connection."""
    if server_type is not None and server_type not in VALID_SERVER_TYPES:
        raise ValueError(
            f"server_type must be in {sorted(VALID_SERVER_TYPES)}, got {server_type!r}"
        )
    if parallelism is not None and parallelism < 1:
        raise ValueError(f"parallelism must be >= 1, got {parallelism!r}")
    if vllm_idle_ttl_s is not None and vllm_idle_ttl_s < 0:
        raise ValueError(f"vllm_idle_ttl_s must be >= 0, got {vllm_idle_ttl_s!r}")

    sql = """
        INSERT INTO worker_controls
            (host_label, queue, llm_server_type, llm_parallelism, vllm_idle_ttl_s,
             updated_at)
        VALUES (
            %(host)s, %(queue)s,
            COALESCE(%(server_type)s, %(def_type)s),
            COALESCE(%(parallelism)s, %(def_par)s),
            COALESCE(%(idle_ttl)s, %(def_ttl)s),
            now()
        )
        ON CONFLICT (host_label, queue) DO UPDATE SET
            llm_server_type = COALESCE(%(server_type)s, worker_controls.llm_server_type),
            llm_parallelism = COALESCE(%(parallelism)s, worker_controls.llm_parallelism),
            vllm_idle_ttl_s = COALESCE(%(idle_ttl)s, worker_controls.vllm_idle_ttl_s),
            updated_at = now()
    """
    params = {
        "host": host_label,
        "queue": queue,
        "server_type": server_type,
        "parallelism": parallelism,
        "idle_ttl": vllm_idle_ttl_s,
        "def_type": DEFAULT_LLM_SERVER_TYPE,
        "def_par": DEFAULT_LLM_PARALLELISM,
        "def_ttl": DEFAULT_VLLM_IDLE_TTL_S,
    }
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(sql, params)
    else:
        with connection() as own, own.cursor() as cur:
            cur.execute(sql, params)


def llm_config_for(host_label: str, queue: str) -> LLMConfig:
    """The effective :class:`LLMConfig` for a worker: the row's LLM columns, or
    the all-defaults config when no row exists.

    Never raises on a partially-migrated DB — both ``UndefinedTable`` (pre-0012,
    no worker_controls) and ``UndefinedColumn`` (0012 applied but not 0013) fall
    back to defaults, so the engine + every consumer runs unchanged before 0013
    is applied (mirrors :func:`get_worker_control`'s pre-0012 tolerance)."""
    import psycopg

    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT llm_server_type, llm_parallelism, vllm_idle_ttl_s "
                "FROM worker_controls WHERE host_label = %s AND queue = %s",
                (host_label, queue),
            )
            row = cur.fetchone()
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
        return LLMConfig()
    if not row:
        return LLMConfig()
    return LLMConfig(
        server_type=row["llm_server_type"],
        parallelism=row["llm_parallelism"],
        vllm_idle_ttl_s=row["vllm_idle_ttl_s"],
    )


# ── stop policies (the extensibility seam) ────────────────────────────────────


def _apply_hard_stop(worker: Any, *, on_exit: Callable[[int], None]) -> None:
    """HARD stop: re-queue this worker's in-flight job(s) and terminate the
    process to free RAM/VRAM immediately.

    Process death is the only thing that reliably stops the inline node body
    (including a wedged CUDA kernel) and reclaims device memory — a cooperative
    in-process unload can't, and is moot once the work is gone. The supervisor
    restarts the container; on boot it re-reads ``worker_controls`` and PARKS
    while still OFF. ``on_exit`` defaults to ``os._exit`` (injected in tests).

    The re-queue (+ busy-ghost clear) runs FIRST so the in-flight job redistributes
    immediately and the GPU-busy gauge drops this worker, mirroring the watchdog
    trip path; a failure there is swallowed — the hard exit must happen regardless
    (the lease-reclaim safety net recovers any row we couldn't re-queue)."""
    try:
        n = worker.requeue_inflight_for_control()
        log.warning(
            "[worker-control:%s] HARD stop for host=%s — re-queued %d in-flight "
            "job(s); exiting(%d) to free RAM (supervisor restart will park it)",
            getattr(worker, "queue", "?"), getattr(worker, "host", "?"),
            n, EXIT_CONTROL_HARD_STOP,
        )
    except Exception:
        log.exception(
            "[worker-control] re-queue before hard stop failed; exiting anyway",
        )
    on_exit(EXIT_CONTROL_HARD_STOP)


#: Stop-policy registry: ``policy name -> handler(worker, *, on_exit)``. The
#: extensibility seam — add ``"drain"`` / ``"pause"`` here (each a new handler)
#: with no migration or API change. ``set_worker_control`` validates a requested
#: policy against these keys, and the watcher dispatches through it.
STOP_POLICIES: dict[str, Callable[..., None]] = {
    "hard": _apply_hard_stop,
}


# ── worker-side watcher ────────────────────────────────────────────────────────


class WorkerControlWatcher:
    """Daemon thread that HARD-STOPS a running worker the instant an operator sets
    its ``(host_label, queue)`` control row to ``desired_state='off'``.

    Mirrors :class:`queue_workflows.claim_worker.JobStatusWatcher`: a dedicated
    autocommit connection LISTENs ``worker_control`` (instant wake) with a periodic
    safety poll (catches a dropped NOTIFY and a row written before this worker
    booted). On OFF it dispatches the row's ``stop_policy`` through
    :data:`STOP_POLICIES`. ``on_exit`` is injectable so tests assert the trip
    without killing the test process; honours ``AI_LEADS_DISABLE_WORKER_CONTROL``
    (tests) to stay inert.

    Runs ONLY while the worker is in its claiming state (started after the boot
    park-gate). The reverse transition — a PARKED worker being turned back ON —
    is handled by the claim worker's park loop, not here."""

    def __init__(
        self, *, worker: Any,
        on_exit: Callable[[int], None] | None = None,
        poll_s: float | None = None,
    ) -> None:
        self._worker = worker
        self._host = worker.host
        self._queue = worker.queue
        self._on_exit = on_exit or os._exit
        self._poll_s = _worker_control_poll_s() if poll_s is None else float(poll_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def _enabled(self) -> bool:
        return not bool(os.environ.get("AI_LEADS_DISABLE_WORKER_CONTROL"))

    def check_once(self) -> bool:
        """Read the control row once; if OFF, dispatch its stop policy. Returns
        ``True`` iff a stop was triggered (the hard path normally never returns —
        ``os._exit`` — but an injected ``on_exit`` in tests does). Best-effort: a
        DB blip is swallowed + retried on the next tick."""
        try:
            row = get_worker_control(self._host, self._queue)
        except Exception:
            log.exception("[worker-control:%s] poll failed; retrying", self._queue)
            return False
        if not row or row.get("desired_state") != STATE_OFF:
            return False
        policy = row.get("stop_policy") or "hard"
        handler = STOP_POLICIES.get(policy)
        if handler is None:
            log.error(
                "[worker-control:%s] OFF requested with unimplemented stop_policy "
                "%r; falling back to hard", self._queue, policy,
            )
            handler = STOP_POLICIES["hard"]
        handler(self._worker, on_exit=self._on_exit)
        return True

    def start(self) -> None:
        if not self._enabled:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"worker-control-{self._queue}",
        )
        self._thread.start()

    def _loop(self) -> None:
        import psycopg

        # Catch a row written BEFORE we subscribed (e.g. set OFF while this worker
        # was mid-boot) before blocking on the NOTIFY.
        if self.check_once():
            return
        try:
            with psycopg.connect(db_url(), autocommit=True) as listen_conn:
                listen_conn.execute(f"LISTEN {NOTIFY_CHANNEL}")
                while not self._stop.is_set():
                    for _ in listen_conn.notifies(
                        timeout=self._poll_s, stop_after=1,
                    ):
                        break
                    if self._stop.is_set():
                        return
                    if self.check_once():
                        return
        except Exception:
            log.exception("[worker-control:%s] watcher loop crashed", self._queue)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ── console entry (ops / standalone) ───────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """``queue-worker-control --queue gpu --off`` / ``--on``. Writes the
    ``worker_controls`` row for a ``(host, queue)`` worker; the row trigger wakes
    the worker. Defaults ``--host`` to this box's label."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="queue-worker-control",
        description="Turn a machine's worker (host_label + queue) ON or OFF.",
    )
    parser.add_argument(
        "--host", default=None,
        help="host_label (default: this host's AI_LEADS_HOST_LABEL / hostname)",
    )
    parser.add_argument("--queue", required=True, help="cpu | gpu | <ingest queue>")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--on", action="store_true", help="enable the worker")
    group.add_argument(
        "--off", action="store_true", help="hard-stop + disable the worker",
    )
    parser.add_argument(
        "--policy", default="hard",
        help=f"stop policy for --off (default hard; registered: "
             f"{sorted(STOP_POLICIES)})",
    )
    parser.add_argument("--requested-by", default=None)
    args = parser.parse_args(argv)

    host = args.host or _default_host()
    state = STATE_OFF if args.off else STATE_ON
    try:
        set_worker_control(
            host, args.queue, desired_state=state,
            stop_policy=args.policy, requested_by=args.requested_by,
        )
    except ValueError as exc:
        parser.error(str(exc))
    log.info(
        "worker_control: %s/%s -> %s (policy=%s)", host, args.queue, state, args.policy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
