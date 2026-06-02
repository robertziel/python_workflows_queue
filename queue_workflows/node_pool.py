"""Node-per-job orchestrator (Postgres-as-queue).

This module owns the *background threads* the orchestrator container runs:

- ``_dispatch_loop`` — periodic sweep that finds freshly-queued ``mode='node'``
  runs and expands their DAG via :func:`dispatcher.start_run`. Each
  newly-ready node is INSERTed as a ``workflow_node_jobs`` row; the
  migration-0006 trigger NOTIFYs a claim worker. The same loop drains the
  dispatch-event outbox and runs the lease-reclaim sweeps (node-job + ingest).
- ``InputListener`` — polls ``workflow_input_submissions``; when Rails inserts
  a user's value for an ``awaiting_input`` node, calls
  :func:`dispatcher.resume_after_input` to unblock the DAG.

The hw-metrics sampler is NOT run here — it lives in the gpu claim worker.

The actual node bodies run in the ``claim_worker`` loops in the worker
containers (one process == one worker, ``SELECT … FOR UPDATE SKIP LOCKED``
claim). On startup the pool reclaims any expired-lease ``running`` rows left by
a worker that died across the restart (:meth:`_await_recovery`).

The builtin-model registration is an INJECTED hook (plan §2b-4):
``register_builtins`` is a ``Callable[[], None] | None`` (default None = skip),
NOT an import of any host's ``builtin_models``. The host's orchestrator passes
its registrar (or relies on ``config.builtin_model_registrar``).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable

from queue_workflows import dispatcher, input_listener, node_queue, run_store
from queue_workflows.db import connection

log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def cpu_worker_count() -> int:
    """Configured CPU-worker count (for the Rails queue snapshot fallback when
    no worker_heartbeats rows exist yet)."""
    return _int_env("AI_LEADS_WORKFLOW_CPU_WORKERS", 5)


def gpu_worker_count() -> int:
    return _int_env("AI_LEADS_WORKFLOW_GPU_WORKERS", 1)


# ── Pool ──────────────────────────────────────────────────────────────────


class NodePool:
    """Orchestrator background threads + input listener."""

    expand_poll_s: float = 0.5

    def __init__(
        self,
        *, cpu_workers: int | None = None, gpu_workers: int | None = None,
        register_builtins: Callable[[], None] | None = None,
    ):
        self._cpu_n = cpu_workers if cpu_workers is not None else cpu_worker_count()
        self._gpu_n = gpu_workers if gpu_workers is not None else gpu_worker_count()
        # Injected registrar — Callable run once at start(), or None to skip.
        # (The plan §2b-4 inversion: the engine never imports a host's
        # builtin_models; the host passes its registrar here.)
        self._register_builtins = register_builtins
        self._dispatch_stop = threading.Event()
        self._dispatch_thread: threading.Thread | None = None
        self._input_listener: input_listener.InputListener | None = None

        # Lease-reclaim sweep — re-queues ``running`` rows whose PG lease lapsed
        # (a dead/wedged worker that stopped renewing). ~5 s cadence
        # (interval-gated, NOT the 0.5 s dispatch tick).
        self._reclaim_interval_s: float = float(
            os.environ.get("AI_LEADS_LEASE_RECLAIM_INTERVAL_S", "5")
        )
        self._reclaim_last_run: float = 0.0

        # Ingest-lease reclaim sweep (twin of the node-job reclaim above) —
        # re-queues ``ingest_jobs`` rows whose lease lapsed. The SOLE recovery
        # for a dead/wedged fetch/load claim worker. Own ``last_run`` so the two
        # sweeps don't suppress each other.
        self._ingest_reclaim_interval_s: float = float(
            os.environ.get("AI_LEADS_LEASE_RECLAIM_INTERVAL_S", "5")
        )
        self._ingest_reclaim_last_run: float = 0.0

        # Dead-worker sweep — flags a worker whose ``worker_heartbeats`` row has
        # gone stale WHILE it still owns a ``running`` job (a GPU-hardware-hang
        # that wedged the worker PROCESS even though the lease-reclaim already
        # re-queued the JOB). The orchestrator is a separate process, so it can
        # see the frozen heartbeat the wedged worker's own GIL-blocked threads
        # cannot act on. Interval-gated like the reclaim sweeps; own ``last_run``.
        self._dead_worker_interval_s: float = float(
            os.environ.get("AI_LEADS_DEAD_WORKER_SWEEP_INTERVAL_S", "5")
        )
        self._dead_worker_last_run: float = 0.0

        # Node-event retention (migration 0011): the append-only
        # ``workflow_node_events`` forensic log is the only table here with no
        # natural terminal-state bound, so prune rows older than N days on a
        # slow (hourly) interval-gated sweep. ON DELETE CASCADE from
        # workflow_runs already covers purge / restart_from / session-delete;
        # this only catches events for runs that are never deleted.
        self._node_event_retention_days: int = int(
            os.environ.get("AI_LEADS_NODE_EVENT_RETENTION_DAYS", "30")
        )
        self._node_event_prune_interval_s: float = float(
            os.environ.get("AI_LEADS_NODE_EVENT_PRUNE_INTERVAL_S", "3600")
        )
        self._node_event_prune_last_run: float = 0.0

        # Orphan-cancel sweep — opt-in (``configure(cancel_orphan_queued_jobs=
        # True)``). Flips ``queued`` jobs whose parent run is already terminal
        # (``cancelled`` / ``failed``) to ``cancelled``. The claim SQL already
        # refuses them, but the rows linger and pollute queue gauges. Disabled
        # by default to preserve pre-0.4 behaviour byte-for-byte.
        self._orphan_cancel_interval_s: float = float(
            os.environ.get("AI_LEADS_ORPHAN_CANCEL_SWEEP_INTERVAL_S", "30")
        )
        self._orphan_cancel_last_run: float = 0.0

        # Stuck-run reconciler — recovery for a run the engine still calls
        # ``queued`` / ``running`` but which has NO live node-job backing it (a
        # ``cancelled`` node dead-ends the DAG; the dispatcher only advances on
        # completed/skipped/failed, then a resume re-queues the run into the
        # dead-end). ``last_run=0`` so a fresh instance reconciles on its FIRST
        # tick — "run instantly after instance start" — then every 5 min. See
        # :func:`dispatcher.reconcile_run`.
        self._stuck_run_interval_s: float = float(
            os.environ.get("AI_LEADS_STUCK_RUN_SWEEP_INTERVAL_S", "300")
        )
        self._stuck_run_last_run: float = 0.0

    def start(self) -> None:
        if self._register_builtins is not None:
            try:
                self._register_builtins()
            except Exception:
                log.exception("[node-pool] builtin model registration failed")

        # Health gate: refuse to start the dispatch thread until the recovery
        # sweep has reconciled stale ``running`` rows. A hard raise is correct
        # — the orchestrator's restart policy is the recovery mechanism.
        self._await_recovery()

        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="node-pool-dispatch",
        )
        self._dispatch_thread.start()

        # Input listener — polls ``workflow_input_submissions`` and calls
        # dispatcher.resume_after_input so downstream nodes get enqueued.
        # Non-fatal on boot failure.
        try:
            self._input_listener = input_listener.InputListener()
            self._input_listener.start()
        except Exception:
            log.exception("[node-pool] input_listener failed to start")

        log.info("[node-pool] started (cpu=%d, gpu=%d)", self._cpu_n, self._gpu_n)

    def stop(self) -> None:
        self._dispatch_stop.set()
        if self._input_listener is not None:
            self._input_listener.stop()
            self._input_listener.join(timeout=2.0)
            self._input_listener = None
        if self._dispatch_thread is not None:
            self._dispatch_thread.join(timeout=2.0)
            self._dispatch_thread = None
        log.info("[node-pool] stopped")

    @property
    def cpu_workers(self) -> int:
        return self._cpu_n

    @property
    def gpu_workers(self) -> int:
        return self._gpu_n

    def current_gpu_model(self) -> str | None:
        """The orchestrator can't see a worker's in-process model cache
        directly; the Rails snapshot reads ``current_model`` off
        ``worker_heartbeats``. This in-process inspector is unused, kept as a
        None stub."""
        return None

    # Cap retries before flagging a dispatch event as poisonous.
    _DISPATCH_MAX_ATTEMPTS: int = 10

    def _drain_dispatch_events(self) -> None:
        """Pop unprocessed events, invoke the dispatcher callback, record
        success or increment ``attempts`` on failure.

        The SELECT + per-event finalisation runs inside a single transaction
        with ``FOR UPDATE SKIP LOCKED`` so two concurrent drainers can't claim
        the same row. Dispatcher callbacks open their own DB connections, so
        they don't deadlock against the locks the outer cursor holds.
        """
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, run_id, node_id, kind, attempts, error
                  FROM workflow_dispatch_events
                 WHERE processed_at IS NULL
                 ORDER BY created_at ASC
                 FOR UPDATE SKIP LOCKED
                 LIMIT 50
                """,
            )
            events = list(cur.fetchall())
            for evt in events:
                evt_id = evt["id"]
                kind = evt["kind"]
                run_id = evt["run_id"]
                node_id = evt["node_id"]
                try:
                    if kind == "completed":
                        dispatcher.on_node_completed(run_id, node_id)
                    elif kind == "failed":
                        dispatcher.on_node_failed(run_id, node_id)
                    elif kind == "awaiting_input":
                        dispatcher.on_node_awaiting_input(run_id, node_id)
                    else:
                        log.error(
                            "[node-pool] unknown dispatch event kind=%r id=%s",
                            kind, evt_id,
                        )
                        cur.execute(
                            "UPDATE workflow_dispatch_events "
                            "SET attempts = attempts + 1, error = %s "
                            "WHERE id = %s",
                            (f"unknown kind={kind!r}", evt_id),
                        )
                        continue
                    cur.execute(
                        "UPDATE workflow_dispatch_events "
                        "SET processed_at = now() WHERE id = %s",
                        (evt_id,),
                    )
                except Exception as exc:
                    attempts = int(evt.get("attempts") or 0) + 1
                    err_text = f"{type(exc).__name__}: {exc}"
                    log.warning(
                        "[node-pool] dispatch event %s failed (attempts=%d): %s",
                        evt_id, attempts, err_text,
                    )
                    cur.execute(
                        "UPDATE workflow_dispatch_events "
                        "SET attempts = attempts + 1, error = %s "
                        "WHERE id = %s",
                        (err_text[:8000], evt_id),
                    )
                    if attempts >= self._DISPATCH_MAX_ATTEMPTS:
                        # Poisonous event — flip the run to ``failed`` so
                        # operators see something instead of a silent stall.
                        log.error(
                            "[node-pool] dispatch event %s exhausted retries; "
                            "marking run %s failed",
                            evt_id, run_id,
                        )
                        cur.execute(
                            "UPDATE workflow_runs "
                            "SET status = 'failed', "
                            "    finished_at = now(), "
                            "    error = %s "
                            "WHERE id = %s "
                            "  AND status NOT IN ("
                            "      'completed', 'failed', 'cancelled'"
                            "  )",
                            (
                                f"dispatch callback {kind!r} for node "
                                f"{node_id!r} failed after "
                                f"{attempts} attempts: {err_text[:500]}",
                                run_id,
                            ),
                        )
                        cur.execute(
                            "UPDATE workflow_dispatch_events "
                            "SET processed_at = now() WHERE id = %s",
                            (evt_id,),
                        )

    def _await_recovery(
        self,
        *,
        max_attempts: int | None = None,
        backoff_s: float | None = None,
    ) -> None:
        """Startup recovery — reclaim every expired lease before the dispatch
        thread starts, with bounded exponential-backoff retries. Raises
        ``RuntimeError`` when every attempt fails — the caller must propagate
        so the worker container restarts via its orchestrator.

        Configurable via env:
            ``AI_LEADS_NODE_POOL_RECOVERY_RETRIES`` (default 5)
            ``AI_LEADS_NODE_POOL_RECOVERY_BACKOFF_S`` (default 2.0)
        """
        if max_attempts is None:
            max_attempts = _int_env("AI_LEADS_NODE_POOL_RECOVERY_RETRIES", 5)
        if backoff_s is None:
            raw = os.environ.get(
                "AI_LEADS_NODE_POOL_RECOVERY_BACKOFF_S", "",
            ).strip()
            try:
                backoff_s = float(raw) if raw else 2.0
            except ValueError:
                backoff_s = 2.0

        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                reclaimed = node_queue.reclaim_expired_leases()
                if reclaimed:
                    log.info(
                        "[node-pool] recovery: reclaimed %d expired-lease job(s)",
                        len(reclaimed),
                    )
                else:
                    log.info("[node-pool] recovery ok")
                return
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "[node-pool] recovery attempt %d/%d failed: %r",
                    attempt + 1, max_attempts, exc,
                )
                if attempt < max_attempts - 1 and backoff_s > 0:
                    # Cap exponential backoff at 30 s.
                    wait = min(backoff_s * (2 ** attempt), 30.0)
                    # Respect ``stop()`` mid-backoff.
                    if self._dispatch_stop.wait(wait):
                        raise RuntimeError(
                            "node pool stopped during recovery backoff"
                        ) from last_exc
        raise RuntimeError(
            f"node pool refused to start: recovery failed after "
            f"{max_attempts} attempts"
        ) from last_exc

    def _dispatch_loop(self) -> None:
        """Periodically find node-mode runs in ``queued`` and expand them into
        initial node-jobs via :func:`dispatcher.start_run`."""
        while not self._dispatch_stop.is_set():
            try:
                self._tick()
            except Exception:
                log.exception("[node-pool] dispatcher tick failed")
            if self._dispatch_stop.wait(self.expand_poll_s):
                return

    def _tick(self) -> None:
        # 1) Expand new node-mode runs.
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM workflow_runs "
                "WHERE mode = 'node' AND status = 'queued' "
                "ORDER BY priority ASC, queued_at ASC NULLS LAST LIMIT 50"
            )
            ids = [r["id"] for r in cur.fetchall()]
        for run_id in ids:
            try:
                n = dispatcher.start_run(run_id)
                if n > 0:
                    run_store.update_run(run_id, status="running")
            except Exception:
                log.exception("[node-pool] start_run %s failed", run_id)

        # 2) Drain durable dispatcher fan-out events. Workers write a
        # ``workflow_dispatch_events`` row in the same txn as their terminal
        # UPDATE, so a callback that fails synchronously on the worker is
        # retried on the next tick instead of stalling the run.
        try:
            self._drain_dispatch_events()
        except Exception:
            log.exception("[node-pool] dispatch-event drain failed")

        # 3) Lease-reclaim sweep — re-queues ``running`` rows whose PG lease
        # lapsed. Interval-gated to ~5 s; the reclaim flips rows back to
        # ``queued`` and migration-0006's trigger fires the ``node_job_ready``
        # NOTIFY so an idle claim worker re-grabs it.
        try:
            self._sweep_expired_leases()
        except Exception:
            log.exception("[node-pool] lease-reclaim sweep failed")

        # 4) Ingest-lease reclaim sweep — the ``ingest_jobs`` twin of step 3.
        try:
            self._sweep_expired_ingest_leases()
        except Exception:
            log.exception("[node-pool] ingest-lease-reclaim sweep failed")

        # 5) Dead-worker sweep — flag a worker whose heartbeat froze while it
        # still owns a ``running`` job (a GPU-hardware-hang that wedged the
        # worker PROCESS even though step 3 already re-queued its JOB). Surfaces
        # the wedged worker for a host-supervisor to bounce — the orchestrator
        # can't safely cross-host-kill it.
        try:
            self._sweep_dead_workers()
        except Exception:
            log.exception("[node-pool] dead-worker sweep failed")

        # 6) Node-event retention — prune workflow_node_events older than the
        # retention window (append-only growth control; hourly-gated).
        try:
            self._sweep_node_event_retention()
        except Exception:
            log.exception("[node-pool] node-event retention sweep failed")

        # 7) Orphan-cancel sweep — opt-in. Flip ``queued`` jobs of cancelled /
        # failed runs to ``cancelled`` so the queue gauges don't read
        # misleadingly. Default-off so the engine's behaviour pre-0.4 is
        # unchanged; hosts that want the cleanup ship
        # ``configure(cancel_orphan_queued_jobs=True)``.
        try:
            self._sweep_orphan_queued_jobs()
        except Exception:
            log.exception("[node-pool] orphan-cancel sweep failed")

        # 8) Stuck-run reconciler — re-drive / re-queue / finalise runs the
        # engine still calls non-terminal but which have no live node-job (a
        # cancelled-node dead-end that a resume re-queued). Interval-gated to
        # 5 min; fires on the first tick after start (instant recovery).
        try:
            self._sweep_stuck_runs()
        except Exception:
            log.exception("[node-pool] stuck-run sweep failed")

    def _sweep_stuck_runs(self) -> None:
        """Reconcile phantom runs: the engine still calls them ``queued`` /
        ``running`` but they have NO live node-job (``queued`` / ``running`` /
        ``awaiting_input``), so a worker has nothing to claim and the run sits
        there forever. Arises when a ``cancelled`` node dead-ends the DAG (the
        dispatcher only advances on completed/skipped/failed) and a resume
        (:func:`run_store.reenqueue_running_for_resume`) re-queues the run into
        that dead-end. :func:`dispatcher.reconcile_run` re-drives each — putting
        the blocked node(s) BACK ON THE QUEUE where it can, finalising the rest.

        Interval-gated (``_stuck_run_interval_s``, default 300 s); ``last_run``
        starts at 0 so a fresh instance reconciles on its FIRST tick (instant
        recovery after a deploy/restart) then every 5 min.
        """
        import time as _time

        now = _time.time()
        if now - self._stuck_run_last_run < self._stuck_run_interval_s:
            return
        self._stuck_run_last_run = now

        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id FROM workflow_runs r
                WHERE r.mode = 'node'
                  AND r.status IN ('queued', 'running')
                  AND NOT EXISTS (
                    SELECT 1 FROM workflow_node_jobs j
                    WHERE j.run_id = r.id
                      AND j.status IN ('queued', 'running', 'awaiting_input')
                  )
                """
            )
            ids = [r["id"] for r in cur.fetchall()]
        for run_id in ids:
            try:
                action = dispatcher.reconcile_run(run_id)
                if action != "noop":
                    log.info(
                        "[node-pool] reconciled stuck run %s → %s",
                        run_id, action,
                    )
            except Exception:
                log.exception(
                    "[node-pool] reconcile of stuck run %s failed", run_id,
                )

    def _sweep_expired_leases(self) -> None:
        """Re-queue ``running`` rows whose PG lease has lapsed.

        Interval-gated (``_reclaim_interval_s``, default 5 s) so the 0.5 s
        dispatch loop doesn't run the reclaim UPDATE every tick.
        """
        import time as _time

        now = _time.time()
        if now - self._reclaim_last_run < self._reclaim_interval_s:
            return
        self._reclaim_last_run = now

        reclaimed = node_queue.reclaim_expired_leases()
        for row in reclaimed:
            log.warning(
                "[node-pool] reclaimed expired-lease job %s "
                "(run=%s node=%s) → re-queued",
                row.get("id"), row.get("run_id"), row.get("node_id"),
            )

    def _sweep_expired_ingest_leases(self) -> None:
        """Re-queue ``ingest_jobs`` rows whose PG lease has lapsed — the ingest
        twin of :meth:`_sweep_expired_leases`. This is the only recovery for a
        dead/wedged fetch/load claim worker."""
        import time as _time

        now = _time.time()
        if now - self._ingest_reclaim_last_run < self._ingest_reclaim_interval_s:
            return
        self._ingest_reclaim_last_run = now

        reclaimed = node_queue.reclaim_expired_ingest_leases()
        for row in reclaimed:
            log.warning(
                "[node-pool] reclaimed expired-lease ingest job %s "
                "(task=%s queue=%s) → re-queued",
                row.get("id"), row.get("task_name"), row.get("queue"),
            )

    def _sweep_dead_workers(self) -> None:
        """Flag a worker whose ``worker_heartbeats`` row has gone stale WHILE it
        still owns a ``running`` job, and surface it for a host-supervisor to
        bounce.

        This closes the gap the lease-reclaim alone can't: a GPU-hardware-hang
        can wedge the worker PROCESS (a torch/HIP call blocked in a dead GPU
        context) so its in-process watchdog can't act and its heartbeat freezes,
        even after :meth:`_sweep_expired_leases` has re-queued the JOB onto a
        healthy host. The orchestrator runs in a SEPARATE process — independent
        of the wedged worker's blocked Python threads — so it can observe the
        frozen heartbeat and flag the dead worker.

        Recovery split:
          * the JOB is already recovered by the lease-reclaim sweep (step 3);
          * the dead PROCESS is flagged here — a clear ERROR log + a durable
            ``worker_heartbeats.last_flagged_dead_at`` marker an operator /
            host-supervisor polls. The orchestrator does NOT kill the worker: a
            cross-host container kill isn't safe/feasible from here (no docker
            socket, different host). See ``docs`` / module note for the
            host-supervisor hook that consumes the flag.

        Interval-gated (``_dead_worker_interval_s``, default 5 s) so the 0.5 s
        dispatch loop doesn't run the detector UPDATE every tick. The detector
        is itself idempotent (re-flags only after a recovered worker goes stale
        again), so the gate is purely a load optimisation."""
        import time as _time

        now = _time.time()
        if now - self._dead_worker_last_run < self._dead_worker_interval_s:
            return
        self._dead_worker_last_run = now

        flagged = node_queue.flag_stale_workers_holding_running_jobs()
        for row in flagged:
            log.error(
                "[node-pool] DEAD WORKER: %s/%s heartbeat stale since %s but "
                "still owns %s running job(s) — worker PROCESS is wedged (GPU "
                "hang?); job(s) re-queued by lease-reclaim, but the container "
                "must be bounced (host-supervisor should restart it)",
                row.get("host_label"), row.get("queue"),
                row.get("last_seen"), row.get("running_jobs"),
            )

    def _sweep_orphan_queued_jobs(self) -> None:
        """Flip ``queued`` jobs of already-terminal runs to ``cancelled``.

        Opt-in via :attr:`EngineConfig.cancel_orphan_queued_jobs` — default
        ``False`` so the engine's pre-0.4 behaviour is preserved byte-for-byte.
        When enabled, runs interval-gated (``_orphan_cancel_interval_s``,
        default 30 s) so the 0.5 s dispatch loop doesn't run the join UPDATE
        every tick. The underlying SQL is idempotent (no-op if there are no
        orphans), so the gate is purely a load optimisation.
        """
        from queue_workflows.config import get_config

        if not get_config().cancel_orphan_queued_jobs:
            return

        import time as _time

        now = _time.time()
        if now - self._orphan_cancel_last_run < self._orphan_cancel_interval_s:
            return
        self._orphan_cancel_last_run = now

        flipped = node_queue.cancel_orphaned_queued_jobs()
        if flipped:
            log.info(
                "[node-pool] cancelled %d orphaned queued job(s) of "
                "terminal runs",
                flipped,
            )

    def _sweep_node_event_retention(self) -> None:
        """Prune ``workflow_node_events`` rows older than the retention window.

        Interval-gated (``_node_event_prune_interval_s``, default 1 h) — the
        append-only event log (migration 0011) is the only table here without a
        natural terminal bound, so an age-based sweep keeps it from growing
        unboundedly for runs that are never deleted. ``ON DELETE CASCADE`` from
        ``workflow_runs`` already covers purge / restart_from / session-delete.
        Retention via ``AI_LEADS_NODE_EVENT_RETENTION_DAYS`` (default 30).
        """
        import time as _time

        now = _time.time()
        if now - self._node_event_prune_last_run < self._node_event_prune_interval_s:
            return
        self._node_event_prune_last_run = now

        deleted = node_queue.prune_node_events(self._node_event_retention_days)
        if deleted:
            log.info(
                "[node-pool] pruned %d node-event row(s) older than %d days",
                deleted, self._node_event_retention_days,
            )
