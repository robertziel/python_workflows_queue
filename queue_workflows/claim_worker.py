"""Postgres-as-queue claim worker — the live job runtime.

A per-worker loop: a Postgres ``SELECT … FOR UPDATE SKIP LOCKED`` claim woken
by ``LISTEN node_job_ready`` (migration 0006) with a 1 s safety poll. One
process per worker (``queue-claim-worker --queue={cpu,gpu,fetch,load}`` or the
host's launcher); the dispatcher INSERTs job rows and the migration-0006/0007
trigger NOTIFYs this loop.

Loop shape (one process == one worker, concurrency-1 structural):

    LISTEN node_job_ready
    while running:
        claimed = claim_next_{cpu,gpu}_job(host, host_priority, current_model,
                                           lease_s)
        if claimed:
            run it through execute_node under a lease-renewer + a per-job
            watchdog, then re-loop immediately (drain greedily)
        else:
            block on the notify with a 1 s timeout, then re-loop

Daemon threads bracket each claimed job:

  * :class:`LeaseRenewer` — every ~10 s pushes ``lease_expires_at`` out so a
    long job keeps its lease; scoped to ``id AND claimed_by`` so it can't
    extend a lease a reclaim already handed to another worker.
  * :class:`Watchdog` — a wall-clock budget. On trip it marks the row failed
    then HARD-exits the process; the lease then lets
    ``reclaim_expired_leases`` re-queue it. Applied to CPU + ingest jobs only.
  * :class:`GpuHealthWatchdog` — the GPU guard, HEALTH-driven not wall-clock:
    it replaces the fixed budget for GPU jobs and kills ONLY a truly-wedged
    worker (no per-container GPU work AND static container RAM over a 5-min
    window). A busy GPU or a > 5 GB RAM move keeps the job alive no matter how
    long it runs — there is NO fixed time cap for GPU renders.
  * :class:`StallWatchdog` — a tight no-progress deadline kept for non-video
    GPU nodes (defense in depth: catches a fast 0%-GPU hang in ~2 min).

GPU jobs are policed by HEALTH, never by elapsed time — a long-but-healthy
render is never killed; a wedged one is caught by the health watchdog (and, for
non-video, the stall watchdog) and re-queued via the lease reclaim.
"""

from __future__ import annotations

import inspect
import logging
import os
import socket
import threading
import time
from typing import Any, Callable

from queue_workflows import node_executor, node_queue
from queue_workflows.config import get_config
from queue_workflows.db import connection, db_url

log = logging.getLogger(__name__)


# ── wall-clock budgets ─────────────────────────────────────────────────────
GPU_DEFAULT_BUDGET_S = 8100      # 2.25 h hard — default GPU job
VIDEO_BUDGET_S = 1800            # 30 min — video render (config.video_model_ids)
CPU_BUDGET_S = 2100              # 35 min hard — default CPU job
INPUT_BUDGET_S = 120             # input (park-and-return) node
# Ingest: the PG run_fetch_all / run_load_all run the whole per-(source,scope)
# sweep INLINE, so the budget must cover the slowest full sweep.
FETCH_BUDGET_S = 2 * 3600        # 2 h — full network sweep across all sources
LOAD_BUDGET_S = 3600             # 1 h — full landing-zone → Postgres sweep

# Lease length + renewal cadence. The lease is renewed every
# ``LEASE_RENEW_INTERVAL_S`` while a job runs, so its absolute length is
# independent of how long the job takes.
LEASE_S = node_queue.DEFAULT_LEASE_S          # 600 s
LEASE_RENEW_INTERVAL_S = 10.0
NOTIFY_POLL_TIMEOUT_S = 1.0                   # safety poll for a dropped NOTIFY

# No-progress (stall) deadline for a GPU node. The wall-clock budget above only
# catches a job that runs too LONG; this catches one that makes NO PROGRESS — a
# Blackwell qwen inference hang sits model-resident at 0 % GPU and would camp
# the full 8100 s budget. A node beats this deadline once per diffusion step
# (via ``status_callback``); the executor beats once more right after the model
# load completes — which ARMS the watchdog. It is inert before that first beat,
# so a multi-minute cold model load (observed ~6 min) can't false-trip it. The
# window only spans the gap between two diffusion steps (~12 s), so it can be
# tight: no beat for this long once armed ⇒ hung ⇒ fail + hard-exit so the lease
# reclaim re-queues the job onto a healthy host.
STALL_TIMEOUT_S = 120.0                        # ≫ inter-step gap (~12 s); load is excluded
STALL_POLL_S = 5.0


# ── GPU health watchdog thresholds (HEALTH-driven, NOT wall-clock) ──────────
# The GPU health watchdog replaces the fixed wall-clock budget for GPU jobs: it
# never kills a job just because time passed. Instead, every
# ``GPU_HEALTH_INTERVAL_S`` it asks "is this worker actually using the GPU, or
# moving memory?" — and only kills a truly-wedged worker (no GPU work AND static
# RAM over the whole window). A busy GPU or a > RAM_DELTA memory move means the
# job is healthy and is NEVER policed by time. Every threshold is env-overridable.
#
#   * GPU_HEALTH_INTERVAL_S  — checkpoint cadence; the window the trip rule spans.
#   * GPU_IDLE_PCT           — per-container GPU util at/under which GPU counts as idle.
#   * GPU_HEALTH_RAM_DELTA_MB — |RAM change| over the window above which the job
#                              counts as doing work (e.g. staging / decode), so it
#                              is NOT killed even if GPU util reads idle.
# GPU util is sampled on the same short ``STALL_POLL_S`` cadence and the MAX over
# the window is the busy signal (a single denoise burst keeps the window healthy).


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, "").strip() or default))
    except (TypeError, ValueError):
        return int(default)


GPU_HEALTH_INTERVAL_S = _env_float("AI_LEADS_GPU_HEALTH_INTERVAL_S", 300.0)
GPU_IDLE_PCT = _env_int("AI_LEADS_GPU_IDLE_PCT", 5)
GPU_HEALTH_RAM_DELTA_MB = _env_int("AI_LEADS_GPU_HEALTH_RAM_DELTA_MB", 5120)  # 5 GiB

# Worker capacity heartbeat refresh cadence. Rails' queue gauge treats a row
# whose ``last_seen`` is older than 30 s as stale, so a 10 s refresh keeps us
# comfortably inside that window.
HEARTBEAT_INTERVAL_S = 10.0

# Lowest engine migration version each queue family's claim loop needs before
# polling (the ``queue_schema_version`` ledger). cpu/gpu draw from
# ``workflow_node_jobs`` and need the migration-0006 lease columns; ingest queues
# draw from ``ingest_jobs`` and need migration 0008 (the ``args`` column + the
# relaxed queue CHECK that lets host-defined queue names exist — without it a
# custom-queue row can't be inserted). A claim worker WAITS for its version
# before polling — the orchestrator owns the migration run (``db.bootstrap`` takes
# no advisory lock), the workers must not race it.
_NODE_REQUIRED_VERSION = 6
_INGEST_REQUIRED_VERSION = 8


def _host_label() -> str:
    return os.environ.get(get_config().host_label_env, "").strip() or socket.gethostname()


def _host_priority() -> int:
    """The claim's cross-host tiebreaker (``config.host_priority_env``):
    high-priority hosts head, overflow hosts tail. See ``node_queue._host_dir``."""
    try:
        return int(os.environ.get(get_config().host_priority_env, "0"))
    except (TypeError, ValueError):
        return 0


def budget_for(job: dict) -> int:
    """Wall-clock budget (s) for a claimed job:

      * GPU job whose ``required_model`` is in the host's video set → 1800 s;
      * any other GPU job → 8100 s;
      * built-in fetch / load ingest jobs → their generous sweep budgets;
      * any other (host-defined, G1) ingest queue → ``config.ingest_default_budget_s``;
      * an input node (``node_module`` starts ``__input__``) → 120 s;
      * any other CPU job → 2100 s.
    """
    queue = job.get("queue")
    if queue == "gpu":
        if (job.get("required_model") or "") in get_config().video_model_ids:
            return VIDEO_BUDGET_S
        return GPU_DEFAULT_BUDGET_S
    if queue == "fetch":
        return FETCH_BUDGET_S
    if queue == "load":
        return LOAD_BUDGET_S
    if queue not in _NODE_QUEUES:
        # host-defined ingest queue (G1) — configurable default budget.
        return get_config().ingest_default_budget_s
    if (job.get("node_module") or "").startswith("__input__"):
        return INPUT_BUDGET_S
    return CPU_BUDGET_S


# ── lease renewal ──────────────────────────────────────────────────────────


_LEASE_TABLES = frozenset({"workflow_node_jobs", "ingest_jobs"})


class LeaseRenewer:
    """Daemon thread that pushes ``lease_expires_at`` out every ``interval_s``
    while a job runs. Scoped to ``id AND claimed_by`` so a reclaim that handed
    the row to another worker is NOT clobbered.

    ``table`` selects which lease table to renew: ``workflow_node_jobs``
    (cpu/gpu) or ``ingest_jobs`` (fetch/load). The table name is validated
    against a fixed allowlist (never interpolated from caller data)."""

    def __init__(
        self, *, job_id: str, claimed_by: str,
        lease_s: int = LEASE_S, interval_s: float = LEASE_RENEW_INTERVAL_S,
        table: str = "workflow_node_jobs",
    ) -> None:
        if table not in _LEASE_TABLES:
            raise ValueError(f"lease table must be in {sorted(_LEASE_TABLES)}, got {table!r}")
        self._job_id = job_id
        self._claimed_by = claimed_by
        self._lease_s = int(lease_s)
        self._interval_s = float(interval_s)
        self._table = table
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"lease-renew-{self._job_id[:8]}",
        )
        self._thread.start()

    def _renew_once(self) -> bool:
        try:
            with connection() as conn, conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {self._table} "
                    "SET lease_expires_at = now() + make_interval(secs => %s) "
                    "WHERE id = %s AND claimed_by = %s AND status = 'running'",
                    (self._lease_s, self._job_id, self._claimed_by),
                )
                return (cur.rowcount or 0) > 0
        except Exception:
            log.exception("[lease-renew] %s renew failed; will retry", self._job_id)
            return False

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._renew_once()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ── watchdog ─────────────────────────────────────────────────────────────


def _fail_job_and_exit(
    *, job_id: str, table: str, error: str,
    on_exit: Callable[[int], None], exit_code: int,
) -> None:
    """Mark a doomed job ``failed`` (+ the dispatch-event outbox row for DAG
    node-jobs) then call ``on_exit``. Shared by :class:`Watchdog` (budget) and
    :class:`StallWatchdog` (no-progress) so the outbox-atomicity contract — the
    terminal mark and the ``failed`` event in ONE txn — is written in exactly
    one place. A mark failure is swallowed + logged: the hard-exit must still
    happen so the lease can expire and a reclaim re-queue the row."""
    try:
        if table == "ingest_jobs":
            # Ingest job: no parent run, no dispatch-event outbox — just mark
            # the row failed so a reclaim/operator sees it.
            node_queue.mark_ingest_failed(job_id, error=error)
        else:
            # DAG node-job: mark failed + write the dispatch-event row in ONE
            # txn so the run fails through to downstream nodes even though we're
            # about to hard-exit.
            with connection() as conn, conn.cursor() as cur:
                row = node_queue.mark_failed_in_txn(cur, job_id, error=error)
                if row is not None:
                    node_queue.enqueue_dispatch_event_in_txn(
                        cur, row["run_id"], row["node_id"], "failed",
                    )
    except Exception:
        log.exception("[watchdog] %s could not mark failed before exit", job_id)
    on_exit(exit_code)


class Watchdog:
    """Daemon thread enforcing a wall-clock budget on a single running job. On
    trip: mark the row failed with the budget-exceeded reason, then call
    ``on_exit`` (default: hard ``os._exit``). A hard exit kills exactly the
    over-budget job; the lease then lets the reclaim sweep re-queue it.

    Applied to CPU + ingest jobs only — GPU jobs are policed by health
    (:class:`GpuHealthWatchdog`), not by a wall-clock cap, so a long-but-healthy
    render is never killed for elapsed time."""

    def __init__(
        self, *, job_id: str, budget_s: float,
        on_exit: Callable[[int], None] | None = None,
        poll_s: float = 1.0, exit_code: int = 75,
        table: str = "workflow_node_jobs",
    ) -> None:
        if table not in _LEASE_TABLES:
            raise ValueError(f"watchdog table must be in {sorted(_LEASE_TABLES)}, got {table!r}")
        self._job_id = job_id
        self._budget_s = float(budget_s)
        self._on_exit = on_exit or os._exit
        self._poll_s = float(poll_s)
        self._exit_code = int(exit_code)
        self._table = table
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._deadline = time.monotonic() + self._budget_s
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"watchdog-{self._job_id[:8]}",
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if time.monotonic() >= self._deadline:
                self._trip()
                return
            # Wake on stop OR at the next poll boundary, whichever first.
            self._stop.wait(self._poll_s)

    def _trip(self) -> None:
        err = (
            f"exceeded wall-clock budget ({int(self._budget_s)}s) — "
            f"watchdog hard-stopped the worker"
        )
        log.error("[watchdog] %s %s", self._job_id, err)
        _fail_job_and_exit(
            job_id=self._job_id, table=self._table, error=err,
            on_exit=self._on_exit, exit_code=self._exit_code,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ── stall (no-progress) watchdog ───────────────────────────────────────────


class StallWatchdog:
    """Daemon thread enforcing a NO-PROGRESS deadline on one running job.

    Where :class:`Watchdog` is a fixed wall-clock budget (catches a job running
    too *long*), this catches a job making *no progress*: it arms a short
    deadline that every :meth:`beat` pushes out. A GPU node beats once per
    diffusion step (threaded in as ``status_callback``); the executor also beats
    once when the model finishes loading, so the cold-load phase gets its own
    fresh window. If no beat arrives within ``stall_timeout_s`` the node is hung
    (model resident, GPU at 0 %) — :func:`_fail_job_and_exit` marks it failed +
    writes the outbox event, then hard-exits so ``reclaim_expired_leases``
    re-queues it onto a healthy host. ``beat`` is tolerant of extra args so it
    can be wired straight in as a node ``status_callback``."""

    def __init__(
        self, *, job_id: str, stall_timeout_s: float = STALL_TIMEOUT_S,
        on_exit: Callable[[int], None] | None = None,
        poll_s: float = STALL_POLL_S, exit_code: int = 76,
        table: str = "workflow_node_jobs",
    ) -> None:
        if table not in _LEASE_TABLES:
            raise ValueError(f"stall-watchdog table must be in {sorted(_LEASE_TABLES)}, got {table!r}")
        self._job_id = job_id
        self._stall_timeout_s = float(stall_timeout_s)
        self._on_exit = on_exit or os._exit
        self._poll_s = float(poll_s)
        self._exit_code = int(exit_code)
        self._table = table
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # None ⇒ not yet armed: the watchdog stays inert until the first beat
        # (the executor's after-model-load beat), so a long cold load is never
        # policed by the no-progress deadline.
        self._deadline: float | None = None

    def beat(self, *args: Any, **kwargs: Any) -> None:
        """Record progress: arm (on the first call) / push the no-progress
        deadline out by one window. Thread-safe; ignores any args so it doubles
        as a node ``status_callback``."""
        with self._lock:
            self._deadline = time.monotonic() + self._stall_timeout_s

    def start(self) -> None:
        # Note: NO initial beat — the watchdog arms on the first beat, not at
        # start, so the model-load phase before that beat is never policed here.
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"stall-watchdog-{self._job_id[:8]}",
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                deadline = self._deadline
            if deadline is not None and time.monotonic() >= deadline:
                self._trip()
                return
            # Wake on stop OR at the next poll boundary, whichever first.
            self._stop.wait(self._poll_s)

    def _trip(self) -> None:
        err = (
            f"no progress for {int(self._stall_timeout_s)}s "
            f"(no GPU step beat) — stall watchdog hard-stopped the worker"
        )
        log.error("[stall-watchdog] %s %s", self._job_id, err)
        _fail_job_and_exit(
            job_id=self._job_id, table=self._table, error=err,
            on_exit=self._on_exit, exit_code=self._exit_code,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ── GPU health watchdog (HEALTH-driven, replaces the wall-clock cap for GPU) ─


class GpuHealthWatchdog:
    """Daemon thread that hard-stops a GPU job ONLY when it's truly wedged —
    health-driven, NOT wall-clock.

    Where :class:`Watchdog` kills a GPU job purely because its time budget
    elapsed (health-blind), this never kills for elapsed time. Every
    ``interval_s`` (default 300 s) it evaluates the window:

      * the MAX per-container GPU utilization seen since the last checkpoint
        (sampled every ``poll_s`` s and remembered), and
      * the change in this container's RAM since the last checkpoint.

    It TRIPS iff over the whole window the GPU stayed idle
    (``max_util <= idle_pct``) AND RAM was static (``|Δram| <= ram_delta_mb``)
    — i.e. no GPU work AND no memory movement ⇒ wedged. If the GPU was busy at
    any point, OR RAM moved more than the delta (staging / decode / model
    swap), the job is HEALTHY: the window resets and it keeps running, no matter
    how long. On trip it reuses :func:`_fail_job_and_exit` (the same
    outbox-atomic mark-failed + hard-exit contract as the other watchdogs) so
    ``reclaim_expired_leases`` re-queues the row onto a healthy host.

    Arming mirrors :class:`StallWatchdog`: the watchdog is INERT until the first
    :meth:`beat` (the executor's post-model-load beat), so a multi-minute cold
    model load is never policed. A node progress ``beat`` (its
    ``status_callback``) also resets the window as an extra liveness signal.
    ``beat`` tolerates any args so it doubles as a node ``status_callback``.

    GPU/RAM samplers are injected (``gpu_sampler`` / ``ram_sampler``) so tests
    feed fakes instead of shelling out to ``nvidia-smi``; production defaults to
    :mod:`queue_workflows.gpu_health` (per-container pmon ``sm%`` + cgroup RAM).
    """

    def __init__(
        self, *, job_id: str,
        interval_s: float = GPU_HEALTH_INTERVAL_S,
        idle_pct: int = GPU_IDLE_PCT,
        ram_delta_mb: int = GPU_HEALTH_RAM_DELTA_MB,
        poll_s: float = STALL_POLL_S,
        gpu_sampler: Callable[[], int] | None = None,
        ram_sampler: Callable[[], int | None] | None = None,
        on_exit: Callable[[int], None] | None = None,
        exit_code: int = 78,
        table: str = "workflow_node_jobs",
    ) -> None:
        if table not in _LEASE_TABLES:
            raise ValueError(f"gpu-health table must be in {sorted(_LEASE_TABLES)}, got {table!r}")
        self._job_id = job_id
        self._interval_s = float(interval_s)
        self._idle_pct = int(idle_pct)
        self._ram_delta_mb = int(ram_delta_mb)
        self._poll_s = float(poll_s)
        if gpu_sampler is None or ram_sampler is None:
            from queue_workflows import gpu_health
            gpu_sampler = gpu_sampler or gpu_health.gpu_util_pct
            ram_sampler = ram_sampler or gpu_health.container_ram_mb
        self._gpu_sampler = gpu_sampler
        self._ram_sampler = ram_sampler
        self._on_exit = on_exit or os._exit
        self._exit_code = int(exit_code)
        self._table = table
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Window state. ``_armed`` stays False until the first beat (post-load),
        # so the cold-load phase is never policed. ``_max_util`` accumulates the
        # peak GPU util seen this window; ``_ram_anchor`` is the RAM at the
        # window's start that the |Δ| is measured against.
        self._armed = False
        self._max_util = 0
        self._ram_anchor: int | None = None
        self._next_checkpoint = 0.0

    # ── window management ────────────────────────────────────────────────

    def _reset_window(self) -> None:
        """Open a fresh window: clear the peak-util accumulator, re-anchor RAM,
        and push the next checkpoint out by one interval. Caller holds the lock."""
        self._max_util = 0
        self._ram_anchor = self._sample_ram()
        self._next_checkpoint = time.monotonic() + self._interval_s

    def beat(self, *args: Any, **kwargs: Any) -> None:
        """Record progress: arm (on the first call) + reset the window. Wired in
        as the node ``status_callback`` (extra liveness) and pulsed once by the
        executor right after the model load (the arming beat). Thread-safe;
        ignores any args so it doubles as a ``status_callback``."""
        with self._lock:
            self._armed = True
            self._reset_window()

    def _sample_gpu(self) -> int:
        try:
            return int(self._gpu_sampler() or 0)
        except Exception:
            log.exception("[gpu-health] %s gpu sample failed; treating as 0", self._job_id)
            return 0

    def _sample_ram(self) -> int | None:
        try:
            v = self._ram_sampler()
            return None if v is None else int(v)
        except Exception:
            log.exception("[gpu-health] %s ram sample failed; treating as None", self._job_id)
            return None

    def start(self) -> None:
        # No initial beat — the watchdog arms on the first beat (post-load), not
        # at start, so the model-load phase is never policed here.
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"gpu-health-{self._job_id[:8]}",
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                armed = self._armed
                if armed:
                    util = self._sample_gpu()
                    if util > self._max_util:
                        self._max_util = util
                    due = time.monotonic() >= self._next_checkpoint
                    if due and self._evaluate_window_locked():
                        # tripped → _trip() already exited; bail just in case the
                        # injected on_exit didn't terminate (tests).
                        return
            self._stop.wait(self._poll_s)

    def _evaluate_window_locked(self) -> bool:
        """At a checkpoint: TRIP iff GPU idle AND RAM static over the window.
        Otherwise reset the window (healthy) and continue. Returns True iff it
        tripped. Caller holds the lock."""
        ram_now = self._sample_ram()
        gpu_idle = self._max_util <= self._idle_pct
        # RAM is "moved" (⇒ healthy) only when we have BOTH anchor and current
        # readings and the |Δ| exceeds the threshold. A missing reading is
        # treated as "no RAM movement detected" so a flaky RAM probe can't, on
        # its own, keep a wedged + GPU-idle worker alive forever — but note GPU
        # idle is still required to trip, so a busy GPU always survives.
        ram_moved = (
            self._ram_anchor is not None
            and ram_now is not None
            and abs(ram_now - self._ram_anchor) > self._ram_delta_mb
        )
        if gpu_idle and not ram_moved:
            self._trip(max_util=self._max_util, ram_anchor=self._ram_anchor, ram_now=ram_now)
            return True
        # Healthy this window — re-anchor and re-arm the next checkpoint.
        self._reset_window()
        return False

    def _trip(self, *, max_util: int, ram_anchor: int | None, ram_now: int | None) -> None:
        err = (
            f"no GPU activity (max sm% {max_util} <= {self._idle_pct}) and static "
            f"RAM (anchor={ram_anchor}MB now={ram_now}MB, |Δ| <= {self._ram_delta_mb}MB) "
            f"for {int(self._interval_s)}s — health watchdog hard-stopped a wedged worker"
        )
        log.error("[gpu-health] %s %s", self._job_id, err)
        _fail_job_and_exit(
            job_id=self._job_id, table=self._table, error=err,
            on_exit=self._on_exit, exit_code=self._exit_code,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ── job-status watcher (abandon a job taken from us) ───────────────────────


class JobStatusWatcher:
    """Daemon thread that HARD-EXITS the worker the instant its claimed job is
    no longer ``claimed_by`` this host — i.e. another actor re-queued it
    (orchestrator restart-resume / lease reclaim) or reassigned it to a peer.
    Polls ``workflow_node_jobs`` every ``poll_s``; on a miss it calls
    ``on_exit`` (default ``os._exit``). systemd then restarts the worker and it
    claims fresh work.

    This is what makes re-queuing a *running* job SAFE: the displaced worker
    kills itself instead of racing the new claimant to a double-run. ``os._exit``
    is the only way to abandon a node body wedged deep in a CUDA kernel.

    Scoped to ``claimed_by`` (not bare ``status``) on purpose: the worker's OWN
    terminal mark — ``mark_completed`` / ``mark_failed`` set ``status`` but leave
    ``claimed_by`` intact — must NOT trip it; only an external hand-off (which
    clears or changes ``claimed_by``) does."""

    def __init__(
        self, *, job_id: str, claimed_by: str,
        on_exit: Callable[[int], None] | None = None,
        poll_s: float = 2.0, exit_code: int = 77,
    ) -> None:
        self._job_id = job_id
        self._claimed_by = claimed_by
        self._on_exit = on_exit or os._exit
        self._poll_s = float(poll_s)
        self._exit_code = int(exit_code)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"job-status-{self._job_id[:8]}",
        )
        self._thread.start()

    def _loop(self) -> None:
        # Wait first, then check — gives the claim a moment to settle and wakes
        # instantly on stop().
        while not self._stop.wait(self._poll_s):
            try:
                row = node_queue.get_node_job(self._job_id)
            except Exception:
                log.exception("[job-status-watcher] %s poll failed; retrying",
                              self._job_id)
                continue
            if row is None or row.get("claimed_by") != self._claimed_by:
                log.warning(
                    "[job-status-watcher] %s no longer ours "
                    "(status=%s claimed_by=%s) — hard-exiting so a fresh worker "
                    "resumes it",
                    self._job_id,
                    None if row is None else row.get("status"),
                    None if row is None else row.get("claimed_by"),
                )
                self._on_exit(self._exit_code)
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ── worker capacity heartbeat ────────────────────────────────────────────────


class HeartbeatEmitter:
    """Daemon thread that keeps this worker's ``worker_heartbeats`` row fresh so
    Rails' queue gauge sees the fleet's real capacity.

    Claim-worker shape:

      * ``concurrency`` is **1** — a claim worker is one poll-claim loop.
      * the **GPU** heartbeat reports ``current_model`` read from the worker's
        :class:`ModelCache` on every tick — the gauge's busy signal. The
        **CPU** heartbeat reports ``current_model=None``.
      * ingest workers (fetch/load + any host-defined queue) also heartbeat —
        migration 0008 dropped the cpu/gpu-only CHECK — reporting
        ``current_model=None`` so a host's queue gauge shows their liveness.
      * honours ``AI_LEADS_DISABLE_WORKER_HEARTBEAT`` (tests).

    Upserts once on :meth:`start` (so the row exists immediately) then refreshes
    every ``interval_s`` until :meth:`stop`."""

    def __init__(
        self, *, queue: str, host_label: str, model_cache: Any = None,
        interval_s: float = HEARTBEAT_INTERVAL_S,
    ) -> None:
        self._queue = queue
        self._host_label = host_label
        self._model_cache = model_cache
        self._interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def _enabled(self) -> bool:
        """Heartbeat runs for every queue family — cpu/gpu node workers AND the
        ingest workers (migration 0008 dropped the cpu/gpu-only CHECK) — unless
        the test opt-out env is set."""
        return not bool(os.environ.get("AI_LEADS_DISABLE_WORKER_HEARTBEAT"))

    def _current_model(self) -> str | None:
        """The GPU busy signal: the warm-model slot, read live each tick. NULL
        for CPU and whenever the GPU slot is empty (cold start / post
        idle-unload)."""
        if self._queue != "gpu" or self._model_cache is None:
            return None
        return getattr(self._model_cache, "current_model", None)

    def emit_once(self) -> None:
        """Upsert this worker's row once (concurrency=1; GPU carries its live
        ``current_model``). Failures are swallowed + logged — a transient DB
        blip must never crash a worker mid-job."""
        try:
            from queue_workflows import model_registry
            node_queue.upsert_worker_heartbeat(
                host_label=self._host_label,
                queue=self._queue,
                concurrency=1,
                current_model=self._current_model(),
                known_models=model_registry.known_ids(),
            )
        except Exception:
            log.exception("[claim-worker:%s] heartbeat upsert failed", self._queue)

    def _loop(self) -> None:
        # First tick already fired in start(); refresh until stopped.
        while not self._stop.wait(self._interval_s):
            self.emit_once()

    def start(self) -> None:
        if not self._enabled:
            return
        self.emit_once()  # row exists immediately, before the first sleep
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"worker-heartbeat-{self._queue}",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ── the claim loop ─────────────────────────────────────────────────────────


#: Queues whose claim worker draws DAG node-jobs from ``workflow_node_jobs``.
_NODE_QUEUES = frozenset({"cpu", "gpu"})
#: Default ingest queues (ai_leads byte-compat). The LIVE ingest set is
#: ``config.ingest_queues`` (host-configurable, G1); any queue NOT in
#: ``_NODE_QUEUES`` is treated as ingest-family (draws from ``ingest_jobs``).
_DEFAULT_INGEST_QUEUES = frozenset({"fetch", "load"})


class ClaimWorker:
    """One Postgres-as-queue worker. ``queue`` ∈ {cpu, gpu, fetch, load}.

    A GPU worker holds a single :class:`ModelCache` (concurrency-1 structural)
    and runs DAG node-jobs. A fetch/load worker runs standalone ingest jobs from
    ``ingest_jobs`` — no model cache, no run-cancel watcher (there's no parent
    run), no ``$from`` resolution. ``run_once`` claims + executes one ready job;
    ``run_forever`` is the LISTEN/poll loop. The two queue families share the
    lease / reclaim / watchdog machinery."""

    def __init__(
        self, *, queue: str, host: str | None = None,
        host_priority: int | None = None, lease_s: int = LEASE_S,
        model_cache: Any = None,
    ) -> None:
        valid = _NODE_QUEUES | get_config().ingest_queues
        if queue not in valid:
            raise ValueError(f"queue must be in {sorted(valid)}, got {queue!r}")
        self.queue = queue
        self.host = host or _host_label()
        self.host_priority = (
            host_priority if host_priority is not None else _host_priority()
        )
        self.lease_s = int(lease_s)
        # GPU worker holds the process-wide warm-model cache; a unit test may
        # inject a fake. CPU + ingest workers have no cache.
        if model_cache is None and queue == "gpu":
            from queue_workflows.gpu_model_cache import gpu_model_cache
            model_cache = gpu_model_cache()
        self.model_cache = model_cache
        self._stop = threading.Event()
        # Worker capacity heartbeat — a no-op for fetch/load.
        self.heartbeat = HeartbeatEmitter(
            queue=self.queue, host_label=self.host,
            model_cache=self.model_cache,
        )
        # This host's hw-metrics sampler — started in ``run_forever`` ONLY when
        # ``queue == 'gpu'`` (one gpu container per host ⇒ one sampler per host).
        self._hw_sampler: Any = None

    @property
    def _is_ingest(self) -> bool:
        # __init__ validated queue ∈ _NODE_QUEUES ∪ config.ingest_queues, so any
        # non-node queue is a (host-configured) ingest queue (G1).
        return self.queue not in _NODE_QUEUES

    # ── claim ────────────────────────────────────────────────────────────

    def _claim(self) -> dict | None:
        if self._is_ingest:
            return node_queue.claim_next_ingest_job(
                self.queue, host=self.host, lease_s=self.lease_s,
            )
        if self.queue == "gpu":
            from queue_workflows import model_registry
            current_model = getattr(self.model_cache, "current_model", None)
            return node_queue.claim_next_gpu_job(
                0, current_model,
                host=self.host, lease_s=self.lease_s,
                host_priority=self.host_priority,
                known_models=model_registry.known_ids(),
            )
        return node_queue.claim_next_cpu_job(
            0, host=self.host, lease_s=self.lease_s,
            host_priority=self.host_priority,
        )

    # ── execute one ───────────────────────────────────────────────────────

    def run_once(self) -> bool:
        """Claim the next ready job and run it to a terminal state. Returns
        ``True`` if a job was claimed + executed, ``False`` when the queue had
        nothing claimable (caller should block on NOTIFY)."""
        job = self._claim()
        if job is None:
            return False
        if self._is_ingest:
            return self._run_ingest(job)
        return self._run_node(job)

    @staticmethod
    def _node_reports_progress(module_name: str) -> bool:
        """True iff the node's ``run(...)`` declares a ``status_callback`` param
        — its opt-in to no-progress policing. A node that doesn't report
        progress can't be told apart from a hung one, so the StallWatchdog is
        NOT armed for it (it's left to the wall-clock :class:`Watchdog`)."""
        try:
            mod = get_config().resolve_node_module(module_name)
            return "status_callback" in inspect.signature(mod.run).parameters
        except Exception:
            return False

    def _run_node(self, job: dict) -> bool:
        """Execute a claimed DAG node-job (cpu/gpu) under a cancel-watcher +
        lease-renewer + a guard set that depends on the queue:

          * CPU jobs keep the wall-clock :class:`Watchdog` (their budgets are
            fine — the user's concern is GPU renders camping forever).
          * GPU jobs get the HEALTH-driven :class:`GpuHealthWatchdog` instead of
            a wall-clock cap (video + non-video): it only kills a worker that's
            truly wedged (no GPU work AND static RAM over a 5-min window), never
            just because time passed.
          * non-video GPU jobs that report per-step progress ALSO keep the tight
            :class:`StallWatchdog` (defense in depth — it catches a fast 0%-GPU
            hang in ~2 min, well before the health watchdog's first checkpoint).
        """
        job_id = job["id"]
        log.info(
            "[claim-worker:%s] claimed %s (node=%s model=%s)",
            self.queue, job_id, job.get("node_id"), job.get("required_model"),
        )

        # Input/await nodes carry a sentinel module name (``__input__<widget>``)
        # and execute NO node module — they park the run for user input. Mirror
        # the completed/failed OUTBOX pattern in node_executor: in one txn, mark
        # the job awaiting_input + enqueue an ``awaiting_input`` dispatch event.
        # NodePool._drain_dispatch_events (which holds the workflow loader) then
        # calls dispatcher.on_node_awaiting_input to build the input_spec.
        # Calling the dispatcher directly here would bypass the durable outbox
        # AND require the loader in every claim-worker process. Without this
        # guard the worker hands the sentinel to execute_node →
        # ``import workflows.nodes.__input__*`` → ModuleNotFoundError.
        if (job.get("node_module") or "").startswith("__input__"):
            with connection() as conn, conn.cursor() as cur:
                node_queue.mark_awaiting_input_in_txn(cur, job_id)
                node_queue.enqueue_dispatch_event_in_txn(
                    cur, job["run_id"], job["node_id"], "awaiting_input",
                )
            return True

        # A cancel-watcher feeds cooperative node bodies the run-cancel signal.
        from queue_workflows.cancel_watcher import _start_run_cancel_watcher
        cancel_event = threading.Event()
        cancel_thread = _start_run_cancel_watcher(
            job["run_id"], cancel_event, interval_s=5.0,
        )

        renewer = LeaseRenewer(
            job_id=job_id, claimed_by=self.host, lease_s=self.lease_s,
        )
        is_gpu = self.queue == "gpu"
        # Wall-clock budget: CPU jobs only. GPU jobs are policed by health, NOT
        # elapsed time — a long-but-healthy render (busy GPU / moving RAM) must
        # never be killed for running too long, so GPU gets NO fixed cap here.
        watchdog: Watchdog | None = None if is_gpu else Watchdog(
            job_id=job_id, budget_s=budget_for(job),
        )
        # Abandon-watcher: hard-exit if this job is re-queued / reassigned out
        # from under us (restart-resume, lease reclaim) so it's never double-run.
        status_watcher = JobStatusWatcher(job_id=job_id, claimed_by=self.host)
        renewer.start()
        if watchdog is not None:
            watchdog.start()
        status_watcher.start()

        # GPU guards. Both arm on the executor's post-load beat (threaded as the
        # node ``status_callback``) so a multi-minute cold model load is never
        # policed. A node's per-step beats then keep both windows open.
        #
        #  * GpuHealthWatchdog — the universal GPU guard (video + non-video),
        #    replacing the removed wall-clock cap. Kills only a wedged worker
        #    (no GPU work AND static RAM over a 5-min window); a busy GPU or a
        #    > 5 GB RAM move keeps it alive forever. Video is the whole point —
        #    a hung wan/ltx render now gets caught on health instead of camping
        #    a (formerly 1800 s) budget that punished healthy long renders.
        #  * StallWatchdog — kept for NON-VIDEO GPU as defense in depth: its
        #    tight 120 s no-progress window catches a fast 0%-GPU hang minutes
        #    before the health watchdog's first 300 s checkpoint. EXCLUDED for
        #    video: a video backend steps slowly (minutes/beat) so 120 s would
        #    false-trip a healthy render — the health watchdog is its only guard.
        is_video = (job.get("required_model") or "") in get_config().video_model_ids
        reports_progress = is_gpu and self._node_reports_progress(job["node_module"])
        stall: StallWatchdog | None = None
        gpu_health: GpuHealthWatchdog | None = None
        beats: list[Callable[..., None]] = []
        if is_gpu:
            gpu_health = GpuHealthWatchdog(job_id=job_id)
            gpu_health.start()
            beats.append(gpu_health.beat)
        if reports_progress and not is_video:
            stall = StallWatchdog(job_id=job_id)
            stall.start()
            beats.append(stall.beat)

        # Thread ONE status_callback to the node that fans a beat out to every
        # armed GPU guard, so each reported step (and the executor's post-load
        # beat) arms + resets all of them at once. ``None`` when no guard wants
        # beats (CPU, or a GPU node that doesn't report progress) — but note the
        # GpuHealthWatchdog still needs the post-load arming beat, so a GPU job
        # always gets a callback whenever a health watchdog is armed.
        status_callback: Callable[..., None] | None = (
            (lambda *a, **k: [b() for b in beats]) if beats else None
        )
        # GPU busy-bracket: hold ``ModelCache._active`` > 0 for the job's
        # lifetime so the cache's idle reaper can't unload the warm model
        # mid-inference. ``mark_idle`` is in the finally so it ALWAYS releases.
        busy = (
            self.queue == "gpu"
            and self.model_cache is not None
            and hasattr(self.model_cache, "mark_busy")
            and hasattr(self.model_cache, "mark_idle")
        )
        if busy:
            self.model_cache.mark_busy()
        try:
            node_executor.execute_node(
                job, model_cache=self.model_cache, cancel_event=cancel_event,
                status_callback=status_callback,
            )
        finally:
            if busy:
                self.model_cache.mark_idle()
            status_watcher.stop()
            if stall is not None:
                stall.stop()
            if gpu_health is not None:
                gpu_health.stop()
            if watchdog is not None:
                watchdog.stop()
            renewer.stop()
            cancel_event.set()
            cancel_thread.join(timeout=2.0)
        return True

    def _run_ingest(self, job: dict) -> bool:
        """Execute a claimed ingest job (fetch/load) under a lease-renewer +
        watchdog. No cancel-watcher (no parent run) and no model cache."""
        from queue_workflows import ingest_executor
        job_id = job["id"]
        log.info(
            "[claim-worker:%s] claimed ingest %s (task=%s reason=%s)",
            self.queue, job_id, job.get("task_name"), job.get("reason"),
        )
        renewer = LeaseRenewer(
            job_id=job_id, claimed_by=self.host, lease_s=self.lease_s,
            table="ingest_jobs",
        )
        watchdog = Watchdog(
            job_id=job_id, budget_s=budget_for(job), table="ingest_jobs",
        )
        renewer.start()
        watchdog.start()
        try:
            ingest_executor.execute_ingest_job(job)
        finally:
            watchdog.stop()
            renewer.stop()
        return True

    # ── the loop ───────────────────────────────────────────────────────────

    @property
    def _wake_channel(self) -> str:
        """The LISTEN channel this worker wakes on. Ingest workers (fetch/load)
        wake on ``ingest_job_ready`` (migration 0007); node workers (cpu/gpu)
        wake on ``node_job_ready`` (migration 0006). Both NOTIFYs carry the
        queue name as payload."""
        return "ingest_job_ready" if self._is_ingest else "node_job_ready"

    def await_schema(self) -> None:
        """Block until the migrations this queue's claim loop depends on are
        applied. The orchestrator owns the migration run (``db.bootstrap``,
        which takes no advisory lock); a claim worker WAITS for the schema
        before it starts polling."""
        from queue_workflows import db
        min_version = (
            _NODE_REQUIRED_VERSION if self.queue in _NODE_QUEUES
            else _INGEST_REQUIRED_VERSION
        )
        log.info(
            "[claim-worker:%s] waiting for queue_schema_version >= %d",
            self.queue, min_version,
        )
        db.wait_for_schema(min_version)

    def run_forever(self) -> None:
        """Block on ``LISTEN <wake_channel>`` and drain the queue greedily on
        each wake (and on the 1 s safety poll for a dropped NOTIFY). Uses a
        dedicated autocommit connection for the LISTEN so it never holds a
        pooled connection across the blocking wait."""
        import psycopg
        # Gate on schema readiness FIRST so the loop never polls a table the
        # orchestrator's bootstrap hasn't created yet.
        self.await_schema()
        log.info(
            "[claim-worker:%s] starting (host=%s priority=%d lease=%ds channel=%s)",
            self.queue, self.host, self.host_priority, self.lease_s,
            self._wake_channel,
        )
        # Advertise capacity (no-op for fetch/load) + keep last_seen fresh.
        self.heartbeat.start()
        # Bring up this HOST's hw-metrics sampler — but ONLY from the gpu
        # worker. There is exactly one gpu-worker container per host, so gating
        # the sampler start to the gpu queue yields exactly one sampler per box.
        # The flock inside the starter is a cheap secondary guard.
        if self.queue == "gpu":
            from queue_workflows import hw_metrics
            self._hw_sampler = hw_metrics.start_hw_metrics_sampler_flocked()
        try:
            with psycopg.connect(db_url(), autocommit=True) as listen_conn:
                listen_conn.execute(f"LISTEN {self._wake_channel}")
                while not self._stop.is_set():
                    # Drain greedily until the queue is empty.
                    try:
                        while not self._stop.is_set() and self.run_once():
                            pass
                    except Exception:
                        log.exception("[claim-worker:%s] run_once failed", self.queue)
                    if self._stop.is_set():
                        break
                    # Idle: block on the wake NOTIFY with a safety-poll timeout.
                    for _ in listen_conn.notifies(
                        timeout=NOTIFY_POLL_TIMEOUT_S, stop_after=1,
                    ):
                        break
        finally:
            self.heartbeat.stop()
            # Stop the sampler iff THIS process won the flock.
            if self._hw_sampler is not None:
                self._hw_sampler.stop()
                self._hw_sampler.join(timeout=2.0)
                self._hw_sampler = None

    def stop(self) -> None:
        self._stop.set()


# ── entrypoint ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse
    import signal

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="queue-claim-worker")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--lease-seconds", type=int, default=LEASE_S)
    args = parser.parse_args(argv)

    # Custom ingest queue names are accepted when the host configured them
    # (queue_workflows.configure(ingest_queues=...)) before calling main(); the
    # bare console script keeps the cpu/gpu/fetch/load default set.
    valid = _NODE_QUEUES | get_config().ingest_queues
    if args.queue not in valid:
        parser.error(
            f"--queue must be in {sorted(valid)} "
            "(call queue_workflows.configure(ingest_queues=...) for custom names)"
        )

    if args.queue == "gpu":
        # Register the model registry up front so the first claim's
        # require_model resolves. The registrar is the engine's configured hook
        # (config.builtin_model_registrar) — a no-op unless a host wired one.
        get_config().builtin_model_registrar()

    worker = ClaimWorker(queue=args.queue, lease_s=args.lease_seconds)

    def _handler(signum, _frame):
        log.info("[claim-worker:%s] signal %s; stopping", args.queue, signum)
        worker.stop()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    worker.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
