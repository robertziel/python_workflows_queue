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
  * :class:`Watchdog` — a wall-clock budget. On trip it RE-QUEUES the node for a
    retry on a fresh worker (DAG jobs; run stays alive) under a per-job cap, then
    fails the run once the cap is reached — and HARD-exits the process either way.
    Applied to CPU + ingest jobs only (ingest trips always mark failed — no run).
  * :class:`GpuHealthWatchdog` — the GPU guard, HEALTH-driven not wall-clock:
    it replaces the fixed budget for GPU jobs and kills ONLY a truly-wedged
    worker (no per-container GPU work AND static container RAM over a 5-min
    window). A busy GPU or a > 5 GB RAM move keeps the job alive no matter how
    long it runs — there is NO fixed time cap for GPU renders. It arms AT job
    start with a generous load-grace first window (so a load-phase hang, and a
    GPU node that never beats, are both bounded), then falls to the 5-min
    cadence on the first beat.
  * :class:`StallWatchdog` — a tight no-progress deadline kept for non-video
    GPU nodes (defense in depth: catches a fast 0%-GPU hang in ~2 min). Its trip
    is GATED on the physical signal: a no-beat timeout only SUSPECTS a stall, then
    it confirms GPU-idle AND RAM-static before killing, so a loading / preparing /
    slow-but-working node is never a false positive.

GPU jobs are policed by HEALTH, never by elapsed time — a long-but-healthy
render is never killed; a wedged one is caught by the health watchdog (and, for
non-video, the stall watchdog). A trip RE-QUEUES the node for a retry on a fresh
worker (the run stays ``running``) up to ``AI_LEADS_WATCHDOG_MAX_RETRIES`` times,
then fails the run — so a single transient wedge no longer kills a whole workflow.
On a hard-exit the trip path also clears this worker's ``current_model`` busy-ghost
so a killed worker doesn't keep inflating the GPU-busy gauge.
"""

from __future__ import annotations

import inspect
import logging
import os
import socket
import threading
import time
from typing import Any, Callable

from queue_workflows import node_executor, node_queue, worker_control
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
# tight. CRITICAL (Part A): no beat for this long once armed is only a SUSPECTED
# stall, not a verdict — the watchdog then CONFIRMS against the physical GPU/RAM
# signal (GPU idle AND RAM static) and trips ONLY if the worker is genuinely
# doing nothing; a loading / preparing / slow-but-working node (busy GPU or moving
# RAM) is re-armed, never killed. A confirmed wedge re-queues the node + hard-exits
# so the lease reclaim retries it on a healthy host (run stays alive, under a cap).
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
# First-window length when the watchdog arms AT job start (before any beat).
# Generous (20 min) so a healthy-but-slow cold model load — which moves RAM as
# weights load and so can't trip the "GPU idle AND RAM static" rule anyway — is
# never policed tightly; it only bounds a load that is GENUINELY hung (idle GPU
# AND static RAM) past this grace. The first beat (executor post-load / per
# step) collapses the window to the normal GPU_HEALTH_INTERVAL_S cadence.
GPU_HEALTH_LOAD_GRACE_S = _env_float("AI_LEADS_GPU_HEALTH_LOAD_GRACE_S", 1200.0)  # 20 min

# ── watchdog re-queue-with-cap policy ───────────────────────────────────────
# A watchdog trip RE-QUEUES the node for a retry on a fresh worker (flip
# running→queued, clear the lease, bump priority to the front, increment the
# per-job ``watchdog_retries`` counter — see node_queue.requeue_job_for_retry)
# rather than failing the whole run. A node that wedges EVERY time would loop
# forever, so the retry is CAPPED: once a job's ``watchdog_retries`` reaches this
# many, the trip falls back to the old mark-failed path (→ run fails). The cap is
# read at trip-time (not captured at import) so an ops env-override takes effect
# without a process restart, and so tests can flip it per-case.
WATCHDOG_MAX_RETRIES = _env_int("AI_LEADS_WATCHDOG_MAX_RETRIES", 3)


def _watchdog_max_retries() -> int:
    """The watchdog re-queue cap, read live from ``AI_LEADS_WATCHDOG_MAX_RETRIES``
    (default 3). Read at trip-time so an env-override needs no restart."""
    return _env_int("AI_LEADS_WATCHDOG_MAX_RETRIES", WATCHDOG_MAX_RETRIES)


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


def _clear_busy_ghost(host_label: str | None, queue: str | None) -> None:
    """Best-effort: clear THIS worker's ``current_model`` busy-ghost before a hard
    exit (Part C). ``os._exit`` skips ``_run_node``'s ``finally`` — so
    ``ModelCache.mark_idle`` and the heartbeat refresh never run, and the worker's
    last ``worker_heartbeats`` row keeps advertising a ``current_model`` even
    though the process is dead, inflating Rails' "N/M GPU busy" gauge (the user's
    "3/2 GPU busy" after a kill). Null it out + age ``last_seen`` past the gauge's
    30 s window so the dead worker drops out at once. Swallows everything — a hung
    DB write must NEVER block the hard-exit. No-op when the worker identity wasn't
    threaded through (e.g. a unit test constructing a watchdog with just a
    ``job_id``), or for a CPU/ingest worker that never advertised a model."""
    if not host_label or not queue:
        return
    try:
        node_queue.clear_worker_current_model(host_label, queue)
    except Exception:
        log.exception(
            "[watchdog] could not clear current_model busy-ghost for %s/%s "
            "before exit (gauge will age it out within ~30s)", host_label, queue,
        )


def _trip_event_type(label: str) -> str:
    """Map a watchdog's label tag to its node-event type (migration 0011)."""
    lbl = (label or "").lower()
    if "stall" in lbl:
        return "stall_trip"
    if "gpu" in lbl:
        return "gpu_health_trip"
    return "budget_trip"


def _emit_node_event(
    job_id: str,
    event_type: str,
    *,
    host_label: str | None = None,
    queue: str | None = None,
    error: str | None = None,
    detail: dict | None = None,
    row: dict | None = None,
) -> None:
    """Best-effort node event from a claim / watchdog site that holds a job_id
    (and usually the row already). Fetches the row for run_id / node_id /
    attempt / model when not supplied; NEVER raises (forensic only — the
    underlying node_queue.record_node_event swallows too). Skips ingest jobs and
    vanished rows (no DAG run_id ⇒ nothing to attach the event to)."""
    try:
        r = row if row is not None else node_queue.get_node_job(job_id)
        if not r or not r.get("run_id"):
            return
        node_queue.record_node_event(
            run_id=r["run_id"], node_id=r["node_id"], job_id=job_id,
            attempt=int(r.get("watchdog_retries") or 0), event_type=event_type,
            host_label=host_label, queue=queue, model=r.get("required_model"),
            error=error, detail=detail,
        )
    except Exception:
        log.exception(
            "[node-event] emit %s for %s failed (ignored)", event_type, job_id,
        )


def _fail_job_and_exit(
    *, job_id: str, table: str, error: str,
    on_exit: Callable[[int], None], exit_code: int,
    host_label: str | None = None, queue: str | None = None,
) -> None:
    """Mark a doomed job ``failed`` (+ the dispatch-event outbox row for DAG
    node-jobs) then call ``on_exit``. Shared by :class:`Watchdog` (budget) and
    :class:`StallWatchdog` (no-progress) so the outbox-atomicity contract — the
    terminal mark and the ``failed`` event in ONE txn — is written in exactly
    one place. A mark failure is swallowed + logged: the hard-exit must still
    happen so the lease can expire and a reclaim re-queue the row.

    Before exiting it also clears this worker's ``current_model`` busy-ghost
    (:func:`_clear_busy_ghost`) so the dead worker doesn't keep inflating the
    GPU-busy gauge (Part C)."""
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
            if row is not None:
                _emit_node_event(
                    job_id, "failed", host_label=host_label, queue=queue,
                    error=error, row=row,
                )
    except Exception:
        log.exception("[watchdog] %s could not mark failed before exit", job_id)
    _clear_busy_ghost(host_label, queue)
    on_exit(exit_code)


def _requeue_job_and_exit(
    *, job_id: str, table: str, error: str,
    on_exit: Callable[[int], None], exit_code: int,
    host_label: str | None = None, queue: str | None = None,
) -> None:
    """Re-queue a watchdog-tripped DAG node-job for a RETRY on a fresh worker,
    then call ``on_exit`` — the re-queue twin of :func:`_fail_job_and_exit`.

    Instead of failing the run, flip the row ``running`` → ``queued`` (clearing
    the lease, bumping priority to the front, incrementing ``watchdog_retries``)
    via :func:`node_queue.requeue_job_for_retry`. Writes NO dispatch event — the
    run stays ``running``; only this node re-runs. The ``node_job_ready`` NOTIFY
    fired by the status flip wakes an idle worker to re-claim it.

    Same exit/atomicity discipline as :func:`_fail_job_and_exit`: a re-queue
    failure is swallowed + logged, but the hard-exit STILL happens — leaving the
    row ``running`` with a lease that the orchestrator's ``reclaim_expired_leases``
    will eventually re-queue anyway, so the node is never silently stranded.

    SAFETY (no double-run): the re-queue sets ``status='queued'`` and clears
    ``claimed_by``, then this process hard-exits. The fresh re-claim is the same
    CAS-guarded ``queued → running`` UPDATE as any claim; and any OTHER worker
    that still thought it held this row self-exits the instant its
    :class:`JobStatusWatcher` sees ``claimed_by`` no longer equals it. So at most
    one worker ever runs the re-queued node, and node bodies are idempotent on
    their out_dir.

    Only meaningful for ``workflow_node_jobs`` — ``requeue_job_for_retry`` targets
    that table. (Ingest jobs never reach here: they take the fail path, which is
    correct — an ingest job has no DAG run to keep alive, and ``ingest_jobs`` has
    its own ``reclaim_expired_ingest_leases`` re-queue.)"""
    try:
        _requeued = node_queue.requeue_job_for_retry(job_id)
        if _requeued is not None:
            _emit_node_event(
                job_id, "requeued", host_label=host_label, queue=queue,
                error=error, row=_requeued,
                detail={"retry": int(_requeued.get("watchdog_retries") or 0),
                        "cap": _watchdog_max_retries()},
            )
    except Exception:
        log.exception(
            "[watchdog] %s could not re-queue before exit (lease reclaim will "
            "recover it)", job_id,
        )
    _clear_busy_ghost(host_label, queue)
    on_exit(exit_code)


def _watchdog_trip(
    *, job_id: str, table: str, error: str, label: str,
    on_exit: Callable[[int], None], exit_code: int,
    host_label: str | None = None, queue: str | None = None,
) -> None:
    """Decide a watchdog trip's outcome: RE-QUEUE the node for a retry (under the
    cap) or FAIL the run (cap reached), then hard-exit. Shared by all three
    watchdogs so the policy lives in exactly one place.

    ``host_label`` / ``queue`` identify the tripping worker so both outcome paths
    can clear its ``current_model`` busy-ghost before the hard-exit (Part C).

    Policy:

      * DAG node-job (``workflow_node_jobs``): read the row's current
        ``watchdog_retries`` (migration 0010). If it's ``< AI_LEADS_WATCHDOG_MAX_
        RETRIES`` (default 3) the trip is treated as a TRANSIENT wedge —
        :func:`_requeue_job_and_exit` re-queues the node + retries it on a fresh
        worker, and the RUN stays alive. Once the counter has reached the cap the
        node is wedging persistently, so we fall back to :func:`_fail_job_and_exit`
        (mark failed + ``failed`` dispatch event → the run fails) rather than loop
        forever.
      * Ingest job (``ingest_jobs``): always :func:`_fail_job_and_exit` — there's
        no DAG run to keep alive and no ``watchdog_retries`` column; the row is
        marked failed and the ingest lease-reclaim handles re-queue separately.

    Logs the chosen path + the attempt count clearly so an operator can see, per
    trip, whether the node was retried (and which attempt) or finally failed.
    ``label`` is the tripping watchdog's tag (e.g. ``stall-watchdog``) for the log
    line."""
    if table == "ingest_jobs":
        log.error(
            "[%s] %s tripped on an ingest job → marking failed (ingest jobs "
            "have no run to keep alive; lease-reclaim re-queues separately)",
            label, job_id,
        )
        _fail_job_and_exit(
            job_id=job_id, table=table, error=error,
            on_exit=on_exit, exit_code=exit_code,
            host_label=host_label, queue=queue,
        )
        return

    cap = _watchdog_max_retries()
    # Read the current re-queue count to decide retry vs fail. A read failure
    # (or a vanished row) is treated as 0 prior retries so we PREFER the
    # non-destructive re-queue — never fail a run just because the count read
    # blipped.
    try:
        row = node_queue.get_node_job(job_id)
        retries = int((row or {}).get("watchdog_retries") or 0)
    except Exception:
        log.exception(
            "[%s] %s could not read watchdog_retries; assuming 0 (re-queue)",
            label, job_id,
        )
        row = None
        retries = 0

    # Forensic: record the trip itself (stall / gpu-health / budget) regardless
    # of the requeue-vs-fail outcome below — the matching requeued / failed
    # event is emitted by the chosen exit path.
    _emit_node_event(
        job_id, _trip_event_type(label), host_label=host_label, queue=queue,
        error=error, row=row,
    )

    if retries < cap:
        attempt = retries + 1
        log.warning(
            "[%s] %s tripped (%s) — RE-QUEUEING for retry %d/%d on a fresh "
            "worker; the run stays alive",
            label, job_id, error, attempt, cap,
        )
        _requeue_job_and_exit(
            job_id=job_id, table=table, error=error,
            on_exit=on_exit, exit_code=exit_code,
            host_label=host_label, queue=queue,
        )
    else:
        log.error(
            "[%s] %s tripped (%s) — watchdog retry cap reached (%d/%d); "
            "FAILING the run",
            label, job_id, error, retries, cap,
        )
        _fail_job_and_exit(
            job_id=job_id, table=table, error=error,
            on_exit=on_exit, exit_code=exit_code,
            host_label=host_label, queue=queue,
        )


class Watchdog:
    """Daemon thread enforcing a wall-clock budget on a single running job. On
    trip: route through :func:`_watchdog_trip` — for a DAG node-job, RE-QUEUE it
    for a retry on a fresh worker under the cap (``AI_LEADS_WATCHDOG_MAX_RETRIES``,
    default 3), then mark failed once the cap is reached; for an ingest job, mark
    failed — then call ``on_exit`` (default: hard ``os._exit``). A hard exit kills
    exactly the over-budget job.

    Applied to CPU + ingest jobs only — GPU jobs are policed by health
    (:class:`GpuHealthWatchdog`), not by a wall-clock cap, so a long-but-healthy
    render is never killed for elapsed time."""

    def __init__(
        self, *, job_id: str, budget_s: float,
        on_exit: Callable[[int], None] | None = None,
        poll_s: float = 1.0, exit_code: int = 75,
        table: str = "workflow_node_jobs",
        host_label: str | None = None, queue: str | None = None,
    ) -> None:
        if table not in _LEASE_TABLES:
            raise ValueError(f"watchdog table must be in {sorted(_LEASE_TABLES)}, got {table!r}")
        self._job_id = job_id
        self._budget_s = float(budget_s)
        self._on_exit = on_exit or os._exit
        self._poll_s = float(poll_s)
        self._exit_code = int(exit_code)
        self._table = table
        self._host_label = host_label
        self._queue = queue
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
        # Re-queue-with-cap (DAG node-jobs) for uniformity with the no-progress /
        # health watchdogs: a CPU over-budget MAY be a genuine runaway, but the
        # cap bounds it either way — at most AI_LEADS_WATCHDOG_MAX_RETRIES retries
        # then the run fails. Ingest over-budget still marks failed (no run to
        # keep alive) — _watchdog_trip branches on the table.
        _watchdog_trip(
            job_id=self._job_id, table=self._table, error=err, label="watchdog",
            on_exit=self._on_exit, exit_code=self._exit_code,
            host_label=self._host_label, queue=self._queue,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ── stall (no-progress) watchdog ───────────────────────────────────────────


#: How many GPU/RAM samples the StallWatchdog takes to CONFIRM a worker is truly
#: wedged before it trips on a no-beat timeout (Part A — gate the trip on the
#: physical signal). A few samples spaced ``STALL_CONFIRM_POLL_S`` apart so a
#: single noisy reading can't flip the verdict either way.
STALL_CONFIRM_SAMPLES = _env_int("AI_LEADS_STALL_CONFIRM_SAMPLES", 3)
STALL_CONFIRM_POLL_S = _env_float("AI_LEADS_STALL_CONFIRM_POLL_S", 1.0)


class StallWatchdog:
    """Daemon thread enforcing a NO-PROGRESS deadline on one running job, GATED on
    the physical GPU/RAM signal so a LOADING / PREPARING / slow-but-working node
    is never killed.

    Where :class:`Watchdog` is a fixed wall-clock budget (catches a job running
    too *long*), this catches a job making *no progress*: it arms a short
    deadline that every :meth:`beat` pushes out. A GPU node beats once per
    diffusion step (threaded in as ``status_callback``); the executor also beats
    once when the model finishes loading, so the cold-load phase gets its own
    fresh window.

    **The no-beat timeout is necessary but NOT sufficient to trip** (the fix for
    the user's false-positive: "should not kill if the GPU model is being loaded
    or preparing to start; only if it does nothing"). When ``stall_timeout_s``
    elapses with no beat, the watchdog does NOT trip outright — beat-absence alone
    can't tell a wedged node from one in a legitimately slow step, mid-load, or
    preparing. Instead it CONFIRMS with the SAME ``gpu_health`` samplers + the SAME
    "GPU idle AND RAM static" predicate :class:`GpuHealthWatchdog` uses, over a
    short confirmation window (``confirm_samples`` reads, ``confirm_poll_s``
    apart):

      * GPU busy at ANY sample (``max sm% > idle_pct``) ⇒ the node IS doing GPU
        work (a slow step) ⇒ NOT wedged — reset the deadline, log, don't trip.
      * RAM moving (``|Δ| > ram_delta_mb`` across the window) ⇒ the node is
        loading weights / staging / preparing ⇒ NOT wedged — reset, log, don't
        trip. (A multi-GB model load moves RAM far beyond the 5 GB delta, so a
        cold/lazy load can NEVER false-trip.)
      * ONLY when GPU stayed idle AND RAM stayed static across the whole
        confirmation window — genuinely "doing nothing" — does it trip.

    On a confirmed trip :func:`_watchdog_trip` RE-QUEUES the node for a retry on a
    fresh worker (run stays alive) under the cap (``AI_LEADS_WATCHDOG_MAX_RETRIES``,
    default 3), then marks it failed once the cap is reached, hard-exiting in
    either case. ``beat`` is tolerant of extra args so it can be wired straight in
    as a node ``status_callback``.

    GPU/RAM samplers are injected (``gpu_sampler`` / ``ram_sampler``), defaulting
    to :mod:`queue_workflows.gpu_health` — the exact same samplers
    :class:`GpuHealthWatchdog` uses, so the two watchdogs' "is this worker idle?"
    verdict is consistent. Tests feed fakes + a tight confirmation window.
    ``host_label`` / ``queue`` are threaded so the trip path can clear this
    worker's ``current_model`` busy-ghost on hard-exit (Part C)."""

    def __init__(
        self, *, job_id: str, stall_timeout_s: float = STALL_TIMEOUT_S,
        on_exit: Callable[[int], None] | None = None,
        poll_s: float = STALL_POLL_S, exit_code: int = 76,
        table: str = "workflow_node_jobs",
        idle_pct: int = GPU_IDLE_PCT,
        ram_delta_mb: int = GPU_HEALTH_RAM_DELTA_MB,
        confirm_samples: int = STALL_CONFIRM_SAMPLES,
        confirm_poll_s: float = STALL_CONFIRM_POLL_S,
        gpu_sampler: Callable[[], int] | None = None,
        ram_sampler: Callable[[], int | None] | None = None,
        host_label: str | None = None,
        queue: str | None = None,
    ) -> None:
        if table not in _LEASE_TABLES:
            raise ValueError(f"stall-watchdog table must be in {sorted(_LEASE_TABLES)}, got {table!r}")
        self._job_id = job_id
        self._stall_timeout_s = float(stall_timeout_s)
        self._on_exit = on_exit or os._exit
        self._poll_s = float(poll_s)
        self._exit_code = int(exit_code)
        self._table = table
        self._idle_pct = int(idle_pct)
        self._ram_delta_mb = int(ram_delta_mb)
        self._confirm_samples = max(1, int(confirm_samples))
        self._confirm_poll_s = float(confirm_poll_s)
        if gpu_sampler is None or ram_sampler is None:
            from queue_workflows import gpu_health
            gpu_sampler = gpu_sampler or gpu_health.gpu_util_pct
            ram_sampler = ram_sampler or gpu_health.container_ram_mb
        self._gpu_sampler = gpu_sampler
        self._ram_sampler = ram_sampler
        self._host_label = host_label
        self._queue = queue
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

    def _rearm(self) -> None:
        """Open a fresh no-progress window (used after a no-beat timeout that the
        physical-signal confirmation cleared as 'busy/loading, not wedged')."""
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
                # No beat for the window — but that ALONE doesn't mean wedged.
                # Confirm against the physical GPU/RAM signal before tripping.
                if self._confirm_wedged():
                    self._trip()
                    return
                # Busy / loading / preparing — a false alarm. Re-arm the window
                # and keep watching; do NOT kill a working node.
                self._rearm()
            # Wake on stop OR at the next poll boundary, whichever first.
            self._stop.wait(self._poll_s)

    def _sample_gpu(self) -> int:
        try:
            return int(self._gpu_sampler() or 0)
        except Exception:
            log.exception("[stall-watchdog] %s gpu sample failed; treating as 0", self._job_id)
            return 0

    def _sample_ram(self) -> int | None:
        try:
            v = self._ram_sampler()
            return None if v is None else int(v)
        except Exception:
            log.exception("[stall-watchdog] %s ram sample failed; treating as None", self._job_id)
            return None

    def _confirm_wedged(self) -> bool:
        """The no-beat timeout fired — is the worker TRULY doing nothing, or just
        loading / preparing / in a slow step? Sample the physical signal over a
        short window and apply the SAME "GPU idle AND RAM static" rule the
        :class:`GpuHealthWatchdog` uses.

        Returns ``True`` (⇒ trip) ONLY when, across the whole confirmation window,
        the MAX GPU util stayed ``<= idle_pct`` AND RAM never moved more than
        ``ram_delta_mb`` from its first reading. A busy GPU at any sample (slow
        step) or a RAM move beyond the delta (weights loading / staging /
        preparing) ⇒ returns ``False`` (NOT wedged) and the caller re-arms.

        A genuinely-loading model moves RAM by GBs >> the delta, so a cold/lazy
        load can never be confirmed as wedged here — exactly the false positive
        the user reported. Interruptible: bails to ``False`` if stopped mid-window
        (the node finished / was reclaimed)."""
        ram_anchor = self._sample_ram()
        max_util = self._sample_gpu()
        ram_now = ram_anchor
        for i in range(self._confirm_samples - 1):
            if self._stop.wait(self._confirm_poll_s):
                return False  # stopped → don't trip
            util = self._sample_gpu()
            if util > max_util:
                max_util = util
            ram_now = self._sample_ram()
        gpu_idle = max_util <= self._idle_pct
        ram_moved = (
            ram_anchor is not None
            and ram_now is not None
            and abs(ram_now - ram_anchor) > self._ram_delta_mb
        )
        if gpu_idle and not ram_moved:
            log.warning(
                "[stall-watchdog] %s no beat for %ds AND confirmed idle "
                "(max sm%% %d <= %d, RAM anchor=%sMB now=%sMB static) — WEDGED, "
                "tripping",
                self._job_id, int(self._stall_timeout_s), max_util,
                self._idle_pct, ram_anchor, ram_now,
            )
            return True
        log.info(
            "[stall-watchdog] %s no beat for %ds BUT GPU/RAM active "
            "(max sm%% %d, RAM anchor=%sMB now=%sMB) — loading/preparing/slow "
            "step, NOT killing; resetting the window",
            self._job_id, int(self._stall_timeout_s), max_util, ram_anchor, ram_now,
        )
        # Highest-value invisible signal today: the stall was SUSPECTED but the
        # worker is healthy (loading / slow step), so we did NOT kill it. Record
        # it so the "why did this node look stuck but wasn't" story is in-app.
        _emit_node_event(
            self._job_id, "stall_suspected",
            host_label=self._host_label, queue=self._queue,
            error=(f"no beat for {int(self._stall_timeout_s)}s but GPU/RAM "
                   f"active — not killing"),
            detail={"max_sm_pct": max_util, "ram_anchor_mb": ram_anchor,
                    "ram_now_mb": ram_now},
        )
        return False

    def _trip(self) -> None:
        err = (
            f"no progress for {int(self._stall_timeout_s)}s (no GPU step beat) "
            f"AND confirmed GPU-idle + static-RAM — stall watchdog hard-stopped "
            f"a wedged worker"
        )
        # Transient-wedge detector (the user's reconstruct/beat_keyframes case):
        # re-queue + retry on a fresh worker under the cap, then fail.
        _watchdog_trip(
            job_id=self._job_id, table=self._table, error=err,
            label="stall-watchdog",
            on_exit=self._on_exit, exit_code=self._exit_code,
            host_label=self._host_label, queue=self._queue,
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
    how long. On trip it routes through :func:`_watchdog_trip` (the same policy as
    the other watchdogs): RE-QUEUE the node for a retry on a fresh worker under
    the cap (``AI_LEADS_WATCHDOG_MAX_RETRIES``, default 3) so the run stays alive,
    then mark failed once the cap is reached — hard-exiting in either case.

    Arming: the watchdog arms AT job start (:meth:`start`) with a generous
    ``load_grace_s`` (20 min) FIRST window — NOT inert-until-first-beat. That is
    safe precisely because the trip rule is "GPU idle AND RAM static": a healthy
    model load MOVES RAM (weights, multiple GB) so it can't trip even while armed
    during load, and a healthy render keeps the GPU busy so it can't trip
    regardless. Only a GENUINELY hung load (idle GPU AND static RAM past the
    grace) trips. This closes the gap the removed wall-clock cap left: a hang
    DURING load (before any beat), and a GPU node with no ``required_model`` /
    no ``status_callback`` that never beats at all, are now both bounded. The
    first :meth:`beat` (the executor's post-model-load beat, then per-step node
    progress beats) collapses the window to the normal ``interval_s`` (5-min)
    cadence. A node progress ``beat`` also resets the window as an extra liveness
    signal. ``beat`` tolerates any args so it doubles as a node
    ``status_callback``.

    GPU/RAM samplers are injected (``gpu_sampler`` / ``ram_sampler``) so tests
    feed fakes instead of shelling out to ``nvidia-smi``; production defaults to
    :mod:`queue_workflows.gpu_health` (per-container pmon ``sm%`` + cgroup RAM).
    """

    def __init__(
        self, *, job_id: str,
        interval_s: float = GPU_HEALTH_INTERVAL_S,
        load_grace_s: float = GPU_HEALTH_LOAD_GRACE_S,
        idle_pct: int = GPU_IDLE_PCT,
        ram_delta_mb: int = GPU_HEALTH_RAM_DELTA_MB,
        poll_s: float = STALL_POLL_S,
        gpu_sampler: Callable[[], int] | None = None,
        ram_sampler: Callable[[], int | None] | None = None,
        on_exit: Callable[[int], None] | None = None,
        exit_code: int = 78,
        table: str = "workflow_node_jobs",
        host_label: str | None = None, queue: str | None = None,
    ) -> None:
        if table not in _LEASE_TABLES:
            raise ValueError(f"gpu-health table must be in {sorted(_LEASE_TABLES)}, got {table!r}")
        self._job_id = job_id
        self._interval_s = float(interval_s)
        self._load_grace_s = float(load_grace_s)
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
        self._host_label = host_label
        self._queue = queue
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Window state. ``_armed`` is set True in ``start()`` with a generous
        # ``load_grace_s`` first checkpoint (NOT inert-until-beat) so a load-phase
        # hang and a never-beating node are both bounded; the first beat collapses
        # the window to ``interval_s``. ``_max_util`` accumulates the peak GPU util
        # seen this window; ``_ram_anchor`` is the RAM at the window's start that
        # the |Δ| is measured against.
        self._armed = False
        self._max_util = 0
        self._ram_anchor: int | None = None
        self._next_checkpoint = 0.0

    # ── window management ────────────────────────────────────────────────

    def _reset_window(self, window_s: float | None = None) -> None:
        """Open a fresh window: clear the peak-util accumulator, re-anchor RAM,
        and push the next checkpoint out by ``window_s`` (default: the normal
        ``interval_s`` cadence; ``start()`` passes the generous ``load_grace_s``
        for the first window). Caller holds the lock."""
        self._max_util = 0
        self._ram_anchor = self._sample_ram()
        self._next_checkpoint = time.monotonic() + (
            self._interval_s if window_s is None else window_s
        )

    def beat(self, *args: Any, **kwargs: Any) -> None:
        """Record progress: arm (idempotent — already armed at start) + reset the
        window to the NORMAL ``interval_s`` cadence. Wired in as the node
        ``status_callback`` (extra liveness) and pulsed once by the executor right
        after the model load — that post-load beat collapses the generous
        load-grace first window opened by ``start()`` down to ``interval_s``.
        Thread-safe; ignores any args so it doubles as a ``status_callback``."""
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
        # Arm AT start with a generous load-grace first window (NOT inert-until-
        # beat). Safe because the trip rule is "GPU idle AND RAM static": a
        # healthy model load moves RAM and a healthy render keeps the GPU busy, so
        # neither trips even while armed during load — only a genuinely hung load
        # (idle GPU AND static RAM past load_grace_s) does. This bounds both a
        # load-phase hang and a GPU node that never beats. The first beat (post-
        # load) collapses the window to the normal interval_s cadence.
        with self._lock:
            self._armed = True
            self._reset_window(window_s=self._load_grace_s)
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
        # Transient-wedge detector (a wedged GPU render): re-queue + retry on a
        # fresh worker under the cap, then fail.
        _watchdog_trip(
            job_id=self._job_id, table=self._table, error=err,
            label="gpu-health",
            on_exit=self._on_exit, exit_code=self._exit_code,
            host_label=self._host_label, queue=self._queue,
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
                if row is not None:
                    _emit_node_event(
                        self._job_id, "reassigned",
                        host_label=self._claimed_by, row=row,
                        detail={"new_claimed_by": row.get("claimed_by"),
                                "status": row.get("status")},
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
                llm_servers_available=get_config().llm_servers_available,
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
        # Operator ON/OFF control watcher — started in ``run_forever`` after the
        # boot park-gate (None until then; stopped in the finally).
        self._control_watcher: Any = None

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
        _emit_node_event(
            job_id, "claimed", host_label=self.host, queue=self.queue, row=job,
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
            host_label=self.host, queue=self.queue,
        )
        # Abandon-watcher: hard-exit if this job is re-queued / reassigned out
        # from under us (restart-resume, lease reclaim) so it's never double-run.
        status_watcher = JobStatusWatcher(job_id=job_id, claimed_by=self.host)
        renewer.start()
        if watchdog is not None:
            watchdog.start()
        status_watcher.start()

        # GPU guards.
        #
        #  * GpuHealthWatchdog — the universal GPU guard (video + non-video),
        #    replacing the removed wall-clock cap. Arms AT job start with a
        #    generous load-grace first window (so a load-phase hang AND a GPU node
        #    that never beats — no required_model / no status_callback, e.g.
        #    sv_detect, scene_build — are both bounded), then the post-load /
        #    per-step beats collapse it to the 5-min cadence. Kills only a wedged
        #    worker (no GPU work AND static RAM over the window); a busy GPU or a
        #    > 5 GB RAM move keeps it alive forever. Video is the whole point —
        #    a hung wan/ltx render now gets caught on health instead of camping
        #    a (formerly 1800 s) budget that punished healthy long renders.
        #  * StallWatchdog — kept for NON-VIDEO GPU as defense in depth: it stays
        #    inert until the first (post-load) beat so a multi-minute cold load is
        #    never policed, then its tight 120 s no-progress window catches a fast
        #    0%-GPU hang minutes before the health watchdog's first checkpoint.
        #    EXCLUDED for video: a video backend steps slowly (minutes/beat) so
        #    120 s would false-trip a healthy render — health is its only guard.
        is_video = (job.get("required_model") or "") in get_config().video_model_ids
        reports_progress = is_gpu and self._node_reports_progress(job["node_module"])
        stall: StallWatchdog | None = None
        gpu_health: GpuHealthWatchdog | None = None
        beats: list[Callable[..., None]] = []
        if is_gpu:
            gpu_health = GpuHealthWatchdog(
                job_id=job_id, host_label=self.host, queue=self.queue,
            )
            gpu_health.start()
            beats.append(gpu_health.beat)
        if reports_progress and not is_video:
            stall = StallWatchdog(
                job_id=job_id, host_label=self.host, queue=self.queue,
            )
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
            host_label=self.host, queue=self.queue,
        )
        renewer.start()
        watchdog.start()
        try:
            ingest_executor.execute_ingest_job(job)
        finally:
            watchdog.stop()
            renewer.stop()
        return True

    # ── operator control (ON/OFF) ────────────────────────────────────────────

    def requeue_inflight_for_control(self) -> int:
        """Re-queue any job this worker is currently running back to the queue and
        clear its GPU busy-ghost — the DB cleanup a control HARD stop does just
        before the process exits.

        ``os._exit`` skips ``_run_node``'s ``finally`` (exactly like a watchdog
        trip), so this mirrors the trip path's pre-exit bookkeeping: release the
        in-flight row (resume-style, NO ``watchdog_retries`` increment — an operator
        turning a machine off is redistribution, not a node failure) so it
        redistributes at once, and null this worker's ``current_model`` busy-ghost
        so the GPU-busy gauge drops it immediately. Returns the rows re-queued."""
        n = node_queue.requeue_running_for_worker(self.host, self.queue)
        _clear_busy_ghost(self.host, self.queue)
        return n

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

    def _park_until_enabled(self) -> bool:
        """Boot-time gate: while this worker's control row is OFF, do NOT claim and
        do NOT advertise capacity — sit idle until an operator turns it back ON.

        Entered from :meth:`run_forever` before the claim loop; returns when the
        control state is ON (or the worker is stopped). RAM is already free here —
        a freshly (re)started process holds no model — so parking is just "stay
        idle, out of the gauge": a parked worker does NOT heartbeat, so it ages out
        of Rails' fresh-heartbeat window within ~30 s and an OFF machine correctly
        shows zero capacity. LISTENs ``worker_control`` for an instant wake on the
        ON flip, with the same safety-poll timeout the watcher uses.

        This is the ONLY place a worker parks: a RUNNING worker that's turned OFF
        hard-exits (the WorkerControlWatcher) and the supervisor restarts it back
        through this gate — there is no in-process running→parked transition.

        Returns ``True`` if the worker should proceed to its claim loop (control
        state is ON — never parked, or parked then turned back ON); ``False`` if it
        was stopped while parked (e.g. SIGTERM during park), so run_forever exits
        without starting the heartbeat / claim loop."""
        import psycopg

        if (
            worker_control.desired_state_for(self.host, self.queue)
            != worker_control.STATE_OFF
        ):
            return True
        log.warning(
            "[claim-worker:%s] control state OFF for host=%s — PARKED (not "
            "claiming, not advertising capacity) until turned back ON",
            self.queue, self.host,
        )
        poll_s = worker_control._worker_control_poll_s()
        with psycopg.connect(db_url(), autocommit=True) as listen_conn:
            listen_conn.execute(f"LISTEN {worker_control.NOTIFY_CHANNEL}")
            while not self._stop.is_set():
                if (
                    worker_control.desired_state_for(self.host, self.queue)
                    != worker_control.STATE_OFF
                ):
                    log.info(
                        "[claim-worker:%s] control state ON for host=%s — resuming",
                        self.queue, self.host,
                    )
                    return True
                for _ in listen_conn.notifies(timeout=poll_s, stop_after=1):
                    break
        return False

    def run_forever(self) -> None:
        """Block on ``LISTEN <wake_channel>`` and drain the queue greedily on
        each wake (and on the 1 s safety poll for a dropped NOTIFY). Uses a
        dedicated autocommit connection for the LISTEN so it never holds a
        pooled connection across the blocking wait."""
        import psycopg
        # Gate on schema readiness FIRST so the loop never polls a table the
        # orchestrator's bootstrap hasn't created yet.
        self.await_schema()
        # Host hw-metrics sampler: start it BEFORE the park gate so a PARKED gpu
        # worker KEEPS streaming this box's telemetry (CPU/GPU/RAM). Otherwise
        # turning the gpu worker OFF also kills the host's telemetry and the queue
        # gauge FREEZES at its last (busy) sample — it looks "still GPU busy" even
        # though the GPU just went idle. The sampler is host-level, independent of
        # whether this worker is claiming; one per host (flock).
        if self.queue == "gpu":
            from queue_workflows import hw_metrics
            self._hw_sampler = hw_metrics.start_hw_metrics_sampler_flocked()
        # Boot-gate: if an operator has this worker turned OFF, PARK (no claim, no
        # heartbeat) until it's turned back ON. A fresh process holds no model so
        # RAM is already free; parking keeps us idle + out of the capacity gauge.
        # Returns False only if stopped while parked ⇒ exit without starting up.
        if not self._park_until_enabled():
            return
        log.info(
            "[claim-worker:%s] starting (host=%s priority=%d lease=%ds channel=%s)",
            self.queue, self.host, self.host_priority, self.lease_s,
            self._wake_channel,
        )
        # Advertise capacity (no-op for fetch/load) + keep last_seen fresh.
        self.heartbeat.start()
        # (hw-metrics sampler already started above, before the park gate, so it
        # survives an OFF→park cycle.)
        # GPU only: arm the LLM-backend factory's config-change LISTEN invalidator
        # so an operator's ollama↔vllm / tunable edit (worker_controls, 0013) is
        # picked up instantly by the co-tenant VLM backend. Cheap, isolated, and
        # env-gated (AI_LEADS_DISABLE_LLM_CONFIG_LISTENER) so it's inert in tests.
        # Wrapped because the LLM backend is an OPTIONAL subsystem (a host may run
        # no VLM at all) — its arming must never take down the claim worker.
        if self.queue == "gpu":
            try:
                from queue_workflows.llm_backends import factory as _llm_factory
                _llm_factory.start()
            except Exception:
                log.exception(
                    "[claim-worker:gpu] LLM backend factory start failed (ignored)"
                )
        # Operator control watcher: HARD-stop this worker the instant it's turned
        # OFF (re-queue in-flight + os._exit to free RAM; the supervisor restart
        # re-enters the park gate above). Honours AI_LEADS_DISABLE_WORKER_CONTROL.
        self._control_watcher = worker_control.WorkerControlWatcher(worker=self)
        self._control_watcher.start()
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
            if self._control_watcher is not None:
                self._control_watcher.stop()
                self._control_watcher = None
            # Stop the LLM-backend factory (invalidator thread + release any cached
            # backends → frees a vllm sidecar's VRAM). GPU only / best-effort, the
            # mirror of the gpu-gated start() above.
            if self.queue == "gpu":
                try:
                    from queue_workflows.llm_backends import factory as _llm_factory
                    _llm_factory.stop()
                except Exception:
                    log.exception(
                        "[claim-worker:gpu] LLM backend factory stop failed (ignored)"
                    )
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
