"""Generic orchestrator entrypoint — bootstrap → NodePool → block.

    queue-orchestrator        (console script)
    python -m queue_workflows.orchestrator

Does:

1. Bootstraps the engine's migration chain (``db.bootstrap`` against the engine
   dir + ``queue_schema_version``) — idempotent, safe on every boot. A host
   that has its OWN domain chain runs that separately (in its launcher) BEFORE
   or AFTER calling this; this orchestrator only owns the queue tables.
2. Re-queues any abandoned ``running`` runs for resume (the startup health
   hook), then starts the :class:`NodePool` (dispatch loop + input listener +
   lease-reclaim sweeps).
3. Blocks until SIGINT / SIGTERM, then stops the pool, closes the DB pool, and
   exits.

No HTTP. The builtin-model registrar handed to the NodePool is the engine's
configured hook (``config.builtin_model_registrar``) — a no-op unless a host
wired one. A host that wants a richer orchestrator (its own SQLAlchemy
disposal, a domain bootstrap) writes a thin launcher that configures the engine
then calls :func:`main` (or replicates this shape).
"""

from __future__ import annotations

import logging
import signal
import sys
import threading

from queue_workflows import db, node_queue, run_store
from queue_workflows.config import get_config
from queue_workflows.node_pool import NodePool

log = logging.getLogger(__name__)
_stop_event = threading.Event()


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        log.info("[orchestrator] received signal %s; stopping", signum)
        _stop_event.set()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    argparse.ArgumentParser(prog="queue-orchestrator").parse_args(argv or [])
    _install_signal_handlers()

    log.info("[orchestrator] bootstrapping engine migrations")
    # Fast path from the snapshot if shipped, then any pending migrations.
    db.bootstrap_from_schema()
    db.bootstrap()

    # Startup health: re-queue any orphan ``running`` runs (a crash left them).
    try:
        n = run_store.reenqueue_running_for_resume()
        if n:
            log.info("[orchestrator] re-queued %d stale running run(s) for resume", n)
    except Exception:
        log.exception("[orchestrator] reenqueue_running_for_resume failed")

    # ...and re-queue every orphan ``running`` NODE JOB. A restart bounces the
    # whole fleet, so any job still ``running`` lost its worker — flip it back
    # to ``queued`` NOW rather than waiting out its (up to 600 s) lease, so the
    # fresh workers resume it immediately. A worker that somehow outlived the
    # restart self-terminates via its JobStatusWatcher when claimed_by clears,
    # so a re-queued row is never double-run.
    try:
        rows = node_queue.reclaim_all_running_for_resume()
        if rows:
            log.info(
                "[orchestrator] re-queued %d orphan running node-job(s) for resume",
                len(rows),
            )
    except Exception:
        log.exception("[orchestrator] reclaim_all_running_for_resume failed")

    log.info("[orchestrator] starting NodePool")
    pool = NodePool(register_builtins=get_config().builtin_model_registrar)
    pool.start()
    log.info(
        "[orchestrator] running (cpu=%d, gpu=%d); claim workers consume the "
        "queues — dispatch loop + input listener running in-process",
        pool.cpu_workers, pool.gpu_workers,
    )

    _stop_event.wait()

    log.info("[orchestrator] stopping pool")
    try:
        pool.stop()
    except Exception:
        log.exception("[orchestrator] error while stopping pool")

    try:
        db.close_pool()
    except Exception:
        log.exception("[orchestrator] error closing psycopg pool")

    log.info("[orchestrator] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
