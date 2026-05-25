"""Run-cancel watcher — feeds a cooperative ``cancel_event`` to a running
node body when its parent run flips to ``cancelled`` / ``failed``.

The claim SQL already refuses jobs of a cancelled run; this catches a cancel
that lands MID-execution so a long-running GPU node can unwind early instead of
running to completion. Reads run status through the engine's
:mod:`queue_workflows.run_store` (the ``repo`` inversion, plan §1d).
"""

from __future__ import annotations

import logging
import threading

from queue_workflows import run_store

log = logging.getLogger(__name__)


def _start_run_cancel_watcher(
    run_id: str,
    cancel_event: "threading.Event",
    *,
    interval_s: float = 5.0,
) -> "threading.Thread":
    """Spawn a daemon thread that polls ``workflow_runs.status`` every
    ``interval_s`` seconds and sets ``cancel_event`` when the run flips to
    ``cancelled`` / ``failed``.

    Used by long-running GPU jobs so a cooperative node body can notice and
    unwind early. The thread exits when ``cancel_event`` is set (either by the
    watcher itself observing a status change, or by the caller's ``finally``
    block on normal completion). Transient DB errors are swallowed and the next
    poll is attempted — a Postgres blip should not crash the watcher mid-job.
    """

    def watch() -> None:
        while not cancel_event.is_set():
            try:
                run = run_store.get_run(run_id) or {}
                if run.get("status") in ("cancelled", "failed"):
                    log.info(
                        "[cancel-watcher] run %s status=%s; signalling cancel",
                        run_id, run.get("status"),
                    )
                    cancel_event.set()
                    return
            except Exception:
                log.exception(
                    "[cancel-watcher] poll failed for run %s; retrying",
                    run_id,
                )
            cancel_event.wait(interval_s)

    t = threading.Thread(
        target=watch,
        daemon=True,
        name=f"cancel-watcher-{run_id[:8]}",
    )
    t.start()
    return t
