"""Poller for user input submissions.

Rails inserts a row into ``workflow_input_submissions`` with
``status='pending'`` when the user submits a value for an ``awaiting_input``
node. This daemon thread polls that table every ``POLL_INTERVAL_S`` seconds and
for each pending row:

1. Claims it by flipping status to ``processing`` (stamping ``claimed_at``) in
   the same transaction so a concurrent poller (on a redeployed worker) can't
   process it twice.
2. Calls :func:`dispatcher.resume_after_input` — writes the value into the
   input node's ``context_delta`` and enqueues downstream nodes.
3. On exception, flips the row to ``status='failed'`` with the error text so an
   operator can see why.

Durable by design: if the poller is down when Rails INSERTs, the row sits in
the table until the poller comes up. No notifications are lost.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from queue_workflows import dispatcher
from queue_workflows.db import connection as _db_connection

log = logging.getLogger(__name__)


POLL_INTERVAL_S = 2.0

# Rows in ``processing`` older than this get reclaimed by the next poll. Sized
# to be: longer than any normal ``dispatcher.resume_after_input`` call
# (sub-second), shorter than user-perceptible delay. Tests override via
# monkeypatch.
INPUT_CLAIM_RECLAIM_S: float = 60.0


class InputListener(threading.Thread):
    """Daemon thread. Polls ``workflow_input_submissions`` on a fixed
    interval. Errors during a single row don't break the loop — the
    row is flagged ``failed`` and the poll continues."""

    def __init__(self) -> None:
        super().__init__(daemon=True, name="input-listener")
        self._stop_evt = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self._poll_once()
            except Exception:
                log.exception("[input-listener] poll failed")
            if self._stop_evt.wait(POLL_INTERVAL_S):
                return

    def _poll_once(self) -> None:
        """Claim and process every pending submission row."""
        rows = self._claim_pending()
        for row in rows:
            self._process(row)

    @staticmethod
    def _claim_pending() -> list[dict[str, Any]]:
        """Atomically claim eligible rows — flip to ``processing``
        and stamp ``claimed_at`` in one UPDATE ... RETURNING.

        Eligibility:
          - ``status='pending'`` (normal case), OR
          - ``status='processing' AND claimed_at < now() - threshold``
            (reclaim case — the previous claimant is presumed dead).

        Without the reclaim half, a listener that died after the commit but
        before ``_mark_processed`` would leave the row stranded forever.
        """
        # Project-scoped (migration 0017): a per-project orchestrator resumes
        # ONLY its own project's input submissions. The table has no project
        # column, so correlate to the parent run's project (= config.project).
        # Without this, on a shared broker project A's listener would claim
        # project B's submission and resume B's run under A's pipeline/resolver.
        # Default "" (single-tenant) matches every run, so behaviour is unchanged.
        from queue_workflows.config import get_config
        project = get_config().project or ""
        with _db_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workflow_input_submissions
                   SET status = 'processing',
                       claimed_at = now()
                 WHERE id IN (
                     SELECT s.id FROM workflow_input_submissions s
                      WHERE (s.status = 'pending'
                          OR (s.status = 'processing'
                              AND s.claimed_at IS NOT NULL
                              AND s.claimed_at < now() - make_interval(secs => %(reclaim)s)))
                        AND EXISTS (
                            SELECT 1 FROM workflow_runs r
                            WHERE r.id = s.run_id AND r.project = %(project)s
                        )
                      ORDER BY s.created_at ASC
                      FOR UPDATE SKIP LOCKED
                      LIMIT 50
                 )
             RETURNING id, run_id, node_id, value
                """,
                {"reclaim": float(INPUT_CLAIM_RECLAIM_S), "project": project},
            )
            rows = cur.fetchall()
            conn.commit()
        return [dict(r) for r in rows]

    @staticmethod
    def _process(row: dict[str, Any]) -> None:
        sub_id = row["id"]
        run_id = row["run_id"]
        node_id = row["node_id"]
        # psycopg returns jsonb as a python value; in-flight tests may
        # pass a string — accept both.
        value = row["value"]
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                pass  # treat as raw string
        try:
            dispatcher.resume_after_input(run_id, node_id, value=value)
        except Exception as exc:
            log.exception(
                "[input-listener] resume failed for %s / %s", run_id, node_id,
            )
            _mark_failed(sub_id, f"{type(exc).__name__}: {exc}")
            return
        _mark_processed(sub_id)


def _mark_processed(sub_id: str) -> None:
    with _db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_input_submissions "
            "SET status='processed', processed_at=now() WHERE id=%s",
            (sub_id,),
        )
        conn.commit()


def _mark_failed(sub_id: str, error: str) -> None:
    with _db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE workflow_input_submissions "
            "SET status='failed', processed_at=now(), error=%s WHERE id=%s",
            (error, sub_id),
        )
        conn.commit()
