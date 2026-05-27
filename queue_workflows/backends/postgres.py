"""PostgreSQL :class:`StorageBackend` — the reference adapter.

This is the canonical implementation: it expresses the contract with the exact
primitives the engine proper uses — ``FOR UPDATE SKIP LOCKED`` for the claim,
``UPDATE … WHERE status NOT IN (terminal) RETURNING *`` for idempotent
transitions, a single transaction for the atomic outbox, and ``pg_notify`` for
the wake. It is the yardstick the redis / mongodb adapters are measured against
by ``tests/test_backend_contract.py``.

It uses its OWN small tables (``qw_jobs`` / ``qw_events`` / ``qw_workers`` /
``qw_controls``), separate from the engine's ``workflow_*`` schema, so enabling
the SPI never collides with — or migrates — a host's existing engine tables.
Every row carries a ``namespace`` column and every query filters on it, so one
Postgres database can host several isolated tenants (the data-leakage guard).
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool

from queue_workflows.backends.base import (
    STATUS_QUEUED,
    STATUS_RUNNING,
    TERMINAL_STATUSES,
    Event,
    Job,
    StorageBackend,
    WakeListener,
)

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS qw_jobs (
    id               TEXT PRIMARY KEY,
    namespace        TEXT NOT NULL,
    queue            TEXT NOT NULL,
    status           TEXT NOT NULL,
    payload          JSONB NOT NULL DEFAULT '{}'::jsonb,
    priority         INTEGER NOT NULL DEFAULT 0,
    attempts         INTEGER NOT NULL DEFAULT 0,
    claimed_by       TEXT,
    lease_expires_at TIMESTAMPTZ,
    result           JSONB,
    error            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS qw_jobs_claim_idx
    ON qw_jobs (namespace, queue, status, priority DESC, created_at);

CREATE TABLE IF NOT EXISTS qw_events (
    seq         BIGSERIAL PRIMARY KEY,
    namespace   TEXT NOT NULL,
    job_id      TEXT NOT NULL,
    queue       TEXT,
    event_type  TEXT NOT NULL,
    detail      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS qw_events_ns_seq_idx ON qw_events (namespace, seq);

CREATE TABLE IF NOT EXISTS qw_workers (
    namespace     TEXT NOT NULL,
    host          TEXT NOT NULL,
    queue         TEXT NOT NULL,
    current_model TEXT,
    last_seen     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, host, queue)
);

CREATE TABLE IF NOT EXISTS qw_controls (
    namespace     TEXT NOT NULL,
    host          TEXT NOT NULL,
    queue         TEXT NOT NULL,
    desired_state TEXT NOT NULL DEFAULT 'on',
    stop_policy   TEXT NOT NULL DEFAULT 'hard',
    requested_by  TEXT,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, host, queue)
);
"""


def _epoch(dt: datetime | None) -> float | None:
    return dt.timestamp() if isinstance(dt, datetime) else None


def _channel(namespace: str) -> str:
    """A NOTIFY channel name scoped to the namespace (so a wake never crosses
    tenants). Sanitized to a valid unquoted identifier; the queue rides the
    payload so one channel per namespace suffices."""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", namespace)[:48]
    return f"qwbe_{safe}"


class PostgresBackend(StorageBackend):
    name = "pg"

    def __init__(self, *, url: str, namespace: str = "") -> None:
        super().__init__(url=url, namespace=namespace)
        self._pool: ConnectionPool | None = None
        self._lock = threading.Lock()

    # ── pool / schema ──────────────────────────────────────────────────────────

    def _ensure_pool(self) -> ConnectionPool:
        if self._pool is None:
            with self._lock:
                if self._pool is None:
                    pool = ConnectionPool(
                        self.url, min_size=1, max_size=5, open=False,
                        kwargs={"row_factory": dict_row},
                    )
                    pool.open(wait=True, timeout=15.0)
                    self._pool = pool
        return self._pool

    @contextmanager
    def _conn(self) -> Iterator[psycopg.Connection]:
        with self._ensure_pool().connection() as conn:
            yield conn

    def ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.close()
                self._pool = None

    # ── enqueue / claim / lease ────────────────────────────────────────────────

    def enqueue(
        self, queue: str, payload: dict[str, Any], *,
        job_id: str | None = None, priority: int = 0,
    ) -> str:
        jid = job_id or uuid.uuid4().hex
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO qw_jobs (id, namespace, queue, status, payload, priority) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (jid, self.namespace, queue, STATUS_QUEUED, Json(payload or {}), priority),
            )
            # The wake rides the same commit as the insert (no "queued but no wake").
            conn.execute("SELECT pg_notify(%s, %s)", (_channel(self.namespace), queue))
            conn.commit()
        return jid

    _CLAIM = """
        UPDATE qw_jobs SET
            status = 'running', claimed_by = %(worker)s,
            lease_expires_at = now() + make_interval(secs => %(lease)s),
            attempts = attempts + 1, updated_at = now()
        WHERE id = (
            SELECT id FROM qw_jobs
            WHERE namespace = %(ns)s AND queue = %(queue)s AND status = 'queued'
            ORDER BY priority DESC, created_at, id
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING *
    """

    def claim(self, queue: str, worker: str, *, lease_s: float) -> Job | None:
        with self._conn() as conn:
            row = conn.execute(
                self._CLAIM,
                {"worker": worker, "lease": float(lease_s), "ns": self.namespace,
                 "queue": queue},
            ).fetchone()
            conn.commit()
        return _row_to_job(row)

    def renew_lease(self, job_id: str, worker: str, *, lease_s: float) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "UPDATE qw_jobs SET lease_expires_at = now() + make_interval(secs => %s), "
                "updated_at = now() "
                "WHERE id = %s AND namespace = %s AND status = 'running' "
                "AND claimed_by = %s RETURNING id",
                (float(lease_s), job_id, self.namespace, worker),
            ).fetchone()
            conn.commit()
        return row is not None

    def reclaim_expired(self, *, queue: str | None = None) -> list[str]:
        sql = (
            "UPDATE qw_jobs SET status = 'queued', claimed_by = NULL, "
            "lease_expires_at = NULL, updated_at = now() "
            "WHERE namespace = %s AND status = 'running' AND lease_expires_at < now()"
        )
        params: list[Any] = [self.namespace]
        if queue is not None:
            sql += " AND queue = %s"
            params.append(queue)
        sql += " RETURNING id, queue"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            for r in rows:  # re-fire the wake for each reclaimed queue
                conn.execute(
                    "SELECT pg_notify(%s, %s)", (_channel(self.namespace), r["queue"])
                )
            conn.commit()
        return [r["id"] for r in rows]

    def requeue_for_retry(self, job_id: str) -> Job | None:
        with self._conn() as conn:
            row = conn.execute(
                "UPDATE qw_jobs SET status = 'queued', claimed_by = NULL, "
                "lease_expires_at = NULL, updated_at = now() "
                "WHERE id = %s AND namespace = %s "
                "AND status NOT IN ('completed', 'failed') RETURNING *",
                (job_id, self.namespace),
            ).fetchone()
            if row is not None:
                conn.execute(
                    "SELECT pg_notify(%s, %s)", (_channel(self.namespace), row["queue"])
                )
            conn.commit()
        return _row_to_job(row)

    # ── terminal transitions ────────────────────────────────────────────────────

    def _mark(self, job_id: str, status: str, *, result, error) -> Job | None:
        with self._conn() as conn:
            row = self._mark_in(conn, job_id, status, result=result, error=error)
            conn.commit()
        return _row_to_job(row)

    def _mark_in(self, conn, job_id, status, *, result, error):
        return conn.execute(
            "UPDATE qw_jobs SET status = %s, result = %s, error = %s, updated_at = now() "
            "WHERE id = %s AND namespace = %s AND status NOT IN ('completed', 'failed') "
            "RETURNING *",
            (status, Json(result) if result is not None else None, error,
             job_id, self.namespace),
        ).fetchone()

    def mark_completed(self, job_id, *, result=None) -> Job | None:
        return self._mark(job_id, "completed", result=result, error=None)

    def mark_failed(self, job_id, *, error=None) -> Job | None:
        return self._mark(job_id, "failed", result=None, error=error)

    def _terminal_with_event(
        self, job_id, status, event_type, *, result, error, detail,
    ) -> Job | None:
        # ONE transaction: the terminal UPDATE and the event INSERT commit
        # together, or (when the row is already terminal → 0 rows updated) neither
        # is written. This is the outbox-atomicity contract in its purest form.
        with self._conn() as conn:
            row = self._mark_in(conn, job_id, status, result=result, error=error)
            if row is None:
                conn.rollback()
                return None
            conn.execute(
                "INSERT INTO qw_events (namespace, job_id, queue, event_type, detail) "
                "VALUES (%s, %s, %s, %s, %s)",
                (self.namespace, job_id, row["queue"], event_type, Json(detail or {})),
            )
            conn.commit()
        return _row_to_job(row)

    def complete_with_event(self, job_id, event_type, *, result=None, detail=None):
        return self._terminal_with_event(
            job_id, "completed", event_type, result=result, error=None, detail=detail
        )

    def fail_with_event(self, job_id, event_type, *, error=None, detail=None):
        return self._terminal_with_event(
            job_id, "failed", event_type, result=None, error=error, detail=detail
        )

    # ── reads ────────────────────────────────────────────────────────────────────

    def get(self, job_id: str) -> Job | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM qw_jobs WHERE id = %s AND namespace = %s",
                (job_id, self.namespace),
            ).fetchone()
        return _row_to_job(row)

    def counts(self, queue: str) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, count(*) AS n FROM qw_jobs "
                "WHERE namespace = %s AND queue = %s GROUP BY status",
                (self.namespace, queue),
            ).fetchall()
        out = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
        for r in rows:
            out[r["status"]] = int(r["n"])
        return out

    def events(self, *, since: int = 0, limit: int = 1000) -> list[Event]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT seq, job_id, queue, event_type, detail, created_at "
                "FROM qw_events WHERE namespace = %s AND seq > %s "
                "ORDER BY seq LIMIT %s",
                (self.namespace, since, limit),
            ).fetchall()
        return [
            Event(
                seq=int(r["seq"]), job_id=r["job_id"], namespace=self.namespace,
                queue=r["queue"], event_type=r["event_type"],
                detail=r["detail"] or {}, created_at=_epoch(r["created_at"]),
            )
            for r in rows
        ]

    # ── wake ──────────────────────────────────────────────────────────────────────

    def notify(self, queue: str) -> None:
        with self._conn() as conn:
            conn.execute("SELECT pg_notify(%s, %s)", (_channel(self.namespace), queue))
            conn.commit()

    def subscribe(self, *queues: str) -> WakeListener:
        return _PgWakeListener(self.url, _channel(self.namespace), frozenset(queues))

    # ── heartbeats + control ───────────────────────────────────────────────────────

    def heartbeat(self, host, queue, *, current_model=None, stale_after_s=30.0) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO qw_workers (namespace, host, queue, current_model, last_seen) "
                "VALUES (%s, %s, %s, %s, now()) "
                "ON CONFLICT (namespace, host, queue) DO UPDATE "
                "SET current_model = EXCLUDED.current_model, last_seen = now()",
                (self.namespace, host, queue, current_model),
            )
            conn.commit()

    def workers(self, queue: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT host, queue, current_model, last_seen FROM qw_workers "
                "WHERE namespace = %s AND queue = %s "
                "AND last_seen > now() - interval '60 seconds'",
                (self.namespace, queue),
            ).fetchall()
        return [
            {"host": r["host"], "queue": r["queue"],
             "current_model": r["current_model"], "last_seen": _epoch(r["last_seen"])}
            for r in rows
        ]

    def set_control(self, host, queue, *, desired_state, stop_policy="hard",
                    requested_by=None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO qw_controls "
                "(namespace, host, queue, desired_state, stop_policy, requested_by, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, now()) "
                "ON CONFLICT (namespace, host, queue) DO UPDATE "
                "SET desired_state = EXCLUDED.desired_state, "
                "stop_policy = EXCLUDED.stop_policy, "
                "requested_by = EXCLUDED.requested_by, updated_at = now()",
                (self.namespace, host, queue, desired_state, stop_policy, requested_by),
            )
            conn.commit()

    def desired_state(self, host: str, queue: str) -> str:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT desired_state FROM qw_controls "
                "WHERE namespace = %s AND host = %s AND queue = %s",
                (self.namespace, host, queue),
            ).fetchone()
        return "off" if (row and row["desired_state"] == "off") else "on"


def _row_to_job(row: dict[str, Any] | None) -> Job | None:
    if row is None:
        return None
    return Job(
        id=row["id"], queue=row["queue"], namespace=row["namespace"],
        status=row["status"], payload=row["payload"] or {},
        priority=int(row["priority"]), attempts=int(row["attempts"]),
        claimed_by=row["claimed_by"], lease_expires_at=_epoch(row["lease_expires_at"]),
        result=row["result"], error=row["error"],
        created_at=_epoch(row["created_at"]), updated_at=_epoch(row["updated_at"]),
    )


class _PgWakeListener:
    """LISTEN on the namespace channel; ``wait`` returns the queue payload if it
    is one we subscribed to (else keeps waiting until the timeout). Uses its own
    autocommit connection, as LISTEN must live outside the pooled txns."""

    def __init__(self, url: str, channel: str, queues: frozenset[str]) -> None:
        self._url = url
        self._channel = channel
        self._queues = queues
        self._conn: psycopg.Connection | None = None

    def __enter__(self) -> "_PgWakeListener":
        self._conn = psycopg.connect(self._url, autocommit=True)
        # channel is a sanitized identifier; quote defensively all the same.
        self._conn.execute(f'LISTEN "{self._channel}"')
        return self

    def __exit__(self, *exc: object) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def wait(self, timeout: float) -> str | None:
        import time

        assert self._conn is not None
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            for note in self._conn.notifies(timeout=remaining, stop_after=1):
                if not self._queues or note.payload in self._queues:
                    return note.payload
                # not one of ours — keep waiting within the remaining budget
            # loop re-checks the deadline
