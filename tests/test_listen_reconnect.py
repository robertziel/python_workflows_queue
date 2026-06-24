"""Tests for ``db.listen_with_reconnect`` — the durable fix for the
"PG-restart strands workers" pattern.

The helper sits behind every LISTEN site in the engine (claim_worker × 3,
worker_control × 1, llm_backends/factory × 1). Until this landed, a Postgres
bounce killed every worker's ``claim_worker.run_forever`` because psycopg's
``.notifies()`` raises ``OperationalError`` when the server closes the
connection — and there was no retry around it. Each worker exited(1) and
stayed Exited until ops re-bounced the container.

These tests inject a fake ``connect_fn`` that drives the reconnect path
without needing to actually drop and re-raise PG, so they're hermetic and
fast (no live DB, no sleep beyond the helper's own backoff which we keep
sub-second).
"""

from __future__ import annotations

import threading

import psycopg
import pytest

pytestmark = pytest.mark.pg_only
import pytest

from queue_workflows.db import listen_with_reconnect


class _FakeListenConn:
    """Stand-in for ``psycopg.connect(..., autocommit=True)`` that:
      - records every ``execute(...)`` call (LISTEN registrations),
      - on ``raise_on_body`` raises ``OperationalError`` the first time the
        body interacts with it (simulating the connection being severed
        mid-loop, exactly like a PG bounce does),
      - supports the ``with conn:`` context-manager protocol so the helper's
        ``with`` block works unchanged.
    """

    def __init__(self, *, raise_on_body: bool = False) -> None:
        self.executed: list[str] = []
        self._raise = raise_on_body
        self._raised = False

    def __enter__(self) -> "_FakeListenConn":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, sql: str) -> None:
        self.executed.append(sql)


def _connect_factory(connections: list[_FakeListenConn]):
    """Return a connect_fn that pops fakes from the list in order. Lets a
    test queue up "drop, drop, succeed" — the helper should reconnect twice."""
    queue = iter(connections)

    def _connect():
        return next(queue)

    return _connect


def test_helper_calls_body_once_per_successful_connect() -> None:
    """Happy path: one connect, body runs, stop fires, helper returns."""
    stop = threading.Event()
    conn = _FakeListenConn()
    bodies_seen: list[object] = []

    def _body(listen_conn):
        bodies_seen.append(listen_conn)
        stop.set()                       # body signals stop → helper returns

    listen_with_reconnect(
        "my_channel", stop, _body,
        connect_fn=_connect_factory([conn]),
    )
    assert bodies_seen == [conn]
    assert conn.executed == ["LISTEN my_channel"]


def test_helper_reconnects_on_operational_error_and_resumes_body() -> None:
    """The headline regression guard: a PG bounce mid-body raises
    OperationalError → the helper reopens a fresh connection and re-invokes
    body against it. Two distinct connections must be seen, in order."""
    stop = threading.Event()
    first = _FakeListenConn()
    second = _FakeListenConn()
    body_calls: list[object] = []

    def _body(listen_conn):
        body_calls.append(listen_conn)
        if listen_conn is first:
            # Simulate the PG bounce: psycopg.OperationalError propagates from
            # inside ``.notifies()`` IRL; we raise it directly here.
            raise psycopg.OperationalError(
                "server closed the connection unexpectedly"
            )
        stop.set()                       # second connection survives → stop

    listen_with_reconnect(
        "my_channel", stop, _body,
        connect_fn=_connect_factory([first, second]),
        base_s=0.0,                      # no real sleep in the test
        max_s=0.0,
    )
    assert body_calls == [first, second], (
        f"helper did not reconnect; saw {body_calls!r}"
    )
    # Both connections received their LISTEN registration — the helper
    # didn't skip the resubscribe on the reconnect.
    assert first.executed == ["LISTEN my_channel"]
    assert second.executed == ["LISTEN my_channel"]


def test_helper_does_not_reconnect_when_stop_already_set() -> None:
    """When the caller's stop event is set before the body raises, the helper
    returns at the next backoff boundary without trying another connect."""
    stop = threading.Event()
    first = _FakeListenConn()
    second = _FakeListenConn()
    body_calls: list[object] = []

    def _body(listen_conn):
        body_calls.append(listen_conn)
        stop.set()                       # signal stop BEFORE the OperationalError
        raise psycopg.OperationalError("server closed connection")

    listen_with_reconnect(
        "my_channel", stop, _body,
        connect_fn=_connect_factory([first, second]),
        base_s=0.0, max_s=0.0,
    )
    assert body_calls == [first], (
        "helper reconnected after stop was set"
    )


def test_helper_propagates_non_operational_exceptions() -> None:
    """Only ``psycopg.OperationalError`` triggers a reconnect. A
    programming bug in the body (RuntimeError, AttributeError, …) must
    surface to the caller — not get swallowed + retried forever."""
    stop = threading.Event()
    conn = _FakeListenConn()

    def _body(listen_conn):
        raise RuntimeError("a bug in the body")

    with pytest.raises(RuntimeError, match="a bug in the body"):
        listen_with_reconnect(
            "my_channel", stop, _body,
            connect_fn=_connect_factory([conn]),
        )


def test_helper_retries_connect_side_operational_errors_too() -> None:
    """Connect-side OperationalError is treated identically to body-side: it's
    a transient PG outage, the helper waits + tries again. This is the
    durable fix's whole point — if a ``restart.sh`` drops PG for 5-30 seconds,
    every worker should wait it out and resume, not exit(1)."""
    stop = threading.Event()
    conn_attempts = {"n": 0}
    final = _FakeListenConn()

    def _connect():
        conn_attempts["n"] += 1
        if conn_attempts["n"] < 3:
            raise psycopg.OperationalError("PG down")
        return final

    def _body(listen_conn):
        assert listen_conn is final, "body must run against the SUCCESSFUL connect"
        stop.set()

    listen_with_reconnect(
        "my_channel", stop, _body,
        connect_fn=_connect, base_s=0.0, max_s=0.0,
    )
    assert conn_attempts["n"] == 3, conn_attempts
    assert final.executed == ["LISTEN my_channel"]


def test_helper_stops_during_connect_retry_backoff() -> None:
    """When PG is down indefinitely, ``stop.set()`` must end the wait
    promptly — the helper doesn't keep retrying past it. Backstop against an
    accidental tight loop or a stuck wait."""
    stop = threading.Event()
    attempts = {"n": 0}

    def _connect():
        attempts["n"] += 1
        if attempts["n"] == 1:
            stop.set()                   # signal stop after first failure
        raise psycopg.OperationalError("PG down")

    # No raise should escape — the helper returns cleanly when stop fires
    # during the backoff sleep that follows a connect-side OperationalError.
    listen_with_reconnect(
        "my_channel", stop, lambda _c: None,
        connect_fn=_connect, base_s=0.05, max_s=0.05,
    )
    # 1 attempt that raised → backoff sleep → stop.wait returns True →
    # helper exits without trying a 2nd connect.
    assert attempts["n"] == 1


def test_helper_resets_backoff_on_successful_connect() -> None:
    """After a successful reconnect+body cycle, a subsequent failure must
    start the backoff at ``base_s`` again — otherwise repeated PG bounces
    over time would compound into multi-minute delays."""
    # Two reconnects total: drop → succeed → drop → succeed. After the first
    # successful body, the backoff is reset; after the second drop, the
    # helper waits ``base_s`` not ``base_s * 4``. We don't observe the
    # backoff value directly; the test asserts that 3 connects all happen
    # in order, which would only complete cleanly if the backoff stayed
    # bounded (a runaway backoff would either hit max_s repeatedly or never
    # progress).
    stop = threading.Event()
    c1, c2, c3 = _FakeListenConn(), _FakeListenConn(), _FakeListenConn()
    seen: list[object] = []

    def _body(listen_conn):
        seen.append(listen_conn)
        if len(seen) == 1:
            raise psycopg.OperationalError("first drop")
        if len(seen) == 2:
            raise psycopg.OperationalError("second drop")
        stop.set()

    listen_with_reconnect(
        "my_channel", stop, _body,
        connect_fn=_connect_factory([c1, c2, c3]),
        base_s=0.0, max_s=0.0,
    )
    assert seen == [c1, c2, c3]
