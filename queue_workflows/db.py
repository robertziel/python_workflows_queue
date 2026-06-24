"""Postgres connection pool + generalised migration runner.

One global ``ConnectionPool`` per process keyed by the DSN read from the
env var named by ``config.db_url_env`` (default ``AI_LEADS_DB_URL`` for
byte-compat with the existing ai_leads deploy; other projects pass their
own via ``queue_workflows.configure(db_url_env=...)``).

Migrations are plain SQL files under a *migrations dir*:

- ``NNNN_name.sql``        — forward migration, applied in ``bootstrap``
- ``NNNN_name.down.sql``   — paired reverse, applied in ``downgrade``

The runner is **generalised** vs the original ai_leads single-chain shape:
``bootstrap`` / ``downgrade`` / ``wait_for_schema`` / ``current_schema_version``
take ``migrations_dir`` + ``version_table`` so the engine owns its OWN chain
(the queue tables, version-ledger ``queue_schema_version``) while a host can
run a SECOND chain (its domain tables) against its own dir + ledger on top.
The engine's defaults point at ``queue_workflows/migrations`` +
``queue_schema_version``.

The highest applied version per chain is tracked in its version table. Both
directions are idempotent — ``bootstrap`` skips versions already recorded;
``downgrade`` skips versions without a ``.down.sql`` pair.

Tests should call :func:`reset_for_tests` (drops + recreates the public
schema, then re-runs the engine migrations) — never run that against
production.
"""

from __future__ import annotations

import json as _json
import logging
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from queue_workflows import config as _config

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
#: The engine's own migration chain (queue tables only). Public via
#: ``queue_workflows.migrations.dir()``.
ENGINE_MIGRATIONS_DIR = _HERE / "migrations"
#: The SQLite-dialect twin of the engine chain (same version numbers; DDL
#: translated, triggers/NOTIFY omitted). Selected when ``db_backend="sqlite"``.
ENGINE_MIGRATIONS_DIR_SQLITE = _HERE / "migrations_sqlite"
ENGINE_VERSION_TABLE = "queue_schema_version"
_ENGINE_SCHEMA_SNAPSHOT = ENGINE_MIGRATIONS_DIR / "schema.sql"


def _engine_migrations_dir() -> Path:
    """The engine migration dir for the active relational backend."""
    return ENGINE_MIGRATIONS_DIR_SQLITE if _engine_is_sqlite() else ENGINE_MIGRATIONS_DIR

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def db_url() -> str:
    env_name = _config.get_config().db_url_env
    url = os.environ.get(env_name)
    if not url:
        raise RuntimeError(
            f"{env_name} is not set; cannot connect to Postgres. "
            "Set it (or pass a different env via "
            "queue_workflows.configure(db_url_env=...))."
        )
    return url


def get_pool() -> ConnectionPool:
    """Lazy-init a process-wide pool. Safe across worker threads."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(
                    db_url(),
                    min_size=1,
                    max_size=int(os.environ.get("AI_LEADS_DB_POOL_MAX", "10")),
                    kwargs={
                        "row_factory": dict_row,
                        # TCP keepalives so workers on remote boxes detect
                        # a control-host network blip in seconds rather
                        # than the OS default ~2 hours. Idle 30 s → probe
                        # every 10 s → kill after 3 missed (≈ 60 s total).
                        # libpq parameters; psycopg passes them through.
                        "keepalives": 1,
                        "keepalives_idle": 30,
                        "keepalives_interval": 10,
                        "keepalives_count": 3,
                    },
                    # Health-check each connection before handing it to
                    # the caller. Critical for forked workers: the pool
                    # inherits dead connections from the parent via fork(),
                    # and without a check the pool hands them out
                    # unconditionally — the caller then sees "the connection
                    # is closed" on first use.
                    check=ConnectionPool.check_connection,
                    open=False,
                )
                _pool.open(wait=True, timeout=15.0)
    return _pool


def close_pool() -> None:
    """Drain the pool. Used by orchestrator shutdown + test teardown. Also drops
    the shared SQLite connection (if any), so a reconfigure/reset starts fresh."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None
    _close_sqlite_conn()


# ── SQLite engine backend (db_backend="sqlite") ─────────────────────────────
#
# The same engine SQL runs on a single SQLite file (a local, daemon-less, low
# RAM/disk deploy) via a thin compatibility layer:
#   * a string-literal-AWARE pyformat→sqlite paramstyle translator (so ``%s``
#     placeholders convert to ``?`` but ``strftime('%s')`` survives),
#   * ``now()``→``datetime('now')``, ``::cast`` strip, ``FOR UPDATE [SKIP
#     LOCKED]`` strip, ``LEAST``/``GREATEST``→``MIN``/``MAX`` — the universal
#     mechanical rewrites (the structural ones — intervals, EXTRACT(EPOCH),
#     ANY(array) — are produced per-call by :mod:`queue_workflows.dialect`),
#   * an explicit row factory that restores psycopg parity: JSONB→dict,
#     text[]→list, TIMESTAMPTZ→aware-UTC ``datetime`` (keyed by the engine's
#     KNOWN column names, so it's robust under ``RETURNING *`` / joins where
#     ``PARSE_DECLTYPES`` is unreliable),
#   * WAL + busy_timeout so multiple worker PROCESSES on one file serialize
#     safely; a per-process shared connection under an RLock serializes THREADS.

def _engine_is_sqlite() -> bool:
    return _config.get_config().db_backend == "sqlite"


def sqlite_path() -> str:
    """Resolve the SQLite file path from the DSN env (``config.db_url_env``).
    Accepts ``sqlite:///rel.db`` / ``sqlite:////abs/path.db`` / a bare path /
    ``:memory:``."""
    raw = (os.environ.get(_config.get_config().db_url_env) or "").strip()
    if not raw:
        raise RuntimeError(
            f"{_config.get_config().db_url_env} is not set; cannot open SQLite. "
            "Set it to a file path (or sqlite:///path, or :memory:)."
        )
    if raw == ":memory:":
        return raw
    if raw.startswith("sqlite://"):
        rest = raw[len("sqlite://"):]
        # sqlite:////abs → /abs ; sqlite:///rel → rel
        return rest[1:] if rest.startswith("///") else rest.lstrip("/") if rest.startswith("//") else rest or ":memory:"
    return raw


# Columns the engine stores as JSON / arrays / timestamps — used by the row
# factory to restore psycopg-equivalent python types on read.
_JSON_OBJ_COLS = frozenset({
    "context", "steps_done", "input_spec", "inputs", "resolved_inputs",
    "context_delta", "args", "result", "detail", "value",
})
_JSON_ARRAY_COLS = frozenset({"known_models", "fits_models", "llm_servers_available"})
_TS_COLS = frozenset({
    "created_at", "updated_at", "queued_at", "started_at", "finished_at",
    "lease_expires_at", "last_seen", "last_flagged_dead_at", "claimed_at",
    "applied_at", "unassignable_at", "processed_at",
})
# SQLite has no boolean type (0/1 INTEGER); restore python ``bool`` parity with
# psycopg for the engine's known boolean columns + derived boolean flags
# (fleet_snapshot's fresh/flagged_dead).
_BOOL_COLS = frozenset({"is_priority", "is_primary", "fresh", "flagged_dead"})


def _parse_ts(value: Any) -> Any:
    """Parse a SQLite TEXT timestamp (``YYYY-MM-DD HH:MM:SS`` UTC, the
    ``datetime('now')`` format) into an aware UTC ``datetime`` — matching what
    psycopg returns for a ``timestamptz`` column."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return value
    s = s.replace("T", " ")
    # drop a trailing tz designator if present
    if s.endswith("Z"):
        s = s[:-1]
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in s else "%Y-%m-%d %H:%M:%S"
    try:
        return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
    except ValueError:
        return value


def _sqlite_row_to_dict(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for idx, col in enumerate(cursor.description):
        name = col[0]
        val = row[idx]
        if val is not None and isinstance(val, str):
            if name in _JSON_OBJ_COLS or name in _JSON_ARRAY_COLS:
                try:
                    val = _json.loads(val)
                except (ValueError, TypeError):
                    pass
            elif name in _TS_COLS:
                val = _parse_ts(val)
        elif name in _BOOL_COLS and isinstance(val, int):
            val = bool(val)
        out[name] = val
    return out


_SQLITE_ADAPTERS_REGISTERED = False


def _register_sqlite_adapters() -> None:
    """Adapt python write-params to SQLite text: psycopg ``Jsonb`` → JSON text,
    aware/naive ``datetime`` → ``YYYY-MM-DD HH:MM:SS`` UTC (matching
    ``datetime('now')`` so stored timestamps compare correctly)."""
    global _SQLITE_ADAPTERS_REGISTERED
    if _SQLITE_ADAPTERS_REGISTERED:
        return
    from psycopg.types.json import Jsonb

    def _adapt_jsonb(j: Jsonb) -> str:
        return _json.dumps(j.obj)

    def _adapt_dt(dt: datetime) -> str:
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    sqlite3.register_adapter(Jsonb, _adapt_jsonb)
    sqlite3.register_adapter(datetime, _adapt_dt)
    _SQLITE_ADAPTERS_REGISTERED = True


# Module-level shared connection per process + RLock (SQLite serializes writers;
# this serializes threads within the process; cross-process safety is WAL +
# busy_timeout on the file). ``_sqlite_depth`` makes ``connection()`` RE-ENTRANT:
# the engine nests connection() calls (e.g. the dispatch drain holds one while a
# callback opens its own) — on one shared SQLite connection that must share a
# single transaction, not BEGIN twice. BEGIN/COMMIT happen only at depth 0.
_sqlite_conn: sqlite3.Connection | None = None
_sqlite_lock = threading.RLock()
_sqlite_depth = 0


def _get_sqlite_conn() -> sqlite3.Connection:
    global _sqlite_conn
    if _sqlite_conn is None:
        _register_sqlite_adapters()
        path = sqlite_path()
        conn = sqlite3.connect(
            path, check_same_thread=False, isolation_level=None,  # autocommit; we manage txns
            timeout=float(os.environ.get("QUEUE_WORKFLOWS_SQLITE_TIMEOUT_S", "30")),
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        _sqlite_conn = conn
    return _sqlite_conn


def _close_sqlite_conn() -> None:
    global _sqlite_conn
    with _sqlite_lock:
        if _sqlite_conn is not None:
            _sqlite_conn.close()
            _sqlite_conn = None


# ── pyformat → sqlite translator (string-literal aware) ──────────────────────

def _split_sql_literals(sql: str) -> list[tuple[bool, str]]:
    """Split ``sql`` into ``(is_literal, chunk)`` segments, where a literal is a
    single-quoted string (with ``''`` escapes). Transforms apply only to
    non-literal chunks so e.g. ``strftime('%s', …)`` is never rewritten."""
    segs: list[tuple[bool, str]] = []
    buf: list[str] = []
    in_str = False
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if not in_str:
            if c == "'":
                if buf:
                    segs.append((False, "".join(buf))); buf = []
                buf.append(c); in_str = True
            else:
                buf.append(c)
        else:
            buf.append(c)
            if c == "'":
                if i + 1 < n and sql[i + 1] == "'":  # '' escape
                    buf.append("'"); i += 2; continue
                segs.append((True, "".join(buf))); buf = []; in_str = False
        i += 1
    if buf:
        segs.append((in_str, "".join(buf)))
    return segs


_CAST_RE = re.compile(r"::[A-Za-z_][A-Za-z0-9_]*(\s*\[\s*\])?")
_NOW_RE = re.compile(r"\bnow\(\)")
_LEAST_RE = re.compile(r"\bLEAST\s*\(")
_GREATEST_RE = re.compile(r"\bGREATEST\s*\(")
_FORUPDATE_RE = re.compile(r"\bFOR\s+UPDATE(\s+SKIP\s+LOCKED)?", re.IGNORECASE)
_NAMED_PARAM_RE = re.compile(r"%\(([A-Za-z_][A-Za-z0-9_]*)\)s")


# Interval arithmetic — SQLite has no make_interval / interval-subtraction
# operator; ``now() ± <n> <unit>`` must FUSE into a single ``datetime('now',
# modifier)`` call. These run BEFORE now()->datetime so they can match ``now()``.
# Placeholders inside the modifier (``%(x)s`` / ``%s``) survive to the paramstyle
# step. (pg never reaches here — the translator runs only on the sqlite path.)
# An interval magnitude is a named param ``%(x)s`` (contains parens!), a
# positional ``%s``, an integer literal, or a bare column/identifier. The engine
# uses ONLY ``now() ± make_interval(<unit> => <n>)`` for interval arithmetic (no
# ``interval 'literal'``, which the string-literal splitter would break apart) —
# so the translator only needs the two make_interval shapes.
_EXPR = r"%\([A-Za-z_]\w*\)s|%s|\d+|[A-Za-z_][\w.]*"
_MI_SECS_RE = re.compile(rf"now\(\)\s*([+-])\s*make_interval\(\s*secs\s*=>\s*({_EXPR})\s*\)")
_MI_DAYS_RE = re.compile(rf"now\(\)\s*([+-])\s*make_interval\(\s*days\s*=>\s*({_EXPR})\s*\)")


def _interval_sub(unit: str):
    """Build a sign-SAFE ``datetime('now', modifier)`` for ``now() ± make_interval
    (<unit> => X)``. The effective offset folds the SQL operator into the value
    (``+`` → X, ``-`` → -X); ``printf('%+d …')`` then emits the correct leading
    sign for any X (incl. negative — test helpers stamp an already-expired lease
    via ``now() + make_interval(secs => <negative>)``). Uses ``printf`` (not a
    CASE) so the placeholder is referenced exactly ONCE — a CASE would duplicate
    it and break positional ``%s`` param counts."""
    def _sub(m):
        op, x = m.group(1), m.group(2)
        eff = f"({x})" if op == "+" else f"(-({x}))"
        # strftime (not datetime) so the offset timestamp keeps millisecond
        # precision — a whole-second lease_expires_at would tie across renewals
        # within the same second.
        return f"strftime('%Y-%m-%d %H:%M:%f', 'now', printf('%+d {unit}', {eff}))"
    return _sub


def _rewrite_intervals(chunk: str) -> str:
    chunk = _MI_SECS_RE.sub(_interval_sub("seconds"), chunk)
    chunk = _MI_DAYS_RE.sub(_interval_sub("days"), chunk)
    return chunk


def _rewrite_chunk(chunk: str) -> str:
    chunk = _FORUPDATE_RE.sub("", chunk)
    chunk = _rewrite_intervals(chunk)
    # Millisecond-precision UTC timestamp (psycopg's now() has sub-second
    # precision; whole-second datetime('now') would tie FIFO-ordered rows created
    # in the same second). Parenthesized → valid even in a column DEFAULT. The
    # ``%Y/%m/%f`` live inside a string literal, so the paramstyle step ignores them.
    chunk = _NOW_RE.sub("(strftime('%Y-%m-%d %H:%M:%f', 'now'))", chunk)
    chunk = _CAST_RE.sub("", chunk)
    chunk = _LEAST_RE.sub("MIN(", chunk)
    chunk = _GREATEST_RE.sub("MAX(", chunk)
    # paramstyle: named first, then positional, then unescape %%
    chunk = _NAMED_PARAM_RE.sub(r":\1", chunk)
    chunk = chunk.replace("%s", "?")
    chunk = chunk.replace("%%", "%")
    return chunk


# ``now() ± interval 'N seconds'`` spans a string literal, so it must be fused
# BEFORE the literal splitter runs (the make_interval forms have no literal and
# are handled per-chunk). Engine SQL uses make_interval; this also covers any
# consumer / test helper that wrote the ``interval 'literal'`` form.
_PRE_INT_LIT_RE = re.compile(r"now\(\)\s*([+-])\s*interval\s*'(\d+)\s+seconds?'")


def _pre_split_intervals(sql: str) -> str:
    return _PRE_INT_LIT_RE.sub(
        lambda m: f"datetime('now', '{m.group(1)}{m.group(2)} seconds')", sql,
    )


def _translate_sql_for_sqlite(sql: str) -> str:
    sql = _pre_split_intervals(sql)
    return "".join(
        seg if is_lit else _rewrite_chunk(seg)
        for is_lit, seg in _split_sql_literals(sql)
    )


class _SqliteCursor:
    """psycopg-cursor-shaped wrapper: context manager, ``execute(sql, params)``
    with pyformat translation, dict ``fetchone``/``fetchall``, ``rowcount``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._cur = conn.cursor()

    def __enter__(self) -> "_SqliteCursor":
        return self

    def __exit__(self, *exc: object) -> None:
        self._cur.close()

    def execute(self, sql: str, params: Any = None) -> "_SqliteCursor":
        sql2 = _translate_sql_for_sqlite(sql)
        if params is None:
            self._cur.execute(sql2)
        else:
            self._cur.execute(sql2, params)
        return self

    def fetchone(self) -> dict[str, Any] | None:
        row = self._cur.fetchone()
        return None if row is None else _sqlite_row_to_dict(self._cur, row)

    def fetchall(self) -> list[dict[str, Any]]:
        rows = self._cur.fetchall()
        return [_sqlite_row_to_dict(self._cur, r) for r in rows]

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount


class _SqliteConn:
    """psycopg-connection-shaped wrapper over the shared sqlite connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def cursor(self) -> _SqliteCursor:
        return _SqliteCursor(self._conn)

    def execute(self, sql: str, params: Any = None) -> _SqliteCursor:
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def executescript(self, sql: str) -> None:
        self._conn.executescript(sql)


@contextmanager
def _sqlite_connection() -> Iterator[_SqliteConn]:
    """SQLite analogue of the pooled ``connection()``: a per-process shared conn
    under a re-entrant RLock. BEGIN on the OUTERMOST entry, commit on its clean
    exit / rollback on error; nested calls join the same transaction (no second
    BEGIN) — matching 'one SQLite connection == one transaction'."""
    global _sqlite_depth
    with _sqlite_lock:
        raw = _get_sqlite_conn()
        outer = _sqlite_depth == 0
        if outer:
            raw.execute("BEGIN")
        _sqlite_depth += 1
        wrapper = _SqliteConn(raw)
        try:
            yield wrapper
        except BaseException:
            if outer:
                raw.rollback()
            raise
        else:
            if outer:
                raw.commit()
        finally:
            _sqlite_depth -= 1


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Borrow a connection for the engine's relational store. Auto-commits on
    clean exit, rolls back on exception. Postgres (default) → pooled psycopg;
    ``db_backend="sqlite"`` → the shared SQLite connection."""
    if _engine_is_sqlite():
        with _sqlite_connection() as conn:
            yield conn
        return
    with get_pool().connection() as conn:
        yield conn


@contextmanager
def cursor() -> Iterator[psycopg.Cursor]:
    """Convenience: borrow a connection + cursor in one block."""
    with connection() as conn:
        with conn.cursor() as cur:
            yield cur


class _SqlitePollConn:
    """A LISTEN-shaped fake for SQLite (no NOTIFY): ``execute('LISTEN …')`` is a
    no-op and ``notifies(timeout=…)`` just waits ``timeout`` (responsive to the
    stop event) and yields nothing — so a wake loop_body falls back to polling at
    its safety-poll cadence."""

    def __init__(self, stop_event: Any) -> None:
        self._stop = stop_event

    def execute(self, *args: object, **kwargs: object) -> "_SqlitePollConn":
        return self

    def notifies(self, *, timeout: float = 1.0, stop_after: int | None = None):
        try:
            self._stop.wait(max(0.0, float(timeout)))
        except Exception:  # noqa: BLE001 — defensive; never break the wake loop
            pass
        return ()


_LISTEN_RECONNECT_BASE_S = 1.0
_LISTEN_RECONNECT_MAX_S = 30.0


def listen_with_reconnect(
    channel: str,
    stop_event: "object",
    loop_body: Callable[[psycopg.Connection], None],
    *,
    base_s: float = _LISTEN_RECONNECT_BASE_S,
    max_s: float = _LISTEN_RECONNECT_MAX_S,
    connect_fn: Callable[[], psycopg.Connection] | None = None,
) -> None:
    """Run ``loop_body(listen_conn)`` against a freshly-opened autocommit
    psycopg connection that has issued ``LISTEN <channel>``. If
    ``psycopg.OperationalError`` escapes ``loop_body`` (e.g. PG was bounced
    and severed the connection mid ``.notifies()``), reconnect with exponential
    backoff and call ``loop_body`` against the fresh connection. Returns when
    ``loop_body`` returns cleanly (caller's stop signal fired) or when
    ``stop_event.wait(...)`` returns True during a backoff sleep.

    This is the durable fix for the documented "PG-restart strands workers"
    pattern: every restart of the central Postgres used to crash every worker's
    ``run_forever`` because psycopg's ``.notifies()`` raises OperationalError
    when the server closes the connection, and ``run_forever`` had no retry
    around it. The five LISTEN sites in the engine (claim_worker × 3,
    worker_control × 1, llm_backends/factory × 1) all funnel through this
    helper so any of them can survive a transient PG outage.

    Parameters:
      channel:     PG NOTIFY channel to LISTEN on.
      stop_event:  threading.Event-like (.is_set(), .wait(s)). When the event
                   fires, the helper returns at the next backoff boundary
                   without trying to reconnect.
      loop_body:   The work to do against the live LISTEN connection. It
                   should respect ``stop_event`` and return when it fires.
      base_s/max_s: Exponential backoff floor/ceiling between reconnect
                   attempts. Successful connect resets backoff to base_s.
      connect_fn:  Test seam — defaults to ``psycopg.connect(db_url(),
                   autocommit=True)``. Tests inject a generator of fake
                   connections so the reconnect path can be exercised
                   without a live PG.
    """
    # SQLite has no LISTEN/NOTIFY: the engine wakes by POLLING. Run loop_body
    # against a poll-only fake connection whose notifies() just waits the safety-
    # poll timeout (so loop_body does its periodic claim/check each cycle). No
    # psycopg connect, no reconnect loop. (Only when no test connect_fn is given.)
    if connect_fn is None and _engine_is_sqlite():
        loop_body(_SqlitePollConn(stop_event))
        return

    if connect_fn is None:
        def connect_fn():
            return psycopg.connect(db_url(), autocommit=True)

    backoff = base_s
    while not stop_event.is_set():
        try:
            with connect_fn() as listen_conn:
                listen_conn.execute(f"LISTEN {channel}")
                backoff = base_s        # successful connect → reset backoff
                loop_body(listen_conn)
            return                       # loop_body returned cleanly (stop)
        except psycopg.OperationalError as exc:
            log.warning(
                "[listen-reconnect:%s] %s: %s — reconnecting in %.1fs",
                channel, exc.__class__.__name__, exc, backoff,
            )
            if stop_event.wait(backoff):
                return                   # stop fired during the backoff sleep
            backoff = min(backoff * 2.0, max_s)


def _forward_migrations(migrations_dir: Path) -> list[tuple[int, Path]]:
    """Sorted list of (version, path) for every forward migration,
    skipping the ``.down.sql`` pairs."""
    out: list[tuple[int, Path]] = []
    for path in sorted(migrations_dir.glob("*.sql")):
        # Skip ``NNNN_name.down.sql``; those are handled by downgrade().
        if path.name.endswith(".down.sql"):
            continue
        if path.name == "schema.sql":
            continue
        n = int(path.stem.split("_", 1)[0])
        out.append((n, path))
    return out


def _down_migration(migrations_dir: Path, version: int) -> Path | None:
    """Locate the ``NNNN_*.down.sql`` file for a given version in
    ``migrations_dir``. Returns None when there's no paired down file."""
    matches = [
        p for p in migrations_dir.glob("*.down.sql")
        if int(p.stem.split("_", 1)[0]) == version
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple down files for version {version}: "
            f"{sorted(p.name for p in matches)}"
        )
    return matches[0]


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_identifier(name: str) -> str:
    """Guard the one SQL identifier the migration runner interpolates
    (``version_table``).

    ``version_table`` is an *identifier* (a table name), so it cannot be bound
    with a ``%s`` placeholder — it has to be interpolated into the statement
    text. It is a developer-supplied parameter (``bootstrap(version_table=...)``),
    never end-user input, so this is defence-in-depth rather than a response to
    an attacker-reachable path: it pins the value to a plain, unqualified
    Postgres identifier (letters/digits/underscore, not starting with a digit)
    so a typo or a hostile caller can't smuggle SQL through the table name.
    Returns the validated name so call sites can interpolate it directly.
    """
    if not _IDENT_RE.match(name):
        raise ValueError(
            f"invalid version_table identifier {name!r} — must match "
            r"^[A-Za-z_][A-Za-z0-9_]*$ (a plain unqualified SQL identifier)"
        )
    return name


def _applied_versions(conn: psycopg.Connection, version_table: str) -> list[int]:
    _check_identifier(version_table)
    with conn.cursor() as cur:
        cur.execute(f"SELECT version FROM {version_table} ORDER BY version")
        return [r["version"] for r in cur.fetchall()]


def _apply_migration_sql(conn: Any, cur: Any, sql_text: str) -> None:
    """Apply a (possibly multi-statement) migration file. psycopg sends the whole
    string via the simple-query protocol; SQLite needs ``executescript``."""
    if _engine_is_sqlite():
        conn.executescript(sql_text)
    else:
        cur.execute(sql_text)


def bootstrap(
    *,
    migrations_dir: Path | None = None,
    version_table: str = ENGINE_VERSION_TABLE,
) -> None:
    """Apply pending migrations from ``migrations_dir`` against the version
    ledger ``version_table``. Idempotent — safe to call on every boot.

    ``migrations_dir`` defaults to the engine chain for the active relational
    backend (``migrations/`` for Postgres, ``migrations_sqlite/`` for SQLite). A
    host applies its domain chain by calling this a SECOND time with its own dir
    + ``schema_version``.

    CONCURRENCY-SAFE on Postgres: a ``pg_advisory_xact_lock`` (keyed on the
    version table) serializes concurrent bootstraps, so MANY processes calling it
    on the SAME database — e.g. every project's orchestrator booting against one
    shared broker after a new migration ships — is safe: the lock holder applies
    the pending chain and commits; every waiter then re-reads the ledger inside
    the lock and finds nothing to do. The lock is acquired **FIRST** — before even
    ``CREATE TABLE IF NOT EXISTS`` (which itself races at the pg catalog) and the
    ``applied`` read — so each waiter's read reflects whatever the winner
    committed (do NOT reorder this).

    On SQLite the advisory lock is a no-op and bootstrap is NOT concurrency-safe
    across PROCESSES (the per-process shared connection + RLock serialize only
    threads; the SQLite migrations use bare ``ADD COLUMN``). That's fine for its
    intended use: SQLite is a single-machine deploy and only the orchestrator
    bootstraps (claim workers/scheduler call :func:`wait_for_schema`, never
    bootstrap), so concurrent SQLite bootstrap does not arise in normal operation.
    """
    _check_identifier(version_table)
    mdir = migrations_dir or _engine_migrations_dir()
    with connection() as conn:
        with conn.cursor() as cur:
            if not _engine_is_sqlite():
                # Serialize concurrent bootstraps on one DB — acquired FIRST, even
                # before CREATE TABLE (concurrent ``CREATE TABLE IF NOT EXISTS``
                # itself races at the pg catalog level). Held until COMMIT; a
                # racing bootstrap blocks here, then the ``applied`` read below
                # reflects whatever the winner committed (so it skips, not double-
                # applies / collides on the ledger PK).
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (version_table,))
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {version_table} ("
                "  version INTEGER PRIMARY KEY,"
                "  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
        applied = set(_applied_versions(conn, version_table))

        for n, path in _forward_migrations(mdir):
            if n in applied:
                continue
            log.info("[queue_workflows.db] applying %s", path.name)
            with conn.cursor() as cur:
                _apply_migration_sql(conn, cur, path.read_text())
                cur.execute(
                    f"INSERT INTO {version_table} (version) VALUES (%s)",
                    (n,),
                )
        conn.commit()


def bootstrap_from_schema(
    path: Path | None = None,
    *,
    version_table: str = ENGINE_VERSION_TABLE,
) -> int:
    """Apply a ``schema.sql`` snapshot in one shot — fast path for tests +
    cold starts. Idempotent: if any version is already recorded in
    ``version_table``, returns without touching the DB.

    Multi-statement SQL is sent via ``psql -f`` because psycopg's
    ``cur.execute()`` only sends one statement at a time under the extended
    query protocol. Returns the highest version after bootstrap, or 0 if the
    snapshot file is missing.
    """
    import subprocess
    import urllib.parse

    _check_identifier(version_table)
    snap = path or _ENGINE_SCHEMA_SNAPSHOT
    if not snap.exists():
        return 0
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) AS t", (f"public.{version_table}",))
        if cur.fetchone()["t"] is not None:
            cur.execute(f"SELECT COALESCE(MAX(version), 0) AS v FROM {version_table}")
            current = int(cur.fetchone()["v"])
            if current > 0:
                return current

    parsed = urllib.parse.urlparse(db_url())
    env = {
        "PGHOST": parsed.hostname or "localhost",
        "PGPORT": str(parsed.port or 5432),
        "PGUSER": parsed.username or "postgres",
        "PGPASSWORD": urllib.parse.unquote(parsed.password or ""),
        "PGDATABASE": parsed.path.lstrip("/") or "postgres",
    }
    proc = subprocess.run(
        ["psql", "-v", "ON_ERROR_STOP=1", "-q", "-f", str(snap)],
        env={**os.environ, **env},
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"psql failed applying {snap}: {proc.stderr.strip()}")

    with connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT COALESCE(MAX(version), 0) AS v FROM {version_table}")
        version = int(cur.fetchone()["v"])
    log.info("[queue_workflows.db] bootstrapped from %s (version=%d)", snap.name, version)
    return version


def current_schema_version(*, version_table: str = ENGINE_VERSION_TABLE) -> int:
    """Highest applied migration version for ``version_table``, or 0 when the
    table doesn't exist yet (a brand-new DB the orchestrator hasn't
    bootstrapped). Never raises ``UndefinedTable`` — the ``to_regclass``
    guard returns NULL instead, which we map to 0."""
    _check_identifier(version_table)
    from queue_workflows.dialect import get_dialect
    exists_sql = "SELECT " + get_dialect().table_exists("%s") + " AS t"
    with connection() as conn, conn.cursor() as cur:
        cur.execute(exists_sql, (version_table,))
        if cur.fetchone()["t"] is None:
            return 0
        cur.execute(f"SELECT COALESCE(MAX(version), 0) AS v FROM {version_table}")
        return int(cur.fetchone()["v"])


def wait_for_schema(
    min_version: int,
    *,
    version_table: str = ENGINE_VERSION_TABLE,
    timeout_s: float = 120.0,
    poll_s: float = 0.5,
    sleep_fn: Callable[[float], None] | None = None,
) -> int:
    """Block until ``version_table`` has migrations applied through
    ``min_version``.

    For processes that DON'T own the migration run (the claim workers /
    scheduler — only the orchestrator calls :func:`bootstrap`): poll
    :func:`current_schema_version` until it reaches ``min_version``, then
    return it. ``bootstrap()`` is NOT concurrency-safe (no advisory lock), so
    a non-owning process must WAIT for the schema rather than apply it itself.

    Raises ``TimeoutError`` if the version isn't reached within ``timeout_s``.
    ``sleep_fn`` is injectable for tests (default ``time.sleep``)."""
    import time as _time

    sleep = sleep_fn or _time.sleep
    deadline = _time.monotonic() + float(timeout_s)
    attempt = 0
    while True:
        current = current_schema_version(version_table=version_table)
        if current >= min_version:
            if attempt:
                log.info(
                    "[queue_workflows.db] schema ready (version=%d >= %d) "
                    "after %d poll(s)", current, min_version, attempt,
                )
            return current
        if _time.monotonic() >= deadline:
            raise TimeoutError(
                f"{version_table} {current} did not reach {min_version} "
                f"within {timeout_s:.0f}s — is the orchestrator's "
                f"bootstrap() running?"
            )
        attempt += 1
        sleep(float(poll_s))


def downgrade(
    *,
    to_version: int = 0,
    migrations_dir: Path | None = None,
    version_table: str = ENGINE_VERSION_TABLE,
) -> list[int]:
    """Roll back every migration whose version is greater than ``to_version``
    in ``migrations_dir`` / ``version_table``. Each step runs the paired
    ``NNNN_*.down.sql`` and removes the row from the version table.

    ``migrations_dir`` defaults to the engine chain for the active relational
    backend. Returns the list of reverted versions (highest-first). Raises
    ``RuntimeError`` when a step has no ``.down.sql`` file.
    """
    _check_identifier(version_table)
    from queue_workflows.dialect import get_dialect
    mdir = migrations_dir or _engine_migrations_dir()
    exists_sql = "SELECT " + get_dialect().table_exists("%s") + " AS t"
    reverted: list[int] = []
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(exists_sql, (version_table,))
            if cur.fetchone()["t"] is None:
                return reverted  # Nothing applied, nothing to revert.
        applied_desc = list(reversed(_applied_versions(conn, version_table)))
        for version in applied_desc:
            if version <= to_version:
                break
            down_path = _down_migration(mdir, version)
            if down_path is None:
                raise RuntimeError(
                    f"cannot revert version {version}: no "
                    f"{version:04d}_*.down.sql file found in {mdir}. "
                    f"Add a paired down migration or bump ``to_version``."
                )
            log.info("[queue_workflows.db] reverting %s", down_path.name)
            with conn.cursor() as cur:
                _apply_migration_sql(conn, cur, down_path.read_text())
                cur.execute(
                    f"DELETE FROM {version_table} WHERE version = %s",
                    (version,),
                )
            reverted.append(version)
        conn.commit()
    return reverted


def reset_for_tests() -> None:
    """TEST-ONLY: drop + recreate the public schema, then re-bootstrap the
    engine chain. Refuses to run if the DB *name* doesn't end in ``_test``."""
    import urllib.parse

    # Match on the parsed db NAME, not a suffix of the whole URL: a socket DSN
    # carries the host in a ``?host=/var/run/postgresql`` query string, so the
    # raw URL ends in the query, not ``_test`` — a whole-URL suffix check would
    # wrongly refuse a legitimate ``*_test`` DB (and, worse, could be fooled by
    # a non-test DB whose URL happened to end in ``_test``).
    db_name = urllib.parse.urlparse(db_url()).path.lstrip("/")
    if not db_name.endswith("_test"):
        raise RuntimeError(
            f"reset_for_tests() refused; DB name does not end in _test: {db_name!r}"
        )
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
            cur.execute("CREATE SCHEMA public")
        conn.commit()
    bootstrap()
