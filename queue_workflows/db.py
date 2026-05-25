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

import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from queue_workflows import config as _config

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
#: The engine's own migration chain (queue tables only). Public via
#: ``queue_workflows.migrations.dir()``.
ENGINE_MIGRATIONS_DIR = _HERE / "migrations"
ENGINE_VERSION_TABLE = "queue_schema_version"
_ENGINE_SCHEMA_SNAPSHOT = ENGINE_MIGRATIONS_DIR / "schema.sql"

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
    """Drain the pool. Used by orchestrator shutdown + test teardown."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Borrow a pooled connection. Auto-commits on clean exit, rolls
    back on exception (psycopg's default for ``with conn:``)."""
    with get_pool().connection() as conn:
        yield conn


@contextmanager
def cursor() -> Iterator[psycopg.Cursor]:
    """Convenience: borrow a connection + cursor in one block."""
    with connection() as conn:
        with conn.cursor() as cur:
            yield cur


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


def _applied_versions(conn: psycopg.Connection, version_table: str) -> list[int]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT version FROM {version_table} ORDER BY version")
        return [r["version"] for r in cur.fetchall()]


def bootstrap(
    *,
    migrations_dir: Path = ENGINE_MIGRATIONS_DIR,
    version_table: str = ENGINE_VERSION_TABLE,
) -> None:
    """Apply pending migrations from ``migrations_dir`` against the version
    ledger ``version_table``. Idempotent — safe to call on every boot.

    The engine's own bootstrap defaults to its migration dir +
    ``queue_schema_version``. A host applies its domain chain by calling this
    a SECOND time with its own dir + ``schema_version``.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {version_table} ("
                "  version INTEGER PRIMARY KEY,"
                "  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
        applied = set(_applied_versions(conn, version_table))

        for n, path in _forward_migrations(migrations_dir):
            if n in applied:
                continue
            log.info("[queue_workflows.db] applying %s", path.name)
            with conn.cursor() as cur:
                cur.execute(path.read_text())
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
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) AS t", (f"public.{version_table}",))
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
    migrations_dir: Path = ENGINE_MIGRATIONS_DIR,
    version_table: str = ENGINE_VERSION_TABLE,
) -> list[int]:
    """Roll back every migration whose version is greater than ``to_version``
    in ``migrations_dir`` / ``version_table``. Each step runs the paired
    ``NNNN_*.down.sql`` and removes the row from the version table.

    Returns the list of reverted versions (highest-first). Raises
    ``RuntimeError`` when a step has no ``.down.sql`` file.
    """
    reverted: list[int] = []
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s) AS t", (f"public.{version_table}",))
            if cur.fetchone()["t"] is None:
                return reverted  # Nothing applied, nothing to revert.
        applied_desc = list(reversed(_applied_versions(conn, version_table)))
        for version in applied_desc:
            if version <= to_version:
                break
            down_path = _down_migration(migrations_dir, version)
            if down_path is None:
                raise RuntimeError(
                    f"cannot revert version {version}: no "
                    f"{version:04d}_*.down.sql file found in {migrations_dir}. "
                    f"Add a paired down migration or bump ``to_version``."
                )
            log.info("[queue_workflows.db] reverting %s", down_path.name)
            with conn.cursor() as cur:
                cur.execute(down_path.read_text())
                cur.execute(
                    f"DELETE FROM {version_table} WHERE version = %s",
                    (version,),
                )
            reverted.append(version)
        conn.commit()
    return reverted


def reset_for_tests() -> None:
    """TEST-ONLY: drop + recreate the public schema, then re-bootstrap the
    engine chain. Refuses to run if the DB name doesn't end in ``_test``."""
    if not db_url().rstrip("/").endswith("_test"):
        raise RuntimeError(
            f"reset_for_tests() refused; DB url does not end in _test: {db_url()!r}"
        )
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
            cur.execute("CREATE SCHEMA public")
        conn.commit()
    bootstrap()
