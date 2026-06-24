"""Engine-only pytest fixtures (plan §5e).

Self-contained: forces a ``*_test`` DB and applies the ENGINE migration chain
only (the queue tables — NO host/domain tables). It must NOT reference any host
table, so the autouse truncate list is trimmed to the engine tables.

DB selection:
  * ``QUEUE_WORKFLOWS_TEST_DB_URL`` if set (preferred — points straight at a
    ``queue_workflows_test`` DB), else
  * ``AI_LEADS_DB_URL`` with its db-name suffixed ``_test`` (so a checkout that
    already has the ai_leads env reaches a sibling ``<db>_test`` DB).

The engine is configured here (``queue_workflows.configure(db_url_env=...)``)
so ``db.db_url()`` reads the test DSN. A tiny fake node module + fake ModelSpec
keep the engine suite domain-free (the domain-coupled tests inject those).
"""

from __future__ import annotations

import os
import urllib.parse
from typing import Iterator

import pytest

# Disable side-effect threads the engine would otherwise spin up under tests
# (mirrors ai_leads' conftest env gates — same env names, byte-compat).
os.environ.setdefault("AI_LEADS_DISABLE_HW_METRICS", "1")
os.environ.setdefault("AI_LEADS_DISABLE_WORKER_HEARTBEAT", "1")
os.environ.setdefault("AI_LEADS_DISABLE_GPU_IDLE_REAPER", "1")
os.environ.setdefault("AI_LEADS_DISABLE_LLM_SUPERVISOR", "1")
os.environ.setdefault("AI_LEADS_DISABLE_LLM_CONFIG_LISTENER", "1")

# The env var the engine reads its DSN from, for THIS test session.
_TEST_DB_ENV = "QUEUE_WORKFLOWS_TEST_DB_URL"


def _resolve_test_db_url() -> str:
    """Resolve the test DSN: ``QUEUE_WORKFLOWS_TEST_DB_URL`` verbatim, else
    ``AI_LEADS_DB_URL`` with a ``_test`` suffix on the db name. Skip the whole
    suite if neither is set (Postgres is required)."""
    raw = os.environ.get(_TEST_DB_ENV)
    if raw:
        return raw
    base = os.environ.get("AI_LEADS_DB_URL")
    if not base:
        pytest.skip(
            f"set {_TEST_DB_ENV} (or AI_LEADS_DB_URL) to a Postgres DSN; "
            "engine tests need Postgres"
        )
    parsed = urllib.parse.urlparse(base)
    db_name = parsed.path.lstrip("/")
    if not db_name.endswith("_test"):
        parsed = parsed._replace(path="/" + db_name + "_test")
    url = urllib.parse.urlunparse(parsed)
    os.environ[_TEST_DB_ENV] = url
    return url


# SQLite cross-backend test mode: ``QUEUE_WORKFLOWS_TEST_SQLITE=1`` runs the WHOLE
# engine suite against a throwaway SQLite file instead of Postgres — the "engine
# runs on SQLite" proof. (NOTIFY/LISTEN tests skip; see pytest_collection_modifyitems.)
_SQLITE_MODE = bool(os.environ.get("QUEUE_WORKFLOWS_TEST_SQLITE"))
_SQLITE_ENV = "QUEUE_WORKFLOWS_TEST_SQLITE_PATH"

import queue_workflows  # noqa: E402

if _SQLITE_MODE:
    import tempfile  # noqa: E402
    _fd, _sqlite_file = tempfile.mkstemp(suffix="_test.db", prefix="qw_sqlite_")
    os.close(_fd)
    os.unlink(_sqlite_file)  # let SQLite create it fresh
    os.environ[_SQLITE_ENV] = _sqlite_file
    _TEST_DB_URL = _sqlite_file
    queue_workflows.configure(db_backend="sqlite", db_url_env=_SQLITE_ENV)
else:
    _TEST_DB_URL = _resolve_test_db_url()
    # Point the engine at the test DSN BEFORE any engine module reads it.
    queue_workflows.configure(db_url_env=_TEST_DB_ENV)

import psycopg  # noqa: E402

from queue_workflows import db as engine_db  # noqa: E402

# Engine tables — the entire schema the engine owns (no host/domain tables).
_ENGINE_TABLES = (
    "workflow_node_jobs",
    "workflow_run_files",
    "workflow_dispatch_events",
    "workflow_node_events",
    "workflow_input_submissions",
    "workflow_runs",
    "worker_heartbeats",
    "worker_controls",
    "ingest_jobs",
)


def _ensure_test_database_exists() -> None:
    """Create the test DB via a maintenance connection if it's missing."""
    parsed = urllib.parse.urlparse(_TEST_DB_URL)
    test_db_name = parsed.path.lstrip("/")
    if not test_db_name.endswith("_test"):
        raise RuntimeError(
            f"refusing to bootstrap a non-_test DB: {test_db_name!r}"
        )
    maint_url = urllib.parse.urlunparse(parsed._replace(path="/postgres"))
    with psycopg.connect(maint_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (test_db_name,),
            )
            if cur.fetchone():
                return
            cur.execute(f'CREATE DATABASE "{test_db_name}"')


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_test_db() -> Iterator[None]:
    """Create the test DB if missing + apply the ENGINE migration chain only."""
    if not _SQLITE_MODE:
        _ensure_test_database_exists()
    engine_db.bootstrap()  # engine dir + queue_schema_version (per-backend)
    yield
    engine_db.close_pool()


@pytest.fixture(autouse=True)
def _truncate_between_tests(_bootstrap_test_db) -> Iterator[None]:
    """Wipe the engine tables between tests so each sees a clean DB."""
    yield
    if _SQLITE_MODE:
        # A test may have repointed db_backend (e.g. backend-factory tests); make
        # sure THIS truncate connects to the SQLite store, not a stray pg pool.
        queue_workflows.configure(db_backend="sqlite", db_url_env=_SQLITE_ENV)
    with engine_db.connection() as conn, conn.cursor() as cur:
        if _SQLITE_MODE:
            # SQLite has no TRUNCATE; DELETE in FK-safe order (children first).
            for tbl in _ENGINE_TABLES:
                cur.execute(f"DELETE FROM {tbl}")
        else:
            cur.execute(
                "TRUNCATE " + ", ".join(_ENGINE_TABLES) + " RESTART IDENTITY CASCADE"
            )
        conn.commit()


@pytest.fixture(autouse=True)
def _reset_engine_config() -> Iterator[None]:
    """Reset injected config (node resolver, registrar, workflow provider,
    ingest map/schedule) between tests, but KEEP the test backend wired so the
    connection keeps working. A test that wires a hook doesn't leak into the next."""
    yield
    from queue_workflows import config as _cfg
    _cfg.reset_for_tests()
    if _SQLITE_MODE:
        queue_workflows.configure(db_backend="sqlite", db_url_env=_SQLITE_ENV)
    else:
        queue_workflows.configure(db_url_env=_TEST_DB_ENV)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "pg_only: test exercises a Postgres-only mechanic (LISTEN/NOTIFY "
        "delivery, FOR UPDATE SKIP LOCKED concurrency, DROP CONSTRAINT via "
        "ALTER) — skipped in SQLite test mode (QUEUE_WORKFLOWS_TEST_SQLITE=1).",
    )


def pytest_collection_modifyitems(config, items):  # noqa: D401
    # Intent-based skip (NOT a substring of a component name — a `listen`
    # substring would wrongly skip the input-LISTENer reclaim INVARIANTS, which
    # exercise the poll+claim+reclaim path SQLite relies on and DO pass there).
    if not _SQLITE_MODE:
        return
    skip = pytest.mark.skip(reason="pg-only mechanic (LISTEN/NOTIFY/SKIP LOCKED) — N/A on SQLite")
    for item in items:
        if "pg_only" in item.keywords:
            item.add_marker(skip)
