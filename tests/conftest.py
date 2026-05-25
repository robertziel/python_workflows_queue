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


_TEST_DB_URL = _resolve_test_db_url()

# Point the engine at the test DSN BEFORE any engine module reads it.
import queue_workflows  # noqa: E402

queue_workflows.configure(db_url_env=_TEST_DB_ENV)

import psycopg  # noqa: E402

from queue_workflows import db as engine_db  # noqa: E402

# Engine tables — the entire schema the engine owns (no host/domain tables).
_ENGINE_TABLES = (
    "workflow_node_jobs",
    "workflow_run_files",
    "workflow_dispatch_events",
    "workflow_input_submissions",
    "workflow_runs",
    "worker_heartbeats",
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
    _ensure_test_database_exists()
    engine_db.bootstrap()  # engine dir + queue_schema_version
    yield
    engine_db.close_pool()


@pytest.fixture(autouse=True)
def _truncate_between_tests(_bootstrap_test_db) -> Iterator[None]:
    """Wipe the engine tables between tests so each sees a clean DB."""
    yield
    with engine_db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "TRUNCATE " + ", ".join(_ENGINE_TABLES) + " RESTART IDENTITY CASCADE"
        )
        conn.commit()


@pytest.fixture(autouse=True)
def _reset_engine_config() -> Iterator[None]:
    """Reset injected config (node resolver, registrar, workflow provider,
    ingest map/schedule) between tests, but KEEP the test DSN wired so the
    pool keeps working. A test that wires a hook doesn't leak into the next."""
    yield
    from queue_workflows import config as _cfg
    _cfg.reset_for_tests()
    queue_workflows.configure(db_url_env=_TEST_DB_ENV)
