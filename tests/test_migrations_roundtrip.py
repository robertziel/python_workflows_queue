"""Migration up/down round-trip + the SQL-identifier guard on ``version_table``.

WHY these contracts matter:

* ``db.py``'s module docstring promises the chain reverses cleanly and that
  "both directions are idempotent", yet nothing in the suite ever invoked
  :func:`db.downgrade` or executed a single ``.down.sql``. A down file that
  references the wrong table/column, drops FKs in the wrong order, or is simply
  missing would ship undetected. ``test_full_chain_downgrade_to_zero_then_rebootstrap``
  exercises every ``.down.sql`` (head → 0) and then re-bootstraps (0 → head),
  proving the full chain is reversible AND replayable. ``test_downgrade_missing_down_file_raises``
  pins the missing-pair ``RuntimeError`` guard (db.py).

* ``version_table`` is an SQL *identifier* interpolated straight into statement
  text (it cannot be a ``%s`` placeholder), so :func:`db._check_identifier` is the
  ONLY thing standing between ``bootstrap(version_table=...)`` and SQL injection.
  ``test_version_table_identifier_is_validated`` pins the reject path that keeps a
  typo'd or hostile table name from smuggling SQL through the ledger name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from queue_workflows import db

# The engine tables the chain creates — every one must vanish at version 0 and
# reappear after a re-bootstrap. (Mirrors conftest's truncate list.)
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


def _head_version() -> int:
    """Highest forward-migration number on disk (the bootstrapped head)."""
    return max(
        int(p.stem.split("_", 1)[0])
        for p in db.ENGINE_MIGRATIONS_DIR.glob("*.sql")
        if not p.name.endswith(".down.sql") and p.name != "schema.sql"
    )


def _to_regclass(table: str):
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) AS t", (f"public.{table}",))
        return cur.fetchone()["t"]


def test_full_chain_downgrade_to_zero_then_rebootstrap() -> None:
    """The whole engine chain must reverse to nothing and rebuild to head.

    Runs every paired ``.down.sql`` (head → 0), asserting the reverted list is
    the full range highest-first and that every engine table is gone, then
    re-applies the forward chain (0 → head) and asserts the tables are back.
    This is the only coverage of ``downgrade()`` + the 16 ``.down.sql`` files;
    a broken down migration (wrong table, bad FK order) fails here.

    The schema is shared session state, so we RESTORE a clean head in
    ``finally`` no matter how the body exits — downstream tests must see the
    bootstrapped chain.
    """
    head = _head_version()
    assert head >= 16  # sanity: the chain we expect to be present
    assert db.current_schema_version() == head

    try:
        reverted = db.downgrade(to_version=0)
        # Highest-first, contiguous, the complete chain.
        assert reverted == sorted(range(1, head + 1), reverse=True)

        # Every engine table is gone at version 0.
        for table in _ENGINE_TABLES:
            assert _to_regclass(table) is None, f"{table} survived downgrade"
        assert db.current_schema_version() == 0

        # Forward chain replays cleanly back to head.
        db.bootstrap()
        assert db.current_schema_version() == head
        for table in _ENGINE_TABLES:
            assert _to_regclass(table) is not None, f"{table} missing after rebootstrap"
    finally:
        # Guarantee a head-version schema for the rest of the session no matter
        # how the body exited. The idempotent forward bootstrap is sufficient
        # here (no need for the heavier DROP+recreate of ``reset_for_tests``).
        db.bootstrap()
        assert db.current_schema_version() == head


def test_downgrade_missing_down_file_raises(tmp_path: Path) -> None:
    """``downgrade`` must hard-fail (not silently skip) a step with no ``.down.sql``.

    Builds a throwaway 2-step chain in its own dir + ledger where step 2 has no
    paired down file, bootstraps it, then asserts the documented ``RuntimeError``
    names the offending version. Uses a throwaway ``version_table`` so the
    engine ledger is untouched.
    """
    ver_table = "qw_rt_ver"
    (tmp_path / "0001_a.sql").write_text(
        "CREATE TABLE qw_rt_a (id INTEGER PRIMARY KEY);"
    )
    (tmp_path / "0001_a.down.sql").write_text("DROP TABLE IF EXISTS qw_rt_a;")
    # NOTE: deliberately NO 0002_b.down.sql — this is the gap under test.
    (tmp_path / "0002_b.sql").write_text(
        "CREATE TABLE qw_rt_b (id INTEGER PRIMARY KEY);"
    )

    try:
        db.bootstrap(migrations_dir=tmp_path, version_table=ver_table)
        assert db.current_schema_version(version_table=ver_table) == 2

        with pytest.raises(RuntimeError, match=r"no .*0002.*down\.sql"):
            db.downgrade(
                to_version=0,
                migrations_dir=tmp_path,
                version_table=ver_table,
            )
    finally:
        with db.connection() as conn, conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS qw_rt_a")
            cur.execute("DROP TABLE IF EXISTS qw_rt_b")
            cur.execute(f"DROP TABLE IF EXISTS {ver_table}")
            conn.commit()


@pytest.mark.parametrize(
    "bad",
    ["evil; DROP TABLE x", "has-dash", "1leading", "has space", "a.b", ""],
)
def test_version_table_identifier_is_validated(bad: str) -> None:
    """A non-identifier ``version_table`` must be rejected BEFORE any SQL runs.

    ``version_table`` is interpolated into raw statement text (it cannot be a
    ``%s`` bind), so this guard is the sole barrier against SQL injection via
    ``bootstrap(version_table=...)`` / ``current_schema_version(version_table=...)``.
    Each bad value — semicolon-injection, dashes, leading digit, whitespace,
    dotted, empty — must raise ``ValueError``.
    """
    with pytest.raises(ValueError, match="invalid version_table identifier"):
        db.bootstrap(version_table=bad)
    with pytest.raises(ValueError, match="invalid version_table identifier"):
        db.current_schema_version(version_table=bad)


def test_check_identifier_passes_plain_name() -> None:
    """The positive control: a plain unqualified identifier is accepted and
    returned verbatim so call sites can interpolate it."""
    assert db._check_identifier("queue_schema_version") == "queue_schema_version"


# ── reset_for_tests destructive-guard (keys on the parsed DB *name*) ─────────


def _forbid_destruction(*_a, **_k):
    """Tripwire: stand in for ``db.connection`` so the schema-dropping body can
    never run during these guard tests (and so a guard that wrongly lets a
    non-``_test`` DB through fails LOUDLY instead of wiping the session schema)."""
    raise AssertionError(
        "reset_for_tests reached the destructive body — guard let a "
        "non-_test (or query-string) DB through"
    )


def test_reset_for_tests_accepts_socket_test_dsn(monkeypatch) -> None:
    """The guard keys on the parsed DB *name*, not a suffix of the whole URL, so a
    socket DSN whose db name ends in ``_test`` but whose URL ends in a
    ``?host=/var/run/postgresql`` query string is ACCEPTED.

    Regression: the old ``db_url().rstrip('/').endswith('_test')`` check matched
    the whole URL and so wrongly REFUSED exactly this (legitimate) DSN. We make the
    destructive body inert (``connection`` raises a sentinel) and assert execution
    REACHED it — i.e. the guard passed — without touching any real database.
    """
    reached = RuntimeError("reached-destructive-body")

    def _sentinel(*_a, **_k):
        raise reached

    monkeypatch.setattr(
        db, "db_url",
        lambda: "postgresql://u@/queue_workflows_test?host=/var/run/postgresql",
    )
    monkeypatch.setattr(db, "connection", _sentinel)
    with pytest.raises(RuntimeError, match="reached-destructive-body"):
        db.reset_for_tests()


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://u@/queue_workflows_prod?host=/var/run/postgresql",
        "postgresql://u:p@host:5432/ai_leads",
        "postgresql://u@/queue_workflows_test_backup",   # name ends in _backup
        "postgresql://u@host:5432/prod?options=db_test",  # _test only in the query
    ],
)
def test_reset_for_tests_refuses_non_test_db(monkeypatch, url: str) -> None:
    """A DB whose parsed *name* doesn't end in ``_test`` must be refused before any
    DROP runs — regardless of URL shape (query string, creds, port). On a
    schema-dropping helper the negatives matter most: a ``*_test_backup`` or a prod
    DB with ``_test`` hiding in the query string is NOT a test DB and must be
    rejected. ``connection`` is a tripwire so a broken guard can't wipe a real DB.
    """
    monkeypatch.setattr(db, "db_url", lambda: url)
    monkeypatch.setattr(db, "connection", _forbid_destruction)
    with pytest.raises(RuntimeError, match="refused"):
        db.reset_for_tests()
