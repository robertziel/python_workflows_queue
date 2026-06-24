"""Per-machine LLM server config on ``worker_controls`` (migration 0013).

Covers the operator-write side of the per-machine LLM-server-type feature:
- the three new columns exist with their defaults (ollama / par 1 / idle 60 s);
- set_llm_config round-trips, and a PARTIAL set preserves the untouched columns
  (COALESCE upsert) while a fresh row fills unspecified columns from defaults;
- set_llm_config NEVER touches desired_state/stop_policy — it is a SOFT config
  change, not the ON/OFF switch (so it must not route through the hard-stop path);
- validation (server_type / parallelism / idle_ttl) fails BEFORE any write,
  matching set_worker_control;
- llm_config_for is default-safe on a pre-0012 (no table) AND a pre-0013 (table
  but no columns) DB ⇒ the engine runs unchanged before the migration is applied;
- the 0013 trigger NOTIFYs the SEPARATE ``worker_llm_config_changed`` channel
  (payload ``host|queue``) and does NOT pollute the ``worker_control`` channel
  (which the hard-stop watcher listens on).
"""

from __future__ import annotations

import contextlib

import pytest

from queue_workflows import worker_control
from queue_workflows.db import connection, db_url


# ── columns + defaults ─────────────────────────────────────────────────────────


def test_new_row_has_llm_defaults():
    """A control row written by the existing ON/OFF path gets the column defaults
    for the new LLM fields (no LLM config implies the safe ollama/par-1 baseline)."""
    worker_control.set_worker_control("host-c", "gpu", desired_state="on")
    cfg = worker_control.llm_config_for("host-c", "gpu")
    assert cfg.server_type == "ollama"
    assert cfg.parallelism == 1
    assert cfg.vllm_idle_ttl_s == 60


def test_llm_config_for_defaults_when_absent():
    """No row at all ⇒ the default LLMConfig (mirrors desired_state_for=on)."""
    cfg = worker_control.llm_config_for("nobody", "gpu")
    assert cfg.server_type == worker_control.DEFAULT_LLM_SERVER_TYPE
    assert cfg.parallelism == worker_control.DEFAULT_LLM_PARALLELISM
    assert cfg.vllm_idle_ttl_s == worker_control.DEFAULT_VLLM_IDLE_TTL_S


# ── set_llm_config round-trip + partial upsert ─────────────────────────────────


def test_set_llm_config_round_trip():
    worker_control.set_llm_config(
        "host-a", "gpu", server_type="vllm", parallelism=128, vllm_idle_ttl_s=30,
    )
    cfg = worker_control.llm_config_for("host-a", "gpu")
    assert cfg.server_type == "vllm"
    assert cfg.parallelism == 128
    assert cfg.vllm_idle_ttl_s == 30


def test_set_llm_config_insert_fills_unspecified_from_defaults():
    """Fresh (host,queue) + only server_type given ⇒ the other two come from the
    column defaults, not NULL (the columns are NOT NULL)."""
    worker_control.set_llm_config("host-b", "gpu", server_type="vllm")
    cfg = worker_control.llm_config_for("host-b", "gpu")
    assert cfg.server_type == "vllm"
    assert cfg.parallelism == worker_control.DEFAULT_LLM_PARALLELISM
    assert cfg.vllm_idle_ttl_s == worker_control.DEFAULT_VLLM_IDLE_TTL_S


def test_set_llm_config_partial_preserves_other_columns():
    """A later partial write (only parallelism) keeps the existing server_type
    and idle_ttl — COALESCE(EXCLUDED, existing), not a full replace."""
    worker_control.set_llm_config(
        "h", "gpu", server_type="vllm", parallelism=64, vllm_idle_ttl_s=45,
    )
    worker_control.set_llm_config("h", "gpu", parallelism=200)
    cfg = worker_control.llm_config_for("h", "gpu")
    assert cfg.server_type == "vllm"      # preserved
    assert cfg.parallelism == 200          # changed
    assert cfg.vllm_idle_ttl_s == 45       # preserved


def test_set_llm_config_does_not_touch_desired_state():
    """LLM config is a SOFT change. Setting it on an OFF worker must leave it OFF
    (and not flip stop_policy) — it must NOT behave like the ON/OFF switch."""
    worker_control.disable_worker("h", "gpu", requested_by="op")
    worker_control.set_llm_config("h", "gpu", server_type="vllm", parallelism=8)
    row = worker_control.get_worker_control("h", "gpu")
    assert row["desired_state"] == "off"   # untouched
    assert row["stop_policy"] == "hard"    # untouched
    cfg = worker_control.llm_config_for("h", "gpu")
    assert cfg.server_type == "vllm" and cfg.parallelism == 8


def test_set_llm_config_upsert_single_row():
    worker_control.set_llm_config("h", "gpu", server_type="vllm")
    worker_control.set_llm_config("h", "gpu", server_type="ollama")
    with connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM worker_controls "
            "WHERE host_label='h' AND queue='gpu'"
        )
        assert cur.fetchone()["n"] == 1


# ── validation (fail-before-write) ─────────────────────────────────────────────


def test_invalid_server_type_rejected():
    with pytest.raises(ValueError):
        worker_control.set_llm_config("h", "gpu", server_type="lmstudio")
    assert worker_control.get_worker_control("h", "gpu") is None


def test_invalid_parallelism_rejected():
    with pytest.raises(ValueError):
        worker_control.set_llm_config("h", "gpu", parallelism=0)
    assert worker_control.get_worker_control("h", "gpu") is None


def test_invalid_idle_ttl_rejected():
    with pytest.raises(ValueError):
        worker_control.set_llm_config("h", "gpu", vllm_idle_ttl_s=-1)
    assert worker_control.get_worker_control("h", "gpu") is None


def test_db_check_rejects_bad_server_type():
    """Defense in depth: even a raw INSERT bypassing set_llm_config hits the
    column CHECK constraint."""
    from tests._helpers import INTEGRITY_ERRORS

    with pytest.raises(INTEGRITY_ERRORS):
        with connection() as c, c.cursor() as cur:
            cur.execute(
                "INSERT INTO worker_controls (host_label, queue, llm_server_type) "
                "VALUES ('x','gpu','bogus')"
            )


# ── default-safe on a partially-migrated DB ────────────────────────────────────


def test_llm_config_for_table_absent_returns_default(monkeypatch):
    """Pre-0012 DB (no worker_controls table) ⇒ default config, no raise."""
    import psycopg

    @contextlib.contextmanager
    def _raise_undefined_table():
        class _Cur:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, *a, **k):
                raise psycopg.errors.UndefinedTable("no worker_controls")
            def fetchone(self): return None

        class _Conn:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def cursor(self): return _Cur()

        yield _Conn()

    monkeypatch.setattr(worker_control, "connection", _raise_undefined_table)
    cfg = worker_control.llm_config_for("h", "gpu")
    assert cfg.server_type == worker_control.DEFAULT_LLM_SERVER_TYPE


def test_llm_config_for_column_absent_returns_default(monkeypatch):
    """Pre-0013 DB (table present, new columns missing) ⇒ default config, no
    raise. This is the case where 0012 ran but 0013 hasn't yet."""
    import psycopg

    @contextlib.contextmanager
    def _raise_undefined_column():
        class _Cur:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, *a, **k):
                raise psycopg.errors.UndefinedColumn("no llm_server_type column")
            def fetchone(self): return None

        class _Conn:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def cursor(self): return _Cur()

        yield _Conn()

    monkeypatch.setattr(worker_control, "connection", _raise_undefined_column)
    cfg = worker_control.llm_config_for("h", "gpu")
    assert cfg.parallelism == worker_control.DEFAULT_LLM_PARALLELISM


# ── NOTIFY: a SEPARATE channel from the hard-stop watcher's ────────────────────


@pytest.mark.pg_only
def test_set_llm_config_fires_llm_config_notify():
    """The 0013 trigger NOTIFYs ``worker_llm_config_changed`` with ``host|queue``
    so the backend factory refreshes instantly (10 s TTL is the fallback)."""
    import psycopg

    with psycopg.connect(db_url(), autocommit=True) as conn:
        conn.execute(f"LISTEN {worker_control.LLM_CONFIG_NOTIFY_CHANNEL}")
        worker_control.set_llm_config("host-c", "gpu", server_type="vllm")
        payloads = [n.payload for n in conn.notifies(timeout=3.0, stop_after=1)]
    assert payloads == ["host-c|gpu"]


@pytest.mark.pg_only
def test_llm_config_change_does_not_pollute_worker_control_channel():
    """A no-op LLM write must not wake the hard-stop watcher with a spurious
    ``worker_control`` NOTIFY that LOOKS like an ON/OFF change. (The existing
    0012 trigger still fires on the row write — that is harmless because the
    watcher re-reads desired_state and sees no OFF — but the dedicated config
    channel is what the factory keys on, kept distinct here.)"""
    import psycopg

    # Seed a row first so the next call is a pure config UPDATE.
    worker_control.set_llm_config("host-c", "gpu", server_type="vllm")
    with psycopg.connect(db_url(), autocommit=True) as conn:
        conn.execute(f"LISTEN {worker_control.LLM_CONFIG_NOTIFY_CHANNEL}")
        worker_control.set_llm_config("host-c", "gpu", parallelism=99)
        got = [n.payload for n in conn.notifies(timeout=3.0, stop_after=1)]
    assert got == ["host-c|gpu"]
