"""The engine's migration chain (queue tables only).

The SQL files (``NNNN_*.sql`` + paired ``.down.sql``) in this package ARE the
engine's schema. They're shipped as package data (see ``pyproject.toml``'s
``force-include``) so an installed wheel can bootstrap a Postgres without the
source tree. :func:`dir` returns the directory so a host can also feed it to
``db.bootstrap(migrations_dir=...)``.
"""

from __future__ import annotations

from pathlib import Path


def dir() -> Path:
    """Return the directory holding the engine's ``NNNN_*.sql`` migrations."""
    return Path(__file__).resolve().parent
