"""Boundary guard for the client/conductor split — the load-bearing invariant.

The repo ships two distributions: the CLIENT (``queue_workflows``, per-project
data plane) and the CONDUCTOR (``queue_workflows_conductor``, control plane). The
dependency edge points **one way — conductor → client** — so the client (which
runs inside every worker/orchestrator) never drags in the conductor. If a client
module ever imported the conductor, the edge would become a cycle and the
per-project client could no longer be deployed without the conductor.

This mirrors ``test_no_ai_leads_import`` (no host coupling) for the new seam.
"""

from __future__ import annotations

import ast
from pathlib import Path

import queue_workflows
import queue_workflows_conductor

_CONDUCTOR_PKG = "queue_workflows_conductor"
_CLIENT_DIR = Path(queue_workflows.__file__).resolve().parent
_CONDUCTOR_DIR = Path(queue_workflows_conductor.__file__).resolve().parent


def _imports_of(path: Path, *, root_name: str) -> list[str]:
    """Every import statement in ``path`` whose top-level module == ``root_name``
    (anywhere — module scope OR inside a function body, since ``ast.walk`` is
    recursive)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == root_name:
                    hits.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 0 and mod.split(".")[0] == root_name:
                names = ", ".join(a.name for a in node.names)
                hits.append(f"from {mod} import {names}")
    return hits


def test_client_never_imports_conductor() -> None:
    offenders: dict[str, list[str]] = {}
    for path in sorted(_CLIENT_DIR.rglob("*.py")):
        bad = _imports_of(path, root_name=_CONDUCTOR_PKG)
        if bad:
            offenders[str(path.relative_to(_CLIENT_DIR))] = bad
    assert not offenders, (
        "client package queue_workflows must NOT import the conductor "
        f"({_CONDUCTOR_PKG}) — the edge is one-way (conductor -> client). "
        f"Offenders: {offenders}"
    )


def test_conductor_depends_on_client() -> None:
    # The edge must actually EXIST in the intended direction: the conductor
    # consumes the client's primitives. (If this ever stops being true, the
    # conductor has no reason to be a separate package.)
    any_client_import = False
    for path in sorted(_CONDUCTOR_DIR.rglob("*.py")):
        if _imports_of(path, root_name="queue_workflows"):
            any_client_import = True
            break
    assert any_client_import, (
        "conductor package is expected to import the client (queue_workflows); "
        "the dependency edge conductor -> client should be real"
    )
