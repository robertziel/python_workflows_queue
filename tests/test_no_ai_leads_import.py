"""Inversion guard — the package must import NOTHING from a host application.

Walks the AST of every ``queue_workflows/*.py`` module and asserts no
``import workflows...`` / ``from workflows... import ...`` statement exists.
This is the load-bearing invariant of the Phase-6 extraction (plan §2b): the
engine is host-agnostic; the host wires into it via the public config hooks,
never the other way round.

Also asserts the package imports clean in this process (it would already, since
conftest imported it — but make it explicit + independent of test ordering).
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import queue_workflows

#: Module-name prefixes that would betray a host coupling. ``workflows`` is the
#: ai_leads source package; ``app`` / ``backend`` are other ai_leads roots.
_FORBIDDEN_PREFIXES = ("workflows", "app", "backend")

_PKG_DIR = Path(queue_workflows.__file__).resolve().parent


def _python_files() -> list[Path]:
    return sorted(_PKG_DIR.rglob("*.py"))


def _forbidden_imports(path: Path) -> list[str]:
    """Return any forbidden import statement text found in ``path``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_PREFIXES:
                    bad.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            # ``from . import x`` (level>0, no module) is intra-package — fine.
            if node.level == 0 and root in _FORBIDDEN_PREFIXES:
                names = ", ".join(a.name for a in node.names)
                bad.append(f"from {mod} import {names}")
    return bad


def test_no_module_imports_a_host_package():
    offenders: dict[str, list[str]] = {}
    for path in _python_files():
        bad = _forbidden_imports(path)
        if bad:
            offenders[str(path.relative_to(_PKG_DIR))] = bad
    assert not offenders, (
        "queue_workflows must import nothing from a host application "
        f"({_FORBIDDEN_PREFIXES}); offenders: {offenders}"
    )


def test_every_submodule_imports_clean():
    """Import every submodule — catches a coupling that only surfaces at import
    time (a runtime ``from workflows...`` inside a function body would NOT be
    caught by the AST top-level scan, but a module-scope one would fail here)."""
    for modinfo in pkgutil.walk_packages(
        queue_workflows.__path__, prefix="queue_workflows."
    ):
        importlib.import_module(modinfo.name)
