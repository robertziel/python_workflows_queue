"""The package root's public API (``__all__``) must list every ``set_*`` /
``register_*`` hook a host wires — a hook that's defined but missing from
``__all__`` is a silent export bug (``from queue_workflows import *`` skips it,
and it reads as private).
"""

from __future__ import annotations

import queue_workflows
from queue_workflows import get_config


def test_all_names_are_resolvable() -> None:
    for name in queue_workflows.__all__:
        assert hasattr(queue_workflows, name), f"{name} in __all__ but not defined"


def test_register_pool_handler_is_exported() -> None:
    # The shared-GPU-pool handler registrar is a public host hook (it has a
    # docstring contract and mutates the config) — it must be in __all__.
    assert "register_pool_handler" in queue_workflows.__all__
    assert callable(queue_workflows.register_pool_handler)


def test_register_pool_handler_registers_into_config() -> None:
    def _handler(*, inputs, output_dir, params):  # pragma: no cover - not invoked
        return {}

    queue_workflows.register_pool_handler("unit_test_handler", _handler)
    assert get_config().gpu_pool_handlers["unit_test_handler"] is _handler
