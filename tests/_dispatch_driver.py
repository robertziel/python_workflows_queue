"""Synchronous dispatcher tick driver for tests.

The production ``NodePool`` runs ``_tick`` on a 0.5 s background thread. This
driver gives the test process a synchronous handle: the ``NodePool`` underneath
is the real one — same code path, just driven manually.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from queue_workflows import node_pool as _np


class DispatchDriver:
    """Test-only NodePool wrapper. Constructs the pool without firing the
    dispatch thread; tests advance state via ``tick()`` or
    ``drain_until_quiescent()``."""

    def __init__(self, *, register_builtins=None) -> None:
        self.pool = _np.NodePool(register_builtins=register_builtins)

    def tick(self) -> None:
        self.pool._tick()

    def drain_until_quiescent(self, *, max_iter: int = 50) -> int:
        from queue_workflows import node_queue
        processed = 0
        for _ in range(max_iter):
            before = node_queue.list_unprocessed_dispatch_events()
            if not before:
                return processed
            before_ids = {e["id"] for e in before}
            self.pool._drain_dispatch_events()
            after_ids = {
                e["id"]
                for e in node_queue.list_unprocessed_dispatch_events()
            }
            this_iter = len(before_ids - after_ids)
            processed += this_iter
            if this_iter == 0 and after_ids == before_ids:
                return processed
        return processed


@pytest.fixture
def dispatch_driver() -> Iterator[DispatchDriver]:
    yield DispatchDriver()
