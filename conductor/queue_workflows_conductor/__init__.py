"""``queue_workflows_conductor`` — the conductor (control-plane) package.

A **separate distribution** (``queue-workflows-conductor``) that depends on
``queue-workflows-client`` and consumes its primitives (``node_queue``,
``worker_control``). The dependency edge points one way — conductor → client —
so the per-project client (worker / orchestrator) never imports the conductor.

Today it ships the read-only fleet view (:mod:`queue_workflows_conductor.conductor`,
console ``queue-conductor``). The networked multi-DB daemon + web UI + inference
proxy accrete here as separate, deployable-apart-from-the-client surfaces.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
