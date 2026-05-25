"""Engine configuration + the dependency-inversion seams.

``queue_workflows`` is a host-agnostic Postgres-as-queue engine. Everything
that used to couple it to a specific application (ai_leads) is now an
*injected* hook held on a process-wide :class:`EngineConfig` singleton. A
host wires those hooks once at startup (``queue_workflows.configure(...)`` +
the ``set_*`` / ``register_*`` helpers) before launching a claim worker /
scheduler / orchestrator.

Every hook has a **safe default** so ``import queue_workflows`` +
``configure()`` + a reachable Postgres is enough to run the engine
standalone (no host wiring required):

  * ``db_url_env``           — env var holding the DSN (default
                               ``AI_LEADS_DB_URL`` for byte-compat with the
                               existing ai_leads deploy; other projects pass
                               their own).
  * ``video_model_ids``      — GPU models on the tight render budget (empty).
  * ``node_module_package``  — dotted package the node-module resolver imports
                               under (empty → the stored ``node_module`` value
                               is imported as a fully-qualified module).
  * ``container_prefix``     — cgroup-attribution container-name prefix
                               (default ``ai_leads-``).
  * the workflow provider    — ``load_workflow`` / ``pipeline_schema`` /
                               ``resolve_ref`` the dispatcher reads the DAG
                               from (defaults raise / use the built-in
                               ``refs.resolve_ref``).
  * the builtin-model registrar — the empty-registry re-registration fallback
                               (default no-op).
  * the ingest dispatch map + schedule — periodic-work callables + cron
                               (default empty).

This module imports NOTHING from the engine's other modules (it's a leaf), so
any engine module can ``from queue_workflows import config`` without a cycle.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

# ── env-var NAMES (configurable; defaults keep ai_leads byte-compat) ─────────
#
# The engine reads these env *names* off the config so a host can rename them
# without touching engine code. The DEFAULTS are the ai_leads names so the
# rendered ``.env`` on the live fleet is unchanged at the cutover.


@dataclass
class EngineConfig:
    """Process-wide engine configuration. One instance lives in this module
    (:data:`_CONFIG`); access it via :func:`get_config`. Mutated only by the
    public ``configure`` / ``set_*`` / ``register_*`` helpers in
    ``queue_workflows.__init__`` — never directly by engine modules at import
    time (so a host can configure AFTER import)."""

    # ── env-var names ────────────────────────────────────────────────────────
    db_url_env: str = "AI_LEADS_DB_URL"
    host_label_env: str = "AI_LEADS_HOST_LABEL"
    host_priority_env: str = "AI_LEADS_GPU_CONSUMER_PRIORITY"

    # ── value config ──────────────────────────────────────────────────────────
    #: GPU model ids on the tight per-job video render budget (claim_worker).
    video_model_ids: frozenset[str] = frozenset()
    #: Dotted package the node-module resolver imports under (e.g.
    #: ``"workflows.nodes"``). Empty ⇒ the stored ``node_module`` is treated as
    #: a fully-qualified importable module name.
    node_module_package: str = ""
    #: cgroup-attribution container-name prefix (hw_metrics per-container slice).
    container_prefix: str = "ai_leads-"

    # ── node-module resolver (overrides node_module_package when set) ──────────
    #: ``Callable[[str], module]`` — resolve a stored ``node_module`` string to
    #: an imported module exposing ``run(...)``. Default builds from
    #: ``node_module_package`` (see :meth:`resolve_node_module`).
    node_resolver: Callable[[str], Any] | None = None

    # ── builtin-model registrar (empty-registry fallback hook) ─────────────────
    #: ``Callable[[], None]`` — idempotently register the host's ModelSpecs into
    #: the engine ``model_registry``. The empty-registry re-registration
    #: fallback in ``model_cache`` calls it; ``node_pool`` / ``claim_worker``
    #: call it once at startup. Default no-op (standalone engine has no models).
    builtin_model_registrar: Callable[[], None] = lambda: None

    # ── workflow / pipeline provider (DAG source) ──────────────────────────────
    #: ``Callable[[str], dict]`` — load a workflow definition by name.
    workflow_loader: Callable[[str], dict] | None = None
    #: ``Callable[[str], dict]`` — load a pipeline schema by name.
    pipeline_schema_loader: Callable[[str], dict] | None = None
    #: ``Callable[[Any, dict], Any]`` — resolve a ``$from``/``$value``/``$filter``
    #: ref against a context. Defaults to the built-in :func:`refs.resolve_ref`
    #: (wired lazily in ``get_resolve_ref`` to keep this module a leaf).
    resolve_ref: Callable[[Any, dict], Any] | None = None

    # ── per-node invoke wrapper (host setup/teardown around each node run) ──────
    #: ``Callable[[dict, dict], ContextManager[Callable[[dict], dict] | None]]`` —
    #: given ``(job, run)``, returns a context manager wrapping the node invoke.
    #: ``__enter__`` does host setup (e.g. pin a run-context ContextVar, capture a
    #: live flag) and yields a ``finalize(context_delta) -> context_delta`` callable
    #: that ``execute_node`` applies ONLY on success (e.g. stamp a per-node marker);
    #: ``__exit__`` does teardown on EVERY exit path (success / failure / skip).
    #: Default ``None`` ⇒ no wrapping (the engine runs the node directly). Lets a
    #: host thread per-node execution state (e.g. a smoke/mock ``_mocked`` stamp)
    #: without forking ``execute_node``.
    invoke_context: Callable[[dict, dict], Any] | None = None

    # ── ingest queue names + budget (host-configurable; G1) ────────────────────
    #: Ingest-family queue names. Migration 0008 dropped the fetch/load DB CHECK;
    #: the host validates ``queue`` against THIS set before enqueue (mirroring the
    #: task_name gate). Default {'fetch','load'} keeps ai_leads byte-compat; a
    #: different project (lm_flood) sets e.g. {'ingest','hydro','hydraulic','gpu'}.
    ingest_queues: frozenset[str] = frozenset({"fetch", "load"})
    #: Wall-clock budget (s) the claim worker applies to ingest queues OTHER than
    #: the built-in fetch/load (``claim_worker.budget_for``). Host-tunable.
    ingest_default_budget_s: int = 3600

    # ── ingest task seam (periodic work) ───────────────────────────────────────
    #: ``task_name -> Callable[[str], dict]`` — the periodic ingest callables the
    #: claim worker runs. The callable takes the ``reason`` string and returns a
    #: JSON-able result dict. Empty ⇒ no ingest work registered.
    ingest_task_map: dict[str, Callable[[str], dict]] = field(default_factory=dict)
    #: The scheduler's periodic schedule (list of ``ScheduleEntry``). Empty ⇒ the
    #: ticker has nothing to fire. Typed ``Any`` here to keep config a leaf (the
    #: ``ScheduleEntry`` type lives in ``scheduler``).
    ingest_schedule: list[Any] = field(default_factory=list)

    # ── lock so configure() from a host thread is safe ─────────────────────────
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ── derived accessors ──────────────────────────────────────────────────────

    def resolve_node_module(self, node_module: str):
        """Import + return the module for a stored ``node_module`` value.

        Honours an injected :attr:`node_resolver` first; otherwise builds the
        dotted name from :attr:`node_module_package` (``"<pkg>.<node_module>"``
        when a package is set, else ``node_module`` verbatim) and imports it.
        """
        if self.node_resolver is not None:
            return self.node_resolver(node_module)
        import importlib

        dotted = (
            f"{self.node_module_package}.{node_module}"
            if self.node_module_package
            else node_module
        )
        return importlib.import_module(dotted)

    def get_resolve_ref(self) -> Callable[[Any, dict], Any]:
        """Return the ref resolver, defaulting to the engine's own
        :func:`queue_workflows.refs.resolve_ref` (imported lazily so this
        config module stays a leaf with no engine-internal imports)."""
        if self.resolve_ref is not None:
            return self.resolve_ref
        from queue_workflows.refs import resolve_ref as _builtin_resolve_ref

        return _builtin_resolve_ref


# Process-wide singleton.
_CONFIG = EngineConfig()


def get_config() -> EngineConfig:
    """Return the process-wide :class:`EngineConfig`."""
    return _CONFIG


def reset_for_tests() -> None:
    """TEST-ONLY. Restore the config to its all-default state so a test that
    mutated a hook doesn't leak into the next."""
    global _CONFIG
    _CONFIG = EngineConfig()
