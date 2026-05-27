"""``queue_workflows`` — a standalone Postgres-as-queue workflow engine.

A host-agnostic engine extracted from ai_leads (Phase 6): a ``SELECT … FOR
UPDATE SKIP LOCKED`` claim loop woken by ``LISTEN``, lease reclaim, a DAG
dispatcher with an outbox, a GPU warm-model cache, periodic ingest work, and
per-host hw-metrics telemetry. Postgres is the only hard dependency.

Usage — a host wires the engine once at startup, then launches a worker /
scheduler / orchestrator::

    import queue_workflows
    from queue_workflows import model_registry
    from queue_workflows.model_registry import ModelSpec

    queue_workflows.configure(
        db_url_env="AI_LEADS_DB_URL",
        video_model_ids=frozenset({"wan_i2v", "ltx_flf"}),
        node_module_package="workflows.nodes",
        container_prefix="ai_leads-",
    )
    queue_workflows.set_workflow_provider(load_workflow, pipeline_schema)
    queue_workflows.set_builtin_model_registrar(register_builtin_models)
    queue_workflows.register_ingest_task("run_fetch_all", run_fetch_all)
    queue_workflows.set_ingest_schedule([ScheduleEntry(...), ...])

    queue_workflows.claim_worker.main(["--queue", "gpu"])

Every hook has a safe default, so ``import queue_workflows`` +
``configure(...)`` + a reachable Postgres is enough to run the engine
standalone (no host wiring required). The package imports NOTHING from any
host application (``workflows.*``) — enforced by ``tests/test_no_ai_leads_import.py``.
"""

from __future__ import annotations

from typing import Any, Callable

from queue_workflows.config import EngineConfig, get_config

__version__ = "0.2.0"

__all__ = [
    "configure",
    "set_node_module_package",
    "set_node_resolver",
    "set_builtin_model_registrar",
    "set_workflow_provider",
    "set_invoke_context",
    "register_ingest_task",
    "set_ingest_schedule",
    "get_config",
    "EngineConfig",
    "__version__",
]


# ── configuration (host calls these once at startup, before launching) ──


def configure(
    *,
    db_url_env: str | None = None,
    video_model_ids: frozenset[str] | None = None,
    node_module_package: str | None = None,
    host_label_env: str | None = None,
    host_priority_env: str | None = None,
    container_prefix: str | None = None,
    ingest_queues: frozenset[str] | None = None,
    ingest_default_budget_s: int | None = None,
    db_backend: str | None = None,
    db_namespace: str | None = None,
) -> EngineConfig:
    """Set engine configuration values. Only the passed keyword args are
    mutated; the rest keep their (ai_leads-byte-compatible) defaults. Returns
    the live :class:`EngineConfig` for chaining/inspection.

    Safe to call once at host startup before launching a worker / scheduler /
    orchestrator. Idempotent and additive — call again to adjust a subset.
    """
    cfg = get_config()
    with cfg._lock:
        if db_url_env is not None:
            cfg.db_url_env = db_url_env
        if video_model_ids is not None:
            cfg.video_model_ids = frozenset(video_model_ids)
        if node_module_package is not None:
            cfg.node_module_package = node_module_package
        if host_label_env is not None:
            cfg.host_label_env = host_label_env
        if host_priority_env is not None:
            cfg.host_priority_env = host_priority_env
        if container_prefix is not None:
            cfg.container_prefix = container_prefix
        if ingest_queues is not None:
            iq = frozenset(ingest_queues)
            reserved = iq & {"cpu", "gpu"}
            if reserved:
                raise ValueError(
                    f"ingest_queues must not reuse the reserved DAG queue names "
                    f"{sorted(reserved)} (cpu/gpu draw from workflow_node_jobs); "
                    "use distinct names for non-DAG ingest queues."
                )
            cfg.ingest_queues = iq
        if ingest_default_budget_s is not None:
            cfg.ingest_default_budget_s = int(ingest_default_budget_s)
        if db_backend is not None:
            # Validate against the backend registry (source of truth), imported
            # lazily so the package root stays import-light and config a leaf.
            from queue_workflows.backends import canonical_backend_name

            cfg.db_backend = canonical_backend_name(db_backend)
        if db_namespace is not None:
            cfg.db_namespace = str(db_namespace)
    return cfg


# ── node registration (the node-module resolver) ──


def set_node_module_package(package: str) -> None:
    """Set the dotted package the node-module resolver imports under (e.g.
    ``"workflows.nodes"``). A stored ``node_module`` value of ``"smoke"`` then
    imports ``workflows.nodes.smoke``. Empty string ⇒ the stored value is
    imported as a fully-qualified module name."""
    cfg = get_config()
    with cfg._lock:
        cfg.node_module_package = package


def set_node_resolver(resolver: Callable[[str], Any]) -> None:
    """Inject a fully custom node-module resolver (``Callable[[str], module]``)
    — overrides :func:`set_node_module_package`. The module must expose a
    ``run(...)`` callable."""
    cfg = get_config()
    with cfg._lock:
        cfg.node_resolver = resolver


# ── model registration (GPU warm-cache registry) ──


def set_builtin_model_registrar(registrar: Callable[[], None]) -> None:
    """Set the idempotent builtin-model registrar — the empty-registry
    re-registration fallback (``model_cache``) + the once-at-startup
    registration (``claim_worker`` / ``orchestrator``). The host wires its
    ``register_builtin_models`` here; it should register ``ModelSpec``s into
    ``queue_workflows.model_registry``."""
    cfg = get_config()
    with cfg._lock:
        cfg.builtin_model_registrar = registrar


# ── workflow/pipeline provider (DAG source) ──


def set_workflow_provider(
    load_workflow: Callable[[str], dict],
    pipeline_schema: Callable[[str], dict],
    *,
    resolve_ref: Callable[[Any, dict], Any] | None = None,
) -> None:
    """Inject the DAG definition source the dispatcher reads from:

    - ``load_workflow(name) -> dict``    — the workflow definition.
    - ``pipeline_schema(name) -> dict``  — the pipeline schema (owns the node
                                           DAG).
    - ``resolve_ref`` (optional)         — a custom ref resolver; defaults to
                                           the engine's
                                           :func:`queue_workflows.refs.resolve_ref`.
    """
    cfg = get_config()
    with cfg._lock:
        cfg.workflow_loader = load_workflow
        cfg.pipeline_schema_loader = pipeline_schema
        if resolve_ref is not None:
            cfg.resolve_ref = resolve_ref


# ── per-node invoke wrapper (host setup/teardown around each node run) ──


def set_invoke_context(factory: Callable[[dict, dict], Any]) -> None:
    """Set the per-node invoke wrapper — a ``Callable[[job, run], ContextManager]``
    whose context manager brackets each node invoke: ``__enter__`` does host setup
    (e.g. pin a run-context ContextVar + capture a live mock flag) and yields a
    ``finalize(context_delta) -> context_delta`` callable applied ONLY on success
    (e.g. stamp a per-node ``_mocked`` marker); ``__exit__`` does teardown on every
    exit path. Default unset ⇒ the engine runs nodes directly (no wrapping). See
    :class:`queue_workflows.config.EngineConfig.invoke_context`."""
    cfg = get_config()
    with cfg._lock:
        cfg.invoke_context = factory


# ── ingest task + schedule registration (periodic work) ──


def register_ingest_task(name: str, callable_: Callable[[str], dict]) -> None:
    """Register a periodic ingest callable under ``name``. The claim worker
    runs it when an ``ingest_jobs`` row with that ``task_name`` is claimed; the
    callable takes the ``reason`` string and returns a JSON-able result dict.
    The registered names are also the valid ``task_name`` set
    ``node_queue.enqueue_ingest_job`` validates against."""
    cfg = get_config()
    with cfg._lock:
        cfg.ingest_task_map[name] = callable_


def set_ingest_schedule(schedule: list) -> None:
    """Set the scheduler's periodic schedule (a list of
    ``queue_workflows.scheduler.ScheduleEntry``). The ``Ticker`` fires it; the
    boot-kick enqueues the non-freshness entries."""
    cfg = get_config()
    with cfg._lock:
        cfg.ingest_schedule = list(schedule)
