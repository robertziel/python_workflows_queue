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

__version__ = "0.5.0"

__all__ = [
    "configure",
    "set_node_module_package",
    "set_node_resolver",
    "set_builtin_model_registrar",
    "set_workflow_provider",
    "set_invoke_context",
    "set_vllm_lifecycle",
    "set_llm_servers_available",
    "register_ingest_task",
    "set_ingest_schedule",
    "register_pool_handler",
    "get_config",
    "EngineConfig",
    "__version__",
]


# ── configuration (host calls these once at startup, before launching) ──


def configure(
    *,
    db_url_env: str | None = None,
    metrics_db_url_env: str | None = None,
    video_model_ids: frozenset[str] | None = None,
    node_module_package: str | None = None,
    host_label_env: str | None = None,
    host_priority_env: str | None = None,
    container_prefix: str | None = None,
    project: str | None = None,
    ingest_queues: frozenset[str] | None = None,
    ingest_default_budget_s: int | None = None,
    db_backend: str | None = None,
    db_namespace: str | None = None,
    cancel_orphan_queued_jobs: bool | None = None,
    vlm_pool_node_modules: frozenset[str] | None = None,
    gpu_self_load_node_modules: frozenset[str] | None = None,
    gpu_pool_backend: str | None = None,
    gpu_pool_url_env: str | None = None,
    gpu_pool_namespace: str | None = None,
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
        if metrics_db_url_env is not None:
            cfg.metrics_db_url_env = metrics_db_url_env
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
        if project is not None:
            cfg.project = str(project)
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
            if db_backend == "sqlite":
                # SQLite is a RELATIONAL engine backend (it hosts the full DAG
                # engine via the dialect seam), not a flat-queue StorageBackend
                # SPI provider — so it bypasses the SPI registry validation. The
                # engine's relational store is "pg" (default) or "sqlite";
                # "redis"/"mongodb" select the SPI flat queue instead.
                cfg.db_backend = "sqlite"
            else:
                # Validate against the backend registry (source of truth),
                # imported lazily so the package root stays import-light.
                from queue_workflows.backends import canonical_backend_name

                cfg.db_backend = canonical_backend_name(db_backend)
        if db_namespace is not None:
            cfg.db_namespace = str(db_namespace)
        if cancel_orphan_queued_jobs is not None:
            cfg.cancel_orphan_queued_jobs = bool(cancel_orphan_queued_jobs)
        if vlm_pool_node_modules is not None:
            cfg.vlm_pool_node_modules = frozenset(vlm_pool_node_modules)
        if gpu_self_load_node_modules is not None:
            cfg.gpu_self_load_node_modules = frozenset(gpu_self_load_node_modules)
        if gpu_pool_backend is not None:
            from queue_workflows.backends import canonical_backend_name

            cfg.gpu_pool_backend = canonical_backend_name(gpu_pool_backend)
        if gpu_pool_url_env is not None:
            cfg.gpu_pool_url_env = str(gpu_pool_url_env)
        if gpu_pool_namespace is not None:
            cfg.gpu_pool_namespace = str(gpu_pool_namespace)
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


def set_llm_servers_available(servers: list[str]) -> None:
    """Declare which LLM server types THIS host can actually run — published in
    the worker heartbeat (migration 0014) so the queue UI gates its per-machine
    server-type control. ``["ollama"]`` is the universal baseline; an NVIDIA host
    with the vllm sidecar rendered passes ``["ollama", "vllm"]``. See
    :attr:`queue_workflows.config.EngineConfig.llm_servers_available`."""
    cfg = get_config()
    with cfg._lock:
        cfg.llm_servers_available = list(servers)


def set_vllm_lifecycle(
    stop_fn: Callable[[], bool],
    start_fn: Callable[[str], None],
) -> None:
    """Wire the vllm-sidecar stop/start the idle supervisor + model-switch drive.

    ``stop_fn() -> bool`` frees the sidecar's VRAM (True iff it stopped one);
    ``start_fn(model_id) -> None`` (re)starts it serving ``model_id``. A host
    that runs vllm as a SEPARATE container wires these so the in-worker supervisor
    can control the SIBLING sidecar without a docker restart policy (ai_leads →
    docker Engine API over the unix socket). Threaded into ``VLLMBackend`` by the
    backend factory. See
    :attr:`queue_workflows.config.EngineConfig.vllm_stop_fn`."""
    cfg = get_config()
    with cfg._lock:
        cfg.vllm_stop_fn = stop_fn
        cfg.vllm_start_fn = start_fn


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


# ── shared GPU pool (pivot B) ──


def register_pool_handler(name: str, callable_: Callable[..., dict]) -> None:
    """Register a GPU-pool handler under ``name`` (deployed on a GPU box). A
    pooled worker resolves a claimed task's ``handler`` to it and runs
    ``fn(*, inputs, output_dir, params) -> dict``. The op CODE lives here on the
    box; the DATA lives on shared NFS (``inputs``/``output_dir`` are paths the
    handler interprets). A submit-only app needn't register any."""
    cfg = get_config()
    with cfg._lock:
        cfg.gpu_pool_handlers[name] = callable_
