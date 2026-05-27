"""Storage-backend registry — resolve ``config.db_backend`` to a concrete
:class:`~queue_workflows.backends.base.StorageBackend`.

Three providers, one file each (split per the request): ``postgres`` (the
reference/default), ``redis``, ``mongodb``. Provider modules are imported
**lazily** — selecting ``pg`` never imports ``redis``/``pymongo``, so those stay
*optional* dependencies (``pip install 'queue_workflows[redis]'`` /
``[mongodb]``) and a pg-only deploy needs neither installed.

Aliases collapse to a canonical name (``postgres``/``postgresql`` → ``pg``;
``mongo`` → ``mongodb``) so ``configure(db_backend=...)`` is forgiving but the
rest of the code only ever sees the canonical form.
"""

from __future__ import annotations

import os
import threading

from queue_workflows.backends.base import (
    Event,
    Job,
    StorageBackend,
    WakeListener,
)
from queue_workflows.config import get_config

__all__ = [
    "StorageBackend",
    "Job",
    "Event",
    "WakeListener",
    "canonical_backend_name",
    "is_known_backend",
    "known_backends",
    "build_backend",
    "get_backend",
    "close_all",
]

#: alias → canonical name. The canonical set is the source of truth ``configure``
#: validates against.
_ALIASES = {
    "pg": "pg",
    "postgres": "pg",
    "postgresql": "pg",
    "redis": "redis",
    "mongo": "mongodb",
    "mongodb": "mongodb",
}

# Dotted path to each provider class, imported only when first selected.
_PROVIDERS = {
    "pg": ("queue_workflows.backends.postgres", "PostgresBackend"),
    "redis": ("queue_workflows.backends.redis", "RedisBackend"),
    "mongodb": ("queue_workflows.backends.mongodb", "MongoBackend"),
}

_instances: dict[tuple[str, str, str], StorageBackend] = {}
_lock = threading.RLock()


def known_backends() -> frozenset[str]:
    """The canonical backend names (``{"pg", "redis", "mongodb"}``)."""
    return frozenset(_PROVIDERS)


def canonical_backend_name(name: str) -> str:
    """Normalize ``name`` to its canonical form, or raise ``ValueError`` listing
    the valid names. Used by ``configure(db_backend=...)`` to fail fast."""
    key = (name or "").strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"unknown db_backend {name!r}; valid: "
            f"{sorted(set(_ALIASES))} (canonical {sorted(known_backends())})"
        )
    return _ALIASES[key]


def is_known_backend(name: str) -> bool:
    return (name or "").strip().lower() in _ALIASES


def _load_provider(canonical: str) -> type[StorageBackend]:
    module_path, cls_name = _PROVIDERS[canonical]
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # missing optional driver
        extra = {"redis": "redis", "mongodb": "mongodb"}.get(canonical, canonical)
        raise ImportError(
            f"the {canonical!r} backend needs its driver: "
            f"pip install 'queue_workflows[{extra}]'  ({exc})"
        ) from exc
    return getattr(module, cls_name)


def build_backend(name: str, *, url: str, namespace: str = "") -> StorageBackend:
    """Construct a backend explicitly (no config / no caching) — the seam tests
    use this to point each provider at its dockerized server + a namespace."""
    cls = _load_provider(canonical_backend_name(name))
    return cls(url=url, namespace=namespace)


def _url_for(canonical: str) -> str:
    cfg = get_config()
    if canonical == "pg":
        from queue_workflows.db import db_url

        return db_url()
    env_name = cfg.redis_url_env if canonical == "redis" else cfg.mongo_url_env
    url = os.environ.get(env_name)
    if not url:
        raise RuntimeError(
            f"db_backend={canonical!r} but {env_name} is not set; "
            f"export the {canonical} DSN there (or pass a different env via "
            f"configure({'redis_url_env' if canonical == 'redis' else 'mongo_url_env'}=...))."
        )
    return url


def get_backend(*, namespace: str | None = None) -> StorageBackend:
    """Return the process-wide backend for the configured ``db_backend`` +
    namespace, building (and caching) it on first use. Cached per
    ``(backend, namespace, url)`` so repeated calls reuse one client/pool."""
    cfg = get_config()
    canonical = canonical_backend_name(cfg.db_backend)
    ns = namespace if namespace is not None else cfg.db_namespace
    url = _url_for(canonical)
    key = (canonical, ns or "", url)
    with _lock:
        be = _instances.get(key)
        if be is None:
            be = _load_provider(canonical)(url=url, namespace=ns or "")
            be.ensure_schema()
            _instances[key] = be
        return be


def close_all() -> None:
    """Close + drop every cached backend (orchestrator shutdown / test teardown)."""
    with _lock:
        for be in _instances.values():
            try:
                be.close()
            except Exception:  # teardown best-effort
                pass
        _instances.clear()
