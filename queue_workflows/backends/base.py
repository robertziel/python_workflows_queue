"""The :class:`StorageBackend` port — a host-agnostic durable work-queue SPI.

WHY THIS EXISTS. The engine proper is, by design, *Postgres-as-queue*: its
claim loop, lease/reclaim, dispatch outbox and wake are written directly in SQL
(``FOR UPDATE SKIP LOCKED``, ``RETURNING`` CAS, ``pg_notify`` in-trigger). That
is deliberate and stays the reference path. This module factors out the **subset
of behaviour that defines "a queue backend"** into an abstract port so the
*database type* becomes selectable (``configure(db_backend="pg"|"redis"|
"mongodb")``) without each call site speaking SQL. It is **additive**: selecting
``pg`` (the default) changes nothing for existing consumers.

DESIGN — what the port is, and what it is NOT.

  * It is a **generic durable queue with a transactional outbox**: enqueue →
    claim-exactly-once → lease/renew → terminal(+atomic event) → reclaim, plus a
    best-effort wake (NOTIFY/pub-sub/change-stream) and the operator heartbeat /
    ON-OFF control rows. These are the contracts every backend must honour
    identically — pinned by ``tests/test_backend_contract.py``.
  * It is **NOT leaky**: no method takes or returns a driver handle (psycopg
    cursor, redis pipeline, pymongo session). The outbox atomicity — "go
    terminal AND append the event, both-or-neither" — is exposed as a single
    high-level call (:meth:`complete_with_event` / :meth:`fail_with_event`) that
    each backend implements atomically *in its own idiom* (PG txn / Redis Lua /
    Mongo multi-doc txn). Hosts never hold a transaction object, so PG internals
    can't bleed into Redis/Mongo call sites (audit dimension: internal data
    leakage).
  * Each backend instance is **bound to one namespace** (constructor arg). Every
    key / row / collection it touches is scoped by that namespace, so two apps
    sharing one Redis/Mongo server cannot claim or read each other's jobs. This
    is the multi-tenant isolation guard — see the cross-namespace test.

A backend is constructed with ``(url, namespace)`` and is safe to share across
threads (each call borrows its own connection / pipeline). ``close()`` releases
pooled resources.
"""

from __future__ import annotations

import abc
from typing import Any, Iterator, Protocol, TypedDict

# ── job status vocabulary (identical across backends) ────────────────────────

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
#: Terminal states a row can never leave — the idempotency guard. A second
#: ``mark_*`` / ``*_with_event`` on a row already in one of these is a no-op that
#: returns ``None`` (and writes NO event), exactly like the PG engine's
#: ``UPDATE … WHERE status NOT IN (...) RETURNING *`` shape.
TERMINAL_STATUSES = frozenset({STATUS_COMPLETED, STATUS_FAILED})

#: ``""`` namespace ⇒ this literal, so a key/collection always has a real scope.
DEFAULT_NAMESPACE = "default"


class Job(TypedDict, total=False):
    """The canonical job shape every backend returns (a plain dict at runtime).

    ``lease_expires_at`` / ``created_at`` / ``updated_at`` are epoch seconds
    (float) so the shape is backend-neutral — each adapter converts to/from its
    native time type. ``payload`` / ``result`` are JSON-able dicts."""

    id: str
    queue: str
    namespace: str
    status: str
    payload: dict[str, Any]
    priority: int
    attempts: int
    claimed_by: str | None
    lease_expires_at: float | None
    result: dict[str, Any] | None
    error: str | None
    created_at: float
    updated_at: float


class Event(TypedDict, total=False):
    """An outbox event row (the durable dispatch-event analog)."""

    seq: int
    job_id: str
    namespace: str
    queue: str
    event_type: str
    detail: dict[str, Any]
    created_at: float


class WakeListener(Protocol):
    """A subscription handle returned by :meth:`StorageBackend.subscribe`.

    Used as a context manager. :meth:`wait` blocks up to ``timeout`` for the
    next wake payload (a queue name) and returns it, or ``None`` on timeout — so
    a worker loops ``while not stop: q = sub.wait(1.0)`` exactly as it loops on
    ``LISTEN`` today (the timeout doubling as the dropped-NOTIFY safety poll)."""

    def __enter__(self) -> WakeListener: ...
    def __exit__(self, *exc: object) -> None: ...
    def wait(self, timeout: float) -> str | None: ...


class StorageBackend(abc.ABC):
    """Abstract durable-queue backend. See module docstring for the contract.

    Subclasses: :class:`~queue_workflows.backends.postgres.PostgresBackend`,
    :class:`~queue_workflows.backends.redis.RedisBackend`,
    :class:`~queue_workflows.backends.mongodb.MongoBackend`.
    """

    #: Registry name (``"pg"`` / ``"redis"`` / ``"mongodb"``); set on subclasses.
    name: str = ""

    def __init__(self, *, url: str, namespace: str = "") -> None:
        self.url = url
        self.namespace = namespace or DEFAULT_NAMESPACE

    # ── schema / lifecycle ────────────────────────────────────────────────────

    @abc.abstractmethod
    def ensure_schema(self) -> None:
        """Idempotently create whatever durable structures the backend needs
        (PG tables / Mongo indexes; a no-op for Redis). Safe to call on boot."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release pooled connections / clients."""

    # ── enqueue / claim / lease ────────────────────────────────────────────────

    @abc.abstractmethod
    def enqueue(
        self, queue: str, payload: dict[str, Any], *,
        job_id: str | None = None, priority: int = 0,
    ) -> str:
        """Append a ``queued`` job and fire the wake NOTIFY for ``queue``.
        Returns the job id (generated when ``job_id`` is None). The wake must be
        emitted as part of the same durable write (no "queued but no wake")."""

    @abc.abstractmethod
    def claim(self, queue: str, worker: str, *, lease_s: float) -> Job | None:
        """Atomically take the single oldest claimable (``queued``, highest
        priority) job on ``queue``: flip it ``running``, stamp ``claimed_by`` +
        ``lease_expires_at = now+lease_s``, bump ``attempts``. Returns the job,
        or ``None`` if none claimable. **Exactly-once under contention**: two
        concurrent callers never receive the same job (the SKIP-LOCKED guarantee
        each backend reproduces in its own idiom)."""

    @abc.abstractmethod
    def renew_lease(self, job_id: str, worker: str, *, lease_s: float) -> bool:
        """Extend ``lease_expires_at`` to ``now+lease_s`` iff the job is still
        ``running`` and still owned by ``worker``. Returns whether it renewed."""

    @abc.abstractmethod
    def reclaim_expired(self, *, queue: str | None = None) -> list[str]:
        """Re-queue every ``running`` job whose lease has lapsed (optionally
        filtered to ``queue``): flip ``running`` → ``queued``, clear the lease /
        owner, re-fire the wake. Returns the reclaimed job ids. This is the sole
        recovery path for an orphaned ``running`` row."""

    @abc.abstractmethod
    def requeue_for_retry(self, job_id: str) -> Job | None:
        """Watchdog re-queue: flip a ``running`` job back to ``queued`` (clear
        lease/owner, keep ``attempts`` as the retry counter), re-fire the wake,
        and write **no** event. Returns the updated job, or ``None`` if it was
        already terminal."""

    # ── terminal transitions (idempotent) ──────────────────────────────────────

    @abc.abstractmethod
    def mark_completed(
        self, job_id: str, *, result: dict[str, Any] | None = None,
    ) -> Job | None:
        """Flip ``job_id`` → ``completed`` iff not already terminal; stamp
        ``result``. Returns the row, or ``None`` if it was already terminal
        (the idempotency guard — a duplicate delivery is a safe no-op)."""

    @abc.abstractmethod
    def mark_failed(self, job_id: str, *, error: str | None = None) -> Job | None:
        """Flip ``job_id`` → ``failed`` iff not already terminal; stamp
        ``error``. Returns the row, or ``None`` if already terminal."""

    @abc.abstractmethod
    def complete_with_event(
        self, job_id: str, event_type: str, *,
        result: dict[str, Any] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> Job | None:
        """ATOMIC outbox: go ``completed`` **and** append one :class:`Event`,
        both-or-neither. Returns the row, or ``None`` if already terminal — in
        which case **no event is written** (the second-delivery no-op)."""

    @abc.abstractmethod
    def fail_with_event(
        self, job_id: str, event_type: str, *,
        error: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> Job | None:
        """ATOMIC outbox twin of :meth:`complete_with_event` for the failure
        path (mark ``failed`` + append the event in one unit, or neither)."""

    # ── reads ──────────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def get(self, job_id: str) -> Job | None:
        """Return the job by id (within this namespace), or ``None``."""

    @abc.abstractmethod
    def counts(self, queue: str) -> dict[str, int]:
        """``{queued, running, completed, failed}`` for ``queue`` (snapshot)."""

    @abc.abstractmethod
    def events(self, *, since: int = 0, limit: int = 1000) -> list[Event]:
        """Outbox events with ``seq > since`` (oldest-first), for an event
        drainer / the atomicity assertions. Namespace-scoped."""

    # ── wake (NOTIFY / pub-sub / change-stream) ─────────────────────────────────

    @abc.abstractmethod
    def notify(self, queue: str) -> None:
        """Best-effort wake for listeners on ``queue`` (out-of-band; ``enqueue``
        already wakes). Mirrors a manual ``pg_notify``."""

    @abc.abstractmethod
    def subscribe(self, *queues: str) -> WakeListener:
        """Open a :class:`WakeListener` for the given queues. Best-effort: a
        dropped wake is covered by the caller's safety-poll timeout, identical to
        the engine's 1 s ``LISTEN`` safety poll."""

    # ── heartbeats + operator ON/OFF control ────────────────────────────────────

    @abc.abstractmethod
    def heartbeat(
        self, host: str, queue: str, *,
        current_model: str | None = None, stale_after_s: float = 30.0,
    ) -> None:
        """Upsert this worker's ``(host, queue)`` liveness row (TTL/last_seen)."""

    @abc.abstractmethod
    def workers(self, queue: str) -> list[dict[str, Any]]:
        """Live workers on ``queue`` (heartbeat fresher than its TTL)."""

    @abc.abstractmethod
    def set_control(
        self, host: str, queue: str, *,
        desired_state: str, stop_policy: str = "hard",
        requested_by: str | None = None,
    ) -> None:
        """Upsert the operator desired-state row for a ``(host, queue)`` worker."""

    @abc.abstractmethod
    def desired_state(self, host: str, queue: str) -> str:
        """Effective desired state: ``"off"`` only when an explicit OFF row
        exists, else ``"on"`` (absent ⇒ ON — the default-on contract)."""


def normalized_namespace(namespace: str) -> str:
    """``""`` → :data:`DEFAULT_NAMESPACE`, else verbatim (trimmed)."""
    return (namespace or "").strip() or DEFAULT_NAMESPACE


def drain_iter(listener: WakeListener, *, timeout: float) -> Iterator[str]:
    """Yield wake payloads until a ``timeout`` elapses with none — a small helper
    for tests/consumers. (Backends needn't override.)"""
    while True:
        q = listener.wait(timeout)
        if q is None:
            return
        yield q
