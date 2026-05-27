"""MongoDB :class:`StorageBackend`.

Mongo maps onto the contract more directly than Redis but with its own caveats,
which the audit flagged up front:

  * **Atomic claim via ``find_one_and_update``** — find the oldest, highest
    priority ``queued`` job and flip it ``running`` in one atomic document op.
    Two concurrent claimers get different jobs because the first's update removes
    it from the ``status:'queued'`` filter before the second matches it
    (the standard Mongo work-queue claim; reproduces SKIP-LOCKED's effect).
  * **Atomic outbox via a MULTI-DOCUMENT TRANSACTION** — go terminal *and* insert
    the event in one transaction, both-or-neither. This is why the backend needs
    a **replica set** (transactions — and change streams — are unavailable on a
    standalone ``mongod``); a single-node RS is enough.
  * **Wake via a change stream** on a capped ``wake`` collection (also RS-only).
    Each enqueue/reclaim/requeue inserts a tiny ``{queue}`` doc; the stream
    surfaces it. Best-effort like every wake — the worker's safety poll covers a
    stream that hasn't resumed yet.

Each namespace is a SEPARATE DATABASE (``qw_<namespace>``), so two tenants on one
server share nothing — the strongest possible isolation (the data-leakage guard).
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import CollectionInvalid

from queue_workflows.backends.base import Event, Job, StorageBackend, WakeListener

_TERMINAL = ("completed", "failed")


def _db_name(namespace: str) -> str:
    return "qw_" + re.sub(r"[^A-Za-z0-9_]", "_", namespace)[:48]


class MongoBackend(StorageBackend):
    name = "mongodb"

    def __init__(self, *, url: str, namespace: str = "") -> None:
        super().__init__(url=url, namespace=namespace)
        self._client: MongoClient = MongoClient(
            url, serverSelectionTimeoutMS=5000, tz_aware=True,
        )
        self._db = self._client[_db_name(self.namespace)]
        self._jobs = self._db["jobs"]
        self._events = self._db["events"]
        self._workers = self._db["workers"]
        self._controls = self._db["controls"]
        self._counters = self._db["counters"]
        self._wake = self._db["wake"]

    # ── schema ──────────────────────────────────────────────────────────────────

    def ensure_schema(self) -> None:
        # Force server selection now so an unreachable server fails the probe.
        self._client.admin.command("ping")
        try:
            self._db.create_collection("wake", capped=True, size=1 << 20)
        except CollectionInvalid:
            pass  # already exists
        self._jobs.create_index(
            [("queue", ASCENDING), ("status", ASCENDING),
             ("priority", DESCENDING), ("seq", ASCENDING)]
        )
        self._jobs.create_index([("status", ASCENDING), ("lease_expires_at", ASCENDING)])
        self._events.create_index([("seq", ASCENDING)])
        self._controls.create_index(
            [("host", ASCENDING), ("queue", ASCENDING)], unique=True
        )
        # TTL: stale heartbeats self-expire (the liveness window).
        self._workers.create_index("last_seen", expireAfterSeconds=60)
        self._workers.create_index([("queue", ASCENDING)])
        for cid in ("seq", "eventseq"):
            self._counters.update_one({"_id": cid}, {"$setOnInsert": {"n": 0}}, upsert=True)

    def close(self) -> None:
        self._client.close()

    # ── counters ──────────────────────────────────────────────────────────────────

    def _next(self, counter_id: str, *, session=None) -> int:
        doc = self._counters.find_one_and_update(
            {"_id": counter_id}, {"$inc": {"n": 1}},
            upsert=True, return_document=ReturnDocument.AFTER, session=session,
        )
        return int(doc["n"])

    def _wake_insert(self, queue: str) -> None:
        try:
            self._wake.insert_one({"queue": queue, "ts": time.time()})
        except Exception:
            pass  # wake is best-effort

    # ── enqueue / claim / lease ────────────────────────────────────────────────

    def enqueue(self, queue, payload, *, job_id=None, priority=0) -> str:
        jid = job_id or uuid.uuid4().hex
        now = time.time()
        self._jobs.insert_one({
            "_id": jid, "namespace": self.namespace, "queue": queue,
            "status": "queued", "payload": payload or {}, "priority": int(priority),
            "attempts": 0, "claimed_by": None, "lease_expires_at": None,
            "result": None, "error": None, "seq": self._next("seq"),
            "created_at": now, "updated_at": now,
        })
        self._wake_insert(queue)
        return jid

    def claim(self, queue, worker, *, lease_s) -> Job | None:
        now = time.time()
        doc = self._jobs.find_one_and_update(
            {"queue": queue, "status": "queued"},
            {"$set": {"status": "running", "claimed_by": worker,
                      "lease_expires_at": now + float(lease_s), "updated_at": now},
             "$inc": {"attempts": 1}},
            sort=[("priority", DESCENDING), ("seq", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )
        return _job(doc)

    def renew_lease(self, job_id, worker, *, lease_s) -> bool:
        now = time.time()
        doc = self._jobs.find_one_and_update(
            {"_id": job_id, "status": "running", "claimed_by": worker},
            {"$set": {"lease_expires_at": now + float(lease_s), "updated_at": now}},
            return_document=ReturnDocument.AFTER,
        )
        return doc is not None

    def reclaim_expired(self, *, queue=None) -> list[str]:
        now = time.time()
        flt: dict[str, Any] = {"status": "running", "lease_expires_at": {"$lt": now}}
        if queue is not None:
            flt["queue"] = queue
        out: list[str] = []
        while True:
            doc = self._jobs.find_one_and_update(
                flt,
                {"$set": {"status": "queued", "claimed_by": None,
                          "lease_expires_at": None, "updated_at": time.time()}},
                return_document=ReturnDocument.AFTER,
            )
            if doc is None:
                break
            out.append(doc["_id"])
            self._wake_insert(doc["queue"])
        return out

    def requeue_for_retry(self, job_id) -> Job | None:
        doc = self._jobs.find_one_and_update(
            {"_id": job_id, "status": {"$nin": list(_TERMINAL)}},
            {"$set": {"status": "queued", "claimed_by": None,
                      "lease_expires_at": None, "updated_at": time.time()}},
            return_document=ReturnDocument.AFTER,
        )
        if doc is not None:
            self._wake_insert(doc["queue"])
        return _job(doc)

    # ── terminal transitions ────────────────────────────────────────────────────

    def _mark(self, job_id, status, *, result, error) -> Job | None:
        doc = self._jobs.find_one_and_update(
            {"_id": job_id, "status": {"$nin": list(_TERMINAL)}},
            {"$set": {"status": status, "result": result, "error": error,
                      "updated_at": time.time()}},
            return_document=ReturnDocument.AFTER,
        )
        return _job(doc)

    def mark_completed(self, job_id, *, result=None) -> Job | None:
        return self._mark(job_id, "completed", result=result, error=None)

    def mark_failed(self, job_id, *, error=None) -> Job | None:
        return self._mark(job_id, "failed", result=None, error=error)

    def _terminal_with_event(self, job_id, status, event_type, *, result, error, detail):
        # One transaction: the terminal flip + the event insert commit together,
        # or — when the job is already terminal (0 docs matched) — neither does.
        with self._client.start_session() as session:
            with session.start_transaction():
                doc = self._jobs.find_one_and_update(
                    {"_id": job_id, "status": {"$nin": list(_TERMINAL)}},
                    {"$set": {"status": status, "result": result, "error": error,
                              "updated_at": time.time()}},
                    return_document=ReturnDocument.AFTER, session=session,
                )
                if doc is None:
                    return None  # already terminal → empty txn, NO event written
                self._events.insert_one({
                    "seq": self._next("eventseq", session=session),
                    "namespace": self.namespace, "job_id": job_id,
                    "queue": doc["queue"], "event_type": event_type,
                    "detail": detail or {}, "created_at": time.time(),
                }, session=session)
        return _job(doc)

    def complete_with_event(self, job_id, event_type, *, result=None, detail=None):
        return self._terminal_with_event(
            job_id, "completed", event_type, result=result, error=None, detail=detail
        )

    def fail_with_event(self, job_id, event_type, *, error=None, detail=None):
        return self._terminal_with_event(
            job_id, "failed", event_type, result=None, error=error, detail=detail
        )

    # ── reads ────────────────────────────────────────────────────────────────────

    def get(self, job_id) -> Job | None:
        return _job(self._jobs.find_one({"_id": job_id}))

    def counts(self, queue) -> dict[str, int]:
        return {
            s: self._jobs.count_documents({"queue": queue, "status": s})
            for s in ("queued", "running", "completed", "failed")
        }

    def events(self, *, since=0, limit=1000) -> list[Event]:
        cur = self._events.find({"seq": {"$gt": since}}).sort("seq", ASCENDING).limit(limit)
        return [
            Event(
                seq=int(d["seq"]), job_id=d["job_id"], namespace=self.namespace,
                queue=d.get("queue"), event_type=d["event_type"],
                detail=d.get("detail") or {}, created_at=float(d.get("created_at") or 0),
            )
            for d in cur
        ]

    # ── wake (change stream) ────────────────────────────────────────────────────

    def notify(self, queue) -> None:
        self._wake_insert(queue)

    def subscribe(self, *queues) -> WakeListener:
        return _MongoWakeListener(self._wake, frozenset(queues))

    # ── heartbeats + control ───────────────────────────────────────────────────────

    def heartbeat(self, host, queue, *, current_model=None, stale_after_s=30.0) -> None:
        self._workers.update_one(
            {"_id": f"{queue}:{host}"},
            {"$set": {"host": host, "queue": queue, "current_model": current_model,
                      "last_seen": datetime.now(timezone.utc)}},
            upsert=True,
        )

    def workers(self, queue) -> list[dict[str, Any]]:
        return [
            {"host": d.get("host"), "queue": d.get("queue"),
             "current_model": d.get("current_model"),
             "last_seen": d["last_seen"].timestamp() if d.get("last_seen") else None}
            for d in self._workers.find({"queue": queue})
        ]

    def set_control(self, host, queue, *, desired_state, stop_policy="hard",
                    requested_by=None) -> None:
        self._controls.update_one(
            {"host": host, "queue": queue},
            {"$set": {"desired_state": desired_state, "stop_policy": stop_policy,
                      "requested_by": requested_by, "updated_at": time.time()}},
            upsert=True,
        )

    def desired_state(self, host, queue) -> str:
        doc = self._controls.find_one({"host": host, "queue": queue})
        return "off" if (doc and doc.get("desired_state") == "off") else "on"


def _job(doc: dict[str, Any] | None) -> Job | None:
    if doc is None:
        return None
    return Job(
        id=doc["_id"], queue=doc["queue"], namespace=doc.get("namespace", ""),
        status=doc["status"], payload=doc.get("payload") or {},
        priority=int(doc.get("priority") or 0), attempts=int(doc.get("attempts") or 0),
        claimed_by=doc.get("claimed_by"), lease_expires_at=doc.get("lease_expires_at"),
        result=doc.get("result"), error=doc.get("error"),
        created_at=float(doc.get("created_at") or 0),
        updated_at=float(doc.get("updated_at") or 0),
    )


class _MongoWakeListener:
    """Change stream over the capped ``wake`` collection; ``wait`` returns the
    inserted queue if it's one we subscribed to."""

    def __init__(self, wake_collection, queues: frozenset[str]) -> None:
        self._coll = wake_collection
        self._queues = queues
        self._stream = None

    def __enter__(self):
        # Open the stream BEFORE the caller enqueues, so the insert is captured.
        self._stream = self._coll.watch(
            [{"$match": {"operationType": "insert"}}], max_await_time_ms=250,
        )
        return self

    def __exit__(self, *exc: object) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            finally:
                self._stream = None

    def wait(self, timeout: float) -> str | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            change = self._stream.try_next()
            if change is None:
                continue  # try_next already blocked up to max_await_time_ms
            q = (change.get("fullDocument") or {}).get("queue")
            if q is not None and (not self._queues or q in self._queues):
                return q
        return None
