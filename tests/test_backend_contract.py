"""The storage-backend CONTRACT — one parametrized suite every provider must
satisfy identically (pg / redis / mongodb).

This is the spec (TDD): the adapters in ``queue_workflows/backends/`` are written
to make these pass. Each backend is exercised against a real server; a backend
whose server isn't reachable (env unset / down) is SKIPPED, not failed, so the
suite runs anywhere — but CI/release must show all three green (see
``docs/storage_backends.md``).

Servers come from env (dockerized in dev):
  * pg      — ``QUEUE_WORKFLOWS_TEST_DB_URL``      (shared with the engine suite)
  * redis   — ``QUEUE_WORKFLOWS_TEST_REDIS_URL``
  * mongodb — ``QUEUE_WORKFLOWS_TEST_MONGO_URL``   (replica set: txns + change streams)

Each test gets a FRESH random namespace, so tests never see each other's jobs
(and the cross-namespace test doubles as the data-leakage guard).
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

from queue_workflows.backends import build_backend
from queue_workflows.backends.base import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
)

_URLS = {
    "pg": os.environ.get("QUEUE_WORKFLOWS_TEST_DB_URL"),
    "redis": os.environ.get("QUEUE_WORKFLOWS_TEST_REDIS_URL"),
    "mongodb": os.environ.get("QUEUE_WORKFLOWS_TEST_MONGO_URL"),
}

# Probe each backend once per session; cache (ok, reason) so we don't reconnect
# for every parametrized test.
_REACHABLE: dict[str, tuple[bool, str]] = {}


def _reachable(name: str) -> tuple[bool, str]:
    if name in _REACHABLE:
        return _REACHABLE[name]
    url = _URLS.get(name)
    if not url:
        res = (False, f"{name}: set QUEUE_WORKFLOWS_TEST_{name.upper()}_URL")
    else:
        try:
            be = build_backend(name, url=url, namespace="probe")
            be.ensure_schema()
            be.close()
            res = (True, "")
        except Exception as exc:  # server down / driver missing
            res = (False, f"{name} unreachable: {type(exc).__name__}: {exc}")
    _REACHABLE[name] = res
    return res


@pytest.fixture(params=["pg", "redis", "mongodb"])
def backend(request):
    name = request.param
    ok, why = _reachable(name)
    if not ok:
        pytest.skip(why)
    ns = f"t_{uuid.uuid4().hex[:12]}"
    be = build_backend(name, url=_URLS[name], namespace=ns)
    be.ensure_schema()
    yield be
    be.close()


# ── enqueue / get ─────────────────────────────────────────────────────────────


def test_enqueue_then_get(backend):
    jid = backend.enqueue("cpu", {"x": 1}, priority=0)
    assert isinstance(jid, str) and jid
    job = backend.get(jid)
    assert job is not None
    assert job["status"] == STATUS_QUEUED
    assert job["queue"] == "cpu"
    assert job["payload"] == {"x": 1}
    assert job["attempts"] == 0
    assert job["namespace"] == backend.namespace


def test_get_missing_returns_none(backend):
    assert backend.get("does-not-exist") is None


# ── claim: exactly-once under contention ───────────────────────────────────────


def test_claim_then_second_claim_is_none(backend):
    backend.enqueue("cpu", {"n": 1})
    first = backend.claim("cpu", "w1", lease_s=30)
    assert first is not None
    assert first["status"] == STATUS_RUNNING
    assert first["claimed_by"] == "w1"
    assert first["attempts"] == 1
    assert backend.claim("cpu", "w2", lease_s=30) is None  # nothing left


def test_claim_returns_distinct_jobs_never_double(backend):
    ids = {backend.enqueue("cpu", {"i": i}) for i in range(5)}
    claimed = []
    for _ in range(5):
        job = backend.claim("cpu", "w", lease_s=30)
        assert job is not None
        claimed.append(job["id"])
    assert backend.claim("cpu", "w", lease_s=30) is None
    assert set(claimed) == ids  # every job claimed exactly once, no repeats
    assert len(claimed) == len(set(claimed))


def test_claim_respects_priority(backend):
    backend.enqueue("cpu", {"p": "low"}, priority=0)
    backend.enqueue("cpu", {"p": "high"}, priority=10)
    job = backend.claim("cpu", "w", lease_s=30)
    assert job["payload"]["p"] == "high"  # higher priority first


def test_claim_isolated_per_queue(backend):
    backend.enqueue("cpu", {"q": "cpu"})
    assert backend.claim("gpu", "w", lease_s=30) is None  # wrong queue


def test_claim_exactly_once_under_thread_contention(backend):
    """The keystone, proven under REAL contention: N jobs, many threads racing
    to claim — every job is claimed exactly once, none twice (SKIP-LOCKED-equiv:
    PG ``FOR UPDATE SKIP LOCKED`` / Redis ``ZPOPMIN`` in Lua / Mongo
    ``find_one_and_update``)."""
    import threading

    n = 40
    ids = {backend.enqueue("cpu", {"i": i}) for i in range(n)}
    claimed: list[str] = []
    lock = threading.Lock()

    def drain(worker: str) -> None:
        while True:
            job = backend.claim("cpu", worker, lease_s=30)
            if job is None:
                return
            with lock:
                claimed.append(job["id"])

    threads = [threading.Thread(target=drain, args=(f"w{t}",)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(claimed) == sorted(ids)        # every job claimed
    assert len(claimed) == len(set(claimed))     # and never twice


# ── lease renew + reclaim ──────────────────────────────────────────────────────


def test_renew_lease_only_by_owner(backend):
    backend.enqueue("cpu", {})
    job = backend.claim("cpu", "w1", lease_s=30)
    assert backend.renew_lease(job["id"], "w1", lease_s=60) is True
    assert backend.renew_lease(job["id"], "someone-else", lease_s=60) is False


def test_reclaim_expired_requeues(backend):
    backend.enqueue("cpu", {})
    job = backend.claim("cpu", "w1", lease_s=0)  # already expired
    assert job["status"] == STATUS_RUNNING
    reclaimed = backend.reclaim_expired(queue="cpu")
    assert job["id"] in reclaimed
    again = backend.get(job["id"])
    assert again["status"] == STATUS_QUEUED
    assert again["claimed_by"] in (None, "")
    # …and it's claimable again
    assert backend.claim("cpu", "w2", lease_s=30)["id"] == job["id"]


def test_reclaim_leaves_live_leases_alone(backend):
    backend.enqueue("cpu", {})
    job = backend.claim("cpu", "w1", lease_s=300)  # healthy lease
    assert backend.reclaim_expired(queue="cpu") == []
    assert backend.get(job["id"])["status"] == STATUS_RUNNING


# ── idempotent terminals ───────────────────────────────────────────────────────


def test_mark_completed_idempotent(backend):
    backend.enqueue("cpu", {})
    job = backend.claim("cpu", "w", lease_s=30)
    done = backend.mark_completed(job["id"], result={"ok": True})
    assert done is not None and done["status"] == STATUS_COMPLETED
    assert done["result"] == {"ok": True}
    assert backend.mark_completed(job["id"], result={"ok": False}) is None  # no-op
    assert backend.get(job["id"])["result"] == {"ok": True}  # not clobbered


def test_mark_failed_idempotent(backend):
    backend.enqueue("cpu", {})
    job = backend.claim("cpu", "w", lease_s=30)
    failed = backend.mark_failed(job["id"], error="boom")
    assert failed is not None and failed["status"] == STATUS_FAILED
    assert backend.mark_failed(job["id"], error="again") is None
    assert backend.mark_completed(job["id"]) is None  # can't resurrect terminal


# ── atomic outbox (the keystone) ───────────────────────────────────────────────


def test_complete_with_event_atomic_and_idempotent(backend):
    backend.enqueue("cpu", {})
    job = backend.claim("cpu", "w", lease_s=30)
    out = backend.complete_with_event(job["id"], "completed", result={"r": 1})
    assert out is not None and out["status"] == STATUS_COMPLETED
    evs = backend.events()
    assert len([e for e in evs if e["job_id"] == job["id"]]) == 1
    ev = [e for e in evs if e["job_id"] == job["id"]][0]
    assert ev["event_type"] == "completed"
    # second delivery: no transition, and crucially NO duplicate event
    assert backend.complete_with_event(job["id"], "completed") is None
    evs2 = backend.events()
    assert len([e for e in evs2 if e["job_id"] == job["id"]]) == 1


def test_fail_with_event_atomic(backend):
    backend.enqueue("cpu", {})
    job = backend.claim("cpu", "w", lease_s=30)
    out = backend.fail_with_event(job["id"], "failed", error="x", detail={"code": 75})
    assert out["status"] == STATUS_FAILED
    evs = [e for e in backend.events() if e["job_id"] == job["id"]]
    assert len(evs) == 1 and evs[0]["event_type"] == "failed"
    assert evs[0]["detail"].get("code") == 75


def test_events_are_ordered_and_filterable_by_seq(backend):
    j1 = backend.enqueue("cpu", {})
    j2 = backend.enqueue("cpu", {})
    backend.fail_with_event(backend.claim("cpu", "w", lease_s=30)["id"], "failed")
    first_seq = backend.events()[0]["seq"]
    backend.complete_with_event(backend.claim("cpu", "w", lease_s=30)["id"], "completed")
    after = backend.events(since=first_seq)
    assert all(e["seq"] > first_seq for e in after)
    assert len(after) == 1


# ── watchdog re-queue (retry) ──────────────────────────────────────────────────


def test_requeue_for_retry_keeps_attempts_no_event(backend):
    backend.enqueue("cpu", {})
    job = backend.claim("cpu", "w1", lease_s=300)
    assert job["attempts"] == 1
    rq = backend.requeue_for_retry(job["id"])
    assert rq is not None and rq["status"] == STATUS_QUEUED
    assert backend.events() == [] or all(
        e["job_id"] != job["id"] for e in backend.events()
    )
    # claimable again; attempts increments on the re-claim
    again = backend.claim("cpu", "w2", lease_s=30)
    assert again["id"] == job["id"]
    assert again["attempts"] == 2


def test_requeue_terminal_returns_none(backend):
    backend.enqueue("cpu", {})
    job = backend.claim("cpu", "w", lease_s=30)
    backend.mark_completed(job["id"])
    assert backend.requeue_for_retry(job["id"]) is None


# ── counts ──────────────────────────────────────────────────────────────────────


def test_counts(backend):
    backend.enqueue("cpu", {})
    backend.enqueue("cpu", {})
    backend.enqueue("cpu", {})
    backend.mark_completed(backend.claim("cpu", "w", lease_s=30)["id"])
    c = backend.counts("cpu")
    assert c["queued"] == 2
    assert c["running"] == 0
    assert c["completed"] == 1
    assert c["failed"] == 0


# ── wake (best-effort NOTIFY / pub-sub / change-stream) ─────────────────────────


def test_wake_on_enqueue(backend):
    with backend.subscribe("cpu") as sub:
        time.sleep(0.2)  # let the subscription establish (pub/sub has no backlog)
        backend.enqueue("cpu", {"wake": 1})
        got = sub.wait(5.0)
    assert got == "cpu"


# ── heartbeats + operator ON/OFF control ────────────────────────────────────────


def test_heartbeat_and_workers(backend):
    backend.heartbeat("hostA", "gpu", current_model="m1")
    workers = backend.workers("gpu")
    assert any(w.get("host") == "hostA" for w in workers)


def test_worker_control_default_on_then_off_then_on(backend):
    assert backend.desired_state("hostA", "gpu") == "on"  # absent ⇒ ON
    backend.set_control("hostA", "gpu", desired_state="off", requested_by="ops")
    assert backend.desired_state("hostA", "gpu") == "off"
    backend.set_control("hostA", "gpu", desired_state="on")
    assert backend.desired_state("hostA", "gpu") == "on"
    # per-queue isolation: turning gpu off must not touch cpu
    backend.set_control("hostA", "gpu", desired_state="off")
    assert backend.desired_state("hostA", "cpu") == "on"


# ── multi-tenant isolation (the data-leakage guard) ─────────────────────────────


def test_namespace_isolation(backend):
    """A second backend on a DIFFERENT namespace, SAME server + queue, must not
    see / claim / count this namespace's jobs."""
    other = build_backend(
        backend.name, url=backend.url, namespace=f"other_{uuid.uuid4().hex[:8]}"
    )
    other.ensure_schema()
    try:
        jid = backend.enqueue("cpu", {"secret": 1})
        assert other.get(jid) is None
        assert other.claim("cpu", "intruder", lease_s=30) is None
        assert other.counts("cpu")["queued"] == 0
        # and the reverse: other's job is invisible here
        ojid = other.enqueue("cpu", {})
        assert backend.get(ojid) is None
    finally:
        other.close()
