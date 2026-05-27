"""Redis :class:`StorageBackend`.

Redis has no SQL, no ``SKIP LOCKED`` and no cross-key ACID transaction, so the
contract is reproduced with the tools Redis *does* give us:

  * **Atomic claim / terminal / re-queue via Lua.** A registered Lua script runs
    server-side as one indivisible step, which is how we get the engine's two
    keystone guarantees on Redis: claim-exactly-once (no two callers pop the same
    job) and the atomic outbox (go terminal *and* append the event, or neither —
    a single script, so a crash can't split them).
  * **Priority + FIFO via a sorted set per queue** (``…:q:<queue>``): score is
    ``-priority`` so ``ZPOPMIN`` takes the highest priority first, and equal
    scores break ties by member — a zero-padded monotonic sequence — i.e. FIFO.
  * **Lease/reclaim via a per-queue running sorted set** scored by lease expiry;
    ``reclaim_expired`` is a ``ZRANGEBYSCORE … now`` sweep.
  * **Wake via pub/sub** on a per-namespace channel (payload = queue). Pub/sub is
    fire-and-forget — a subscriber that is down misses the message — which is why
    the worker keeps a safety poll exactly as it does behind PG ``LISTEN``.

Everything is under the key prefix ``qw:<namespace>:`` so two tenants on one
Redis server are fully isolated (the data-leakage guard). NOTE: keys are derived
server-side inside the scripts, so this targets a **single Redis instance**, not
Cluster (which would require all keys in one hash slot).
"""

from __future__ import annotations

import json
import time
from typing import Any

import redis  # the redis-py client (absolute import; not this module)

from queue_workflows.backends.base import Event, Job, StorageBackend, WakeListener

# ── Lua (each script is ONE atomic server-side step) ─────────────────────────

_LUA_ENQUEUE = """
local p = ARGV[1]
local seq = redis.call('INCR', p..'seq')
local jobkey = p..'job:'..ARGV[2]
redis.call('HSET', jobkey, 'id',ARGV[2],'namespace',ARGV[7],'queue',ARGV[3],
  'status','queued','payload',ARGV[4],'priority',ARGV[5],'attempts','0',
  'claimed_by','','lease_expires_at','','result','','error','',
  'created_at',ARGV[6],'updated_at',ARGV[6])
redis.call('ZADD', KEYS[1], -tonumber(ARGV[5]), string.format('%020d|%s', seq, ARGV[2]))
redis.call('SADD', p..'queues', ARGV[3])
redis.call('INCR', p..'cnt:'..ARGV[3]..':queued')
redis.call('PUBLISH', ARGV[8], ARGV[3])
return ARGV[2]
"""

_LUA_CLAIM = """
local p = ARGV[1]
local popped = redis.call('ZPOPMIN', KEYS[1])
if popped[1] == nil then return nil end
local member = popped[1]
local bar = string.find(member, '|', 1, true)
local jobid = string.sub(member, bar + 1)
local jobkey = p..'job:'..jobid
local attempts = tonumber(redis.call('HGET', jobkey, 'attempts') or '0') + 1
local lease_at = tonumber(ARGV[5]) + tonumber(ARGV[4])
redis.call('HSET', jobkey, 'status','running','claimed_by',ARGV[3],
  'lease_expires_at', tostring(lease_at), 'attempts', tostring(attempts),
  'updated_at', ARGV[5])
redis.call('ZADD', p..'running:'..ARGV[2], lease_at, jobid)
redis.call('DECR', p..'cnt:'..ARGV[2]..':queued')
redis.call('INCR', p..'cnt:'..ARGV[2]..':running')
return redis.call('HGETALL', jobkey)
"""

_LUA_RENEW = """
local p = ARGV[1]
local st = redis.call('HGET', KEYS[1], 'status')
local cb = redis.call('HGET', KEYS[1], 'claimed_by')
if st == 'running' and cb == ARGV[2] then
  local q = redis.call('HGET', KEYS[1], 'queue')
  local lease_at = tonumber(ARGV[4]) + tonumber(ARGV[3])
  redis.call('HSET', KEYS[1], 'lease_expires_at', tostring(lease_at), 'updated_at', ARGV[4])
  redis.call('ZADD', p..'running:'..q, lease_at, ARGV[5])
  return 1
end
return 0
"""

# Mark terminal (+ optionally append one event) — the atomic outbox. ARGV[7]
# empty ⇒ plain mark_* with no event. Already-terminal/missing ⇒ nil (no-op, and
# NO event written): the idempotency guard.
_LUA_TERMINAL = """
local p = ARGV[1]
local st = redis.call('HGET', KEYS[1], 'status')
if (not st) or st == 'completed' or st == 'failed' then return nil end
local q = redis.call('HGET', KEYS[1], 'queue')
redis.call('HSET', KEYS[1], 'status',ARGV[2],'result',ARGV[3],'error',ARGV[4],'updated_at',ARGV[5])
redis.call('ZREM', p..'running:'..q, ARGV[6])
redis.call('DECR', p..'cnt:'..q..':'..st)
redis.call('INCR', p..'cnt:'..q..':'..ARGV[2])
if ARGV[7] ~= '' then
  local eseq = redis.call('INCR', p..'eventseq')
  redis.call('HSET', p..'event:'..eseq, 'seq',eseq,'job_id',ARGV[6],'queue',q,
    'event_type',ARGV[7],'detail',ARGV[8],'created_at',ARGV[5])
  redis.call('RPUSH', p..'events', eseq)
end
return redis.call('HGETALL', KEYS[1])
"""

_LUA_REQUEUE = """
local p = ARGV[1]
local st = redis.call('HGET', KEYS[1], 'status')
if (not st) or st == 'completed' or st == 'failed' then return nil end
local q = redis.call('HGET', KEYS[1], 'queue')
local pri = tonumber(redis.call('HGET', KEYS[1], 'priority') or '0')
redis.call('HSET', KEYS[1], 'status','queued','claimed_by','','lease_expires_at','','updated_at',ARGV[2])
redis.call('ZREM', p..'running:'..q, ARGV[4])
local seq = redis.call('INCR', p..'seq')
redis.call('ZADD', p..'q:'..q, -pri, string.format('%020d|%s', seq, ARGV[4]))
redis.call('DECR', p..'cnt:'..q..':'..st)
redis.call('INCR', p..'cnt:'..q..':queued')
redis.call('PUBLISH', ARGV[3], q)
return redis.call('HGETALL', KEYS[1])
"""

_LUA_RECLAIM = """
local p = ARGV[1]
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[3])
local out = {}
for i, jobid in ipairs(expired) do
  local jobkey = p..'job:'..jobid
  if redis.call('HGET', jobkey, 'status') == 'running' then
    redis.call('HSET', jobkey, 'status','queued','claimed_by','','lease_expires_at','','updated_at',ARGV[3])
    redis.call('ZREM', KEYS[1], jobid)
    local seq = redis.call('INCR', p..'seq')
    local pri = tonumber(redis.call('HGET', jobkey, 'priority') or '0')
    redis.call('ZADD', p..'q:'..ARGV[2], -pri, string.format('%020d|%s', seq, jobid))
    redis.call('DECR', p..'cnt:'..ARGV[2]..':running')
    redis.call('INCR', p..'cnt:'..ARGV[2]..':queued')
    redis.call('PUBLISH', ARGV[4], ARGV[2])
    out[#out + 1] = jobid
  end
end
return out
"""


class RedisBackend(StorageBackend):
    name = "redis"

    def __init__(self, *, url: str, namespace: str = "") -> None:
        super().__init__(url=url, namespace=namespace)
        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._prefix = f"qw:{self.namespace}:"
        self._wake_channel = f"{self._prefix}wake"
        self._enqueue = self._r.register_script(_LUA_ENQUEUE)
        self._claim = self._r.register_script(_LUA_CLAIM)
        self._renew = self._r.register_script(_LUA_RENEW)
        self._terminal = self._r.register_script(_LUA_TERMINAL)
        self._requeue = self._r.register_script(_LUA_REQUEUE)
        self._reclaim = self._r.register_script(_LUA_RECLAIM)

    def _k(self, suffix: str) -> str:
        return self._prefix + suffix

    def ensure_schema(self) -> None:
        self._r.ping()  # nothing to create; just verify connectivity

    def close(self) -> None:
        try:
            self._r.close()
        except Exception:
            pass

    # ── enqueue / claim / lease ────────────────────────────────────────────────

    def enqueue(self, queue, payload, *, job_id=None, priority=0) -> str:
        import uuid

        jid = job_id or uuid.uuid4().hex
        self._enqueue(
            keys=[self._k(f"q:{queue}")],
            args=[self._prefix, jid, queue, json.dumps(payload or {}),
                  str(int(priority)), repr(time.time()), self.namespace,
                  self._wake_channel],
        )
        return jid

    def claim(self, queue, worker, *, lease_s) -> Job | None:
        reply = self._claim(
            keys=[self._k(f"q:{queue}")],
            args=[self._prefix, queue, worker, repr(float(lease_s)), repr(time.time())],
        )
        return self._job(reply)

    def renew_lease(self, job_id, worker, *, lease_s) -> bool:
        return bool(self._renew(
            keys=[self._k(f"job:{job_id}")],
            args=[self._prefix, worker, repr(float(lease_s)), repr(time.time()), job_id],
        ))

    def reclaim_expired(self, *, queue=None) -> list[str]:
        queues = [queue] if queue is not None else list(self._r.smembers(self._k("queues")))
        now = repr(time.time())
        out: list[str] = []
        for q in queues:
            ids = self._reclaim(
                keys=[self._k(f"running:{q}")],
                args=[self._prefix, q, now, self._wake_channel],
            )
            out.extend(ids or [])
        return out

    def requeue_for_retry(self, job_id) -> Job | None:
        reply = self._requeue(
            keys=[self._k(f"job:{job_id}")],
            args=[self._prefix, repr(time.time()), self._wake_channel, job_id],
        )
        return self._job(reply)

    # ── terminal transitions ────────────────────────────────────────────────────

    def _do_terminal(self, job_id, status, *, result, error, event_type, detail):
        reply = self._terminal(
            keys=[self._k(f"job:{job_id}")],
            args=[self._prefix, status,
                  json.dumps(result) if result is not None else "",
                  error or "", repr(time.time()), job_id,
                  event_type or "", json.dumps(detail or {})],
        )
        return self._job(reply)

    def mark_completed(self, job_id, *, result=None) -> Job | None:
        return self._do_terminal(job_id, "completed", result=result, error=None,
                                 event_type="", detail=None)

    def mark_failed(self, job_id, *, error=None) -> Job | None:
        return self._do_terminal(job_id, "failed", result=None, error=error,
                                 event_type="", detail=None)

    def complete_with_event(self, job_id, event_type, *, result=None, detail=None):
        return self._do_terminal(job_id, "completed", result=result, error=None,
                                 event_type=event_type, detail=detail)

    def fail_with_event(self, job_id, event_type, *, error=None, detail=None):
        return self._do_terminal(job_id, "failed", result=None, error=error,
                                 event_type=event_type, detail=detail)

    # ── reads ────────────────────────────────────────────────────────────────────

    def get(self, job_id) -> Job | None:
        h = self._r.hgetall(self._k(f"job:{job_id}"))
        return self._job_from_dict(h) if h else None

    def counts(self, queue) -> dict[str, int]:
        keys = [self._k(f"cnt:{queue}:{s}") for s in
                ("queued", "running", "completed", "failed")]
        vals = self._r.mget(keys)
        return {
            s: max(0, int(v or 0))
            for s, v in zip(("queued", "running", "completed", "failed"), vals)
        }

    def events(self, *, since=0, limit=1000) -> list[Event]:
        seqs = self._r.lrange(self._k("events"), 0, -1)
        wanted = [s for s in seqs if int(s) > since][:limit]
        out: list[Event] = []
        for s in wanted:
            h = self._r.hgetall(self._k(f"event:{s}"))
            if not h:
                continue
            out.append(Event(
                seq=int(h["seq"]), job_id=h["job_id"], namespace=self.namespace,
                queue=h.get("queue") or None, event_type=h["event_type"],
                detail=json.loads(h.get("detail") or "{}"),
                created_at=float(h.get("created_at") or 0),
            ))
        return out

    # ── wake ──────────────────────────────────────────────────────────────────────

    def notify(self, queue) -> None:
        self._r.publish(self._wake_channel, queue)

    def subscribe(self, *queues) -> WakeListener:
        return _RedisWakeListener(self._r, self._wake_channel, frozenset(queues))

    # ── heartbeats + control ───────────────────────────────────────────────────────

    def heartbeat(self, host, queue, *, current_model=None, stale_after_s=30.0) -> None:
        key = self._k(f"worker:{queue}:{host}")
        self._r.hset(key, mapping={
            "host": host, "queue": queue, "current_model": current_model or "",
            "last_seen": repr(time.time()),
        })
        self._r.expire(key, max(1, int(stale_after_s)))

    def workers(self, queue) -> list[dict[str, Any]]:
        out = []
        for key in self._r.scan_iter(match=self._k(f"worker:{queue}:*"), count=100):
            h = self._r.hgetall(key)
            if h:
                out.append({"host": h.get("host"), "queue": h.get("queue"),
                            "current_model": h.get("current_model") or None,
                            "last_seen": float(h.get("last_seen") or 0)})
        return out

    def set_control(self, host, queue, *, desired_state, stop_policy="hard",
                    requested_by=None) -> None:
        self._r.hset(self._k(f"control:{host}:{queue}"), mapping={
            "desired_state": desired_state, "stop_policy": stop_policy,
            "requested_by": requested_by or "", "updated_at": repr(time.time()),
        })

    def desired_state(self, host, queue) -> str:
        st = self._r.hget(self._k(f"control:{host}:{queue}"), "desired_state")
        return "off" if st == "off" else "on"

    # ── helpers ──────────────────────────────────────────────────────────────────

    def _job(self, reply) -> Job | None:
        """Script HGETALL replies arrive as a flat ``[k, v, k, v, …]`` list."""
        if not reply:
            return None
        if isinstance(reply, list):
            reply = dict(zip(reply[::2], reply[1::2]))
        return self._job_from_dict(reply)

    def _job_from_dict(self, h: dict[str, str]) -> Job:
        return Job(
            id=h["id"], queue=h["queue"], namespace=h.get("namespace", self.namespace),
            status=h["status"], payload=json.loads(h.get("payload") or "{}"),
            priority=int(h.get("priority") or 0), attempts=int(h.get("attempts") or 0),
            claimed_by=(h.get("claimed_by") or None),
            lease_expires_at=(float(h["lease_expires_at"]) if h.get("lease_expires_at") else None),
            result=(json.loads(h["result"]) if h.get("result") else None),
            error=(h.get("error") or None),
            created_at=float(h.get("created_at") or 0),
            updated_at=float(h.get("updated_at") or 0),
        )


class _RedisWakeListener:
    """pub/sub subscription; ``wait`` returns the queue payload if subscribed."""

    def __init__(self, client, channel: str, queues: frozenset[str]) -> None:
        self._client = client
        self._channel = channel
        self._queues = queues
        self._pubsub = None

    def __enter__(self):
        self._pubsub = self._client.pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe(self._channel)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._pubsub is not None:
            try:
                self._pubsub.close()
            finally:
                self._pubsub = None

    def wait(self, timeout: float) -> str | None:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            msg = self._pubsub.get_message(timeout=remaining)
            if msg is None:
                continue
            if msg.get("type") != "message":
                continue
            payload = msg.get("data")
            if not self._queues or payload in self._queues:
                return payload
