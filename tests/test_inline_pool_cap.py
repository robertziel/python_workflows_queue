"""The INLINE diffusion lane must respect the per-machine PAR cap.

A GPU worker runs two claim lanes on one host process: the INLINE diffusion
lane (``run_once`` → ``_claim`` → ``require_model=True``, serial) and the VLM
POOL lane (a PAR-sized feeder, ``require_model=False``). A machine's TOTAL
concurrent node-jobs (inline + pool) must never exceed PAR
(= ``worker_controls.llm_parallelism``).

The pool feeder already enforces its half: ``_pool_budget = PAR - inline_running``
(it yields a slot to a running inline diffusion). But the inline lane was the
MISSING mirror — ``run_once`` claimed + ran with NO regard for how many pool
jobs were already in flight. So when the pool had claimed a job while the inline
lane was momentarily idle, and then a model-backed job arrived, the inline lane
ran it ON TOP → the machine held ``pool_inflight + 1`` jobs, i.e. PAR+1.

Operator symptom: PAR=1 machines (box-b, box-a) showing ``2/1 node jobs`` —
one inline diffusion stacked on one pool job — while an idle peer (box-a2) sat
at ``0/1``. The over-claim drains the shared queue past the machine's own
capacity, so a busy box keeps taking work that should spill to an idle one.

These tests pin the inline lane to the SAME per-machine budget the pool uses:
a machine at PAR claims no further inline job (returns False without claiming);
below PAR it claims normally.
"""

from __future__ import annotations

from queue_workflows import claim_worker, worker_control


class _Cache:
    """Stand-in for the GPU ModelCache (no warm model loaded)."""

    current_model = None


def _worker(host: str = "solo") -> claim_worker.ClaimWorker:
    return claim_worker.ClaimWorker(queue="gpu", host=host, model_cache=_Cache())


def _set_par(monkeypatch, par: int) -> None:
    monkeypatch.setattr(
        worker_control, "llm_config_for",
        lambda h, q: worker_control.LLMConfig(parallelism=par),
    )


def test_inline_lane_defers_when_pool_fills_par(monkeypatch):
    """PAR=1 with one pool job already in flight ⇒ the machine is at capacity,
    so ``run_once`` must NOT even attempt an inline claim (it returns False
    without calling ``_claim``). This is the regression: pre-fix, ``run_once``
    ignored ``_pool_inflight`` and claimed a 2nd (inline) job → ``2/1``."""
    _set_par(monkeypatch, 1)
    w = _worker()
    # Pool lane already holds the machine's single PAR slot.
    with w._pool_lock:
        w._pool_inflight = 1

    claim_calls = {"n": 0}

    def spy_claim():
        claim_calls["n"] += 1
        return None

    monkeypatch.setattr(w, "_claim", spy_claim)

    assert w.run_once() is False
    assert claim_calls["n"] == 0, (
        "inline lane claimed while the machine was already at PAR — this is the "
        "inline+pool over-claim (the 2/1 bug)"
    )
    # A skipped inline claim must not leak a reserved slot.
    assert w._inline_running is False


def test_inline_lane_claims_when_pool_idle(monkeypatch):
    """PAR=1 and the pool idle ⇒ the inline lane claims + runs as normal (the
    fix must not block legitimate diffusion work)."""
    _set_par(monkeypatch, 1)
    w = _worker()
    assert w._pool_inflight == 0

    fake = {"id": "j1", "run_id": "r1", "node_id": "n1", "node_module": "x"}
    monkeypatch.setattr(w, "_claim", lambda: fake)
    ran = {"job": None}

    def spy_run(job):
        ran["job"] = job
        # Inline must be marked busy while the node runs (so the pool budgets
        # PAR-1 alongside it).
        assert w._inline_running is True
        return True

    monkeypatch.setattr(w, "_run_node", spy_run)

    assert w.run_once() is True
    assert ran["job"] is fake
    # Slot released after the job finishes.
    assert w._inline_running is False


def test_inline_lane_claims_below_par_alongside_pool(monkeypatch):
    """PAR=2 with one pool job in flight ⇒ the inline lane MAY still claim one
    (1 inline + 1 pool = 2 = PAR). The cap is the TOTAL, not 1."""
    _set_par(monkeypatch, 2)
    w = _worker()
    with w._pool_lock:
        w._pool_inflight = 1   # one of two PAR slots taken by the pool

    fake = {"id": "j2", "run_id": "r2", "node_id": "n2", "node_module": "x"}
    monkeypatch.setattr(w, "_claim", lambda: fake)
    monkeypatch.setattr(w, "_run_node", lambda job: True)

    assert w.run_once() is True


def test_inline_lane_defers_when_pool_fills_all_par_slots(monkeypatch):
    """PAR=2 with TWO pool jobs in flight ⇒ the machine is full; the inline lane
    must defer (no inline claim), capping the machine at PAR=2 total."""
    _set_par(monkeypatch, 2)
    w = _worker()
    with w._pool_lock:
        w._pool_inflight = 2   # both PAR slots taken by the pool

    claim_calls = {"n": 0}
    monkeypatch.setattr(
        w, "_claim", lambda: claim_calls.__setitem__("n", claim_calls["n"] + 1),
    )

    assert w.run_once() is False
    assert claim_calls["n"] == 0
    assert w._inline_running is False
