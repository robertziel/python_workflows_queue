"""Only VLM-facade node modules may run in the PAR-concurrent VLM pool.

The GPU worker's POOL lane was for lightweight VLM-facade jobs (no in-process
model — they POST to the per-host vLLM server, which batches up to PAR). It
claimed EVERY no-model gpu job (``required_model IS NULL``). But many no-model
gpu jobs are NOT facade work — they self-load a heavy GPU model in-process
(``fence_remove`` / ``car_remove`` erasers, the GroundingDINO+SAM detectors,
``scene_build``, ...). Running several of those concurrently in the pool at
PAR>1 thrashes / OOMs the GPU.

Fix: ``claim_next_gpu_job`` takes ``pool_modules`` — the set of node modules
that ARE genuine VLM-facade (pool-eligible). When non-empty:

  * the POOL lane (``require_model=False``) claims a no-model job ONLY if its
    ``node_module`` is in ``pool_modules``;
  * the INLINE lane (``require_model=True``) additionally claims no-model jobs
    whose module is NOT in ``pool_modules`` (heavy in-process GPU work routes to
    the conc-1 serial lane).

Empty / unset ``pool_modules`` ⇒ byte-identical to the legacy split (every
no-model job is pool-eligible) so other consumers + tests are unaffected. The
two lanes stay disjoint and jointly exhaustive over the gpu queue.
"""

from __future__ import annotations

from queue_workflows import node_queue
from tests._helpers import make_run

POOL = ("facade_parse_walls_vlm",)   # the one genuine VLM-facade module


def _enqueue(node_module: str, *, required_model: str | None = None) -> str:
    run_id = make_run()
    return node_queue.enqueue_node_job(
        run_id=run_id, node_id=f"n-{node_module}", node_module=node_module,
        queue="gpu", required_model=required_model,
    )


def test_inprocess_gpu_job_routes_to_inline_not_pool():
    """A no-model in-process GPU job (``fence_remove``) is NOT pool-eligible:
    the pool lane skips it and the inline lane claims it."""
    job_id = _enqueue("fence_remove")

    # Pool lane refuses it.
    assert node_queue.claim_next_gpu_job(
        0, None, host="h", require_model=False, pool_modules=POOL,
    ) is None

    # Inline lane takes it (heavy in-process work → conc-1 serial lane).
    claimed = node_queue.claim_next_gpu_job(
        0, None, host="h", require_model=True, pool_modules=POOL,
    )
    assert claimed is not None and claimed["id"] == job_id
    assert claimed["claimed_by"] == "h" and claimed["status"] == "running"


def test_vlm_facade_job_stays_in_pool():
    """A pool-eligible VLM-facade job (``facade_parse_walls_vlm``) is claimed by
    the pool lane and NOT by the inline lane."""
    job_id = _enqueue("facade_parse_walls_vlm")

    # Inline lane leaves it for the pool.
    assert node_queue.claim_next_gpu_job(
        0, None, host="h", require_model=True, pool_modules=POOL,
    ) is None

    # Pool lane claims it.
    claimed = node_queue.claim_next_gpu_job(
        0, None, host="h", require_model=False, pool_modules=POOL,
    )
    assert claimed is not None and claimed["id"] == job_id


def test_model_backed_job_unaffected_by_pool_modules():
    """A model-backed job always belongs to the inline lane regardless of
    ``pool_modules``."""
    job_id = _enqueue("qwen_fence_edit", required_model="qwen_edit")

    assert node_queue.claim_next_gpu_job(
        0, None, host="h", require_model=False, pool_modules=POOL,
    ) is None
    claimed = node_queue.claim_next_gpu_job(
        0, "qwen_edit", host="h", require_model=True,
        known_models=["qwen_edit"], pool_modules=POOL,
    )
    assert claimed is not None and claimed["id"] == job_id


def test_empty_pool_modules_is_legacy_split():
    """Unset ``pool_modules`` ⇒ legacy behaviour: EVERY no-model job is
    pool-eligible (pool claims ``fence_remove``), inline claims none of them."""
    job_id = _enqueue("fence_remove")

    # Legacy: inline (require_model=True) does NOT claim a no-model job.
    assert node_queue.claim_next_gpu_job(
        0, None, host="h", require_model=True,
    ) is None
    # Legacy: pool (require_model=False) claims it.
    claimed = node_queue.claim_next_gpu_job(
        0, None, host="h", require_model=False,
    )
    assert claimed is not None and claimed["id"] == job_id


def test_lanes_disjoint_and_exhaustive_over_mixed_queue():
    """With a mix of (facade, heavy-no-model, model-backed) queued, the two
    lanes partition them: pool gets the facade, inline gets the other two."""
    facade = _enqueue("facade_parse_walls_vlm")
    heavy = _enqueue("car_remove")
    backed = _enqueue("osediff_node", required_model="osediff")

    pool_ids, inline_ids = set(), set()
    while (j := node_queue.claim_next_gpu_job(
            0, None, host="h", require_model=False, pool_modules=POOL)) is not None:
        pool_ids.add(j["id"])
    while (j := node_queue.claim_next_gpu_job(
            0, None, host="h", require_model=True,
            known_models=["osediff"], pool_modules=POOL)) is not None:
        inline_ids.add(j["id"])

    assert pool_ids == {facade}
    assert inline_ids == {heavy, backed}
