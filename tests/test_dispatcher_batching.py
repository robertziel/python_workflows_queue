"""Fan-out batching by model affinity.

When ``_process_ready`` finds N ready nodes in one tick, they're sorted by
``required_model`` before being enqueued — so consecutive same-model jobs land
contiguously (the affinity router prefers the same target). Without this, FIFO
scheduling on a fan-out of [A, B, A, B] sends each worker through unnecessary
swaps. Tests pin the ordering contract.
"""

from __future__ import annotations

from queue_workflows import dispatcher


def _node(node_id: str, model: str | None = None, gpu: bool = True) -> dict:
    """Minimal node-dict for the sort path. Only ``model`` and ``id`` matter."""
    return {
        "id": node_id,
        "node": node_id,
        "gpu": gpu,
        "model": model,
        "depends_on": [],
    }


def test_ready_sorted_groups_by_model_in_process_ready(monkeypatch):
    seen: list[str] = []

    monkeypatch.setattr(dispatcher, "_jobs_by_node_id", lambda *_a, **_k: {})
    monkeypatch.setattr(dispatcher, "_should_skip_node", lambda *a, **k: False)
    monkeypatch.setattr(
        dispatcher, "_enqueue",
        lambda run_id, node, run: seen.append(node["id"]),
    )
    calls = {"n": 0}

    def fake_find_ready(_wf, _existing, *, run):
        calls["n"] += 1
        if calls["n"] == 1:
            return [
                _node("n_qwen_1", model="qwen_edit"),
                _node("n_flux_1", model="flux_kontext"),
                _node("n_qwen_2", model="qwen_edit"),
                _node("n_flux_2", model="flux_kontext"),
            ]
        return []
    monkeypatch.setattr(dispatcher, "_find_ready_nodes", fake_find_ready)

    n = dispatcher._process_ready(
        run_id="r", wf={"steps": []}, run={"priority": 100},
    )
    assert n == 4
    qwen_idx = [i for i, name in enumerate(seen) if name.startswith("n_qwen")]
    flux_idx = [i for i, name in enumerate(seen) if name.startswith("n_flux")]
    assert max(qwen_idx) < min(flux_idx) or max(flux_idx) < min(qwen_idx), (
        f"models interleaved: {seen!r}"
    )


def test_ready_sorted_preserves_in_model_order(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(dispatcher, "_jobs_by_node_id", lambda *a, **k: {})
    monkeypatch.setattr(dispatcher, "_should_skip_node", lambda *a, **k: False)
    monkeypatch.setattr(
        dispatcher, "_enqueue",
        lambda run_id, node, run: seen.append(node["id"]),
    )
    calls = {"n": 0}

    def fake_find_ready(_wf, _existing, *, run):
        calls["n"] += 1
        if calls["n"] == 1:
            return [
                _node("first", model="qwen_edit"),
                _node("second", model="qwen_edit"),
                _node("third", model="qwen_edit"),
            ]
        return []
    monkeypatch.setattr(dispatcher, "_find_ready_nodes", fake_find_ready)

    dispatcher._process_ready(
        run_id="r", wf={"steps": []}, run={"priority": 100},
    )
    assert seen == ["first", "second", "third"]


def test_ready_sorted_pushes_modelless_to_end(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(dispatcher, "_jobs_by_node_id", lambda *a, **k: {})
    monkeypatch.setattr(dispatcher, "_should_skip_node", lambda *a, **k: False)
    monkeypatch.setattr(
        dispatcher, "_enqueue",
        lambda run_id, node, run: seen.append(node["id"]),
    )
    calls = {"n": 0}

    def fake_find_ready(_wf, _existing, *, run):
        calls["n"] += 1
        if calls["n"] == 1:
            return [
                _node("modelless"),
                _node("with_model", model="qwen_edit"),
            ]
        return []
    monkeypatch.setattr(dispatcher, "_find_ready_nodes", fake_find_ready)

    dispatcher._process_ready(
        run_id="r", wf={"steps": []}, run={"priority": 100},
    )
    assert seen == ["with_model", "modelless"]
