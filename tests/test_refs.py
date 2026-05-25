"""``refs.resolve_ref`` behaviour (the ref-resolution mini-language).

(The workflow-loading half of ai_leads' ``test_registry`` stays in ai_leads;
only the ``resolve_ref``/``_dig``/``_match`` cases move here.)
"""

from __future__ import annotations

import pytest

from queue_workflows.refs import resolve_ref


def test_value_literal():
    assert resolve_ref({"$value": 42}, {}) == 42
    assert resolve_ref({"$value": "hello"}, {}) == "hello"
    assert resolve_ref({"$value": [1, 2]}, {}) == [1, 2]


def test_from_simple_path():
    ctx = {"parcel": {"lat": 50.5, "lon": 22.0, "label": "p1"}}
    assert resolve_ref({"$from": "parcel.lat"}, ctx) == 50.5
    assert resolve_ref({"$from": "parcel.label"}, ctx) == "p1"


def test_from_nested_path():
    ctx = {
        "svinfer": {
            "primary_file": "views/comparison.png",
            "summary": {"top_score": 0.72, "n": 3},
        }
    }
    assert resolve_ref({"$from": "svinfer.primary_file"}, ctx) == "views/comparison.png"
    assert resolve_ref({"$from": "svinfer.summary.top_score"}, ctx) == 0.72


def test_from_missing_path_raises():
    ctx = {"parcel": {"lat": 1.0}}
    with pytest.raises(KeyError, match="missing segment"):
        resolve_ref({"$from": "parcel.lon"}, ctx)


# ── $eq / $ne (skip_if backing) ───────────────────────────────────────────


def test_eq_returns_true_on_match():
    ctx = {"turnout": {"path": "rotate"}}
    assert resolve_ref({"$from": "turnout.path", "$eq": "rotate"}, ctx) is True


def test_eq_returns_false_on_mismatch():
    ctx = {"turnout": {"path": "rotate"}}
    assert resolve_ref({"$from": "turnout.path", "$eq": "direct"}, ctx) is False


def test_ne_inverts_eq():
    ctx = {"turnout": {"path": "rotate"}}
    assert resolve_ref({"$from": "turnout.path", "$ne": "direct"}, ctx) is True
    assert resolve_ref({"$from": "turnout.path", "$ne": "rotate"}, ctx) is False


def test_eq_handles_non_string_values():
    ctx = {"settings": {"count": 4, "active": True}}
    assert resolve_ref({"$from": "settings.count", "$eq": 4}, ctx) is True
    assert resolve_ref({"$from": "settings.count", "$eq": 5}, ctx) is False
    assert resolve_ref({"$from": "settings.active", "$eq": True}, ctx) is True


def test_filter_eq():
    ctx = {"svinfer": {"files": [
        {"rel_path": "inputs/views.json", "kind": "json"},
        {"rel_path": "views/annotated_pano_0.png", "kind": "image"},
        {"rel_path": "views/annotated_pano_1.png", "kind": "image"},
        {"rel_path": "comparison.png", "kind": "image"},
    ]}}
    result = resolve_ref(
        {"$from": "svinfer.files", "$filter": {"kind:eq": "image"}},
        ctx,
    )
    assert len(result) == 3
    assert all(item["kind"] == "image" for item in result)


def test_filter_regex():
    ctx = {"svinfer": {"files": [
        {"rel_path": "views/annotated_pano_0.png", "kind": "image"},
        {"rel_path": "comparison.png", "kind": "image"},
        {"rel_path": "views/annotated_pano_1.png", "kind": "image"},
    ]}}
    result = resolve_ref(
        {
            "$from": "svinfer.files",
            "$filter": {"rel_path:matches": "annotated_pano_\\d+\\.png$"},
        },
        ctx,
    )
    assert len(result) == 2
    assert all("annotated_pano_" in item["rel_path"] for item in result)


def test_filter_combined():
    ctx = {"step": {"files": [
        {"rel_path": "views/a.png", "kind": "image"},
        {"rel_path": "views/a.json", "kind": "json"},
        {"rel_path": "other/b.png", "kind": "image"},
    ]}}
    result = resolve_ref(
        {
            "$from": "step.files",
            "$filter": {"kind:eq": "image", "rel_path:matches": "^views/"},
        },
        ctx,
    )
    assert len(result) == 1
    assert result[0]["rel_path"] == "views/a.png"


def test_plain_dict_recurses():
    ctx = {"parcel": {"lat": 1.5}}
    out = resolve_ref(
        {
            "lat": {"$from": "parcel.lat"},
            "count": {"$value": 3},
            "name": "literal",
        },
        ctx,
    )
    assert out == {"lat": 1.5, "count": 3, "name": "literal"}


def test_literal_scalar_returns_as_is():
    assert resolve_ref(42, {}) == 42
    assert resolve_ref("hello", {}) == "hello"
    assert resolve_ref([1, 2, 3], {}) == [1, 2, 3]


def test_filter_on_non_list_raises():
    with pytest.raises(TypeError, match="\\$filter on non-list"):
        resolve_ref(
            {"$from": "parcel", "$filter": {"kind:eq": "x"}},
            {"parcel": {"lat": 1.0}},
        )
