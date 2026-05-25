"""Unit tests for ``node_executor._invoke`` — the introspection adapter that
dispatches a ``workflow_node_jobs`` row's ``inputs`` dict to the matching node
module's ``run()`` function.

(Renamed from the ai_leads ``test_tasks_invoke`` — ``tasks.py`` is gone.)

Covers the three contract modes:
- **Explicit kwargs** — one kwarg per schema input name, with ``str → Path``
  coercion when the annotation says ``Path`` (or a union containing ``Path``).
- **Runtime-injected** — ``out``, ``model_handle``, ``status_callback`` filled
  by the dispatcher.
- **Catch-all** — ``run(*, inputs, out, ...)`` consumes the whole inputs dict.

The fake node modules resolve via ``set_node_module_package``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import queue_workflows
from queue_workflows.node_executor import _invoke


@pytest.fixture(autouse=True)
def _fake_node_pkg():
    queue_workflows.set_node_module_package("qwf_invoke_nodes")
    yield


def _install_fake_node(name: str, run_fn):
    mod = types.ModuleType(f"qwf_invoke_nodes.{name}")
    mod.run = run_fn
    sys.modules[f"qwf_invoke_nodes.{name}"] = mod


def test_invoke_coerces_str_to_path_for_path_annotated_param():
    captured: dict = {}

    def run(parcel_file: Path, out: Path):
        captured["parcel_file"] = parcel_file
        captured["out"] = out
        return {}

    _install_fake_node("_test_path_coerce", run)
    result = _invoke(
        "_test_path_coerce",
        inputs={"parcel_file": "/tmp/x.json"},
        out=Path("/tmp/out"),
        handle=None,
    )

    assert isinstance(captured["parcel_file"], Path)
    assert captured["parcel_file"] == Path("/tmp/x.json")
    assert result == {"context_delta": {}}


def test_invoke_coerces_inside_path_pipe_none_union():
    captured: dict = {}

    def run(maybe_path: Path | None, out: Path):
        captured["maybe_path"] = maybe_path
        return {}

    _install_fake_node("_test_path_union", run)
    _invoke(
        "_test_path_union",
        inputs={"maybe_path": "/tmp/foo"},
        out=Path("/tmp/out"),
        handle=None,
    )
    assert isinstance(captured["maybe_path"], Path)


def test_invoke_passes_runtime_injected_kwargs():
    captured: dict = {}

    def run(x: int, out: Path, model_handle, status_callback):
        captured.update(dict(x=x, out=out, mh=model_handle, sc=status_callback))
        return {}

    _install_fake_node("_test_runtime_inject", run)
    handle = object()
    _invoke(
        "_test_runtime_inject",
        inputs={"x": 7},
        out=Path("/tmp/out"),
        handle=handle,
    )

    assert captured["x"] == 7
    assert captured["out"] == Path("/tmp/out")
    assert captured["mh"] is handle
    assert captured["sc"] is None


def test_invoke_catchall_inputs_contract():
    captured: dict = {}

    def run(*, inputs=None, out=None, model_handle=None, status_callback=None):
        captured["inputs"] = inputs
        captured["out"] = out
        return {"context_delta": {"foo": "bar"}}

    _install_fake_node("_test_catchall", run)
    result = _invoke(
        "_test_catchall",
        inputs={"a": 1, "b": 2},
        out=Path("/tmp"),
        handle=None,
    )

    assert captured["inputs"] == {"a": 1, "b": 2}
    assert result == {"context_delta": {"foo": "bar"}}


def test_invoke_wraps_non_dict_return_in_context_delta():
    def run(x: int, out: Path):
        return {"some_other_key": "v"}

    _install_fake_node("_test_wrap", run)
    result = _invoke(
        "_test_wrap",
        inputs={"x": 1},
        out=Path("/tmp"),
        handle=None,
    )
    assert result == {"context_delta": {"some_other_key": "v"}}


def test_invoke_wraps_none_return_in_empty_context_delta():
    def run(x: int, out: Path):
        return None

    _install_fake_node("_test_none", run)
    result = _invoke(
        "_test_none",
        inputs={"x": 1},
        out=Path("/tmp"),
        handle=None,
    )
    assert result == {"context_delta": {}}


def test_invoke_null_input_falls_through_to_signature_default():
    captured: dict = {}

    def run(n: int, fov: float = 90.0, size: int = 640, out: Path = None):
        captured["n"] = n
        captured["fov"] = fov
        captured["size"] = size
        return {}

    _install_fake_node("_test_null_fallthrough", run)
    _invoke(
        "_test_null_fallthrough",
        inputs={"n": 4, "fov": None, "size": None},
        out=Path("/tmp"),
        handle=None,
    )
    assert captured["n"] == 4
    assert captured["fov"] == 90.0
    assert captured["size"] == 640


def test_invoke_missing_required_arg_raises_typeerror():
    def run(views_file: Path, parcel_file: Path, out: Path):
        return {}

    _install_fake_node("_test_missing", run)
    with pytest.raises(TypeError, match="parcel_file"):
        _invoke(
            "_test_missing",
            inputs={"views_file": "/tmp/v.json"},
            out=Path("/tmp"),
            handle=None,
        )
