"""Ollama-specific behaviour beyond the shared contract.

The shared request-accounting + config-echo invariants live in
``tests/test_llm_backend_contract.py`` (which already parametrizes over
``OllamaBackend``). This module pins the bits that are ollama's OWN: the two
vendor endpoint paths, and the deliberate inertness of the lifecycle — ollama is
an externally-managed daemon, so the engine never starts it, never stops it, and
its readiness check performs no I/O. None of these touch a daemon; they assert
pure return values only (the backend makes no network call).
"""

from __future__ import annotations

from queue_workflows.llm_backends.ollama import OllamaBackend


def _make(base_url="http://host:11434") -> OllamaBackend:
    return OllamaBackend(base_url=base_url, parallelism=1, idle_ttl_s=0.0)


def test_server_type_is_ollama():
    assert _make().server_type == "ollama"


def test_chat_url_is_api_chat():
    b = _make()
    assert b.chat_url == "http://host:11434/api/chat"
    assert b.chat_url.endswith("/api/chat")


def test_health_url_is_api_tags():
    b = _make()
    assert b.health_url == "http://host:11434/api/tags"
    assert b.health_url.endswith("/api/tags")


def test_ensure_ready_is_a_noop_and_returns_none():
    """ensure_ready must NOT contact a daemon — it returns None without raising,
    for any model id."""
    assert _make().ensure_ready("anything") is None


def test_is_running_is_true():
    """The external daemon is assumed reachable; is_running is unconditionally True."""
    assert _make().is_running() is True


def test_stop_server_is_false():
    """The library never manages the ollama daemon, so stop_server is a no-op."""
    assert _make().stop_server() is False


def test_base_url_trailing_slash_handling():
    """A base_url WITH a trailing slash still yields a single-slash chat_url."""
    b = _make(base_url="http://host:11434/")
    assert b.base_url == "http://host:11434"
    assert b.chat_url == "http://host:11434/api/chat"
    assert b.health_url == "http://host:11434/api/tags"
