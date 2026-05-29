"""Per-machine LLM server backends — the ollama / vllm abstraction.

A worker reads its ``(host_label, queue)`` LLM config (migration 0013, via
:func:`queue_workflows.worker_control.llm_config_for`) and drives the matching
:class:`LLMBackend`. The backend owns request accounting (so the
:class:`LLMSupervisor` can free a vllm sidecar's VRAM after an idle window) and
exposes ``chat_url`` for the HOST node to POST against — the library never makes
the LLM HTTP call itself, keeping prompt/response shaping in the consumer.

The factory that reads Postgres and constructs the right backend is built in a
later phase; the backends here are deliberately DB-free and unit-testable with
injected config + a virtual clock.
"""

from __future__ import annotations

from queue_workflows.llm_backends.base import LLMBackend
from queue_workflows.llm_backends.factory import BackendFactory, get_backend
from queue_workflows.llm_backends.ollama import OllamaBackend
from queue_workflows.llm_backends.supervisor import LLMSupervisor, vllm_should_stop
from queue_workflows.llm_backends.vllm import VLLMBackend, VLLMState

__all__ = [
    "LLMBackend",
    "OllamaBackend",
    "VLLMBackend",
    "VLLMState",
    "LLMSupervisor",
    "vllm_should_stop",
    "BackendFactory",
    "get_backend",
]
