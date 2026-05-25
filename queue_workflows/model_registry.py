"""GPU-worker model registry.

A ``ModelSpec`` names a model loadable by the long-lived GPU worker.
Nodes with ``gpu: true`` in the workflow JSON declare ``model: "<id>"``
pointing here. The GPU worker resolves the id → loader and caches the
result across consecutive jobs so same-model back-to-back work skips
reload.

This is the registration *target*: the engine owns the registry; the host
registers ITS ModelSpecs INTO it (via the injected builtin-model registrar,
plan §2c). Keep this module cheap to import — no torch, no diffusers at module
load. Loader functions defer heavy imports until called.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ModelHandle = Any  # whatever the loader returns — a diffusers pipe, a bespoke object, etc.
ModelLoader = Callable[[], ModelHandle]
ModelUnloader = Callable[[ModelHandle], None]


@dataclass
class ModelSpec:
    """One entry in the model registry.

    - ``id``           — stable string node JSONs point at.
    - ``loader``       — zero-arg factory returning the loaded handle.
    - ``unloader``     — optional; defaults to the GPU-worker's generic
                          drop-ref + gc + empty_cache + malloc_trim path.
    - ``est_vram_gb``  — scheduler hint (today informational only;
                          future: admission control).
    """
    id: str
    loader: ModelLoader
    unloader: ModelUnloader | None = None
    est_vram_gb: float = 0.0


# Populated at import by registry consumers. Keep the dict in one
# place so tests can swap entries via ``MODELS[id] = fake_spec``.
MODELS: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> None:
    """Register a model spec. Replaces any existing entry for the id."""
    MODELS[spec.id] = spec


def get(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id]
    except KeyError as e:
        raise KeyError(
            f"unknown model id {model_id!r}. "
            f"Known: {sorted(MODELS)!r}. "
            f"Register via model_registry.register(ModelSpec(...))."
        ) from e


def known_ids() -> list[str]:
    return sorted(MODELS)


def clear_for_tests() -> None:
    """TEST-ONLY. Drops the registry so a test can populate it freshly."""
    MODELS.clear()
