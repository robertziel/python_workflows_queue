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


def fits_within(vram_total_mb: int | None, *, headroom: float = 1.0) -> list[str]:
    """The registered model ids whose ``est_vram_gb`` fits a machine with
    ``vram_total_mb`` of total GPU VRAM — sorted, the worker's advertised
    ``fits_models``.

    A model fits iff ``est_vram_gb * 1024 * headroom <= vram_total_mb``.
    ``headroom`` (>= 1.0) reserves margin over the bare estimate for runtime
    overshoot (activations, fragmentation); default 1.0 = the raw estimate.

    Semantics of the edges:
      * ``vram_total_mb is None`` (capacity unknown — no GPU probe / cold worker)
        ⇒ return ALL known ids. The claim gate must NOT wedge the queue on a
        worker that simply hasn't measured its VRAM yet; the existing capability
        gate already falls back to claim-any on an empty/uninformative set, and
        a "fits everything" advertisement keeps that behaviour.
      * A model with ``est_vram_gb <= 0`` (unset / informational) fits anywhere —
        it carries no capacity claim, so it is never filtered out.
    """
    if vram_total_mb is None:
        return known_ids()
    cap = int(vram_total_mb)
    hr = max(1.0, float(headroom))
    out = [
        mid for mid, spec in MODELS.items()
        if (spec.est_vram_gb or 0.0) <= 0.0
        or float(spec.est_vram_gb) * 1024.0 * hr <= cap
    ]
    return sorted(out)


def clear_for_tests() -> None:
    """TEST-ONLY. Drops the registry so a test can populate it freshly."""
    MODELS.clear()
