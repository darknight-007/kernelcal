"""
distinction_game
================

Multi-source annotation as kernel dynamics. Implements the per-tile pipeline
specified in ``docs/distinction-game-design.md``: each external label
source (OSM, Grounding-DINO, SAM, Mask R-CNN heads, human annotators) is
treated as a *distinction kernel* `k_s` over a shared ground set of regions
``R_t``; per-source label semantics live in a small confusion matrix
``Q_s(ŷ | c)``; per-source weights ``λ_s`` are MaxCal-fitted to free
ground-truth (OSM-anchored regions). Output is a fused
:class:`SceneGraph` whose nodes carry both a category posterior over a
fixed taxonomy ``c*`` and full provenance (which source said what, with
what native confidence).

Phasing
-------

- **PR-1 (this version, v0.4):** Schema + uniform-λ fusion baseline. No
  Lagrange fit, no spectral diagnostics, no Q_s refit. Just enough for
  ``deepgis-xr`` to start producing fused scene graphs from the live
  kernel containers.

- **PR-2:** ``deepgis-xr`` orchestrator endpoint that calls every
  configured kernel for a viewport, wraps each output as a
  :class:`KernelClaim`, calls :func:`build_scene_graph`, and serves the
  result to the Cesium frontend.

- **PR-3:** Real :func:`fit_kernel_mix` (Lagrange fit against
  OSM-anchor + bounded-disagreement + coverage-matching constraints,
  per §4 of the design doc); spectral diagnostics on the per-tile
  region graph (§5); ``Q_s`` refit from supervised pairs (§8.1, R1
  rung).

Public API
----------

The module re-exports the minimal surface needed by the orchestrator
in ``deepgis-xr``::

    from kernelcal.distinction_game import (
        Taxonomy, PHX_URBAN_V0,
        Region, KernelClaim,
        ConfusionMatrix, default_q_s,
        KernelMixFit, fit_kernel_mix,
        SceneNode, SceneEdge, SceneGraph,
        build_scene_graph,
    )

See ``docs/distinction-game-design.md`` §3, §4, §11 for the math
identities each piece implements.
"""

from __future__ import annotations

from .taxonomy import (
    Taxonomy,
    PHX_URBAN_V0,
)
from .region import (
    Region,
    KernelClaim,
)
from .q_s import (
    ConfusionMatrix,
    default_q_s,
    default_q_s_table,
    available_sources,
)
from .kernel_mix import (
    KernelMixFit,
    fit_kernel_mix,
    uniform_lambdas,
)
from .scene_graph import (
    SceneNode,
    SceneEdge,
    SceneGraph,
    Viewport,
    build_scene_graph,
)

__all__ = [
    "Taxonomy",
    "PHX_URBAN_V0",
    "Region",
    "KernelClaim",
    "ConfusionMatrix",
    "default_q_s",
    "default_q_s_table",
    "available_sources",
    "KernelMixFit",
    "fit_kernel_mix",
    "uniform_lambdas",
    "SceneNode",
    "SceneEdge",
    "SceneGraph",
    "Viewport",
    "build_scene_graph",
]
