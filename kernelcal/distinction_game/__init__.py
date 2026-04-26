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
    JointFitResult,
    KernelMixFit,
    fit_kernel_mix,
    fit_kernel_mix_em,
    manual_lambdas,
    uniform_lambdas,
)
from .scene_graph import (
    SceneNode,
    SceneEdge,
    SceneGraph,
    Viewport,
    build_scene_graph,
)
from .q_s_fit import (
    DEFAULT_OSM_ANCHOR_SOURCES,
    QsFitResult,
    bayesian_refit_q_s,
    bayesian_refit_q_s_table,
    count_evidence_triples,
    evidence_triples_from_many_scene_graphs,
    evidence_triples_from_scene_graph,
)
from .spectral import (
    cg_node_index_map,
    graph_smooth_posteriors,
    posteriors_array_from_scene_graph,
    spectral_consistency_score,
)
from .fit_pipeline import (
    DistinctionGameFit,
    fit_distinction_game,
)
from .factor_graph import (
    BPResult,
    Factor,
    FactorGraph,
    PairwiseAssociationFactor,
    PairwiseSpatialFactor,
    PairwiseTemporalFactor,
    UnaryPerceptualFactor,
    Variable,
    loopy_bp,
)
from .collapse import (
    FusedSceneGraph,
    collapse_scene_graphs,
    data_associate,
    temporal_links,
)
from .geometry import (
    # Superquadric core
    EPS_MAX,
    EPS_MIN,
    FitDiagnostics,
    PACKED_BYTES,
    PACKED_SPECTRUM_BYTES,
    Superquadric,
    SuperquadricFit,
    fit_superquadric,
    fit_tree,
    pack_superquadric,
    packed_size,
    superquadric_box,
    superquadric_cylinder,
    superquadric_ellipsoid,
    superquadric_sphere,
    unpack_superquadric,
    # Properties (PR-5.5)
    PropertyId,
    PropertySpec,
    decode_property_trailer,
    encode_property_trailer,
    get_spec,
    # Accumulators (PR-5.5)
    SuperquadricPropertyStore,
    WelfordAccumulator,
    merge_property_stores,
    store_from_decoded_trailer,
    # Spectrum sidecar (PR-5.5)
    SpectrumAccumulator,
    SpectrumPacket,
    compress_spectrum,
    decompress_spectrum,
    # Attribution (PR-5.5)
    LidarIntensityAttributor,
    MicaSenseAttributor,
    OceanOpticsAttributor,
    SQSpatialIndex,
    evi,
    gndvi,
    ndre,
    ndvi,
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
    "JointFitResult",
    "fit_kernel_mix",
    "fit_kernel_mix_em",
    "manual_lambdas",
    "uniform_lambdas",
    "SceneNode",
    "SceneEdge",
    "SceneGraph",
    "Viewport",
    "build_scene_graph",
    # PR-3 surface
    "DEFAULT_OSM_ANCHOR_SOURCES",
    "QsFitResult",
    "bayesian_refit_q_s",
    "bayesian_refit_q_s_table",
    "count_evidence_triples",
    "evidence_triples_from_many_scene_graphs",
    "evidence_triples_from_scene_graph",
    "graph_smooth_posteriors",
    "spectral_consistency_score",
    "cg_node_index_map",
    "posteriors_array_from_scene_graph",
    "DistinctionGameFit",
    "fit_distinction_game",
    # PR-4 factor graph + collapse surface
    "BPResult",
    "Factor",
    "FactorGraph",
    "PairwiseAssociationFactor",
    "PairwiseSpatialFactor",
    "PairwiseTemporalFactor",
    "UnaryPerceptualFactor",
    "Variable",
    "loopy_bp",
    "FusedSceneGraph",
    "collapse_scene_graphs",
    "data_associate",
    "temporal_links",
    # PR-5 geometry surface (Superquadric primitives + fit + codec)
    "EPS_MIN",
    "EPS_MAX",
    "Superquadric",
    "superquadric_box",
    "superquadric_cylinder",
    "superquadric_ellipsoid",
    "superquadric_sphere",
    "FitDiagnostics",
    "SuperquadricFit",
    "fit_superquadric",
    "fit_tree",
    "PACKED_BYTES",
    "PACKED_SPECTRUM_BYTES",
    "pack_superquadric",
    "unpack_superquadric",
    "packed_size",
    # PR-5.5 appearance: properties + accumulators + spectrum + attribution
    "PropertyId",
    "PropertySpec",
    "encode_property_trailer",
    "decode_property_trailer",
    "get_spec",
    "WelfordAccumulator",
    "SuperquadricPropertyStore",
    "merge_property_stores",
    "store_from_decoded_trailer",
    "SpectrumAccumulator",
    "SpectrumPacket",
    "compress_spectrum",
    "decompress_spectrum",
    "SQSpatialIndex",
    "LidarIntensityAttributor",
    "MicaSenseAttributor",
    "OceanOpticsAttributor",
    "ndvi",
    "ndre",
    "gndvi",
    "evi",
]
