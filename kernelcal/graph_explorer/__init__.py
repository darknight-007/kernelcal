"""Shared exploration-policy primitives for graph-based adaptive mappers.

This subpackage hosts the code that is identical between
``examples/controller/drone_dem_betti_adaptive_experiment.py`` (DEM stream
graphs over pixel patches) and
``examples/bishop/bishop_rocks_graph_explorer.py`` (k-NN graphs over rock
centroids).  Both are concrete instances of the same pattern documented in
the companion paper §P4: split the current field of view into four diagonal
quadrants, score each by its local graph Betti topology plus an exploration
bonus, and move to the outer corner of the winner.

Public API
----------
:class:`BettiWeights`
    Weights for the canonical quadrant-Betti score.

:class:`Candidate` / :class:`ScoredCandidate`
    Dataclass carrying one candidate quadrant's graph summary
    (β₀, β₁, n_nodes, unseen_frac) and, after ranking, its post-penalty
    score.

:func:`score_betti_candidate`
    Pure score function:
    ``w_beta1·clip(β₁/n) − w_beta0·clip(β₀/n) + w_unseen·unseen_frac``.

:func:`choose_best_candidate`
    Ranker with revisit penalty and cyclic tie-break (identical to
    ``choose_next_location`` in the DEM explorer).

:data:`QUADRANT_NAMES`, :data:`QUADRANT_OFFSETS_IMAGE`, :data:`QUADRANT_OFFSETS_METRIC`
    Convention helpers so callers in image-coordinates (DEM) and
    metric-coordinates (bishop) can share the same quadrant names.

:class:`CameraModel`
    Nadir square-footprint camera: ``altitude + fov → footprint_side_m``
    used by both explorers to size their scan window.

:class:`CoverageRaster`
    Bool visited-mask over a metric bbox at a fixed resolution; bishop
    analog of the drone-DEM explorer's ``visited`` numpy array.  Lets
    bishop compute ``unseen_frac`` with the exact same
    ``1 − mean(visited[target_patch])`` semantics as DEM.

Each example is still responsible for extracting the per-quadrant sub-graph
from its domain — only the scoring / ranking / revisit-penalty logic and
the camera+coverage bookkeeping are shared here.
"""

from __future__ import annotations

from .camera import CameraModel
from .coverage import CoverageRaster
from .planner import (
    BettiWeights,
    Candidate,
    QUADRANT_NAMES,
    QUADRANT_OFFSETS_IMAGE,
    QUADRANT_OFFSETS_METRIC,
    ScoredCandidate,
    choose_best_candidate,
    score_betti_candidate,
)


__all__ = [
    "BettiWeights",
    "Candidate",
    "CameraModel",
    "CoverageRaster",
    "ScoredCandidate",
    "QUADRANT_NAMES",
    "QUADRANT_OFFSETS_IMAGE",
    "QUADRANT_OFFSETS_METRIC",
    "score_betti_candidate",
    "choose_best_candidate",
]
