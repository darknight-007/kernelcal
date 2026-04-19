"""
kernelcal.semantic — Mask R-CNN / SAM / panoptic + HITL labelling via DeepGIS.

Subpackage overview
-------------------
* :mod:`~kernelcal.semantic.registry` — live class vocabulary with prototypes
  and motion tags (static / semi-static / dynamic).
* :mod:`~kernelcal.semantic.segmenters` — adapters for Mask R-CNN, SAM,
  Mask2Former, Grounding-DINO, and a ``StubSegmenter`` for tests.
* :mod:`~kernelcal.semantic.ensemble` — MaxCal-arbitrated three-layer
  ensemble (closed-set → panoptic → open-vocab) with per-instance status
  (``known | uncertain | unknown``).
* :mod:`~kernelcal.semantic.novelty` — combines six signals from kernelcal
  (status, ensemble novelty, prototype gap, HS novelty, spectral P2 triple,
  leakage D_t) into a scalar per candidate.
* :mod:`~kernelcal.semantic.active_query` — MaxCal D-optimal selector over
  candidates given a human-time budget and coverage constraints.

Typical usage::

    from kernelcal.semantic import (
        ClassRegistry, SegmenterEnsemble, StubSegmenter,
        score_frame, ActiveQuerySampler, QueryBudget,
    )

    registry = ClassRegistry.urban_default()
    ensemble = SegmenterEnsemble(
        registry=registry,
        closed_set=StubSegmenter(name="maskrcnn"),
    )
    result = ensemble.process_frame(image)
    reports = score_frame(result, spectral_context={"d_entropy": 0.3})
    plan = ActiveQuerySampler(QueryBudget(max_queries=5)).plan(
        ActiveQuerySampler().build_candidates(reports)
    )
"""

from .registry import (
    ClassRegistry,
    ClassSpec,
    MOTION_STATIC,
    MOTION_SEMI_STATIC,
    MOTION_DYNAMIC,
)
from .segmenters import (
    InstanceMask,
    Segmenter,
    StubSegmenter,
    MaskRCNNSegmenter,
    SAMSegmenter,
    Mask2FormerSegmenter,
    GroundingDINOSegmenter,
    KIND_CLOSED,
    KIND_PANOPTIC,
    KIND_OPEN_VOCAB,
)
from .ensemble import (
    SegmenterEnsemble,
    ResolvedInstance,
    EnsembleFrameResult,
    STATUS_KNOWN,
    STATUS_UNCERTAIN,
    STATUS_UNKNOWN,
)
from .novelty import (
    NoveltyWeights,
    NoveltyReport,
    score_frame,
    filter_candidates,
)
from .active_query import (
    QueryCandidate,
    QueryBudget,
    QueryPlan,
    ActiveQuerySampler,
    default_info_gain,
)

__all__ = [
    # registry
    "ClassRegistry",
    "ClassSpec",
    "MOTION_STATIC",
    "MOTION_SEMI_STATIC",
    "MOTION_DYNAMIC",
    # segmenters
    "InstanceMask",
    "Segmenter",
    "StubSegmenter",
    "MaskRCNNSegmenter",
    "SAMSegmenter",
    "Mask2FormerSegmenter",
    "GroundingDINOSegmenter",
    "KIND_CLOSED",
    "KIND_PANOPTIC",
    "KIND_OPEN_VOCAB",
    # ensemble
    "SegmenterEnsemble",
    "ResolvedInstance",
    "EnsembleFrameResult",
    "STATUS_KNOWN",
    "STATUS_UNCERTAIN",
    "STATUS_UNKNOWN",
    # novelty
    "NoveltyWeights",
    "NoveltyReport",
    "score_frame",
    "filter_candidates",
    # active query
    "QueryCandidate",
    "QueryBudget",
    "QueryPlan",
    "ActiveQuerySampler",
    "default_info_gain",
]
