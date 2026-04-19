"""
kernelcal.semantic.ensemble — layered, MaxCal-arbitrated segmentation.

The ensemble runs up to three layers per keyframe:

1. **closed-set** (Mask R-CNN, fast, high-precision) on every frame,
2. **panoptic** (SAM / Mask2Former) every ``panoptic_every`` frames,
3. **open-vocab** (Grounding-DINO) only on regions that Layers 1–2 failed to
   claim confidently.

Which backends run is gated by a ``ModelKernelSelector`` — the MaxCal
arbiter from ``kernelcal.models.selector``.  The arbiter distributes
probability mass across backends so that their mean cost respects a budget
and their mean MI gain (measured as kernel novelty) meets a target.

Outputs from the three layers are *merged* into a single labelled pool.
Each surviving instance carries:

- a :class:`~kernelcal.semantic.segmenters.InstanceMask`,
- a *final label* in ``{known-class-name | None}`` — ``None`` means the pool
  could not place it in the current ontology with enough confidence,
- a *status* in ``{"known", "uncertain", "unknown"}``,
- a *novelty-proxy* score in ``[0, 1]`` (higher ⇒ more likely to need a
  human label).

This module never raises LabelRequests by itself; that's done downstream in
:mod:`kernelcal.semantic.active_query`.  It only produces the per-instance
status + novelty proxy that the query sampler consumes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..kernel.space import hilbert_schmidt_distance, kernel_from_embeddings
from ..models.selector import ModelKernelSelector
from .registry import ClassRegistry
from .segmenters import (
    InstanceMask,
    KIND_CLOSED,
    KIND_OPEN_VOCAB,
    KIND_PANOPTIC,
    Segmenter,
)

STATUS_KNOWN = "known"
STATUS_UNCERTAIN = "uncertain"
STATUS_UNKNOWN = "unknown"


@dataclass
class ResolvedInstance:
    """One instance after ensemble merging and prototype classification."""

    instance: InstanceMask
    final_label: Optional[str]
    status: str
    novelty_proxy: float
    supporting_segmenters: List[str] = field(default_factory=list)
    prototype_similarity: float = -1.0


@dataclass
class EnsembleFrameResult:
    """Per-frame bundle of ensemble outputs."""

    frame_index: int
    resolved: List[ResolvedInstance]
    selector_distribution: Dict[str, float]
    frame_kernel: Optional[np.ndarray] = None
    hs_novelty_vs_prev: float = 0.0
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def unknown_count(self) -> int:
        return sum(1 for r in self.resolved if r.status == STATUS_UNKNOWN)

    @property
    def uncertain_count(self) -> int:
        return sum(1 for r in self.resolved if r.status == STATUS_UNCERTAIN)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    if a.shape != b.shape:
        return 0.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / max(float(union), 1.0)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


class SegmenterEnsemble:
    """Layered semantic segmenter ensemble with MaxCal arbitration.

    Parameters
    ----------
    registry : :class:`~kernelcal.semantic.registry.ClassRegistry`
        Live class vocabulary; mutated externally by DeepGIS responses.
    closed_set : Segmenter
        Required.  Mask R-CNN or equivalent on ``registry`` classes.
    panoptic : Segmenter or None
        Optional class-agnostic partitioner.
    open_vocab : Segmenter or None
        Optional Grounding-DINO / CLIP model for unclaimed regions.
    confident_score : float
        Closed-set score above which an instance is accepted directly
        (``STATUS_KNOWN``) without consulting prototypes.
    min_prototype_cosine : float
        Cosine similarity to a class prototype above which an open-vocab or
        panoptic mask is placed as that class (``STATUS_KNOWN``).
    uncertain_score : float
        Lower bound for ``STATUS_UNCERTAIN``; below that, ``STATUS_UNKNOWN``.
    iou_overlap_thresh : float
        IoU above which a panoptic mask is considered "claimed" by a
        closed-set detection.
    panoptic_every : int
        Run panoptic backend every N frames (0 disables).
    cost_budget_seconds : float
        Target mean cost per frame for the MaxCal arbiter.
    """

    def __init__(
        self,
        registry: ClassRegistry,
        closed_set: Segmenter,
        panoptic: Optional[Segmenter] = None,
        open_vocab: Optional[Segmenter] = None,
        confident_score: float = 0.80,
        min_prototype_cosine: float = 0.60,
        uncertain_score: float = 0.35,
        iou_overlap_thresh: float = 0.50,
        panoptic_every: int = 5,
        cost_budget_seconds: float = 1.5,
    ) -> None:
        if closed_set is None:
            raise ValueError("closed_set segmenter is required")
        self.registry = registry
        self.closed_set = closed_set
        self.panoptic = panoptic
        self.open_vocab = open_vocab
        self.confident_score = float(confident_score)
        self.min_prototype_cosine = float(min_prototype_cosine)
        self.uncertain_score = float(uncertain_score)
        self.iou_overlap_thresh = float(iou_overlap_thresh)
        self.panoptic_every = int(panoptic_every)

        # MaxCal arbiter — one model per registered backend
        names = [closed_set.name]
        if panoptic is not None:
            names.append(panoptic.name)
        if open_vocab is not None:
            names.append(open_vocab.name)
        self._selector = ModelKernelSelector(
            model_names=names,
            cost_budget_seconds=cost_budget_seconds,
            temperature=1.0,
        )
        self._frame_index = -1
        self._prev_kernel: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def selector(self) -> ModelKernelSelector:
        return self._selector

    def process_frame(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None,
        force_open_vocab: bool = False,
    ) -> EnsembleFrameResult:
        """Run the ensemble on a single RGB frame."""
        self._frame_index += 1
        timings: Dict[str, float] = {}

        prompt = prompt or self.registry.grounding_prompt()

        closed_instances = self._timed_segment(self.closed_set, image, None, timings)

        run_panoptic = (
            self.panoptic is not None
            and self.panoptic_every > 0
            and (self._frame_index % self.panoptic_every == 0)
        )
        panoptic_instances: List[InstanceMask] = []
        if run_panoptic:
            panoptic_instances = self._timed_segment(
                self.panoptic, image, None, timings
            )

        resolved_closed = [
            self._resolve_closed(inst) for inst in closed_instances
        ]

        resolved_panoptic: List[ResolvedInstance] = []
        unclaimed_panoptic: List[InstanceMask] = []
        for p_inst in panoptic_instances:
            claimed = any(
                _iou(p_inst.mask, r.instance.mask) > self.iou_overlap_thresh
                for r in resolved_closed
            )
            if claimed:
                continue
            r = self._resolve_open(p_inst)
            resolved_panoptic.append(r)
            if r.status != STATUS_KNOWN:
                unclaimed_panoptic.append(p_inst)

        resolved_openvocab: List[ResolvedInstance] = []
        if (
            self.open_vocab is not None
            and (force_open_vocab or unclaimed_panoptic)
        ):
            ov = self._timed_segment(self.open_vocab, image, prompt, timings)
            for o_inst in ov:
                claimed = any(
                    _iou(o_inst.mask, r.instance.mask) > self.iou_overlap_thresh
                    for r in resolved_closed + resolved_panoptic
                )
                if claimed:
                    continue
                resolved_openvocab.append(self._resolve_open(o_inst))

        resolved = resolved_closed + resolved_panoptic + resolved_openvocab

        frame_kernel = self._frame_kernel(resolved)
        hs_novelty = 0.0
        if self._prev_kernel is not None and frame_kernel is not None:
            if self._prev_kernel.shape == frame_kernel.shape:
                hs_novelty = hilbert_schmidt_distance(
                    self._prev_kernel, frame_kernel
                )
        self._prev_kernel = frame_kernel

        self._update_selector(resolved, timings, frame_kernel)

        return EnsembleFrameResult(
            frame_index=self._frame_index,
            resolved=resolved,
            selector_distribution=dict(
                zip(
                    self._selector.model_names,
                    self._selector.distribution().tolist(),
                )
            ),
            frame_kernel=frame_kernel,
            hs_novelty_vs_prev=float(hs_novelty),
            timings=timings,
        )

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def _resolve_closed(self, inst: InstanceMask) -> ResolvedInstance:
        label = inst.proposed_label
        score = inst.score
        in_reg = label is not None and label in self.registry
        if in_reg and score >= self.confident_score:
            novelty = 1.0 - score
            return ResolvedInstance(
                instance=inst,
                final_label=label,
                status=STATUS_KNOWN,
                novelty_proxy=float(novelty),
                supporting_segmenters=[inst.source],
            )
        if in_reg and score >= self.uncertain_score:
            return ResolvedInstance(
                instance=inst,
                final_label=label,
                status=STATUS_UNCERTAIN,
                novelty_proxy=float(1.0 - score),
                supporting_segmenters=[inst.source],
            )
        # Below threshold or label not in registry → try prototype matching
        proto_name, sim = (None, -1.0)
        if inst.embedding is not None:
            proto_name, sim = self.registry.classify_by_prototype(
                inst.embedding,
                min_cosine=self.min_prototype_cosine,
            )
        if proto_name is not None:
            return ResolvedInstance(
                instance=inst,
                final_label=proto_name,
                status=STATUS_KNOWN,
                novelty_proxy=float(max(0.0, 1.0 - sim)),
                supporting_segmenters=[inst.source, "prototype"],
                prototype_similarity=sim,
            )
        return ResolvedInstance(
            instance=inst,
            final_label=None,
            status=STATUS_UNKNOWN,
            novelty_proxy=1.0,
            supporting_segmenters=[inst.source],
            prototype_similarity=float(sim),
        )

    def _resolve_open(self, inst: InstanceMask) -> ResolvedInstance:
        """Resolve a panoptic / open-vocab instance that was not overridden
        by a closed-set detection.  Tries proposed label first, then
        prototype matching."""
        lbl = inst.proposed_label
        if lbl is not None and lbl in self.registry and inst.score >= self.uncertain_score:
            status = (
                STATUS_KNOWN if inst.score >= self.confident_score else STATUS_UNCERTAIN
            )
            return ResolvedInstance(
                instance=inst,
                final_label=lbl,
                status=status,
                novelty_proxy=float(1.0 - inst.score),
                supporting_segmenters=[inst.source],
            )
        proto_name, sim = (None, -1.0)
        if inst.embedding is not None:
            proto_name, sim = self.registry.classify_by_prototype(
                inst.embedding,
                min_cosine=self.min_prototype_cosine,
            )
        if proto_name is not None:
            return ResolvedInstance(
                instance=inst,
                final_label=proto_name,
                status=STATUS_KNOWN,
                novelty_proxy=float(max(0.0, 1.0 - sim)),
                supporting_segmenters=[inst.source, "prototype"],
                prototype_similarity=sim,
            )
        return ResolvedInstance(
            instance=inst,
            final_label=None,
            status=STATUS_UNKNOWN,
            novelty_proxy=1.0,
            supporting_segmenters=[inst.source],
            prototype_similarity=float(sim),
        )

    # ------------------------------------------------------------------
    # MaxCal arbiter update + frame kernel
    # ------------------------------------------------------------------

    def _frame_kernel(
        self,
        resolved: Sequence[ResolvedInstance],
    ) -> Optional[np.ndarray]:
        embs = [r.instance.embedding for r in resolved if r.instance.embedding is not None]
        if not embs:
            return None
        dims = {e.shape[0] for e in embs}
        if len(dims) != 1:
            return None
        E = np.stack(embs, axis=0)
        return kernel_from_embeddings(E)

    def _update_selector(
        self,
        resolved: Sequence[ResolvedInstance],
        timings: Dict[str, float],
        frame_kernel: Optional[np.ndarray],
    ) -> None:
        per_source: Dict[str, List[ResolvedInstance]] = {}
        for r in resolved:
            per_source.setdefault(r.instance.source, []).append(r)
        for name in self._selector.model_names:
            cost = float(timings.get(name, 0.0))
            group = per_source.get(name, [])
            mi = float(np.mean([r.novelty_proxy for r in group])) if group else 0.0
            if cost == 0.0 and not group:
                continue
            self._selector.register_outcome(
                model_name=name,
                cost_seconds=cost,
                mi_gain_nats=mi,
                kernel_matrix=frame_kernel,
            )
        self._selector.update()

    # ------------------------------------------------------------------
    # Timed segmenter runner
    # ------------------------------------------------------------------

    @staticmethod
    def _timed_segment(
        segmenter: Segmenter,
        image: np.ndarray,
        prompt: Optional[str],
        timings: Dict[str, float],
    ) -> List[InstanceMask]:
        t0 = time.perf_counter()
        out = segmenter.segment(image, prompt=prompt)
        timings[segmenter.name] = time.perf_counter() - t0
        return out


__all__ = [
    "SegmenterEnsemble",
    "ResolvedInstance",
    "EnsembleFrameResult",
    "STATUS_KNOWN",
    "STATUS_UNCERTAIN",
    "STATUS_UNKNOWN",
]
