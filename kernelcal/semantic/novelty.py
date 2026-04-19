"""
kernelcal.semantic.novelty — scalar novelty detector for unknown regions.

Turns the six signals we identified in the assessment into a single
``novelty_score ∈ [0, 1]`` per resolved instance.  Inputs beyond what
``SegmenterEnsemble`` already provides:

- ``spectral_context``: optional per-frame dict with
  ``H[h*]``, ``lambda1``, ``delta_prime``, ``beta1``, ``D_t``
  coming from :mod:`kernelcal.spectral.dynamics` / :mod:`kernelcal.geo3d`.
- ``hs_novelty_vs_prev``: already supplied by :class:`EnsembleFrameResult`.

The combination is a *soft-or* of sigmoid-squashed signals — no single
signal can suppress the others.  Weights are exposed so the caller can
re-weight for their sensor platform without touching the logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .ensemble import (
    EnsembleFrameResult,
    ResolvedInstance,
    STATUS_KNOWN,
    STATUS_UNCERTAIN,
    STATUS_UNKNOWN,
)


@dataclass
class NoveltyWeights:
    """Per-signal multiplicative weights in the combined score."""

    w_status: float = 1.0       # STATUS_UNKNOWN > UNCERTAIN > KNOWN
    w_proxy: float = 1.0        # ensemble.novelty_proxy
    w_proto: float = 0.8        # 1 − prototype_similarity
    w_hs: float = 0.6           # frame-level HS novelty vs previous frame
    w_spectral: float = 0.6     # P2 triple (entropy rise, λ₁ drop, β₁ anomaly)
    w_leakage: float = 0.7      # |D_t| / |Δ'|  — honest-map leakage


@dataclass
class NoveltyReport:
    """Novelty bundle attached to a resolved instance."""

    resolved: ResolvedInstance
    score: float
    signals: Dict[str, float] = field(default_factory=dict)

    @property
    def final_label(self) -> Optional[str]:
        return self.resolved.final_label

    @property
    def status(self) -> str:
        return self.resolved.status


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: float, scale: float = 1.0) -> float:
    return float(1.0 / (1.0 + np.exp(-x / max(scale, 1e-9))))


def _status_score(status: str) -> float:
    if status == STATUS_UNKNOWN:
        return 1.0
    if status == STATUS_UNCERTAIN:
        return 0.5
    return 0.0  # STATUS_KNOWN


def _spectral_signal(ctx: Optional[Dict[str, float]]) -> float:
    """Combine the P2 triple diagnostic into a scalar in [0, 1].

    Each sub-signal is sigmoid-squashed against a baseline-relative scale.
    We expect the caller to pass deltas vs. a rolling baseline::

        d_entropy   = H[h*_t] - H[h*_baseline]    (positive = surprising)
        d_lambda1   = (λ1_baseline - λ1_t) / λ1_baseline  (positive = connectivity loss)
        d_beta1     = β1_t - β1_baseline          (integer delta)

    All absent keys are treated as zero (no evidence).
    """
    if not ctx:
        return 0.0
    d_h = float(ctx.get("d_entropy", 0.0))
    d_l = float(ctx.get("d_lambda1", 0.0))
    d_b = float(ctx.get("d_beta1", 0.0))
    s_h = _sigmoid(d_h, scale=0.25)
    s_l = _sigmoid(d_l, scale=0.25)
    s_b = _sigmoid(abs(d_b), scale=1.0)
    # Soft-OR combination
    return float(1.0 - (1.0 - s_h) * (1.0 - s_l) * (1.0 - s_b))


def _leakage_signal(ctx: Optional[Dict[str, float]]) -> float:
    """Normalised Dt / Δ' ratio, sigmoid-squashed.

    P2 Prop. 2 says a healthy map satisfies D_m = -Δ'.  Deviations of |D_t|
    from the baseline |Δ'| indicate leakage or out-of-regime behaviour.
    """
    if not ctx:
        return 0.0
    Dt = ctx.get("D_t")
    delta = ctx.get("delta_prime")
    if Dt is None or delta is None or abs(delta) < 1e-9:
        return 0.0
    ratio = abs(abs(Dt) - abs(delta)) / abs(delta)
    return _sigmoid(ratio, scale=0.5)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def score_frame(
    result: EnsembleFrameResult,
    *,
    spectral_context: Optional[Dict[str, float]] = None,
    weights: Optional[NoveltyWeights] = None,
    hs_novelty_scale: float = 0.5,
) -> List[NoveltyReport]:
    """Compute a ``NoveltyReport`` for every resolved instance in a frame.

    The frame-level signals (``hs_novelty_vs_prev``, spectral context, leakage)
    are identical across all instances in the frame — they act as a *frame
    prior* that is combined with per-instance signals (status, novelty_proxy,
    prototype similarity gap).
    """
    w = weights or NoveltyWeights()
    frame_hs = _sigmoid(result.hs_novelty_vs_prev, scale=hs_novelty_scale)
    frame_spec = _spectral_signal(spectral_context)
    frame_leak = _leakage_signal(spectral_context)

    reports: List[NoveltyReport] = []
    for r in result.resolved:
        s_status = _status_score(r.status)
        s_proxy = float(np.clip(r.novelty_proxy, 0.0, 1.0))
        if r.prototype_similarity >= -1.0:
            s_proto = float(np.clip(1.0 - max(r.prototype_similarity, 0.0), 0.0, 1.0))
        else:
            s_proto = 0.0

        signals = {
            "status": s_status,
            "proxy": s_proxy,
            "prototype_gap": s_proto,
            "frame_hs": frame_hs,
            "spectral": frame_spec,
            "leakage": frame_leak,
        }

        # Weighted soft-OR: P(not-novel) = Π (1 − w_i · s_i)
        not_novel = 1.0
        for (key, s), weight in zip(
            signals.items(),
            [w.w_status, w.w_proxy, w.w_proto, w.w_hs, w.w_spectral, w.w_leakage],
        ):
            contribution = min(weight, 1.0) * float(s)
            not_novel *= max(0.0, 1.0 - contribution)
        score = float(np.clip(1.0 - not_novel, 0.0, 1.0))
        reports.append(NoveltyReport(resolved=r, score=score, signals=signals))
    return reports


def filter_candidates(
    reports: Sequence[NoveltyReport],
    min_score: float = 0.5,
    min_area_px: int = 200,
) -> List[NoveltyReport]:
    """Return reports worth submitting to the active-query sampler."""
    out = []
    for rep in reports:
        if rep.score < min_score:
            continue
        if rep.resolved.instance.area_px < min_area_px:
            continue
        out.append(rep)
    return out


__all__ = [
    "NoveltyWeights",
    "NoveltyReport",
    "score_frame",
    "filter_candidates",
]
