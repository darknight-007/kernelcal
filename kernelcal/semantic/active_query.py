"""
kernelcal.semantic.active_query — MaxCal D-optimal query sampler.

The novelty detector produces a pool of labelling candidates.  The human
(via DeepGIS) can answer only a few per cycle.  We therefore choose the
subset that maximises expected information gain under explicit cost and
coverage constraints:

    p(i) ∝ q(i) · exp(−Σ_k λ_k f_k(i))

Constraint features:

* ``cost(i)``          — expected human time for the query, target ≤ T_budget,
* ``-info_gain(i)``    — MI-style gain from labelling i (minimised ⇒ higher p),
* ``spatial_load(j)``  — number of queries already in the same tile j
                          (soft coverage penalty).

This mirrors :class:`kernelcal.maxcal.sampler.MaxCalSampler` but works on
heterogeneous candidate instances rather than a 2D waypoint grid.  It reuses
``kernelcal.maxcal.functional.fit_lagrange_multipliers`` so the arithmetic is
identical to the rest of the library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..maxcal.functional import (
    fit_lagrange_multipliers,
    maxcal_log_weights,
)
from .novelty import NoveltyReport


@dataclass
class QueryCandidate:
    """A single candidate LabelRequest prior to MaxCal selection."""

    candidate_id: str
    report: NoveltyReport
    expected_cost_seconds: float = 8.0
    info_gain_nats: float = 0.0
    spatial_bucket: Optional[str] = None
    reason: str = ""

    @property
    def novelty_score(self) -> float:
        return self.report.score


@dataclass
class QueryBudget:
    """Operational budget for a labelling cycle."""

    max_queries: int = 8
    cost_budget_seconds: float = 60.0
    info_gain_target_nats: Optional[float] = None
    max_per_bucket: int = 3


@dataclass
class QueryPlan:
    """MaxCal-sampled set of queries to emit to DeepGIS."""

    selected: List[QueryCandidate]
    probabilities: np.ndarray
    lambdas: np.ndarray
    cycle_index: int = 0
    diagnostics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spatial_load(candidates: Sequence[QueryCandidate]) -> np.ndarray:
    """Number of candidates sharing the same spatial bucket, per candidate."""
    counts: Dict[str, int] = {}
    for c in candidates:
        k = c.spatial_bucket or "<none>"
        counts[k] = counts.get(k, 0) + 1
    return np.array(
        [counts[c.spatial_bucket or "<none>"] for c in candidates],
        dtype=float,
    )


def default_info_gain(report: NoveltyReport) -> float:
    """Default MI gain proxy: novelty × (log area + 1).

    A larger unknown region tends to carry more Shannon information; multiply
    by the novelty scalar so well-explained large regions aren't oversampled.
    """
    area = max(report.resolved.instance.area_px, 1)
    return float(report.score * np.log1p(area))


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class ActiveQuerySampler:
    """MaxCal-governed selector over a pool of LabelRequest candidates.

    Parameters
    ----------
    budget : QueryBudget
    info_gain_fn : callable (NoveltyReport) → float nats
        Defaults to :func:`default_info_gain`.
    """

    def __init__(
        self,
        budget: Optional[QueryBudget] = None,
        info_gain_fn: Optional[Callable[[NoveltyReport], float]] = None,
    ) -> None:
        self.budget = budget or QueryBudget()
        self.info_gain_fn = info_gain_fn or default_info_gain
        self._cycle_index = 0
        self._lambdas = np.zeros(3)

    # ------------------------------------------------------------------
    # Candidate construction
    # ------------------------------------------------------------------

    def build_candidates(
        self,
        reports: Sequence[NoveltyReport],
        *,
        spatial_bucket_fn: Optional[Callable[[NoveltyReport], str]] = None,
        cost_fn: Optional[Callable[[NoveltyReport], float]] = None,
    ) -> List[QueryCandidate]:
        out: List[QueryCandidate] = []
        for k, rep in enumerate(reports):
            cost = (cost_fn or (lambda r: 8.0))(rep)
            info = self.info_gain_fn(rep)
            bucket = (
                spatial_bucket_fn(rep) if spatial_bucket_fn else None
            )
            reason = self._default_reason(rep)
            out.append(QueryCandidate(
                candidate_id=f"q_{self._cycle_index:04d}_{k:04d}",
                report=rep,
                expected_cost_seconds=float(cost),
                info_gain_nats=float(info),
                spatial_bucket=bucket,
                reason=reason,
            ))
        return out

    @staticmethod
    def _default_reason(rep: NoveltyReport) -> str:
        sig = rep.signals
        pairs = sorted(sig.items(), key=lambda kv: kv[1], reverse=True)
        top = pairs[0] if pairs else ("none", 0.0)
        return f"{rep.status}/{top[0]}={top[1]:.2f}"

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def plan(
        self,
        candidates: Sequence[QueryCandidate],
        reference_weights: Optional[np.ndarray] = None,
    ) -> QueryPlan:
        """Fit MaxCal and draw up to ``budget.max_queries`` candidates."""
        self._cycle_index += 1
        if not candidates:
            return QueryPlan(
                selected=[],
                probabilities=np.zeros(0),
                lambdas=np.zeros(3),
                cycle_index=self._cycle_index,
            )

        costs = np.array(
            [c.expected_cost_seconds for c in candidates], dtype=float
        )
        gains = np.array([c.info_gain_nats for c in candidates], dtype=float)
        load = _spatial_load(candidates)

        # Reference prior: proportional to novelty score (so obvious
        # non-candidates don't dominate)
        if reference_weights is None:
            q = np.array([c.novelty_score for c in candidates], dtype=float) + 1e-6
        else:
            q = np.asarray(reference_weights, dtype=float)
            if q.shape[0] != len(candidates):
                raise ValueError("reference_weights length mismatch")
            q = np.clip(q, 1e-9, None)
        q = q / q.sum()
        log_q = np.log(q)

        F_matrix = np.column_stack([costs, -gains, load])
        ig_target = self.budget.info_gain_target_nats
        if ig_target is None:
            ig_target = float(np.mean(gains))
        targets = np.array([
            self.budget.cost_budget_seconds / max(self.budget.max_queries, 1),
            -ig_target,
            float(np.mean(load)),
        ])

        lam0 = self._lambdas if self._lambdas.shape[0] == F_matrix.shape[1] else np.zeros(F_matrix.shape[1])
        try:
            lambdas, _ = fit_lagrange_multipliers(
                log_q, F_matrix, targets, lambda0=lam0
            )
        except Exception:
            lambdas = np.zeros(F_matrix.shape[1])
        log_p = maxcal_log_weights(log_q, lambdas, F_matrix)
        p = np.exp(log_p - log_p.max())
        p = p / p.sum()
        self._lambdas = lambdas

        selected = self._draw(candidates, p)
        diagnostics = {
            "n_candidates": len(candidates),
            "cost_mean": float(np.sum(p * costs)),
            "gain_mean": float(np.sum(p * gains)),
            "load_mean": float(np.sum(p * load)),
            "entropy_nats": float(-np.sum(p * np.log(np.clip(p, 1e-300, None)))),
        }
        return QueryPlan(
            selected=selected,
            probabilities=p,
            lambdas=lambdas,
            cycle_index=self._cycle_index,
            diagnostics=diagnostics,
        )

    def _draw(
        self,
        candidates: Sequence[QueryCandidate],
        probs: np.ndarray,
    ) -> List[QueryCandidate]:
        order = np.argsort(-probs)
        picked: List[QueryCandidate] = []
        per_bucket: Dict[str, int] = {}
        total_cost = 0.0
        for idx in order:
            c = candidates[int(idx)]
            bucket = c.spatial_bucket or "<none>"
            if per_bucket.get(bucket, 0) >= self.budget.max_per_bucket:
                continue
            if total_cost + c.expected_cost_seconds > self.budget.cost_budget_seconds:
                continue
            picked.append(c)
            per_bucket[bucket] = per_bucket.get(bucket, 0) + 1
            total_cost += c.expected_cost_seconds
            if len(picked) >= self.budget.max_queries:
                break
        return picked


__all__ = [
    "QueryCandidate",
    "QueryBudget",
    "QueryPlan",
    "ActiveQuerySampler",
    "default_info_gain",
]
