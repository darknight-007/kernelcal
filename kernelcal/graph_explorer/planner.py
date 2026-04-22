"""Shared quadrant-Betti exploration policy.

This module holds the scoring rule, tie-break, and revisit penalty used by
both the drone-DEM explorer
(``examples/controller/drone_dem_betti_adaptive_experiment.py``) and the
bishop-rocks k-NN graph explorer
(``examples/bishop/bishop_rocks_graph_explorer.py``).  Both scripts describe
the same underlying algorithm in the companion paper §P4 (Δβ₁ biosignature
signal): split the current field of view into four diagonal quadrants
(NW / NE / SW / SE), compute per-quadrant graph Betti numbers, and move
toward the outer corner of the quadrant that maximises ::

    score = w_beta1 * clip(β₁ / n, 0, 1)          # braided-loop density
          - w_beta0 * clip(β₀ / n, 0, 1)          # fragmentation penalty
          + w_unseen * unseen_frac                # exploration bonus

A revisit penalty is subtracted for candidates whose position was seen in the
recent past, and ties are broken by a cyclic rotation (so the planner is
deterministic but does not bias toward the first candidate in list order).

Domain-specific work — how the quadrant sub-graphs are extracted and what a
"position" means — is left to each caller.  A quadrant sub-graph comes from a
DEM sub-patch in the drone explorer, and from a geometric quadrant of rocks
inside the scan window in the bishop explorer; both produce :class:`Candidate`
instances that are then ranked by :func:`choose_best_candidate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Sequence

import numpy as np


__all__ = [
    "BettiWeights",
    "Candidate",
    "ScoredCandidate",
    "QUADRANT_NAMES",
    "QUADRANT_OFFSETS_IMAGE",
    "QUADRANT_OFFSETS_METRIC",
    "score_betti_candidate",
    "choose_best_candidate",
]


# ---------------------------------------------------------------------------
# Quadrant conventions
# ---------------------------------------------------------------------------

#: Canonical quadrant order used by both explorers.  Kept as a tuple so
#: callers can rely on deterministic iteration.
QUADRANT_NAMES: tuple[str, str, str, str] = ("NW", "NE", "SW", "SE")

#: Diagonal offsets with *image* row/col convention (row grows south).
#: ``(dr, dc)`` where ``dr = -1`` means move toward smaller row index (= up
#: on screen / north).  Used by the drone-DEM explorer.
QUADRANT_OFFSETS_IMAGE: dict[str, tuple[int, int]] = {
    "NW": (-1, -1),
    "NE": (-1, +1),
    "SW": (+1, -1),
    "SE": (+1, +1),
}

#: Diagonal offsets with *metric* (x east, y north) convention.  ``(dx, dy)``.
#: Used by the bishop-rocks explorer where ``y_m`` grows northward.
QUADRANT_OFFSETS_METRIC: dict[str, tuple[int, int]] = {
    "NW": (-1, +1),
    "NE": (+1, +1),
    "SW": (-1, -1),
    "SE": (+1, -1),
}


# ---------------------------------------------------------------------------
# Weights + candidate datamodel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BettiWeights:
    """Weights for the quadrant-Betti exploration score.

    The score formula lives in :func:`score_betti_candidate`.  Defaults
    match the drone-DEM explorer's CLI defaults:

    - ``w_beta1 = 2.5``  — reward per unit braided-loop density (β₁/n).
    - ``w_beta0 = 0.5``  — penalty per unit fragmentation (β₀/n).
    - ``w_unseen = 5.0`` — reward per unit of unexplored coverage at the
      candidate position (``unseen_frac`` ∈ [0, 1]).
    - ``revisit_penalty = 0.5`` — subtracted from a candidate's score when
      its position matches a recently-visited one.
    """

    w_beta1: float = 2.5
    w_beta0: float = 0.5
    w_unseen: float = 5.0
    revisit_penalty: float = 0.5


@dataclass
class Candidate:
    """A candidate next waypoint with its local-graph summary statistics.

    Attributes
    ----------
    name
        Human-readable label (e.g. ``"NW"``, ``"NE"``).  Used by logging /
        visualisation and for the CSV summary row; the planner does not
        interpret it.
    position
        Any hashable waypoint identifier.  For the DEM explorer it is the
        integer ``(row, col)`` of the target pixel; for the bishop explorer
        it is a rounded ``(x_m, y_m)`` tuple or similar.  Equality against
        ``recent_positions`` drives the revisit penalty, so positions must
        be comparable with ``==``.
    beta0, beta1, n_nodes
        Connected-component count, cycle count, and node count of the
        sub-graph seen from this candidate.  ``n_nodes`` is clamped to at
        least 1 inside the score formula so an empty sub-graph yields a
        well-defined (zero) score contribution.
    unseen_frac
        Fraction of the candidate footprint that has *not* been observed
        yet.  Must lie in ``[0, 1]``; values outside that range are still
        used unchanged but will produce scores outside the documented
        range.
    extra
        Free-form dictionary for caller-specific metadata (raw Fiedler
        value, novelty vector, etc.).  The planner passes it through
        untouched; downstream code is free to read / ignore it.
    """

    name: str
    position: Hashable
    beta0: int
    beta1: int
    n_nodes: int
    unseen_frac: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredCandidate:
    """A :class:`Candidate` paired with its final (post-penalty) score.

    Returned as part of :func:`choose_best_candidate`'s diagnostic list so
    callers can plot / log / verify the full ranking rather than just the
    winner.
    """

    candidate: Candidate
    score: float


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_betti_candidate(cand: Candidate, weights: BettiWeights) -> float:
    """Canonical quadrant-Betti score used by both explorers.

    Identical to the scoring block in ``choose_next_location`` of the
    drone-DEM explorer::

        n = max(1, n_nodes)
        score = w_beta1 * clip(β₁ / n, 0, 1)
              - w_beta0 * clip(β₀ / n, 0, 1)
              + w_unseen * unseen_frac

    ``β₁/n`` rewards topological complexity (braided-loop density,
    §P4 Δβ₁ signal).  ``β₀/n`` is *penalised* because high β₀ indicates a
    fragmented / noisy graph; a single connected network (β₀ = 1) is
    preferred.  Both are clamped to ``[0, 1]`` so they stay on the same
    scale as ``unseen_frac`` regardless of graph size.

    The revisit penalty is applied by :func:`choose_best_candidate`, not
    here, so this function is a pure function of ``cand`` and ``weights``.
    """
    n = max(1.0, float(cand.n_nodes))
    b1n = float(np.clip(float(cand.beta1) / n, 0.0, 1.0))
    b0n = float(np.clip(float(cand.beta0) / n, 0.0, 1.0))
    return (
        float(weights.w_beta1) * b1n
        - float(weights.w_beta0) * b0n
        + float(weights.w_unseen) * float(cand.unseen_frac)
    )


def choose_best_candidate(
    candidates: Iterable[Candidate],
    weights: BettiWeights,
    *,
    recent_positions: Sequence[Hashable] = (),
    tie_break_index: int = 0,
    score_fn: Callable[[Candidate, BettiWeights], float] = score_betti_candidate,
) -> tuple[Candidate | None, float, list[ScoredCandidate]]:
    """Rank ``candidates`` and pick the best under the quadrant-Betti rule.

    Parameters
    ----------
    candidates
        Iterable of :class:`Candidate` to rank.  If empty the function
        returns ``(None, 0.0, [])``.
    weights
        Score weights (see :class:`BettiWeights`).
    recent_positions
        Positions visited in the recent past (typically the last ~20
        waypoints).  Any candidate whose ``position`` equals one of these
        gets ``weights.revisit_penalty`` subtracted from its score after
        the base score is computed.  Equality is checked with ``in`` so
        positions must be hashable / comparable with ``==``.
    tie_break_index
        When several candidates share the top score, the winner is
        ``tied[tie_break_index % len(tied)]``.  Both explorers pass
        ``len(records)`` so ties rotate deterministically across steps
        instead of always picking the first one.
    score_fn
        Alternative score function for experiments / tests.  Must accept
        ``(candidate, weights) -> float``.  Defaults to
        :func:`score_betti_candidate`.

    Returns
    -------
    best
        The winning :class:`Candidate`, or ``None`` if ``candidates`` was
        empty.
    best_score
        The winner's final (post-penalty) score, or ``0.0`` when there is
        no winner.
    scored
        Full list of :class:`ScoredCandidate` in input order with
        post-penalty scores.  Useful for diagnostics / plotting; callers
        that only want the winner can ignore it.
    """
    recent_list = list(recent_positions) if recent_positions else []
    scored: list[ScoredCandidate] = []
    for cand in candidates:
        base = float(score_fn(cand, weights))
        if recent_list and cand.position in recent_list:
            base -= float(weights.revisit_penalty)
        scored.append(ScoredCandidate(cand, base))

    if not scored:
        return None, 0.0, []

    best = max(sc.score for sc in scored)
    eps = 1e-6 * max(1.0, abs(best))
    tied = [sc for sc in scored if abs(sc.score - best) <= eps]
    chosen = tied[int(tie_break_index) % len(tied)]
    return chosen.candidate, float(chosen.score), scored
