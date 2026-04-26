"""
Q_s refit (PR-3, §8.1 R1 rung).

Given a prior :class:`ConfusionMatrix` ``Q_s^0`` and observed
``(native_label, true_category)`` count pairs, return the
posterior-mean confusion matrix under a column-wise Dirichlet
conjugate prior::

    Q_s[r, c]_post  =  (alpha * Q_s^0[r, c] + N[r, c])
                       /
                       (alpha * 1            + N[:, c].sum())

where ``alpha`` is the prior pseudo-count strength (large => trust the
prior, small => trust the data). Each column of the posterior matrix
is column-stochastic by construction, matching the
:class:`ConfusionMatrix` invariant.

The "true category" can come from three places, in increasing order of
how supervised they are:

1. **Bootstrap** — the consensus argmax over *other* sources for the
   same region. Self-consistent but biased toward the prior.
2. **OSM-anchored** — wherever an OSM building/road claim is present
   on the same region, treat its native label as ground truth. Free
   supervision in the urban setting.
3. **Human-labelled** — a CategoryAnnotation / supervised pair pulled
   from the deepgis-xr DB. The strongest signal.

This module is policy-agnostic: it just consumes a stream of
``(source_id, native_label, true_category_index)`` triples (with an
optional weight per triple) and returns the refit table. The caller
decides where the triples come from.

The pipeline composition is::

    counts = count_evidence_triples(triples, prior_table, taxonomy)
    posterior_table = bayesian_refit_q_s_table(prior_table, counts,
                                               alpha=10.0)

Both functions are fully covered by tests in
``tests/test_distinction_game_q_s_fit.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .q_s import ConfusionMatrix
from .taxonomy import Taxonomy


# ---------------------------------------------------------------------------
# Triple typing
# ---------------------------------------------------------------------------

# (source_id, native_label, true_category_index, weight)
EvidenceTriple = Tuple[str, str, int, float]


@dataclass(frozen=True)
class QsFitResult:
    """Bundle of artefacts from :func:`bayesian_refit_q_s_table`."""

    table: Dict[str, ConfusionMatrix]
    counts: Dict[str, np.ndarray]      # source_id -> (n_native, n_categories)
    alpha: float
    n_evidence: Dict[str, int]         # per-source total observed weight
    method: str = "bayesian_dirichlet"

    def to_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "alpha": float(self.alpha),
            "n_evidence": {k: int(v) for k, v in self.n_evidence.items()},
            "counts": {k: v.tolist() for k, v in self.counts.items()},
            "table": {k: m.to_dict() for k, m in self.table.items()},
        }


# ---------------------------------------------------------------------------
# Count gathering
# ---------------------------------------------------------------------------

def count_evidence_triples(
    triples: Iterable[EvidenceTriple],
    prior_table: Mapping[str, ConfusionMatrix],
    taxonomy: Taxonomy,
    *,
    drop_unknown_native: bool = True,
) -> Dict[str, np.ndarray]:
    """Aggregate ``(source, native_label, true_c, w)`` triples into
    per-source ``(|Y_s|, |c*|)`` count matrices.

    Triples whose ``source`` is not in ``prior_table`` are dropped
    (the prior dictates which kernels we know how to refit). Triples
    whose ``native_label`` is not in the prior's native vocabulary
    are dropped iff ``drop_unknown_native``; otherwise they raise.

    Parameters
    ----------
    triples
        Iterable of ``(source_id, native_label, true_c_idx, weight)``.
        Weight defaults to 1.0 — useful for soft EM weights.
    prior_table
        Per-source prior matrices. Defines (a) the set of legal
        sources and (b) the row-vocabulary per source.
    taxonomy
        Defines the column-vocabulary. All ``true_c_idx`` must lie in
        ``[0, taxonomy.n)``.
    drop_unknown_native
        If True, silently skip claims whose native label isn't in the
        prior's vocabulary. If False, raise ``KeyError``. Use False
        in tests; True in production where kernel native vocabularies
        drift across versions.

    Returns
    -------
    Dict[source_id, ndarray]
        Keys are exactly ``prior_table.keys()``. Each value is a
        ``(|Y_s|, |c*|)`` count matrix (float64; weights are real-valued).
    """
    counts: Dict[str, np.ndarray] = {
        sid: np.zeros((qm.n_native, taxonomy.n), dtype=np.float64)
        for sid, qm in prior_table.items()
    }
    for sid, native_label, true_c_idx, weight in triples:
        if sid not in prior_table:
            continue
        qm = prior_table[sid]
        if not (0 <= true_c_idx < taxonomy.n):
            raise IndexError(
                f"true_c_idx={true_c_idx} out of range [0, {taxonomy.n})"
            )
        try:
            r = qm.native_index(native_label)
        except KeyError:
            if drop_unknown_native:
                continue
            raise
        counts[sid][r, true_c_idx] += float(weight)
    return counts


# ---------------------------------------------------------------------------
# Bayesian refit
# ---------------------------------------------------------------------------

def bayesian_refit_q_s(
    prior: ConfusionMatrix,
    counts: np.ndarray,
    *,
    alpha: float = 10.0,
) -> ConfusionMatrix:
    """Closed-form Dirichlet refit of a single ``ConfusionMatrix``.

    For each category column ``c``, the posterior mean is

    .. math::

       Q_s[:, c]_{\\text{post}} = \\frac{\\alpha \\, Q_s[:, c]_{\\text{prior}} + N[:, c]}
            {\\alpha + \\sum_r N[r, c]}

    which is column-stochastic by construction.

    Parameters
    ----------
    prior
        Prior confusion matrix. Defines shape and labels.
    counts
        ``(|Y_s|, |c*|)`` non-negative count matrix aligned with
        ``prior.matrix``.
    alpha
        Dirichlet pseudo-count strength. ``alpha=0`` reduces to the
        empirical MLE; very large alpha returns the prior unchanged.
        Default 10.0 — modest concession to data, robust against
        noisy single-tile evidence.

    Returns
    -------
    ConfusionMatrix
        Same ``source_id`` / ``taxonomy`` / ``native_labels`` as
        ``prior``; refit ``matrix``; ``description`` annotated with
        the alpha and total observed mass.
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    if counts_arr.shape != prior.matrix.shape:
        raise ValueError(
            f"counts shape {counts_arr.shape} mismatches prior matrix "
            f"shape {prior.matrix.shape}"
        )
    if (counts_arr < 0).any():
        raise ValueError("counts must be non-negative")
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative; got {alpha}")

    pseudo = alpha * prior.matrix + counts_arr  # (R, C)
    col_sums = pseudo.sum(axis=0, keepdims=True)  # (1, C)
    # Defensive: a column with all-zero pseudo-counts (alpha=0 and
    # zero data) would otherwise produce NaN.
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    posterior = pseudo / col_sums

    n_obs_total = float(counts_arr.sum())
    return ConfusionMatrix(
        source_id=prior.source_id,
        taxonomy=prior.taxonomy,
        native_labels=prior.native_labels,
        matrix=posterior,
        description=(
            f"{prior.description.rstrip()}\n"
            f"Bayesian-Dirichlet refit on {n_obs_total:.1f} observed pairs "
            f"(alpha={alpha:g})."
        ).strip(),
    )


def bayesian_refit_q_s_table(
    prior_table: Mapping[str, ConfusionMatrix],
    counts_by_source: Mapping[str, np.ndarray],
    *,
    alpha: float = 10.0,
) -> QsFitResult:
    """Refit a whole ``q_s_table`` against per-source count matrices.

    Sources missing from ``counts_by_source`` are kept at their prior
    (zero-evidence). Sources missing from ``prior_table`` raise.
    """
    out: Dict[str, ConfusionMatrix] = {}
    n_evidence: Dict[str, int] = {}
    counts_record: Dict[str, np.ndarray] = {}
    for sid, prior in prior_table.items():
        N = counts_by_source.get(
            sid,
            np.zeros((prior.n_native, prior.n_categories), dtype=np.float64),
        )
        out[sid] = bayesian_refit_q_s(prior, N, alpha=alpha)
        counts_record[sid] = np.asarray(N, dtype=np.float64).copy()
        n_evidence[sid] = int(np.asarray(N).sum())
    extra = set(counts_by_source) - set(prior_table)
    if extra:
        raise KeyError(
            f"counts_by_source has sources not in prior_table: {sorted(extra)}"
        )
    return QsFitResult(
        table=out,
        counts=counts_record,
        alpha=float(alpha),
        n_evidence=n_evidence,
    )


# ---------------------------------------------------------------------------
# SceneGraph -> evidence triples
# ---------------------------------------------------------------------------

# Sources whose native labels we treat as gold ground truth when
# anchoring evidence. OSM tags directly inherit from the OSM schema
# and are the strongest free supervision in the urban setting.
DEFAULT_OSM_ANCHOR_SOURCES: Tuple[str, ...] = ("osm_buildings", "osm_roads")


def evidence_triples_from_scene_graph(
    scene_graph_dict: Mapping[str, object],
    prior_table: Mapping[str, ConfusionMatrix],
    *,
    osm_anchor_sources: Sequence[str] = DEFAULT_OSM_ANCHOR_SOURCES,
    use_consensus_fallback: bool = False,
    min_score: float = 0.0,
) -> List[EvidenceTriple]:
    """Convert one persisted SceneGraph dict into evidence triples.

    Strategy:

    1. For each node in the scene graph:
       a. If any of ``osm_anchor_sources`` claimed this node, take the
          fused ``category_index`` as the anchor (OSM's own native
          label is what produced that argmax in the first place, so
          this is consistent supervision rather than circular).
       b. Else, if ``use_consensus_fallback``, take the fused
          ``category_index`` regardless. Bootstrapping mode: useful
          early but biased toward the prior.
       c. Else, skip the node — no anchor, no evidence.
    2. For every claim on the anchored node, emit one triple
       ``(source_id, native_label, anchor_c, score)``.

    Score becomes the weight: a low-confidence claim contributes less
    to the count than a high-confidence one.
    """
    nodes = scene_graph_dict.get("nodes") or []
    out: List[EvidenceTriple] = []
    anchor_set = set(osm_anchor_sources)
    for node in nodes:
        claims = node.get("claims") or []
        if not claims:
            continue
        sources_present = {c.get("source_id") for c in claims}
        anchor_c = None
        if sources_present & anchor_set:
            anchor_c = node.get("category_index")
        elif use_consensus_fallback:
            anchor_c = node.get("category_index")
        if anchor_c is None:
            continue
        try:
            anchor_c = int(anchor_c)
        except (TypeError, ValueError):
            continue
        for claim in claims:
            sid = claim.get("source_id")
            if sid not in prior_table:
                continue
            native_label = claim.get("native_label")
            if not isinstance(native_label, str):
                continue
            score = float(claim.get("score", 1.0) or 0.0)
            if score < min_score:
                continue
            out.append((sid, native_label, anchor_c, score))
    return out


def evidence_triples_from_many_scene_graphs(
    scene_graph_dicts: Iterable[Mapping[str, object]],
    prior_table: Mapping[str, ConfusionMatrix],
    *,
    osm_anchor_sources: Sequence[str] = DEFAULT_OSM_ANCHOR_SOURCES,
    use_consensus_fallback: bool = False,
    min_score: float = 0.0,
) -> List[EvidenceTriple]:
    """Vector form of :func:`evidence_triples_from_scene_graph`."""
    out: List[EvidenceTriple] = []
    for sg in scene_graph_dicts:
        out.extend(evidence_triples_from_scene_graph(
            sg, prior_table,
            osm_anchor_sources=osm_anchor_sources,
            use_consensus_fallback=use_consensus_fallback,
            min_score=min_score,
        ))
    return out


__all__ = [
    "EvidenceTriple",
    "QsFitResult",
    "DEFAULT_OSM_ANCHOR_SOURCES",
    "count_evidence_triples",
    "bayesian_refit_q_s",
    "bayesian_refit_q_s_table",
    "evidence_triples_from_scene_graph",
    "evidence_triples_from_many_scene_graphs",
]
