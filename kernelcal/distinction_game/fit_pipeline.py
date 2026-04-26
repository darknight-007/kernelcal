"""
Top-level distinction-game fit pipeline.

Consumes a sequence of persisted :class:`SceneGraph` dicts (the JSON
artefacts deepgis-xr writes per request) and runs the full PR-3 fit:

1. Reconstruct :class:`KernelClaim` instances from each node's
   ``claims`` list.
2. Decide an anchor category per node: OSM-claimed regions inherit the
   fused argmax (free supervision); regions where ``use_consensus_fallback=True``
   use the fused argmax regardless; otherwise the region is left
   unsupervised and feeds the EM E-step only.
3. Run :func:`fit_kernel_mix_em` to update ``λ`` and ``Q_s`` jointly.

The result is a :class:`DistinctionGameFit` bundle; the deepgis-xr
admin / management command serialises this to disk and tags the
contributing SceneGraph rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .kernel_mix import JointFitResult, KernelMixFit, fit_kernel_mix_em
from .q_s import ConfusionMatrix
from .q_s_fit import DEFAULT_OSM_ANCHOR_SOURCES
from .region import KernelClaim
from .taxonomy import Taxonomy


@dataclass(frozen=True)
class DistinctionGameFit:
    """End-to-end fit artefact returned by :func:`fit_distinction_game`."""

    mix: KernelMixFit
    q_s_table: Mapping[str, ConfusionMatrix]
    n_regions: int
    n_anchored_regions: int
    n_unsupervised_regions: int
    n_scene_graphs: int
    sources: Tuple[str, ...]
    log_likelihood_history: List[float] = field(default_factory=list)
    converged: bool = False
    used_osm_anchor_sources: Tuple[str, ...] = field(default_factory=tuple)
    used_consensus_fallback: bool = False
    contributing_scene_graph_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mix": self.mix.to_dict(),
            "q_s_table": {k: v.to_dict() for k, v in self.q_s_table.items()},
            "n_regions": int(self.n_regions),
            "n_anchored_regions": int(self.n_anchored_regions),
            "n_unsupervised_regions": int(self.n_unsupervised_regions),
            "n_scene_graphs": int(self.n_scene_graphs),
            "sources": list(self.sources),
            "log_likelihood_history": list(self.log_likelihood_history),
            "converged": bool(self.converged),
            "used_osm_anchor_sources": list(self.used_osm_anchor_sources),
            "used_consensus_fallback": bool(self.used_consensus_fallback),
            "contributing_scene_graph_ids": list(self.contributing_scene_graph_ids),
        }


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------

def _claims_from_node(node: Mapping[str, Any]) -> List[KernelClaim]:
    """Reconstruct :class:`KernelClaim` instances from a serialised
    SceneGraph node payload."""
    raw = node.get("claims") or []
    out: List[KernelClaim] = []
    for c in raw:
        try:
            out.append(KernelClaim.from_dict(c))
        except Exception:
            continue
    return out


def _node_anchor(
    node: Mapping[str, Any],
    *,
    osm_anchor_sources: Sequence[str],
    use_consensus_fallback: bool,
) -> Optional[int]:
    """Compute the supervised anchor category for one node, or None."""
    claims = node.get("claims") or []
    sources_present = {c.get("source_id") for c in claims}
    if sources_present & set(osm_anchor_sources):
        cat = node.get("category_index")
    elif use_consensus_fallback:
        cat = node.get("category_index")
    else:
        return None
    try:
        return int(cat)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fit_distinction_game(
    scene_graph_dicts: Iterable[Mapping[str, Any]],
    *,
    prior_q_s_table: Mapping[str, ConfusionMatrix],
    taxonomy: Optional[Taxonomy] = None,
    sources: Optional[Sequence[str]] = None,
    prior_lambdas: Optional[Mapping[str, float]] = None,
    osm_anchor_sources: Sequence[str] = DEFAULT_OSM_ANCHOR_SOURCES,
    use_consensus_fallback: bool = False,
    fit_q_s: bool = True,
    alpha_q_s: float = 10.0,
    max_iter: int = 20,
    tol: float = 1e-5,
    min_score: float = 0.0,
    exclude_anchor_sources_from_fit: bool = True,
) -> DistinctionGameFit:
    """End-to-end EM fit of ``(λ, Q_s)`` against persisted SceneGraph dicts.

    Parameters
    ----------
    scene_graph_dicts
        Iterable of SceneGraph dicts as returned by
        :meth:`SceneGraph.to_dict`. Tolerates the orchestrator's
        ``payload_json`` format (top-level dict with the SceneGraph
        fields).
    prior_q_s_table
        The prior confusion matrices to update.
    taxonomy
        Optional override (defaults to ``prior_q_s_table``'s).
    sources
        Optional ordered source list; defaults to
        ``sorted(prior_q_s_table.keys())``.
    prior_lambdas
        Optional warm start for ``λ`` (mapping source_id -> weight).
    osm_anchor_sources
        Source ids whose presence on a node triggers OSM anchoring.
    use_consensus_fallback
        Use the fused argmax as the anchor for non-OSM regions too.
        Useful early but biased toward the prior; off by default.
    fit_q_s
        Update ``Q_s`` each EM iteration (defaults True).
    alpha_q_s
        Dirichlet pseudo-count strength for the Q_s refit.
    max_iter, tol
        EM outer loop knobs.
    min_score
        Drop claims with score below this floor (PR-3 §4 robustness).
    exclude_anchor_sources_from_fit
        If True (default), strip claims whose ``source_id`` is in
        ``osm_anchor_sources`` from each region *after* the anchor has
        been determined. Without this, the EM fit would collapse onto
        ``λ_anchor → 1`` by trivial circularity (the anchor source
        perfectly predicts its own contribution to the fused argmax).
        With it, the anchor source provides only the supervision
        label and the fitted ``λ`` describes how much each *other*
        source agrees with that label. The fitted ``sources`` list is
        also pruned accordingly so the output ``λ`` vector is aligned
        to the non-anchor kernels only.

    Returns
    -------
    DistinctionGameFit
    """
    sg_list = list(scene_graph_dicts)
    if not sg_list:
        raise ValueError("fit_distinction_game requires at least one scene graph")

    if taxonomy is None:
        taxonomy = next(iter(prior_q_s_table.values())).taxonomy
    if sources is None:
        sources = sorted(prior_q_s_table.keys())
    sources = list(sources)

    anchor_set = set(osm_anchor_sources)
    if exclude_anchor_sources_from_fit:
        # Anchor sources provide labels, not features — strip them from
        # the fitted source list so the result λ-vector describes the
        # other kernels' agreement with the anchor.
        sources = [s for s in sources if s not in anchor_set]
        if not sources:
            raise ValueError(
                "exclude_anchor_sources_from_fit=True removed every "
                "source from the fit. Either disable the flag or expand "
                "the prior_q_s_table beyond the anchor sources."
            )

    regions: List[List[KernelClaim]] = []
    anchors: List[Optional[int]] = []
    contributing: List[str] = []
    for sg in sg_list:
        sg_id = (
            sg.get("session_id")
            or sg.get("id")
            or (sg.get("viewport") or {}).get("capture_metadata", {}).get("request_id")
            or "<no-id>"
        )
        contributing.append(str(sg_id))
        for node in sg.get("nodes") or []:
            # Compute the anchor *before* filtering so the anchor logic
            # still sees the OSM (or other anchor-source) claims.
            anchor = _node_anchor(
                node,
                osm_anchor_sources=osm_anchor_sources,
                use_consensus_fallback=use_consensus_fallback,
            )
            claims = _claims_from_node(node)
            if min_score > 0:
                claims = [c for c in claims if c.score >= min_score]
            if exclude_anchor_sources_from_fit:
                claims = [c for c in claims if c.source_id not in anchor_set]
            if not claims:
                continue
            regions.append(claims)
            anchors.append(anchor)

    if not regions:
        raise ValueError(
            "No usable regions found across the supplied scene graphs."
        )

    joint: JointFitResult = fit_kernel_mix_em(
        regions, prior_q_s_table,
        taxonomy=taxonomy,
        sources=sources,
        anchors=anchors,
        prior_lambdas=prior_lambdas,
        fit_q_s=fit_q_s,
        alpha_q_s=alpha_q_s,
        max_iter=max_iter,
        tol=tol,
    )

    return DistinctionGameFit(
        mix=joint.mix,
        q_s_table=joint.q_s_table,
        n_regions=len(regions),
        n_anchored_regions=joint.n_anchored_regions,
        n_unsupervised_regions=joint.n_unsupervised_regions,
        n_scene_graphs=len(sg_list),
        sources=tuple(sources),
        log_likelihood_history=joint.log_likelihood_history,
        converged=joint.converged,
        used_osm_anchor_sources=tuple(osm_anchor_sources),
        used_consensus_fallback=bool(use_consensus_fallback),
        contributing_scene_graph_ids=contributing,
    )


__all__ = [
    "DistinctionGameFit",
    "fit_distinction_game",
]
