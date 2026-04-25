"""
SceneGraph: the per-tile fused output of the multi-kernel pipeline.

Takes a flat list of :class:`KernelClaim` from every available kernel
(OSM, Grounding-DINO, SAM2, MR-rocks, MR-house, …) on a single
viewport, spatially associates them into a shared region set, and
returns a :class:`SceneGraph` whose nodes carry:

- a *fused category posterior* over the shared taxonomy ``c*``;
- the per-source claims that voted for that node, with native label
  and native confidence (provenance);
- a single ``category`` argmax + score for downstream rendering.

Edges are first-pass spatial adjacency in this PR (bbox-touch /
centroid-proximity); §5 spectral diagnostics on the resulting region
graph land in PR-3.

Public API
----------

The single entry point is :func:`build_scene_graph`. Everything else
is the data classes the orchestrator needs to serialize and the
Cesium frontend will consume::

    SceneGraph
    ├── viewport            # bbox + image_size + capture metadata
    ├── taxonomy            # which c* this graph lives in
    ├── kernels_queried     # which sources contributed (or were silent)
    ├── nodes : list[SceneNode]
    │       ├── id
    │       ├── geometry (Region)
    │       ├── category (argmax of category_posterior)
    │       ├── category_posterior : list[float]   # length |c*|
    │       ├── score
    │       └── claims : list[KernelClaim]
    ├── edges : list[SceneEdge]
    └── fusion_metadata     # mixer params, lambdas, version, ...
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .kernel_mix import KernelMixFit, uniform_lambdas
from .q_s import ConfusionMatrix
from .region import KernelClaim, Region, bbox_iou
from .taxonomy import Taxonomy

SCHEMA_VERSION = "0.1"  # bump when the on-the-wire schema changes


# ---------------------------------------------------------------------------
# Viewport spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Viewport:
    """Spec of the tile this scene graph was built on.

    Attributes
    ----------
    bbox_geo
        Geographic bounding box ``(west, south, east, north)`` in
        degrees, or ``None`` if the tile is in pure image coordinates
        (e.g. for synthetic tests).
    image_size
        ``(width, height)`` of the source tile in pixels.
    capture_metadata
        Free-form dict for things the orchestrator wants to round-trip
        back to the frontend (capture timestamp, camera state, layer
        ids, etc.).
    """

    image_size: Optional[Tuple[int, int]] = None
    bbox_geo: Optional[Tuple[float, float, float, float]] = None
    capture_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_size": list(self.image_size) if self.image_size else None,
            "bbox_geo": list(self.bbox_geo) if self.bbox_geo else None,
            "capture_metadata": dict(self.capture_metadata),
        }


# ---------------------------------------------------------------------------
# SceneNode / SceneEdge
# ---------------------------------------------------------------------------

@dataclass
class SceneNode:
    """One fused entity on the tile.

    Each node is anchored to a :class:`Region` (the consensus
    geometry) and carries the list of :class:`KernelClaim` instances
    that voted for it. The category posterior is computed by the
    :class:`KernelMixFit` against those claims.
    """

    id: str
    region: Region
    category: str
    category_index: int
    category_posterior: np.ndarray
    score: float
    claims: List[KernelClaim] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def sources(self) -> List[str]:
        """Distinct source ids that contributed claims to this node."""
        return sorted({c.source_id for c in self.claims})

    def to_dict(self, taxonomy: Taxonomy) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "category_index": int(self.category_index),
            "category_posterior": [
                {"category": taxonomy.name_of(i), "p": float(p)}
                for i, p in enumerate(self.category_posterior)
            ],
            "score": float(self.score),
            "region": self.region.to_dict(),
            "sources": self.sources,
            "claims": [c.to_dict() for c in self.claims],
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class SceneEdge:
    """An undirected edge between two :class:`SceneNode` instances.

    PR-1 produces a single relation type, ``"adjacent"`` (set when
    two nodes' bbox centroids are close), to keep the schema concrete.
    PR-3 will add ``"contains"``, ``"served_by"`` (building → road),
    etc.
    """

    source: str
    target: str
    relation: str = "adjacent"
    weight: float = 1.0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": float(self.weight),
            "attributes": dict(self.attributes),
        }


# ---------------------------------------------------------------------------
# SceneGraph
# ---------------------------------------------------------------------------

@dataclass
class SceneGraph:
    """Per-tile fused multi-kernel scene representation.

    Returned by :func:`build_scene_graph`. JSON-serializable via
    :meth:`to_dict`; the deepgis-xr orchestrator returns this dict
    directly to the Cesium frontend.
    """

    taxonomy: Taxonomy
    viewport: Viewport
    nodes: List[SceneNode] = field(default_factory=list)
    edges: List[SceneEdge] = field(default_factory=list)
    kernels_queried: List[str] = field(default_factory=list)
    fusion_metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def category_histogram(self) -> Dict[str, int]:
        """Argmax-category counts across all nodes."""
        out: Dict[str, int] = {c: 0 for c in self.taxonomy.categories}
        for n in self.nodes:
            out[n.category] = out.get(n.category, 0) + 1
        return out

    def nodes_by_category(self, category: str) -> List[SceneNode]:
        return [n for n in self.nodes if n.category == category]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "taxonomy": self.taxonomy.to_dict(),
            "viewport": self.viewport.to_dict(),
            "kernels_queried": list(self.kernels_queried),
            "nodes": [n.to_dict(self.taxonomy) for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "fusion_metadata": dict(self.fusion_metadata),
            "category_histogram": self.category_histogram(),
        }


# ---------------------------------------------------------------------------
# Spatial association
# ---------------------------------------------------------------------------

def _associate_claims(
    claims: Sequence[KernelClaim],
    *,
    iou_threshold: float = 0.5,
) -> List[List[KernelClaim]]:
    """Greedy single-link clustering of claims by bbox-IoU.

    Algorithm: sort claims by area descending; for each claim, attach
    it to the existing cluster with the highest bbox-IoU above
    ``iou_threshold``; otherwise start a new cluster anchored on this
    claim. Cluster bbox is the largest claim's bbox so the threshold
    has stable semantics ("does this new claim cover at least half of
    the anchor?").

    PR-1 keeps this O(N²) since the per-tile claim count is in the
    hundreds at most. PR-3 may swap in an R-tree for >10⁴ claims.
    """
    if iou_threshold <= 0.0 or iou_threshold > 1.0:
        raise ValueError(
            f"iou_threshold must be in (0, 1]; got {iou_threshold}"
        )
    ordered = sorted(claims, key=lambda c: -c.area)
    clusters: List[List[KernelClaim]] = []
    anchor_bboxes: List[Tuple[float, float, float, float]] = []
    for claim in ordered:
        best_idx = -1
        best_iou = iou_threshold
        for i, anchor in enumerate(anchor_bboxes):
            iou = bbox_iou(claim.bbox, anchor)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx >= 0:
            clusters[best_idx].append(claim)
        else:
            clusters.append([claim])
            anchor_bboxes.append(claim.bbox)
    return clusters


def _consensus_region(claims: Sequence[KernelClaim]) -> Region:
    """Pick the largest claim's geometry as the consensus region.

    Trivial in PR-1 — picking the biggest polygon is the most stable
    baseline when sources disagree on shape (a building footprint
    from OSM vs a slightly-cropped building roof from MR-house). PR-3
    can compute a per-pixel union or a shapely-based intersection.
    """
    if not claims:
        raise ValueError("_consensus_region requires at least one claim")
    biggest = max(claims, key=lambda c: c.area)
    return Region(
        polygon=biggest.polygon.copy(),
        bbox=biggest.bbox,
        area=biggest.area,
        geo_polygon=(
            biggest.geo_polygon.copy() if biggest.geo_polygon is not None else None
        ),
        mask_rle=biggest.mask_rle,
        image_size=biggest.image_size,
        attributes={
            "anchor_claim_id": biggest.id,
            "anchor_source": biggest.source_id,
            "n_claims": len(claims),
        },
    )


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------

def _bbox_centroid(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _adjacency_edges(
    nodes: Sequence[SceneNode],
    *,
    proximity: float = 0.05,
) -> List[SceneEdge]:
    """Build undirected adjacency edges between nodes whose bbox
    centroids are within ``proximity`` (in normalised image
    coordinates).

    Cheap O(N²) pass; for typical PHX viewports with ≲ 200 nodes this
    is well under a millisecond. PR-3 will replace centroid-proximity
    with the SAM-segment-adjacency graph specified in §5.
    """
    out: List[SceneEdge] = []
    centroids = [_bbox_centroid(n.region.bbox) for n in nodes]
    for i in range(len(nodes)):
        cx_i, cy_i = centroids[i]
        for j in range(i + 1, len(nodes)):
            cx_j, cy_j = centroids[j]
            d = float(np.hypot(cx_i - cx_j, cy_i - cy_j))
            if d <= proximity:
                # Weight ∈ (0, 1] decays with distance: 1.0 at zero,
                # 0 at proximity. Useful as an edge weight for
                # downstream graph kernels.
                w = max(0.0, 1.0 - d / proximity)
                out.append(SceneEdge(
                    source=nodes[i].id,
                    target=nodes[j].id,
                    relation="adjacent",
                    weight=w,
                    attributes={"centroid_distance": d},
                ))
    return out


# ---------------------------------------------------------------------------
# build_scene_graph: the public entry point
# ---------------------------------------------------------------------------

def build_scene_graph(
    claims: Iterable[KernelClaim],
    *,
    taxonomy: Taxonomy,
    q_s_table: Mapping[str, ConfusionMatrix],
    fit: Optional[KernelMixFit] = None,
    viewport: Optional[Viewport] = None,
    iou_threshold: float = 0.5,
    edge_proximity: float = 0.05,
    min_score: float = 0.0,
    kernels_queried: Optional[Sequence[str]] = None,
) -> SceneGraph:
    """Fuse a flat list of :class:`KernelClaim` into a :class:`SceneGraph`.

    Pipeline:

    1. Drop claims below ``min_score``.
    2. Greedy spatial association by bbox-IoU (``iou_threshold``).
    3. For each cluster, pick the consensus region (largest claim).
    4. Compute the per-region category posterior using ``fit``
       (defaults to uniform-λ over the sources present).
    5. Argmax → ``category``; ``score`` is the posterior mass on the
       argmax.
    6. Build adjacency edges (centroid-proximity).
    7. Wrap in a :class:`SceneGraph` and return.

    Parameters
    ----------
    claims
        Flat iterable of all kernels' claims for this tile.
    taxonomy
        Shared category vocabulary ``c*``.
    q_s_table
        Confusion matrices keyed by source id. Every source that
        appears in ``claims`` should also appear here, otherwise its
        contribution is silently zeroed (the orchestrator should
        warn).
    fit
        Pre-computed mixer fit. If ``None``, a uniform-λ fit is built
        over the source ids that appear in ``q_s_table`` ∩ ``claims``.
    viewport
        Tile metadata (bbox, image size, capture timestamp, …). If
        ``None``, an empty placeholder is used.
    iou_threshold
        Minimum bbox-IoU for two claims to merge into the same node.
    edge_proximity
        Maximum centroid distance (in normalised image coords) for
        two nodes to be linked by an ``"adjacent"`` edge.
    min_score
        Drop claims with native confidence below this floor before
        association.
    kernels_queried
        Optional list of source ids that the orchestrator *queried*,
        regardless of whether they returned anything. Used by the
        frontend to tell silent kernels apart from un-queried ones.

    Returns
    -------
    SceneGraph
        Fully populated, JSON-serializable scene representation.
    """
    t0 = time.perf_counter()

    all_claims = list(claims)
    raw = [c for c in all_claims if c.score >= min_score]
    n_dropped = len(all_claims) - len(raw)

    sources_in_claims = sorted({c.source_id for c in raw})
    sources_for_fit = [s for s in sources_in_claims if s in q_s_table]

    if fit is None and sources_for_fit:
        fit = uniform_lambdas(sources_for_fit, taxonomy)

    clusters = _associate_claims(raw, iou_threshold=iou_threshold)

    nodes: List[SceneNode] = []
    for cluster in clusters:
        region = _consensus_region(cluster)
        if fit is None:
            posterior = np.full(taxonomy.n, 1.0 / taxonomy.n)
        else:
            posterior = fit.posterior(cluster, q_s_table)

        cat_idx = int(np.argmax(posterior))
        cat_name = taxonomy.name_of(cat_idx)
        node_score = float(posterior[cat_idx])

        nodes.append(SceneNode(
            id=f"n-{uuid.uuid4().hex[:12]}",
            region=region,
            category=cat_name,
            category_index=cat_idx,
            category_posterior=posterior,
            score=node_score,
            claims=list(cluster),
            attributes={
                "n_claims": len(cluster),
                "n_distinct_sources": len({c.source_id for c in cluster}),
            },
        ))

    edges = _adjacency_edges(nodes, proximity=edge_proximity)

    if kernels_queried is None:
        kernels_queried = sources_in_claims

    fusion_metadata: Dict[str, Any] = {
        "n_claims_in":   len(raw),
        "n_claims_dropped_below_min_score": n_dropped,
        "n_clusters":    len(clusters),
        "iou_threshold": iou_threshold,
        "edge_proximity": edge_proximity,
        "min_score":     min_score,
        "wall_clock_s":  time.perf_counter() - t0,
        "schema_version": SCHEMA_VERSION,
    }
    if fit is not None:
        fusion_metadata["mixer"] = fit.to_dict()

    return SceneGraph(
        taxonomy=taxonomy,
        viewport=viewport or Viewport(),
        nodes=nodes,
        edges=edges,
        kernels_queried=list(kernels_queried),
        fusion_metadata=fusion_metadata,
    )
