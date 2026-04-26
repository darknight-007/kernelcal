"""
Collapse multiple persisted SceneGraph dictionaries into one fused graph.

The collapse path is intentionally a semantic-only PGM: observations are
associated into world entities using cheap geometry / OSM-id signals,
then a discrete factor graph infers the category posterior for each
entity from all contributing claims plus spatial consistency factors.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .factor_graph import (
    BPResult,
    FactorGraph,
    PairwiseTemporalFactor,
    PairwiseSpatialFactor,
    UnaryPerceptualFactor,
    loopy_bp,
)
from .q_s import ConfusionMatrix
from .region import KernelClaim, bbox_iou
from .taxonomy import Taxonomy


NodeRef = Tuple[int, str]


def _node_id(node: Mapping[str, Any], sg_idx: int, node_idx: int) -> str:
    return str(node.get("id") or f"sg{sg_idx}:node{node_idx}")


def _node_bbox(node: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    region = node.get("region") or {}
    bbox = region.get("bbox") or node.get("bbox")
    if bbox is None:
        claims = node.get("claims") or []
        for claim in claims:
            if claim.get("bbox") is not None:
                bbox = claim.get("bbox")
                break
    if bbox is None or len(bbox) != 4:
        return None
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def _bbox_from_points(points: Sequence[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    try:
        arr = np.asarray(points, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != 2:
        return None
    return (
        float(arr[:, 0].min()),
        float(arr[:, 1].min()),
        float(arr[:, 0].max()),
        float(arr[:, 1].max()),
    )


def _node_geo_bbox(node: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    region = node.get("region") or {}
    if region.get("geo_bbox") is not None:
        bbox = region["geo_bbox"]
        if len(bbox) == 4:
            return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if region.get("geo_polygon") is not None:
        bbox = _bbox_from_points(region["geo_polygon"])
        if bbox is not None:
            return bbox
    for claim in node.get("claims") or []:
        if claim.get("geo_polygon") is not None:
            bbox = _bbox_from_points(claim["geo_polygon"])
            if bbox is not None:
                return bbox
    return None


def _bbox_centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def _centroid_dist(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay = _bbox_centroid(a)
    bx, by = _bbox_centroid(b)
    return float(np.hypot(ax - bx, ay - by))


def _grid_candidate_pairs(
    refs: Sequence[NodeRef],
    bboxes: Mapping[NodeRef, Optional[Tuple[float, float, float, float]]],
    *,
    cell_size: float,
) -> List[Tuple[NodeRef, NodeRef]]:
    """Return candidate bbox pairs sharing a coarse spatial grid cell."""
    cell_size = max(float(cell_size), 1e-9)
    cells: Dict[Tuple[int, int], List[NodeRef]] = {}
    for ref in refs:
        bbox = bboxes.get(ref)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        ix1, ix2 = int(np.floor(x1 / cell_size)), int(np.floor(x2 / cell_size))
        iy1, iy2 = int(np.floor(y1 / cell_size)), int(np.floor(y2 / cell_size))
        # Guard against pathological giant boxes filling the world grid.
        if (ix2 - ix1 + 1) * (iy2 - iy1 + 1) > 512:
            ix1 = ix2 = int(np.floor(((x1 + x2) * 0.5) / cell_size))
            iy1 = iy2 = int(np.floor(((y1 + y2) * 0.5) / cell_size))
        for ix in range(ix1, ix2 + 1):
            for iy in range(iy1, iy2 + 1):
                cells.setdefault((ix, iy), []).append(ref)

    seen = set()
    out: List[Tuple[NodeRef, NodeRef]] = []
    for members in cells.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                key = tuple(sorted((a, b)))
                if key in seen:
                    continue
                seen.add(key)
                out.append((key[0], key[1]))
    return out


def _osm_feature_ids(node: Mapping[str, Any]) -> List[str]:
    ids: List[str] = []
    for claim in node.get("claims") or []:
        attrs = claim.get("attributes") or {}
        for key in ("osm_id", "id", "feature_id", "osm_way_id", "osm_node_id"):
            if key in attrs and attrs[key] is not None:
                ids.append(str(attrs[key]))
        if claim.get("source_id") == "osm" and claim.get("id"):
            ids.append(str(claim.get("id")))
    return sorted(set(ids))


def _claims_from_node(node: Mapping[str, Any]) -> List[KernelClaim]:
    out: List[KernelClaim] = []
    for claim in node.get("claims") or []:
        try:
            out.append(KernelClaim.from_dict(claim))
        except Exception:
            continue
    return out


class _UnionFind:
    def __init__(self, items: Iterable[NodeRef]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: NodeRef) -> NodeRef:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: NodeRef, b: NodeRef) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> List[List[NodeRef]]:
        out: Dict[NodeRef, List[NodeRef]] = {}
        for item in self.parent:
            out.setdefault(self.find(item), []).append(item)
        return list(out.values())


@dataclass(frozen=True)
class FusedSceneGraph:
    """JSON-friendly collapsed SceneGraph artifact."""

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    provenance: Dict[str, Any]
    bp_diagnostics: Dict[str, Any]
    taxonomy: str
    fusion_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "0.1",
            "taxonomy": self.taxonomy,
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "provenance": dict(self.provenance),
            "bp_diagnostics": dict(self.bp_diagnostics),
            "fusion_metadata": dict(self.fusion_metadata),
        }


def data_associate(
    scene_graph_dicts: Sequence[Mapping[str, Any]],
    *,
    iou_thresh: float = 0.5,
    osm_id_match: bool = True,
    centroid_eps: float = 0.0002,
) -> List[List[NodeRef]]:
    """Cluster raw observations into provisional world entities."""
    refs: List[NodeRef] = []
    nodes_by_ref: Dict[NodeRef, Mapping[str, Any]] = {}
    for sg_idx, sg in enumerate(scene_graph_dicts):
        for node_idx, node in enumerate(sg.get("nodes") or []):
            ref = (sg_idx, _node_id(node, sg_idx, node_idx))
            refs.append(ref)
            nodes_by_ref[ref] = node
    uf = _UnionFind(refs)

    # Hard equality for identical OSM feature identifiers.
    if osm_id_match:
        by_osm: Dict[str, NodeRef] = {}
        for ref, node in nodes_by_ref.items():
            for osm_id in _osm_feature_ids(node):
                if osm_id in by_osm:
                    uf.union(ref, by_osm[osm_id])
                else:
                    by_osm[osm_id] = ref

    # Geometry fallback. Cross-row association must use geographic
    # lon/lat boxes; normalised image coordinates are comparable only
    # inside the same SceneGraph row.
    geo_bboxes = {ref: _node_geo_bbox(node) for ref, node in nodes_by_ref.items()}
    img_bboxes = {ref: _node_bbox(node) for ref, node in nodes_by_ref.items()}
    for a, b in _grid_candidate_pairs(
        refs,
        geo_bboxes,
        cell_size=max(centroid_eps * 4.0, 0.001),
    ):
        ga = geo_bboxes.get(a)
        gb = geo_bboxes.get(b)
        if ga is not None and gb is not None:
            if bbox_iou(ga, gb) >= iou_thresh or _centroid_dist(ga, gb) <= centroid_eps:
                uf.union(a, b)

    # Same-row image-space fallback only; normalised image coordinates
    # are not comparable across captures.
    by_row: Dict[int, List[NodeRef]] = {}
    for ref in refs:
        by_row.setdefault(ref[0], []).append(ref)
    for row_refs in by_row.values():
        for a, b in _grid_candidate_pairs(row_refs, img_bboxes, cell_size=0.05):
            ia = img_bboxes.get(a)
            ib = img_bboxes.get(b)
            if ia is not None and ib is not None and bbox_iou(ia, ib) >= iou_thresh:
                uf.union(a, b)

    return [sorted(group) for group in uf.groups()]


def temporal_links(
    scene_graph_dicts: Sequence[Mapping[str, Any]],
    clusters: Optional[Sequence[Sequence[NodeRef]]] = None,
) -> List[Tuple[NodeRef, NodeRef]]:
    """Return adjacent-in-time observation pairs for repeated entities.

    With the current artifact schema we do not have per-node timestamps,
    only per-SceneGraph capture order. A temporal link therefore means:
    two observations from different SceneGraph rows landed in the same
    associated world-entity cluster. The links are sorted by SceneGraph
    index and connect consecutive observations only, avoiding dense
    all-pairs temporal factors for long-lived entities.
    """
    if clusters is None:
        clusters = data_associate(scene_graph_dicts)
    links: List[Tuple[NodeRef, NodeRef]] = []
    for cluster in clusters:
        ordered = sorted(cluster, key=lambda ref: (ref[0], ref[1]))
        prev: Optional[NodeRef] = None
        for ref in ordered:
            if prev is not None and prev[0] != ref[0]:
                links.append((prev, ref))
            prev = ref
    return links


def _lambda_mapping(
    lambdas: Union[Mapping[str, float], Sequence[float]],
    sources: Optional[Sequence[str]],
) -> Dict[str, float]:
    if isinstance(lambdas, Mapping):
        return {str(k): float(v) for k, v in lambdas.items()}
    if sources is None:
        raise ValueError("sources is required when lambdas is a sequence")
    return {str(s): float(l) for s, l in zip(sources, lambdas)}


def _representative_region(nodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    best = max(
        nodes,
        key=lambda n: float(((n.get("region") or {}).get("area") or n.get("area") or n.get("score") or 0.0)),
    )
    region = dict(best.get("region") or {})
    if not region:
        bbox = _node_bbox(best)
        if bbox is not None:
            region["bbox"] = list(bbox)
    return region


def collapse_scene_graphs(
    scene_graph_dicts: Iterable[Mapping[str, Any]],
    *,
    q_s_table: Mapping[str, ConfusionMatrix],
    lambdas: Union[Mapping[str, float], Sequence[float]],
    taxonomy: Optional[Taxonomy] = None,
    sources: Optional[Sequence[str]] = None,
    beta_spatial: float = 1.0,
    spatial_degree_cap: Optional[int] = 8,
    iou_thresh: float = 0.5,
    osm_id_match: bool = True,
    centroid_eps: float = 0.0002,
    persistence_alpha: float = 0.95,
    bp_max_iter: int = 30,
    bp_damping: float = 0.5,
    bp_tol: float = 1e-4,
) -> FusedSceneGraph:
    """Build and infer a collapsed semantic SceneGraph."""
    sg_list = list(scene_graph_dicts)
    if not sg_list:
        raise ValueError("collapse_scene_graphs requires at least one SceneGraph")
    if taxonomy is None:
        taxonomy = next(iter(q_s_table.values())).taxonomy
    lam = _lambda_mapping(lambdas, sources)

    clusters = data_associate(
        sg_list,
        iou_thresh=iou_thresh,
        osm_id_match=osm_id_match,
        centroid_eps=centroid_eps,
    )
    t_links = temporal_links(sg_list, clusters)
    ref_to_cluster: Dict[NodeRef, str] = {}
    cluster_nodes: Dict[str, List[Mapping[str, Any]]] = {}
    cluster_claims: Dict[str, List[KernelClaim]] = {}
    for idx, cluster in enumerate(clusters):
        cid = f"w-{idx:06d}"
        cluster_nodes[cid] = []
        cluster_claims[cid] = []
        for ref in cluster:
            sg_idx, node_id = ref
            ref_to_cluster[ref] = cid
            node = next(
                n for n_i, n in enumerate(sg_list[sg_idx].get("nodes") or [])
                if _node_id(n, sg_idx, n_i) == node_id
            )
            cluster_nodes[cid].append(node)
            cluster_claims[cid].extend(_claims_from_node(node))

    graph = FactorGraph()
    for cid in cluster_nodes:
        graph.add_variable(cid, taxonomy.n)
        graph.add_factor(UnaryPerceptualFactor(
            cid,
            cluster_claims[cid],
            q_s_table=q_s_table,
            lambdas=lam,
            taxonomy=taxonomy,
        ))

    edge_seen = set()
    edge_skipped_degree_cap = 0
    degree_by_cluster: Dict[str, int] = {cid: 0 for cid in cluster_nodes}
    out_edges: List[Dict[str, Any]] = []
    for sg_idx, sg in enumerate(sg_list):
        node_refs = {
            _node_id(node, sg_idx, node_idx): (sg_idx, _node_id(node, sg_idx, node_idx))
            for node_idx, node in enumerate(sg.get("nodes") or [])
        }
        for edge in sg.get("edges") or []:
            src = node_refs.get(str(edge.get("source")))
            dst = node_refs.get(str(edge.get("target")))
            if src is None or dst is None:
                continue
            a = ref_to_cluster.get(src)
            b = ref_to_cluster.get(dst)
            if a is None or b is None or a == b:
                continue
            key = tuple(sorted((a, b)))
            weight = float(edge.get("weight", 1.0) or 1.0)
            if key not in edge_seen:
                if spatial_degree_cap is not None and spatial_degree_cap >= 0:
                    if (
                        degree_by_cluster.get(key[0], 0) >= spatial_degree_cap
                        or degree_by_cluster.get(key[1], 0) >= spatial_degree_cap
                    ):
                        edge_skipped_degree_cap += 1
                        continue
                graph.add_factor(PairwiseSpatialFactor(
                    key[0],
                    key[1],
                    taxonomy=taxonomy,
                    weight=weight,
                    beta=beta_spatial,
                ))
                out_edges.append({
                    "id": f"e-{uuid.uuid4().hex[:12]}",
                    "source": key[0],
                    "target": key[1],
                    "relation": edge.get("relation", "adjacent"),
                    "weight": weight,
                    "attributes": {
                        "merged_from": [{
                            "scene_graph_index": sg_idx,
                            "source": edge.get("source"),
                            "target": edge.get("target"),
                        }],
                    },
                })
                edge_seen.add(key)
                degree_by_cluster[key[0]] = degree_by_cluster.get(key[0], 0) + 1
                degree_by_cluster[key[1]] = degree_by_cluster.get(key[1], 0) + 1

    temporal_factor_count = 0
    temporal_seen = set()
    for src_ref, dst_ref in t_links:
        a = ref_to_cluster.get(src_ref)
        b = ref_to_cluster.get(dst_ref)
        if a is None or b is None or a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in temporal_seen:
            continue
        graph.add_factor(PairwiseTemporalFactor(
            key[0],
            key[1],
            n_states=taxonomy.n,
            alpha=persistence_alpha,
        ))
        temporal_seen.add(key)
        temporal_factor_count += 1

    bp: BPResult = loopy_bp(
        graph,
        max_iter=bp_max_iter,
        damping=bp_damping,
        tol=bp_tol,
    )

    out_nodes: List[Dict[str, Any]] = []
    for cid, nodes in cluster_nodes.items():
        posterior = bp.posteriors[cid]
        cat_idx = int(np.argmax(posterior))
        out_nodes.append({
            "id": cid,
            "category_index": cat_idx,
            "category": taxonomy.name_of(cat_idx),
            "category_posterior": posterior.tolist(),
            "score": float(posterior[cat_idx]),
            "region": _representative_region(nodes),
            "sources": sorted({c.source_id for c in cluster_claims[cid]}),
            "claims": [c.to_dict() for c in cluster_claims[cid]],
            "merged_from": [
                {
                    "scene_graph_index": sg_idx,
                    "scene_graph_id": (
                        sg_list[sg_idx].get("session_id")
                        or sg_list[sg_idx].get("id")
                        or f"scene_graph_{sg_idx}"
                    ),
                    "node_id": node_id,
                }
                for sg_idx, node_id in clusters[int(cid.split("-")[1])]
            ],
        })

    provenance = {
        "n_scene_graphs": len(sg_list),
        "scene_graph_ids": [
            sg.get("session_id") or sg.get("id") or f"scene_graph_{i}"
            for i, sg in enumerate(sg_list)
        ],
        "n_input_nodes": sum(len(sg.get("nodes") or []) for sg in sg_list),
        "n_clusters": len(clusters),
        "association": {
            "iou_thresh": iou_thresh,
            "osm_id_match": osm_id_match,
            "centroid_eps": centroid_eps,
            "persistence_alpha": persistence_alpha,
        },
        "n_temporal_links": len(t_links),
    }
    diagnostics = {
        "n_iter": bp.n_iter,
        "converged": bp.converged,
        "max_delta": bp.max_delta,
        "map_energy": bp.map_energy,
        "history": bp.history,
        "n_variables": len(graph.variables),
        "n_factors": len(graph.factors),
        "n_temporal_factors": temporal_factor_count,
        "n_spatial_edges_used": len(edge_seen),
        "n_spatial_edges_skipped_degree_cap": edge_skipped_degree_cap,
    }
    return FusedSceneGraph(
        nodes=out_nodes,
        edges=out_edges,
        provenance=provenance,
        bp_diagnostics=diagnostics,
        taxonomy=taxonomy.name,
        fusion_metadata={
            "method": "factor_graph_bp",
            "lambdas": dict(lam),
            "beta_spatial": beta_spatial,
            "spatial_degree_cap": spatial_degree_cap,
            "persistence_alpha": persistence_alpha,
        },
    )


__all__ = [
    "FusedSceneGraph",
    "NodeRef",
    "collapse_scene_graphs",
    "data_associate",
    "temporal_links",
]
