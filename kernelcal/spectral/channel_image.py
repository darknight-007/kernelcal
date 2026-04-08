"""
Image-to-channel-network extraction and flow-aware topology analysis.

This module provides a practical bridge from orthomap-like channel images to
graph-based analysis compatible with the rest of kernelcal.spectral.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import deque, defaultdict
import heapq
from pathlib import Path
import json

import numpy as np
from scipy import ndimage


Pixel = Tuple[int, int]  # (row, col)
EdgeKey = Tuple[int, int]  # (upstream_node, downstream_node)


@dataclass(frozen=True)
class ChannelEdge:
    """Topological edge extracted from a skeletonized channel image."""

    u: int
    v: int
    length_px: float
    mean_width_px: float
    pixels: Tuple[Pixel, ...]


@dataclass(frozen=True)
class ChannelGraphExtraction:
    """Result of image -> skeleton -> graph conversion."""

    binary_mask: np.ndarray
    skeleton_mask: np.ndarray
    node_pixels: Tuple[Pixel, ...]
    edges: Tuple[ChannelEdge, ...]


@dataclass(frozen=True)
class FlowTopologyAnalysis:
    """Flow-aware diagnostics computed on the extracted graph."""

    n_nodes: int
    n_edges: int
    outlet_node: int
    source_nodes: Tuple[int, ...]
    max_flow_value: float
    edge_capacity: Dict[EdgeKey, float]
    edge_flow: Dict[EdgeKey, float]
    edge_utilization: Dict[EdgeKey, float]
    bottleneck_edges: Tuple[EdgeKey, ...]
    strahler_order_node: Dict[int, int]
    strahler_order_edge: Dict[EdgeKey, int]


@dataclass(frozen=True)
class ChannelVerificationArtifacts:
    """Paths to generated visual and JSON verification artifacts."""

    overview_png: str
    graph_overlay_png: str
    labeled_overlay_png: str
    summary_json: str


def _load_image_rgb(path: str) -> np.ndarray:
    """Load image from disk as float RGB in [0, 1]."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "Pillow is required for image loading. Install with `pip install pillow`."
        ) from exc

    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr


def _foreground_from_rgb(
    rgb: np.ndarray,
    white_threshold: float = 0.93,
    saturation_threshold: float = 0.08,
    include_dark: bool = False,
) -> np.ndarray:
    """
    Heuristic foreground extraction for channel maps on mostly white backgrounds.

    Keeps:
    - colored pixels (cyan/green/yellow/red channels),
    - dark pixels (black labels/lines).
    """
    value = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = value - min_channel
    is_colored = (saturation > saturation_threshold) & (value < 0.99)
    is_dark = value < 0.78
    foreground = (is_colored | (include_dark and is_dark)) & (value < white_threshold)
    return foreground


def _keep_large_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return mask

    counts = np.bincount(labels.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    for idx, c in enumerate(counts):
        if idx == 0:
            continue
        keep[idx] = c >= min_size
    return keep[labels]


def _morphological_skeleton(mask: np.ndarray, max_iters: int = 1024) -> np.ndarray:
    """Binary morphological skeletonization using iterative erosion/opening."""
    mask = mask.astype(bool)
    skel = np.zeros_like(mask, dtype=bool)
    structure = np.ones((3, 3), dtype=np.uint8)
    current = mask.copy()

    for _ in range(max_iters):
        if not current.any():
            break
        eroded = ndimage.binary_erosion(current, structure=structure)
        opened = ndimage.binary_dilation(eroded, structure=structure)
        skel |= current & ~opened
        current = eroded
    return skel


def _neighbors8(pixel: Pixel, shape: Tuple[int, int]) -> List[Pixel]:
    r, c = pixel
    h, w = shape
    out: List[Pixel] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _skeleton_neighbors(skel: np.ndarray) -> Dict[Pixel, List[Pixel]]:
    pix = list(map(tuple, np.argwhere(skel)))
    pix_set = set(pix)
    nbrs: Dict[Pixel, List[Pixel]] = {}
    for p in pix:
        candidates = _neighbors8(p, skel.shape)
        nbrs[p] = [q for q in candidates if q in pix_set]
    return nbrs


def extract_channel_graph_from_image(
    image_path: str,
    *,
    min_component_size: int = 32,
    closing_iterations: int = 1,
    opening_iterations: int = 1,
    include_dark: bool = False,
) -> ChannelGraphExtraction:
    """
    Extract a channel topology graph from a channel-network image.

    Steps:
    1) RGB foreground extraction,
    2) morphology cleanup,
    3) skeletonization,
    4) node/edge extraction by compressing degree-2 chains.
    """
    rgb = _load_image_rgb(image_path)
    mask = _foreground_from_rgb(rgb, include_dark=include_dark)

    if opening_iterations > 0:
        mask = ndimage.binary_opening(mask, iterations=opening_iterations)
    if closing_iterations > 0:
        mask = ndimage.binary_closing(mask, iterations=closing_iterations)
    mask = _keep_large_components(mask, min_size=min_component_size)

    skel = _morphological_skeleton(mask)
    skel = _keep_large_components(skel, min_size=max(8, min_component_size // 2))

    distance = ndimage.distance_transform_edt(mask)
    nbrs = _skeleton_neighbors(skel)
    degrees = {p: len(v) for p, v in nbrs.items()}

    key_pixels = {p for p, d in degrees.items() if d != 2 and d > 0}
    if not key_pixels:
        # fallback: use all pixels as one pseudo-node if extraction is degenerate
        pts = tuple(map(tuple, np.argwhere(skel)))
        return ChannelGraphExtraction(
            binary_mask=mask,
            skeleton_mask=skel,
            node_pixels=pts[:1],
            edges=tuple(),
        )

    node_pixels = tuple(sorted(key_pixels))
    node_id = {p: i for i, p in enumerate(node_pixels)}

    traversed_segments = set()
    edges: List[ChannelEdge] = []

    for start in node_pixels:
        for nxt in nbrs[start]:
            seg = tuple(sorted((start, nxt)))
            if seg in traversed_segments:
                continue
            path = [start, nxt]
            prev = start
            cur = nxt
            traversed_segments.add(seg)

            while cur not in key_pixels:
                cur_neighbors = [x for x in nbrs[cur] if x != prev]
                if not cur_neighbors:
                    break
                nxt2 = cur_neighbors[0]
                seg2 = tuple(sorted((cur, nxt2)))
                if seg2 in traversed_segments:
                    break
                path.append(nxt2)
                traversed_segments.add(seg2)
                prev, cur = cur, nxt2

            if cur == start:
                continue
            if cur not in key_pixels:
                continue

            u = node_id[start]
            v = node_id[cur]
            if u == v:
                continue

            # Geometric edge length in pixels (8-neighbor metric)
            length = 0.0
            for a, b in zip(path[:-1], path[1:]):
                dr = float(a[0] - b[0])
                dc = float(a[1] - b[1])
                length += float(np.hypot(dr, dc))

            mean_width = float(2.0 * np.mean([distance[p] for p in path]))
            edges.append(
                ChannelEdge(
                    u=min(u, v),
                    v=max(u, v),
                    length_px=max(length, 1e-6),
                    mean_width_px=max(mean_width, 1e-6),
                    pixels=tuple(path),
                )
            )

    # deduplicate parallel repeats from opposite traversal starts
    unique = {}
    for e in edges:
        key = (e.u, e.v)
        if key not in unique or e.length_px < unique[key].length_px:
            unique[key] = e

    return ChannelGraphExtraction(
        binary_mask=mask,
        skeleton_mask=skel,
        node_pixels=node_pixels,
        edges=tuple(unique.values()),
    )


def _coarsen_extraction(
    extraction: ChannelGraphExtraction,
    *,
    bin_size_px: int = 6,
    min_edge_length_px: float = 2.0,
) -> ChannelGraphExtraction:
    """
    Spatially coarsen an extracted graph to suppress pixel-scale branching noise.
    """
    if bin_size_px <= 1 or len(extraction.node_pixels) <= 2:
        return extraction

    bin_to_nodes: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for nid, (r, c) in enumerate(extraction.node_pixels):
        key = (r // bin_size_px, c // bin_size_px)
        bin_to_nodes[key].append(nid)

    coarse_nodes: List[Pixel] = []
    node_map: Dict[int, int] = {}
    for members in bin_to_nodes.values():
        rs = [extraction.node_pixels[i][0] for i in members]
        cs = [extraction.node_pixels[i][1] for i in members]
        coarse_id = len(coarse_nodes)
        coarse_nodes.append((int(round(float(np.mean(rs)))), int(round(float(np.mean(cs))))))
        for i in members:
            node_map[i] = coarse_id

    agg: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
    for e in extraction.edges:
        u = node_map[e.u]
        v = node_map[e.v]
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        agg[key].append((e.length_px, e.mean_width_px))

    coarse_edges: List[ChannelEdge] = []
    for (u, v), vals in agg.items():
        lengths = [x[0] for x in vals]
        widths = [x[1] for x in vals]
        mean_len = float(np.mean(lengths))
        if mean_len < min_edge_length_px:
            continue
        mean_w = float(np.mean(widths))
        coarse_edges.append(
            ChannelEdge(
                u=u,
                v=v,
                length_px=max(mean_len, 1e-6),
                mean_width_px=max(mean_w, 1e-6),
                pixels=tuple(),
            )
        )

    return ChannelGraphExtraction(
        binary_mask=extraction.binary_mask,
        skeleton_mask=extraction.skeleton_mask,
        node_pixels=tuple(coarse_nodes),
        edges=tuple(coarse_edges),
    )


def _compact_to_largest_component(
    extraction: ChannelGraphExtraction,
) -> ChannelGraphExtraction:
    """Drop isolated nodes and keep the largest connected component."""
    if not extraction.edges:
        return extraction

    adj: Dict[int, List[int]] = defaultdict(list)
    for e in extraction.edges:
        adj[e.u].append(e.v)
        adj[e.v].append(e.u)

    visited = set()
    components: List[List[int]] = []
    for n in adj.keys():
        if n in visited:
            continue
        comp = []
        q = deque([n])
        visited.add(n)
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        components.append(comp)

    keep_nodes = set(max(components, key=len))
    old_to_new = {old: i for i, old in enumerate(sorted(keep_nodes))}
    new_nodes = tuple(extraction.node_pixels[old] for old in sorted(keep_nodes))

    new_edges: List[ChannelEdge] = []
    for e in extraction.edges:
        if e.u not in keep_nodes or e.v not in keep_nodes:
            continue
        new_edges.append(
            ChannelEdge(
                u=old_to_new[e.u],
                v=old_to_new[e.v],
                length_px=e.length_px,
                mean_width_px=e.mean_width_px,
                pixels=e.pixels,
            )
        )

    return ChannelGraphExtraction(
        binary_mask=extraction.binary_mask,
        skeleton_mask=extraction.skeleton_mask,
        node_pixels=new_nodes,
        edges=tuple(new_edges),
    )


def _build_undirected_adjacency(
    n_nodes: int, edges: Tuple[ChannelEdge, ...]
) -> Dict[int, List[Tuple[int, float, float]]]:
    adj = {i: [] for i in range(n_nodes)}
    for e in edges:
        adj[e.u].append((e.v, e.length_px, e.mean_width_px))
        adj[e.v].append((e.u, e.length_px, e.mean_width_px))
    return adj


def _shortest_distance_to_outlet(
    n_nodes: int, adj: Dict[int, List[Tuple[int, float, float]]], outlet: int
) -> Dict[int, float]:
    dist = {i: float("inf") for i in range(n_nodes)}
    dist[outlet] = 0.0
    pq = [(0.0, outlet)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, length, _ in adj[u]:
            nd = d + length
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def _edmonds_karp(
    capacity: Dict[int, Dict[int, float]],
    source: int,
    sink: int,
) -> Tuple[float, Dict[int, Dict[int, float]]]:
    residual: Dict[int, Dict[int, float]] = defaultdict(dict)
    nodes = set(capacity.keys())
    for u, nbrs in capacity.items():
        nodes.update(nbrs.keys())
    for u in nodes:
        residual[u] = defaultdict(float, residual[u])
    for u, nbrs in capacity.items():
        for v, cap in nbrs.items():
            residual[u][v] += float(cap)
            residual[v][u] += 0.0

    max_flow = 0.0
    while True:
        parent = {source: None}
        q = deque([source])
        while q and sink not in parent:
            u = q.popleft()
            for v, cap in residual[u].items():
                if cap > 1e-12 and v not in parent:
                    parent[v] = u
                    q.append(v)
        if sink not in parent:
            break

        path_cap = float("inf")
        cur = sink
        while cur != source:
            prev = parent[cur]
            path_cap = min(path_cap, residual[prev][cur])
            cur = prev

        cur = sink
        while cur != source:
            prev = parent[cur]
            residual[prev][cur] -= path_cap
            residual[cur][prev] += path_cap
            cur = prev

        max_flow += path_cap

    return max_flow, residual


def _infer_outlet(node_pixels: Tuple[Pixel, ...], adj: Dict[int, List[Tuple[int, float, float]]]) -> int:
    endpoints = [n for n, nbrs in adj.items() if len(nbrs) == 1]
    if not endpoints:
        # fallback: pick lowest pixel (largest row index)
        return int(np.argmax([p[0] for p in node_pixels]))
    return max(endpoints, key=lambda n: node_pixels[n][0])


def _strahler_orders(
    n_nodes: int,
    directed_edges: List[EdgeKey],
    dist_to_outlet: Dict[int, float],
) -> Tuple[Dict[int, int], Dict[EdgeKey, int]]:
    preds: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
    for u, v in directed_edges:
        preds[v].append(u)

    node_order: Dict[int, int] = {}
    # upstream first: larger distance to outlet
    for n in sorted(range(n_nodes), key=lambda x: dist_to_outlet[x], reverse=True):
        incoming = preds[n]
        if not incoming:
            node_order[n] = 1
            continue
        incoming_orders = [node_order.get(p, 1) for p in incoming]
        m = max(incoming_orders)
        if incoming_orders.count(m) >= 2:
            node_order[n] = m + 1
        else:
            node_order[n] = m

    edge_order = {(u, v): node_order[u] for (u, v) in directed_edges}
    return node_order, edge_order


def analyze_channel_network_image(
    image_path: str,
    *,
    source_supply: float = 1.0,
    capacity_width_power: float = 1.25,
    capacity_length_power: float = 1.0,
    min_component_size: int = 32,
    include_dark: bool = False,
    coarsen_bin_px: int = 1,
) -> Tuple[ChannelGraphExtraction, FlowTopologyAnalysis]:
    """
    End-to-end analysis from channel image to flow/topology diagnostics.

    Capacity model (dimensionless):
        c_e = (mean_width_px ** capacity_width_power) / (length_px ** capacity_length_power)
    Direction convention:
        edges are oriented from larger shortest-path distance to the inferred
        outlet node toward smaller distance.
    """
    extraction = extract_channel_graph_from_image(
        image_path=image_path,
        min_component_size=min_component_size,
        include_dark=include_dark,
    )

    extraction = _coarsen_extraction(extraction, bin_size_px=coarsen_bin_px)
    extraction = _compact_to_largest_component(extraction)
    n_nodes = len(extraction.node_pixels)
    if n_nodes == 0:
        raise ValueError("No channel graph nodes extracted from image.")

    adj = _build_undirected_adjacency(n_nodes, extraction.edges)
    outlet = _infer_outlet(extraction.node_pixels, adj)
    dist = _shortest_distance_to_outlet(n_nodes, adj, outlet)

    directed_edges: List[EdgeKey] = []
    edge_capacity: Dict[EdgeKey, float] = {}
    for e in extraction.edges:
        du, dv = dist[e.u], dist[e.v]
        if du > dv:
            u, v = e.u, e.v
        elif dv > du:
            u, v = e.v, e.u
        else:
            # tie-break by row: downstream has larger row index
            ru, rv = extraction.node_pixels[e.u][0], extraction.node_pixels[e.v][0]
            if ru >= rv:
                u, v = e.v, e.u
            else:
                u, v = e.u, e.v

        cap = (e.mean_width_px ** capacity_width_power) / (
            e.length_px ** capacity_length_power + 1e-9
        )
        directed_edges.append((u, v))
        edge_capacity[(u, v)] = float(max(cap, 1e-9))

    sources = tuple(sorted(n for n, nbrs in adj.items() if len(nbrs) == 1 and n != outlet))
    if not sources:
        # fallback for cyclic/non-tree extraction
        sources = tuple(sorted(n for n in range(n_nodes) if n != outlet))

    super_source = n_nodes
    super_sink = outlet
    capacity: Dict[int, Dict[int, float]] = defaultdict(dict)
    for (u, v), cap in edge_capacity.items():
        capacity[u][v] = cap
    for s in sources:
        capacity[super_source][s] = float(source_supply)

    max_flow, residual = _edmonds_karp(capacity, super_source, super_sink)

    edge_flow: Dict[EdgeKey, float] = {}
    edge_util: Dict[EdgeKey, float] = {}
    for (u, v), cap in edge_capacity.items():
        rem = residual[u][v] if u in residual and v in residual[u] else 0.0
        flow = cap - rem
        util = flow / cap if cap > 0 else 0.0
        edge_flow[(u, v)] = float(max(flow, 0.0))
        edge_util[(u, v)] = float(max(util, 0.0))

    bottlenecks = tuple(
        key
        for key, _ in sorted(edge_util.items(), key=lambda kv: kv[1], reverse=True)
        if edge_util[key] >= 0.90
    )

    strahler_node, strahler_edge = _strahler_orders(n_nodes, directed_edges, dist)

    analysis = FlowTopologyAnalysis(
        n_nodes=n_nodes,
        n_edges=len(extraction.edges),
        outlet_node=outlet,
        source_nodes=sources,
        max_flow_value=float(max_flow),
        edge_capacity=edge_capacity,
        edge_flow=edge_flow,
        edge_utilization=edge_util,
        bottleneck_edges=bottlenecks,
        strahler_order_node=strahler_node,
        strahler_order_edge=strahler_edge,
    )
    return extraction, analysis


def save_channel_extraction_verification(
    image_path: str,
    output_dir: str,
    *,
    prefix: str = "channel_verify",
    source_supply: float = 1.0,
    capacity_width_power: float = 1.25,
    capacity_length_power: float = 1.0,
    min_component_size: int = 32,
    include_dark: bool = False,
    coarsen_bin_px: int = 1,
    max_labeled_edges: int = 200,
) -> ChannelVerificationArtifacts:
    """
    Save verifiable artifacts for extracted channel topology.

    Produces:
      1) overview panel (original / mask / skeleton / overlay),
      2) high-resolution overlay with nodes/edges/outlet/sources,
      3) machine-readable JSON summary.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "matplotlib is required for visual verification output. "
            "Install with `pip install matplotlib`."
        ) from exc

    extraction, analysis = analyze_channel_network_image(
        image_path=image_path,
        source_supply=source_supply,
        capacity_width_power=capacity_width_power,
        capacity_length_power=capacity_length_power,
        min_component_size=min_component_size,
        include_dark=include_dark,
        coarsen_bin_px=coarsen_bin_px,
    )
    rgb = _load_image_rgb(image_path)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overview_path = out_dir / f"{prefix}_overview.png"
    overlay_path = out_dir / f"{prefix}_graph_overlay.png"
    labeled_overlay_path = out_dir / f"{prefix}_labeled_overlay.png"
    summary_path = out_dir / f"{prefix}_summary.json"

    rows = np.array([p[0] for p in extraction.node_pixels], dtype=float)
    cols = np.array([p[1] for p in extraction.node_pixels], dtype=float)

    def _edge_util(e: ChannelEdge) -> float:
        return max(
            analysis.edge_utilization.get((e.u, e.v), 0.0),
            analysis.edge_utilization.get((e.v, e.u), 0.0),
        )

    def _edge_strahler(e: ChannelEdge) -> int:
        return int(
            analysis.strahler_order_edge.get(
                (e.u, e.v),
                analysis.strahler_order_edge.get((e.v, e.u), 1),
            )
        )

    # High-resolution overlay.
    fig = plt.figure(figsize=(10, 6), dpi=160)
    ax = fig.add_subplot(111)
    ax.imshow(rgb)
    ax.set_title("Extracted channel graph overlay")
    for e in extraction.edges:
        ru, cu = extraction.node_pixels[e.u]
        rv, cv = extraction.node_pixels[e.v]
        util = _edge_util(e)
        color = plt.cm.plasma(min(1.0, util))
        lw = 0.8 + 2.2 * util
        ax.plot([cu, cv], [ru, rv], color=color, linewidth=lw, alpha=0.9)
    if len(cols) > 0:
        ax.scatter(cols, rows, s=6, c="white", edgecolors="black", linewidths=0.25, alpha=0.8)
    o_r, o_c = extraction.node_pixels[analysis.outlet_node]
    ax.scatter([o_c], [o_r], s=80, c="red", marker="*", label="outlet")
    if analysis.source_nodes:
        s_rows = [extraction.node_pixels[i][0] for i in analysis.source_nodes]
        s_cols = [extraction.node_pixels[i][1] for i in analysis.source_nodes]
        ax.scatter(s_cols, s_rows, s=20, c="lime", marker="o", edgecolors="black", linewidths=0.3, label="sources")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(overlay_path, bbox_inches="tight")
    plt.close(fig)

    # Labeled topology overlay for manual auditing (edge ID + Strahler order).
    fig = plt.figure(figsize=(11, 7), dpi=180)
    ax = fig.add_subplot(111)
    ax.imshow(rgb)
    ax.set_title("Labeled extracted topology (edge_id / Strahler order)")
    edges_with_id = list(enumerate(extraction.edges))
    for edge_id, e in edges_with_id:
        ru, cu = extraction.node_pixels[e.u]
        rv, cv = extraction.node_pixels[e.v]
        util = _edge_util(e)
        color = plt.cm.cividis(min(1.0, util))
        lw = 0.8 + 2.0 * util
        ax.plot([cu, cv], [ru, rv], color=color, linewidth=lw, alpha=0.95)

    # Limit text labels when graph is very dense; prefer long edges.
    label_candidates = sorted(
        edges_with_id,
        key=lambda item: item[1].length_px,
        reverse=True,
    )[: max(1, int(max_labeled_edges))]
    for edge_id, e in label_candidates:
        ru, cu = extraction.node_pixels[e.u]
        rv, cv = extraction.node_pixels[e.v]
        mr = 0.5 * (ru + rv)
        mc = 0.5 * (cu + cv)
        order = _edge_strahler(e)
        ax.text(
            mc,
            mr,
            f"{edge_id}/{order}",
            color="white",
            fontsize=5.5,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="black", alpha=0.55, linewidth=0.0),
        )

    if len(cols) > 0:
        ax.scatter(cols, rows, s=4, c="yellow", alpha=0.65)
    ax.scatter([o_c], [o_r], s=85, c="red", marker="*", label="outlet")
    ax.set_axis_off()
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(labeled_overlay_path, bbox_inches="tight")
    plt.close(fig)

    # 2x2 verification overview.
    fig = plt.figure(figsize=(12, 9), dpi=140)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(rgb)
    ax1.set_title("Original image")
    ax1.set_axis_off()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(extraction.binary_mask, cmap="gray")
    ax2.set_title("Foreground mask")
    ax2.set_axis_off()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(extraction.skeleton_mask, cmap="gray")
    ax3.set_title("Skeleton")
    ax3.set_axis_off()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(rgb)
    for e in extraction.edges:
        ru, cu = extraction.node_pixels[e.u]
        rv, cv = extraction.node_pixels[e.v]
        util = _edge_util(e)
        color = plt.cm.viridis(min(1.0, util))
        lw = 0.6 + 1.8 * util
        ax4.plot([cu, cv], [ru, rv], color=color, linewidth=lw, alpha=0.95)
    if len(cols) > 0:
        ax4.scatter(cols, rows, s=5, c="yellow", alpha=0.8)
    ax4.scatter([o_c], [o_r], s=70, c="red", marker="*", label="outlet")
    ax4.set_title(
        f"Extracted graph (N={analysis.n_nodes}, E={analysis.n_edges}, "
        f"max_flow={analysis.max_flow_value:.3f})"
    )
    ax4.set_axis_off()
    fig.tight_layout()
    fig.savefig(overview_path, bbox_inches="tight")
    plt.close(fig)

    top_edges = sorted(
        analysis.edge_utilization.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[:20]
    summary = {
        "image_path": str(image_path),
        "overview_png": str(overview_path),
        "graph_overlay_png": str(overlay_path),
        "labeled_overlay_png": str(labeled_overlay_path),
        "n_nodes": int(analysis.n_nodes),
        "n_edges": int(analysis.n_edges),
        "outlet_node": int(analysis.outlet_node),
        "outlet_pixel": [
            int(extraction.node_pixels[analysis.outlet_node][0]),
            int(extraction.node_pixels[analysis.outlet_node][1]),
        ],
        "source_nodes": [int(x) for x in analysis.source_nodes],
        "source_pixels": [
            [int(extraction.node_pixels[i][0]), int(extraction.node_pixels[i][1])]
            for i in analysis.source_nodes[:50]
        ],
        "max_flow_value": float(analysis.max_flow_value),
        "bottleneck_edges": [[int(u), int(v)] for (u, v) in analysis.bottleneck_edges],
        "top_edge_utilization": [
            {"edge": [int(u), int(v)], "utilization": float(vu)}
            for ((u, v), vu) in top_edges
        ],
        "edge_index_map": [
            {
                "edge_id": int(edge_id),
                "u": int(e.u),
                "v": int(e.v),
                "length_px": float(e.length_px),
                "mean_width_px": float(e.mean_width_px),
                "utilization": float(_edge_util(e)),
                "strahler_order": int(_edge_strahler(e)),
            }
            for edge_id, e in edges_with_id
        ],
        "max_strahler_order": int(max(analysis.strahler_order_node.values()))
        if analysis.strahler_order_node
        else 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    return ChannelVerificationArtifacts(
        overview_png=str(overview_path),
        graph_overlay_png=str(overlay_path),
        labeled_overlay_png=str(labeled_overlay_path),
        summary_json=str(summary_path),
    )


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract channel topology from image and save verification artifacts."
    )
    parser.add_argument("--image", required=True, help="Path to channel-network image")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write verification PNG/JSON outputs",
    )
    parser.add_argument("--prefix", default="channel_verify", help="Output filename prefix")
    parser.add_argument("--min-component-size", type=int, default=32)
    parser.add_argument("--include-dark", action="store_true")
    parser.add_argument("--coarsen-bin-px", type=int, default=1)
    parser.add_argument("--max-labeled-edges", type=int, default=200)
    args = parser.parse_args()

    artifacts = save_channel_extraction_verification(
        image_path=args.image,
        output_dir=args.output_dir,
        prefix=args.prefix,
        min_component_size=args.min_component_size,
        include_dark=args.include_dark,
        coarsen_bin_px=args.coarsen_bin_px,
        max_labeled_edges=args.max_labeled_edges,
    )
    print("Verification artifacts written:")
    print("  overview:", artifacts.overview_png)
    print("  overlay :", artifacts.graph_overlay_png)
    print("  labels  :", artifacts.labeled_overlay_png)
    print("  summary :", artifacts.summary_json)


if __name__ == "__main__":
    _main()
