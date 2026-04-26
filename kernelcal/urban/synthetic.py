"""kernelcal.urban.synthetic -- generate synthetic CityGraphs for tests
and structural-prediction receipts.

Used by the PR-C (CR-2026-04-26) planned-vs-organic ΔH receipt to
exercise the spectral pipeline without depending on live OSM /
Overpass.  The two layouts mirror the structural archetypes the Field
Notes 38/39 prediction is about:

* :func:`make_grid_layout` / :func:`make_grid_road_segments` -- a
  Manhattan-style rectilinear block lattice.  Buildings sit on
  intersections of a regular street grid.

* :func:`make_fringe_layout` / :func:`make_fringe_road_segments` -- a
  branching tree-like road network with buildings clustered near its
  endpoints.  Mirrors organic / fringe street fabric where roads
  splay outwards from a few seed nodes.

Both layouts are turned into :class:`CityGraph` instances by
:func:`synthetic_city_graph`, which is dependency-light: no osmnx, no
geopandas, no networkx -- only numpy / scipy and the existing
``CityGraph`` dataclass.

The σ used by the Gaussian-weighted Laplacian is computed identically
to the live ``buildings_to_graph`` / ``buildings_to_graph_via_roads``
flows so that synthetic spectra are directly comparable to live ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree

from .city_graph import CityGraph


# ---------------------------------------------------------------------------\
# layout generators
# ---------------------------------------------------------------------------\

def make_grid_layout(
    n_blocks_x: int = 8,
    n_blocks_y: int = 8,
    block_size_m: float = 80.0,
    jitter_m: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """Manhattan-style rectilinear lattice of building centroids.

    Returns positions array of shape ``(n_blocks_x * n_blocks_y, 2)``
    in metres.  ``jitter_m`` controls per-position Gaussian noise to
    keep the kd-tree from being degenerate.
    """
    rng = np.random.default_rng(seed)
    xs = np.arange(n_blocks_x, dtype=float) * block_size_m
    ys = np.arange(n_blocks_y, dtype=float) * block_size_m
    X, Y = np.meshgrid(xs, ys)
    pos = np.column_stack([X.ravel(), Y.ravel()])
    pos += rng.normal(scale=float(jitter_m), size=pos.shape)
    return pos


def make_grid_road_segments(
    n_blocks_x: int = 8,
    n_blocks_y: int = 8,
    block_size_m: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rectilinear road network: every interior block edge is a road
    segment.

    Returns
    -------
    nodes
        ``(n_road_nodes, 2)`` UTM-metres positions of road
        intersections.
    edges
        ``(n_road_edges, 2)`` int array of node-index pairs.  Edges
        are undirected; lengths are recovered from ``nodes`` by
        ``np.linalg.norm``.
    """
    xs = np.arange(n_blocks_x, dtype=float) * block_size_m
    ys = np.arange(n_blocks_y, dtype=float) * block_size_m
    X, Y = np.meshgrid(xs, ys)
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    def _id(i: int, j: int) -> int:
        return j * n_blocks_x + i

    edges = []
    for j in range(n_blocks_y):
        for i in range(n_blocks_x):
            if i + 1 < n_blocks_x:
                edges.append((_id(i, j), _id(i + 1, j)))
            if j + 1 < n_blocks_y:
                edges.append((_id(i, j), _id(i, j + 1)))
    return nodes, np.asarray(edges, dtype=int)


def make_fringe_layout(
    n_buildings: int = 64,
    n_seeds: int = 6,
    scale_m: float = 800.0,
    cluster_sigma_m: float = 25.0,
    seed: int = 42,
) -> np.ndarray:
    """Branching / clustered building layout.

    A small number of seed points are scattered in a square of side
    ``scale_m``.  Each seed spawns a Gaussian cluster of buildings;
    the number of buildings per cluster is roughly
    ``n_buildings / n_seeds`` with multinomial allocation.  This mimics
    organic settlement: dense clumps near road endpoints, voids in
    between.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0.0, scale_m, size=(int(n_seeds), 2))
    weights = rng.dirichlet(np.ones(int(n_seeds)) * 1.5)
    counts = rng.multinomial(int(n_buildings), weights)
    chunks = []
    for s, c in zip(seeds, counts):
        if int(c) <= 0:
            continue
        chunks.append(s + rng.normal(scale=float(cluster_sigma_m), size=(int(c), 2)))
    if not chunks:
        chunks.append(seeds[:1].copy())
    return np.concatenate(chunks, axis=0)


def make_fringe_road_segments(
    n_seeds: int = 6,
    scale_m: float = 800.0,
    branch_length_m: float = 120.0,
    n_branches_per_seed: int = 3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Branching road network: a small backbone connecting seed points,
    with short dead-end branches off each seed.

    Reflects the Field Note 107 motivation for ``road_knn``: organic
    fabric has roads that physically *don't* connect adjacent
    centroids, so the network distance can be much larger than
    Euclidean.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0.0, scale_m, size=(int(n_seeds), 2))

    nodes = list(seeds)
    edges = []

    # Backbone: connect seeds in order (a sparse path, not a complete
    # graph; this is what makes the fringe topology *fringe*).
    for i in range(int(n_seeds) - 1):
        edges.append((i, i + 1))

    # Branches: each seed sprouts a few short spurs.
    next_id = int(n_seeds)
    for s_idx, s_pos in enumerate(seeds):
        for _b in range(int(n_branches_per_seed)):
            theta = rng.uniform(0.0, 2.0 * math.pi)
            end = s_pos + branch_length_m * np.array([math.cos(theta), math.sin(theta)])
            nodes.append(end)
            edges.append((s_idx, next_id))
            next_id += 1

    return np.asarray(nodes, dtype=float), np.asarray(edges, dtype=int)


# ---------------------------------------------------------------------------\
# pure-numerics CityGraph build
# ---------------------------------------------------------------------------\

def _adaptive_sigma(positions: np.ndarray, sigma_frac: float) -> float:
    """σ used by the Gaussian-weighted Laplacian.

    Mirrors :func:`buildings_to_graph` exactly so spectra stay
    comparable.
    """
    n = positions.shape[0]
    xmin, ymin = positions.min(axis=0)
    xmax, ymax = positions.max(axis=0)
    diag = math.hypot(xmax - xmin, ymax - ymin)
    tree = cKDTree(positions)
    nn_dists, _ = tree.query(positions, k=min(2, n))
    median_nn = float(np.median(nn_dists[:, -1])) if n > 1 else 1.0
    return max(sigma_frac * max(diag, 1.0), 2.0 * max(median_nn, 1e-3))


def _euclidean_knn_W(
    positions: np.ndarray,
    k: int,
    sigma: float,
) -> np.ndarray:
    """Symmetric Gaussian-weighted k-NN adjacency on Euclidean distance."""
    n = positions.shape[0]
    tree = cKDTree(positions)
    dists, inds = tree.query(positions, k=min(k + 1, n))
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j_idx in range(1, dists.shape[1]):
            j = inds[i, j_idx]
            d = dists[i, j_idx]
            w = math.exp(-d * d / (sigma * sigma))
            if w > W[i, j]:
                W[i, j] = w
                W[j, i] = w
    return W


def _network_distance_matrix_synthetic(
    positions: np.ndarray,
    road_nodes: np.ndarray,
    road_edges: np.ndarray,
    cutoff_m: float,
) -> np.ndarray:
    """Pairwise building-to-building network distance via
    ``scipy.sparse.csgraph.shortest_path``.

    Bypasses osmnx / networkx entirely.  Buildings snap to their
    nearest road node by Euclidean distance.
    """
    n_b = positions.shape[0]
    n_r = road_nodes.shape[0]

    seg_lengths = np.linalg.norm(
        road_nodes[road_edges[:, 0]] - road_nodes[road_edges[:, 1]],
        axis=1,
    )

    rows = np.concatenate([road_edges[:, 0], road_edges[:, 1]])
    cols = np.concatenate([road_edges[:, 1], road_edges[:, 0]])
    data = np.concatenate([seg_lengths, seg_lengths])
    A = csr_matrix((data, (rows, cols)), shape=(n_r, n_r))

    snap_tree = cKDTree(road_nodes)
    snap_offsets, snap_nodes = snap_tree.query(positions, k=1)

    unique_snaps = np.unique(snap_nodes)
    dist_from_snap = shortest_path(A, indices=unique_snaps, method="D")
    snap_to_row = {int(s): r for r, s in enumerate(unique_snaps)}

    d_net = np.full((n_b, n_b), np.inf, dtype=float)
    np.fill_diagonal(d_net, 0.0)
    for i in range(n_b):
        s_i = int(snap_nodes[i])
        row_i = snap_to_row[s_i]
        for j in range(n_b):
            if i == j:
                continue
            s_j = int(snap_nodes[j])
            d = dist_from_snap[row_i, s_j]
            if not np.isfinite(d) or d > cutoff_m:
                continue
            d_net[i, j] = float(snap_offsets[i] + d + snap_offsets[j])

    return np.minimum(d_net, d_net.T)


def _network_knn_W(
    d_net: np.ndarray,
    k: int,
    sigma: float,
) -> Tuple[np.ndarray, int, int]:
    """Build the symmetric Gaussian-weighted k-NN adjacency from a
    network-distance matrix; returns ``(W, n_edges_added, n_isolated)``.
    """
    n = d_net.shape[0]
    W = np.zeros((n, n), dtype=float)
    n_edges_added = 0
    n_isolated = 0
    for i in range(n):
        row = d_net[i].copy()
        row[i] = np.inf
        order = np.argsort(row)
        picked = 0
        for j in order:
            d = row[j]
            if not np.isfinite(d):
                break
            if picked >= k:
                break
            w = math.exp(-d * d / (sigma * sigma))
            if w > W[i, j]:
                W[i, j] = w
                W[j, i] = w
                n_edges_added += 1
            picked += 1
        if picked == 0:
            n_isolated += 1
    return W, n_edges_added, n_isolated


def synthetic_city_graph(
    name: str,
    place: str,
    positions: np.ndarray,
    *,
    road_nodes: np.ndarray | None = None,
    road_edges: np.ndarray | None = None,
    k: int = 8,
    sigma_frac: float = 0.05,
    max_network_dist: float | None = None,
) -> CityGraph:
    """Build a :class:`CityGraph` from a synthetic layout.

    If ``road_nodes`` / ``road_edges`` are provided, the adjacency is
    built in ``road_knn`` mode using Dijkstra on the supplied road
    network; otherwise it falls back to Euclidean k-NN.  σ-matching
    follows :func:`buildings_to_graph` exactly.
    """
    n = positions.shape[0]
    if n < 2:
        raise ValueError("synthetic_city_graph needs at least 2 positions")

    sigma = _adaptive_sigma(positions, sigma_frac)
    use_roads = road_nodes is not None and road_edges is not None

    road_meta: dict = {}
    if use_roads:
        cutoff = float(max_network_dist) if max_network_dist is not None else 5.0 * sigma
        d_net = _network_distance_matrix_synthetic(
            positions, road_nodes, road_edges, cutoff_m=cutoff
        )
        W, n_edges_added, n_isolated = _network_knn_W(d_net, k=k, sigma=sigma)
        reachable = np.isfinite(d_net)
        np.fill_diagonal(reachable, False)
        road_meta = {
            "n_road_nodes": int(road_nodes.shape[0]),
            "n_road_edges": int(road_edges.shape[0]),
            "cutoff_m": float(cutoff),
            "n_isolated_buildings": int(n_isolated),
            "n_reachable_pairs": int(reachable.sum() // 2),
            "synthetic": True,
        }
    else:
        W = _euclidean_knn_W(positions, k=k, sigma=sigma)

    D = np.diag(W.sum(axis=1))
    L = D - W
    eigvals, eigvecs = scipy_eigh(L)
    eigvals = np.maximum(eigvals, 0.0)

    xmin, ymin = positions.min(axis=0)
    xmax, ymax = positions.max(axis=0)
    traits = np.zeros((n, 4), dtype=float)

    return CityGraph(
        name=name,
        place=place,
        positions=positions,
        traits=traits,
        L=L,
        W=W,
        eigvals=eigvals,
        eigvecs=eigvecs,
        n_buildings=int(n),
        bounds_m=(float(xmin), float(ymin), float(xmax), float(ymax)),
        raw_gdf=None,
        graph_mode="road_knn" if use_roads else "knn",
        road_meta=road_meta,
        raw_road_graph=None,
    )


__all__ = [
    "make_fringe_layout",
    "make_fringe_road_segments",
    "make_grid_layout",
    "make_grid_road_segments",
    "synthetic_city_graph",
]
