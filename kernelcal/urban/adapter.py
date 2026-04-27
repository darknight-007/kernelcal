"""kernelcal.urban.adapter -- bridge from urban.CityGraph to fluid.FluidGraph.

PR-A.1 of CR-2026-04-26.

A :class:`kernelcal.urban.CityGraph` carries a Gaussian-weighted
adjacency ``W`` (Euclidean k-NN or road-aware k-NN over building
centroids) plus a graph Laplacian ``L`` and a UTM-projected node
position list.  PR-A's multi-component fluid lift (A.2) will run
``simulate_kernel_fluid_sparse`` on the same node set, so the bridge
needs to:

1. Reuse the *exact* connectivity from ``W`` (not, e.g., re-run k-NN
   on positions) so that PR-C's spectral receipts and PR-A's fluid
   simulations talk about the same graph.
2. Translate Gaussian *weights* into fluid *edge lengths* — the fluid
   solver consumes ``edge_lengths`` as a divisor for gradients
   (``grad_p = (p[j] - p[i]) / ell``).  High weight (close in
   substrate) ⇒ short fluid edge; low weight ⇒ long edge.  We use
   ``ell = 1 / max(W_ij, weight_floor)``.
3. Preserve connectivity invariants: the resulting :class:`FluidGraph`
   has the same connected-component count β₀ as the source CityGraph
   Laplacian (acceptance criterion A1).

The adapter is pure-numpy / pure-scipy and does not depend on osmnx,
geopandas, or networkx, so it can run inside CI on the synthetic
CityGraph builders shipped with PR-C.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ..fluid import FluidGraph, SparseFluidGraph
from .city_graph import CityGraph


def to_fluid_graph(
    city: CityGraph,
    *,
    weight_floor: float = 1e-6,
    return_sparse: bool = False,
) -> Union[FluidGraph, Tuple[FluidGraph, SparseFluidGraph]]:
    """Convert a :class:`CityGraph` to a :class:`FluidGraph`.

    Edge convention: every nonzero strict-upper-triangle entry of
    ``city.W`` becomes a canonical FluidGraph edge with
    ``edge_length = 1 / max(W[i, j], weight_floor)``.  Diagonal entries
    are ignored (graph Laplacians have ``W_ii = 0`` by construction).

    Parameters
    ----------
    city
        Source :class:`CityGraph`.  Must have a square, symmetric,
        non-negative ``W``.
    weight_floor
        Lower clip applied to ``W`` before inversion to avoid
        ``1/eps`` blow-ups for sub-numerical-noise weights.  Does
        **not** determine which entries become edges -- any strictly
        positive ``W_ij`` produces an edge.  Default ``1e-6``.
    return_sparse
        Also return a pre-built :class:`SparseFluidGraph` so downstream
        solvers (e.g. PR-A.2's multi-component lift) can reuse the
        cached incidence matrix across categories without rebuilding
        it per simulate call.

    Returns
    -------
    FluidGraph
        When ``return_sparse=False``.
    (FluidGraph, SparseFluidGraph)
        When ``return_sparse=True``.

    Raises
    ------
    ValueError
        If ``city.W`` is not a square symmetric non-negative matrix or
        has no nonzero off-diagonal entries (FluidGraph requires at
        least one edge).
    """
    W = np.asarray(city.W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(
            f"city.W must be a square matrix, got shape {W.shape}"
        )
    n = int(W.shape[0])
    if n < 2:
        raise ValueError("city.W must have at least 2 nodes")
    if not np.allclose(W, W.T, atol=1e-9, rtol=0.0):
        raise ValueError(
            "city.W must be symmetric; got max asymmetry "
            f"{float(np.max(np.abs(W - W.T))):.3e}"
        )
    if np.any(W < 0):
        raise ValueError(
            "city.W must be non-negative; got min "
            f"{float(np.min(W)):.3e}"
        )

    triu_i, triu_j = np.triu_indices(n, k=1)
    weights = W[triu_i, triu_j]
    nonzero = weights > 0
    if not np.any(nonzero):
        raise ValueError(
            "city.W has no nonzero off-diagonal entries; FluidGraph "
            "requires at least one edge"
        )

    edge_i = triu_i[nonzero]
    edge_j = triu_j[nonzero]
    edge_w = weights[nonzero]
    edge_lengths = 1.0 / np.maximum(edge_w, weight_floor)

    edges = list(zip(edge_i.tolist(), edge_j.tolist()))
    fluid_graph = FluidGraph.from_edges(
        num_nodes=n,
        edges=edges,
        edge_lengths=edge_lengths.tolist(),
    )

    if return_sparse:
        sparse_graph = SparseFluidGraph.from_fluid_graph(fluid_graph)
        return fluid_graph, sparse_graph
    return fluid_graph


def fluid_graph_connected_components(graph: FluidGraph) -> int:
    """Connected-component count of a :class:`FluidGraph`.

    Built from ``graph.adjacency_mask`` via
    :func:`scipy.sparse.csgraph.connected_components`.  Used by the
    PR-A.1 acceptance test (β₀ must agree with the source CityGraph
    Laplacian's near-zero-eigenvalue count) and exposed here so other
    consumers (PR-B's ledger closure test, PR-A.5's pipeline smoke)
    can compute it without re-deriving the adjacency.
    """
    adj_mask = np.asarray(graph.adjacency_mask, dtype=bool)
    if adj_mask.shape != (graph.num_nodes, graph.num_nodes):
        raise ValueError(
            f"adjacency_mask shape {adj_mask.shape} does not match "
            f"num_nodes={graph.num_nodes}"
        )
    sparse_adj = csr_matrix(adj_mask.astype(np.int8))
    n_components, _ = connected_components(sparse_adj, directed=False)
    return int(n_components)


__all__ = [
    "fluid_graph_connected_components",
    "to_fluid_graph",
]
