"""River channel graph analysis and spectral diagnostics.

Builds drainage network graphs from D8 flow routing and applies the
triple spectral diagnostic of Proposition 3 (P2):

    D_channel = [H[h*] < H*] ∧ [E_curl > E*_c] ∧ [β₁ ≥ β*_1]

Physical context
----------------
A branching drainage network of order n has β₁ ≥ n - 1 independent cycles.
The spectral signature distinguishes channeled from unchanneled terrain:

  1. Fiedler concentration  — spectral entropy H[h*] < H_flat
  2. Elevated curl energy   — E_curl = ‖B₂ω*‖² > E_curl,flat
  3. Anomalous β₁           — β₁ ≥ n - 1 > β_flat_1 = 0

The module also implements Strahler ordering (mapped to eigenvalue bands)
and the Max-Flow Min-Cut spectral phase-transition criterion (P3, §7.5).
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass
from itertools import combinations
import math

from .dem import (
    TerrainGraph, dem_to_graph, terrain_graph_laplacian,
    d8_flow_direction, flow_accumulation, channel_mask,
    _D8_OFFSETS,
)


# ---------------------------------------------------------------------------
# Drainage network graph construction
# ---------------------------------------------------------------------------

@dataclass
class DrainageGraph:
    """Directed drainage network extracted from D8 flow routing.

    Attributes
    ----------
    nodes           : list of (row, col) tuples
    node_index      : dict (row,col) → int
    directed_edges  : list of (from_node, to_node) int pairs
    undirected_edges: numpy (E, 2) int array (for Laplacian construction)
    accumulation    : upstream area at each node
    strahler        : Strahler stream order at each node
    beta0           : number of connected components
    beta1           : number of independent cycles (braided channels)
    """
    nodes:            list[tuple[int, int]]
    node_index:       dict[tuple[int, int], int]
    directed_edges:   list[tuple[int, int]]
    undirected_edges: np.ndarray
    accumulation:     np.ndarray
    strahler:         np.ndarray
    beta0:            int
    beta1:            int


def drainage_network_graph(
    dem: np.ndarray,
    threshold: int = 10,
    dx: float = 1.0,
    dy: float = 1.0,
) -> DrainageGraph:
    """Extract a drainage network graph from a DEM.

    Parameters
    ----------
    dem       : (nrows, ncols) elevation array
    threshold : minimum upstream area (cells) to be considered a channel
    dx, dy    : cell spacings in metres

    Returns
    -------
    DrainageGraph
    """
    nrows, ncols = dem.shape
    fdir = d8_flow_direction(dem, dx=dx, dy=dy)
    acc  = flow_accumulation(fdir)
    chan = channel_mask(acc, threshold)

    # Channel cells become nodes
    nodes: list[tuple[int, int]] = []
    node_index: dict[tuple[int, int], int] = {}
    for r in range(nrows):
        for c in range(ncols):
            if chan[r, c]:
                node_index[(r, c)] = len(nodes)
                nodes.append((r, c))

    # Directed edges: each channel cell → its D8 downslope neighbour if also a channel
    directed_edges: list[tuple[int, int]] = []
    undirected_set: set[tuple[int, int]] = set()

    for r, c in nodes:
        d = int(fdir[r, c])
        if d >= 0:
            dr, dc = _D8_OFFSETS[d]
            nr, nc = r + dr, c + dc
            if (nr, nc) in node_index:
                i = node_index[(r, c)]
                j = node_index[(nr, nc)]
                directed_edges.append((i, j))
                pair = (min(i, j), max(i, j))
                undirected_set.add(pair)

    undirected_edges = (np.array(sorted(undirected_set), dtype=np.int32)
                        if undirected_set else np.empty((0, 2), dtype=np.int32))

    # Accumulation and Strahler at nodes
    node_acc = np.array([int(acc[r, c]) for r, c in nodes], dtype=np.int32)
    strahler_arr = _compute_strahler(nodes, node_index, directed_edges)

    # Betti numbers (graph level)
    n = len(nodes)
    e = len(undirected_set)
    beta0 = _count_components(n, undirected_edges) if n > 0 else 0
    beta1 = max(0, e - n + beta0)

    return DrainageGraph(
        nodes=nodes,
        node_index=node_index,
        directed_edges=directed_edges,
        undirected_edges=undirected_edges,
        accumulation=node_acc,
        strahler=strahler_arr,
        beta0=beta0,
        beta1=beta1,
    )


def _count_components(n: int, edges: np.ndarray) -> int:
    """Union-find component count."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[rj] = ri
    return len(set(find(i) for i in range(n)))


def _compute_strahler(
    nodes:         list[tuple[int, int]],
    node_index:    dict[tuple[int, int], int],
    directed_edges: list[tuple[int, int]],
) -> np.ndarray:
    """Strahler stream ordering via topological sort.

    Rules:
      - Headwater nodes (no upstream): order = 1
      - Node with exactly one upstream of order k: order = k
      - Node with two or more upstream of order k: order = k + 1
      - Node with upstream of different orders: order = max order

    Returns
    -------
    (N,) int array of Strahler orders
    """
    n = len(nodes)
    strahler = np.zeros(n, dtype=np.int32)
    # Build upstream adjacency
    upstream: list[list[int]] = [[] for _ in range(n)]
    indegree = np.zeros(n, dtype=np.int32)
    for i, j in directed_edges:
        upstream[j].append(i)
        indegree[j] += 1

    # Process sources first
    queue = deque(i for i in range(n) if indegree[i] == 0)
    for i in queue:
        strahler[i] = 1

    # Kahn's topological sort
    processed = np.zeros(n, dtype=bool)
    order_list: list[int] = []
    in_q = np.zeros(n, dtype=bool)
    q2 = deque(i for i in range(n) if indegree[i] == 0)
    for i in q2:
        in_q[i] = True

    while q2:
        i = q2.popleft()
        processed[i] = True
        order_list.append(i)
        for j_i, j in directed_edges:
            if j_i == i:
                indegree[j] -= 1
                if indegree[j] == 0 and not in_q[j]:
                    q2.append(j)
                    in_q[j] = True

    # Assign Strahler in topological order
    for i in order_list:
        ups = upstream[i]
        if not ups:
            strahler[i] = 1
        else:
            up_orders = sorted([strahler[u] for u in ups], reverse=True)
            if len(up_orders) >= 2 and up_orders[0] == up_orders[1]:
                strahler[i] = up_orders[0] + 1
            else:
                strahler[i] = up_orders[0]

    return strahler


def drainage_graph_laplacian(dg: DrainageGraph) -> np.ndarray:
    """Combinatorial Laplacian of the undirected drainage network."""
    n = len(dg.nodes)
    W = np.zeros((n, n), dtype=float)
    for i, j in dg.undirected_edges:
        W[i, j] = 1.0
        W[j, i] = 1.0
    D = np.diag(W.sum(axis=1))
    return D - W


# ---------------------------------------------------------------------------
# Hodge decomposition on drainage graphs (edge signals)
# ---------------------------------------------------------------------------

def hodge_edge_decompose(
    edge_signal: np.ndarray,
    n_nodes:     int,
    edges:       np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose an edge signal f = grad + curl + harmonic on the channel graph.

    For a graph (no faces), the Hodge decomposition reduces to:
      - Gradient component: projection onto image(B₁ᵀ)  [irrotational]
      - Harmonic component: ker(L₁) = ker(L₀) restricted to edge space
      - Curl component: residual (requires triangle faces for full curl;
        here approximated as the residual after gradient extraction)

    Parameters
    ----------
    edge_signal : (E,) float array — signal on edges
    n_nodes     : number of nodes
    edges       : (E, 2) int array of (i, j) node pairs

    Returns
    -------
    grad, curl_approx, harmonic   each (E,) float arrays
    """
    f = np.asarray(edge_signal, dtype=float)
    n_E = len(f)
    if n_E == 0:
        return f, f, f

    # Build signed incidence matrix B₁: (n_nodes, n_E)
    B1 = np.zeros((n_nodes, n_E), dtype=float)
    for eid, (i, j) in enumerate(edges):
        B1[i, eid] = -1.0
        B1[j, eid] = +1.0

    # L₀ = B₁ᵀ B₁
    L0 = B1.T @ B1  # (n_E, n_E)  — edge Laplacian from vertex structure

    # Gradient component: f_grad = B₁ᵀ ψ where ψ = (B₁ B₁ᵀ)⁺ B₁ f
    # Equivalently: solve min ‖B₁ᵀ ψ - f‖ for ψ
    B1_dense = B1                    # (n_nodes, n_E)
    B1T = B1.T                       # (n_E, n_nodes)
    ψ, *_ = np.linalg.lstsq(B1T, f, rcond=None)
    grad = B1T @ ψ                   # (n_E,)

    harmonic = f - grad              # residual (true harmonic if no faces)
    curl_approx = np.zeros_like(f)   # 0 for pure graph (no triangles)

    return grad, curl_approx, harmonic


def curl_energy(
    dg:          DrainageGraph,
    edge_signal: np.ndarray | None = None,
) -> float:
    """Compute the curl energy of a flow signal on the drainage network.

    For a graph with no faces, curl is zero. This function instead measures
    the deviation from pure gradient flow by projecting the flow signal onto
    the gradient subspace and computing the residual energy.  Non-zero residual
    indicates recirculation / braided channel structure.

    If ``edge_signal`` is None, uses the flow-accumulation difference as the
    natural drainage flow signal.

    Returns E_curl = ‖f - f_grad‖² / ‖f‖²   (normalised residual energy)
    """
    n_E = len(dg.undirected_edges)
    if n_E == 0:
        return 0.0

    if edge_signal is None:
        # Default: flow-accumulation difference across each edge
        sig = np.zeros(n_E, dtype=float)
        for eid, (i, j) in enumerate(dg.undirected_edges):
            sig[eid] = abs(float(dg.accumulation[i]) - float(dg.accumulation[j]))
    else:
        sig = np.asarray(edge_signal, dtype=float)

    n = len(dg.nodes)
    grad, curl_a, _ = hodge_edge_decompose(sig, n, dg.undirected_edges)
    total = float(np.dot(sig, sig))
    if total < 1e-12:
        return 0.0
    residual = float(np.dot(sig - grad, sig - grad))
    return residual / total


# ---------------------------------------------------------------------------
# Triple spectral diagnostic (P2, Proposition 3)
# ---------------------------------------------------------------------------

@dataclass
class ChannelDiagnostic:
    """Result of the triple spectral diagnostic for channel detection."""
    H_spectral:      float   # spectral entropy of channel graph
    E_curl:          float   # normalised curl energy of drainage flow
    beta1:           int     # graph β₁
    fiedler:         float   # Fiedler value λ₁
    n_nodes:         int
    n_edges:         int
    strahler_max:    int
    # Comparison against a flat (null) terrain
    H_flat:          float | None = None
    E_curl_flat:     float | None = None
    beta1_flat:      int | None   = None
    # Triple diagnostic flags
    fiedler_concentrated: bool | None = None
    curl_elevated:        bool | None = None
    beta1_anomalous:      bool | None = None

    @property
    def is_channeled(self) -> bool | None:
        """True if all three diagnostic criteria are met."""
        if None in (self.fiedler_concentrated, self.curl_elevated, self.beta1_anomalous):
            return None
        return self.fiedler_concentrated and self.curl_elevated and self.beta1_anomalous


@dataclass
class CriticalNodeResult:
    """Critical-node solution for a fixed group size k."""
    k: int
    nodes: np.ndarray                 # selected node ids
    method_used: str                  # "exact" or "greedy"
    pairwise_connectivity: int        # connected node pairs after node removal
    disconnected_pairs: int           # removed-pair count = total_pairs - pairwise_connectivity
    subbasins: int                    # connected components after removal


@dataclass
class CriticalNodeCurve:
    """Fragmentation curves under critical-node and central-node deletion."""
    k_values: np.ndarray
    critical_nodes: list[np.ndarray]
    central_nodes: list[np.ndarray] | None
    pairwise_connectivity_critical: np.ndarray
    pairwise_connectivity_central: np.ndarray | None
    subbasins_critical: np.ndarray
    subbasins_central: np.ndarray | None
    powerlaw_slope_pc_critical: float
    powerlaw_slope_pc_central: float | None
    linear_slope_subbasins_critical: float
    linear_slope_subbasins_central: float | None
    mean_pc_advantage_critical: float | None


def triple_spectral_diagnostic(
    dg:      DrainageGraph,
    dg_flat: DrainageGraph | None = None,
    n_modes: int = 20,
) -> ChannelDiagnostic:
    """Apply the triple spectral diagnostic (P2 Proposition 3) to a drainage graph.

    Parameters
    ----------
    dg      : drainage graph of the terrain patch under examination
    dg_flat : drainage graph of a flat / unchanneled reference terrain
              (if None, theoretical flat values are used)
    n_modes : number of Laplacian eigenvalues to compute

    Returns
    -------
    ChannelDiagnostic
    """
    n = len(dg.nodes)
    if n == 0:
        return ChannelDiagnostic(H_spectral=0., E_curl=0., beta1=0,
                                 fiedler=0., n_nodes=0, n_edges=0, strahler_max=0)

    # Spectral entropy
    L = drainage_graph_laplacian(dg)
    k = min(n_modes, n)
    eigvals = np.linalg.eigvalsh(L)[:k]
    eigvals = np.maximum(eigvals, 0.0)
    pos = eigvals[eigvals > 1e-10]
    if len(pos) > 0:
        h_bar = pos / pos.sum()
        H_chan = float(-np.sum(h_bar * np.log(h_bar + 1e-12)))
        fiedler = float(eigvals[1]) if len(eigvals) > 1 else 0.0
    else:
        H_chan = 0.0
        fiedler = 0.0

    E_curl_chan = curl_energy(dg)

    # Flat reference
    H_flat = E_flat = None
    beta1_flat = 0
    if dg_flat is not None and len(dg_flat.nodes) > 1:
        L_f = drainage_graph_laplacian(dg_flat)
        ev_f = np.maximum(np.linalg.eigvalsh(L_f), 0.0)
        pos_f = ev_f[ev_f > 1e-10]
        if len(pos_f) > 0:
            h_f = pos_f / pos_f.sum()
            H_flat = float(-np.sum(h_f * np.log(h_f + 1e-12)))
        E_flat = curl_energy(dg_flat)
        beta1_flat = dg_flat.beta1
    else:
        # Theoretical: flat terrain has maximum entropy and zero curl
        H_flat = float(np.log(max(k, 1)))
        E_flat = 0.0

    # Diagnostic flags
    fiedler_conc = H_chan < H_flat if H_flat is not None else None
    curl_elev    = E_curl_chan > (E_flat + 1e-6) if E_flat is not None else None
    b1_anom      = dg.beta1 > beta1_flat

    return ChannelDiagnostic(
        H_spectral=H_chan,
        E_curl=E_curl_chan,
        beta1=dg.beta1,
        fiedler=fiedler,
        n_nodes=n,
        n_edges=len(dg.undirected_edges),
        strahler_max=int(dg.strahler.max()) if len(dg.strahler) > 0 else 0,
        H_flat=H_flat,
        E_curl_flat=E_flat,
        beta1_flat=beta1_flat,
        fiedler_concentrated=fiedler_conc,
        curl_elevated=curl_elev,
        beta1_anomalous=b1_anom,
    )


# ---------------------------------------------------------------------------
# Abiotic null model for drainage networks
# ---------------------------------------------------------------------------

def abiotic_beta1_channels(n_junctions: int) -> dict[str, int]:
    """Abiotic β₁ prediction for a fluvial network.

    Fluvial geomorphology (P2, P3):
      - A drainage tree with n independent junctions has β₁ = n - 1.
      - A flat / un-channeled surface has β₁ = 0.
      - Braided or anastomosing networks exceed this lower bound.

    Parameters
    ----------
    n_junctions : number of channel junctions in the drainage network

    Returns
    -------
    dict with 'beta1_abio', 'kmin_abio'
    """
    beta1_abio = max(0, n_junctions - 1)
    return {
        "beta1_abio": beta1_abio,
        "beta0_abio": 1,
        "kmin_abio":  1 + beta1_abio,
        "n_junctions": n_junctions,
    }


# ---------------------------------------------------------------------------
# Spectral bandwidth budget (kmin)
# ---------------------------------------------------------------------------

def topology_budget(dg: DrainageGraph) -> dict[str, int]:
    """Compute the spectral topology budget kmin = β₀ + β₁ for a drainage graph.

    kmin is the minimum number of spectral modes required to preserve the
    full topological structure of the drainage network (P2 Theorem 1,
    P3 Corollary 9.3).

    Returns
    -------
    dict with 'beta0', 'beta1', 'kmin', 'strahler_max', 'n_nodes', 'n_edges'
    """
    return {
        "beta0":        dg.beta0,
        "beta1":        dg.beta1,
        "kmin":         dg.beta0 + dg.beta1,
        "strahler_max": int(dg.strahler.max()) if len(dg.strahler) > 0 else 0,
        "n_nodes":      len(dg.nodes),
        "n_edges":      len(dg.undirected_edges),
    }


# ---------------------------------------------------------------------------
# Critical nodes: pairwise connectivity and group betweenness (P3/H2)
# ---------------------------------------------------------------------------

def _adjacency_list_from_dg(dg: DrainageGraph) -> list[list[int]]:
    """Build an undirected adjacency list from DrainageGraph edges."""
    n = len(dg.nodes)
    adj = [[] for _ in range(n)]
    for i, j in dg.undirected_edges:
        ii, jj = int(i), int(j)
        adj[ii].append(jj)
        adj[jj].append(ii)
    return adj


def _component_sizes_after_removal(
    n_nodes: int,
    adj: list[list[int]],
    removed: set[int],
) -> list[int]:
    """Connected-component sizes after deleting a node set."""
    seen = np.zeros(n_nodes, dtype=bool)
    for r in removed:
        if 0 <= r < n_nodes:
            seen[r] = True

    sizes: list[int] = []
    for s in range(n_nodes):
        if seen[s]:
            continue
        q = deque([s])
        seen[s] = True
        size = 0
        while q:
            u = q.popleft()
            size += 1
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        sizes.append(size)
    return sizes


def pairwise_connectivity_after_removal(
    dg: DrainageGraph,
    removed_nodes: np.ndarray | list[int] | tuple[int, ...] | None = None,
) -> int:
    """Pairwise connectivity after removing nodes.

    Pairwise connectivity is the number of connected node pairs in the
    remaining graph:

        PC = sum_c |C_c| (|C_c| - 1) / 2

    where C_c are connected components after deletion.
    """
    n = len(dg.nodes)
    if n <= 1:
        return 0
    removed = set() if removed_nodes is None else set(int(x) for x in removed_nodes)
    adj = _adjacency_list_from_dg(dg)
    sizes = _component_sizes_after_removal(n, adj, removed)
    return int(sum(s * (s - 1) // 2 for s in sizes))


def subbasins_after_removal(
    dg: DrainageGraph,
    removed_nodes: np.ndarray | list[int] | tuple[int, ...] | None = None,
) -> int:
    """Number of connected components (sub-basins) after node deletion."""
    n = len(dg.nodes)
    if n == 0:
        return 0
    removed = set() if removed_nodes is None else set(int(x) for x in removed_nodes)
    adj = _adjacency_list_from_dg(dg)
    return int(len(_component_sizes_after_removal(n, adj, removed)))


def betweenness_centrality_undirected(dg: DrainageGraph) -> np.ndarray:
    """Unweighted undirected node betweenness centrality (Brandes algorithm)."""
    n = len(dg.nodes)
    if n == 0:
        return np.array([], dtype=float)

    adj = _adjacency_list_from_dg(dg)
    bc = np.zeros(n, dtype=float)

    for s in range(n):
        stack: list[int] = []
        pred: list[list[int]] = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=float)
        sigma[s] = 1.0
        dist = -np.ones(n, dtype=np.int32)
        dist[s] = 0

        q = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    q.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta = np.zeros(n, dtype=float)
        while stack:
            w = stack.pop()
            if sigma[w] > 0:
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]

    # undirected graph correction
    bc *= 0.5
    return bc


def most_central_nodes(dg: DrainageGraph, k: int) -> np.ndarray:
    """Top-k nodes by individual betweenness centrality."""
    n = len(dg.nodes)
    if n == 0 or k <= 0:
        return np.array([], dtype=int)
    k = int(min(k, n))
    bc = betweenness_centrality_undirected(dg)
    order = np.argsort(-bc, kind="mergesort")  # stable tie break by index
    return order[:k].astype(int)


def group_betweenness_score(
    dg: DrainageGraph,
    nodes: np.ndarray | list[int] | tuple[int, ...],
) -> int:
    """Group-betweenness proxy on trees via disconnected shortest-path pairs.

    On a tree, there is exactly one shortest path between any pair of nodes.
    Therefore, group betweenness of a node set S equals the number of node
    pairs disconnected by deleting S, i.e. total_pairs - pairwise_connectivity.
    """
    n = len(dg.nodes)
    total_pairs = n * (n - 1) // 2
    pc = pairwise_connectivity_after_removal(dg, nodes)
    return int(total_pairs - pc)


def identify_critical_nodes(
    dg: DrainageGraph,
    k: int,
    method: str = "auto",
    exact_max_combinations: int = 200000,
) -> CriticalNodeResult:
    """Identify critical nodes minimizing remaining pairwise connectivity.

    Parameters
    ----------
    dg : DrainageGraph
    k : number of nodes to remove
    method : {"auto", "exact", "greedy"}
        - auto: exact when combinatorics are moderate, else greedy.
    exact_max_combinations : max combinations allowed in exact mode.
    """
    n = len(dg.nodes)
    if n == 0 or k <= 0:
        return CriticalNodeResult(
            k=max(0, int(k)),
            nodes=np.array([], dtype=int),
            method_used="exact",
            pairwise_connectivity=0,
            disconnected_pairs=0,
            subbasins=0,
        )

    k = int(min(k, n))
    total_pairs = n * (n - 1) // 2

    if method not in {"auto", "exact", "greedy"}:
        raise ValueError("method must be one of: 'auto', 'exact', 'greedy'.")

    if method == "auto":
        n_comb = int(math.comb(n, k))
        method_used = "exact" if n_comb <= exact_max_combinations else "greedy"
    else:
        method_used = method

    best_nodes: tuple[int, ...] | None = None
    best_pc = np.inf

    if method_used == "exact":
        for cand in combinations(range(n), k):
            pc = pairwise_connectivity_after_removal(dg, cand)
            if pc < best_pc:
                best_pc = pc
                best_nodes = cand
    else:
        # greedy maximization of pairwise-connectivity drop
        selected: list[int] = []
        current_pc = pairwise_connectivity_after_removal(dg, selected)
        candidate_set = set(range(n))
        for _ in range(k):
            best_node = None
            best_drop = -1
            best_next_pc = current_pc
            for v in sorted(candidate_set):
                pc = pairwise_connectivity_after_removal(dg, selected + [v])
                drop = current_pc - pc
                if drop > best_drop:
                    best_drop = drop
                    best_next_pc = pc
                    best_node = v
            if best_node is None:
                break
            selected.append(best_node)
            candidate_set.remove(best_node)
            current_pc = best_next_pc
        best_nodes = tuple(selected)
        best_pc = current_pc

    if best_nodes is None:
        best_nodes = tuple()
        best_pc = pairwise_connectivity_after_removal(dg, best_nodes)

    subbasins = subbasins_after_removal(dg, best_nodes)
    disconnected_pairs = int(total_pairs - int(best_pc))
    return CriticalNodeResult(
        k=k,
        nodes=np.array(best_nodes, dtype=int),
        method_used=method_used,
        pairwise_connectivity=int(best_pc),
        disconnected_pairs=disconnected_pairs,
        subbasins=int(subbasins),
    )


def _powerlaw_slope(k_values: np.ndarray, y_values: np.ndarray) -> float:
    """Slope of log(y) vs log(k), returning NaN if not enough support."""
    k = np.asarray(k_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = (k > 0) & (y > 0)
    if int(mask.sum()) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(k[mask]), np.log(y[mask]), 1)
    return float(slope)


def critical_fragmentation_curve(
    dg: DrainageGraph,
    k_max: int = 10,
    method: str = "auto",
    exact_max_combinations: int = 200000,
    compare_central: bool = True,
) -> CriticalNodeCurve:
    """Compute fragmentation curves for critical vs central nodes.

    Returns pairwise-connectivity and sub-basin trajectories vs removed node
    count k, plus fitted slopes matching the analysis protocol in
    Sarker et al. (2019): power-law slope for pairwise connectivity and
    linear slope for sub-basin count.
    """
    n = len(dg.nodes)
    if n == 0:
        empty = np.array([], dtype=float)
        return CriticalNodeCurve(
            k_values=np.array([], dtype=int),
            critical_nodes=[],
            central_nodes=[] if compare_central else None,
            pairwise_connectivity_critical=empty,
            pairwise_connectivity_central=empty if compare_central else None,
            subbasins_critical=empty,
            subbasins_central=empty if compare_central else None,
            powerlaw_slope_pc_critical=float("nan"),
            powerlaw_slope_pc_central=float("nan") if compare_central else None,
            linear_slope_subbasins_critical=float("nan"),
            linear_slope_subbasins_central=float("nan") if compare_central else None,
            mean_pc_advantage_critical=float("nan") if compare_central else None,
        )

    k_max = int(max(1, min(k_max, n)))
    k_values = np.arange(1, k_max + 1, dtype=int)

    crit_nodes_list: list[np.ndarray] = []
    pc_crit = np.zeros(k_max, dtype=float)
    sub_crit = np.zeros(k_max, dtype=float)
    for i, k in enumerate(k_values):
        res = identify_critical_nodes(
            dg, int(k), method=method, exact_max_combinations=exact_max_combinations
        )
        crit_nodes_list.append(res.nodes)
        pc_crit[i] = float(res.pairwise_connectivity)
        sub_crit[i] = float(res.subbasins)

    pc_cent = None
    sub_cent = None
    cent_nodes_list = None
    slope_pc_cent = None
    slope_sub_cent = None
    mean_adv = None
    if compare_central:
        bc_rank = most_central_nodes(dg, k_max)
        cent_nodes_list = []
        pc_cent = np.zeros(k_max, dtype=float)
        sub_cent = np.zeros(k_max, dtype=float)
        for i, k in enumerate(k_values):
            nodes_k = bc_rank[:k].astype(int)
            cent_nodes_list.append(nodes_k)
            pc_cent[i] = float(pairwise_connectivity_after_removal(dg, nodes_k))
            sub_cent[i] = float(subbasins_after_removal(dg, nodes_k))
        slope_pc_cent = _powerlaw_slope(k_values, pc_cent)
        if len(k_values) >= 2:
            slope_sub_cent = float(np.polyfit(k_values.astype(float), sub_cent, 1)[0])
        else:
            slope_sub_cent = float("nan")
        mean_adv = float(np.mean(pc_cent - pc_crit))

    slope_pc_crit = _powerlaw_slope(k_values, pc_crit)
    if len(k_values) >= 2:
        slope_sub_crit = float(np.polyfit(k_values.astype(float), sub_crit, 1)[0])
    else:
        slope_sub_crit = float("nan")

    return CriticalNodeCurve(
        k_values=k_values,
        critical_nodes=crit_nodes_list,
        central_nodes=cent_nodes_list,
        pairwise_connectivity_critical=pc_crit,
        pairwise_connectivity_central=pc_cent,
        subbasins_critical=sub_crit,
        subbasins_central=sub_cent,
        powerlaw_slope_pc_critical=slope_pc_crit,
        powerlaw_slope_pc_central=slope_pc_cent,
        linear_slope_subbasins_critical=slope_sub_crit,
        linear_slope_subbasins_central=slope_sub_cent,
        mean_pc_advantage_critical=mean_adv,
    )
