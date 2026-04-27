"""
Discrete kernel-fluid dynamics on graph-structured kernel space.

This module implements a lightweight "kernel hydrodynamics" toy model:

    - density rho_i(t) on graph nodes (kernel states),
    - antisymmetric edge velocity u_ij(t) = -u_ji(t),
    - continuity update for rho,
    - momentum-like edge update driven by pressure and potential gradients.

The implementation is intentionally simple and numerically robust so it can be
used for synthetic experiments and intuition-building before heavier models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Edge = Tuple[int, int]


@dataclass(frozen=True)
class FluidGraph:
    """Undirected graph used as discretized kernel space."""

    num_nodes: int
    edges: Tuple[Edge, ...]
    edge_lengths: np.ndarray
    adjacency: Tuple[Tuple[int, ...], ...]
    adjacency_mask: np.ndarray

    @staticmethod
    def from_edges(
        num_nodes: int,
        edges: Sequence[Edge],
        edge_lengths: Sequence[float] | None = None,
    ) -> "FluidGraph":
        """Create a graph from undirected edges.

        Parameters
        ----------
        num_nodes
            Number of nodes.
        edges
            Sequence of undirected edges (i, j), with i != j.
        edge_lengths
            Optional edge lengths matching `edges`.
        """
        if num_nodes <= 1:
            raise ValueError("num_nodes must be >= 2")
        if not edges:
            raise ValueError("edges must be non-empty")

        canonical_edges: List[Edge] = []
        seen: set[Edge] = set()
        for i, j in edges:
            if i == j:
                raise ValueError(f"self-loop edge ({i}, {j}) is not allowed")
            if i < 0 or j < 0 or i >= num_nodes or j >= num_nodes:
                raise ValueError(f"edge ({i}, {j}) is out of range for {num_nodes} nodes")
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                canonical_edges.append((a, b))
                seen.add((a, b))

        if edge_lengths is None:
            lengths = np.ones(len(canonical_edges), dtype=float)
        else:
            if len(edge_lengths) != len(canonical_edges):
                raise ValueError("edge_lengths length must match number of unique edges")
            lengths = np.asarray(edge_lengths, dtype=float)
            if np.any(lengths <= 0.0):
                raise ValueError("all edge lengths must be > 0")

        neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
        mask = np.zeros((num_nodes, num_nodes), dtype=bool)
        for i, j in canonical_edges:
            neighbors[i].append(j)
            neighbors[j].append(i)
            mask[i, j] = True
            mask[j, i] = True

        adjacency = tuple(tuple(sorted(nbrs)) for nbrs in neighbors)
        return FluidGraph(
            num_nodes=num_nodes,
            edges=tuple(canonical_edges),
            edge_lengths=lengths,
            adjacency=adjacency,
            adjacency_mask=mask,
        )

    @staticmethod
    def ring_with_chords(num_nodes: int = 20, chords: Sequence[Edge] = ((2, 12), (7, 17))) -> "FluidGraph":
        """Reference graph: ring plus optional long-range chords."""
        ring_edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
        return FluidGraph.from_edges(num_nodes=num_nodes, edges=[*ring_edges, *chords])


@dataclass(frozen=True)
class PotentialLandscape:
    """Node-wise potential terms used by the fluid dynamics."""

    loss: np.ndarray
    cost: np.ndarray
    info: np.ndarray

    def validate(self, num_nodes: int) -> None:
        if self.loss.shape != (num_nodes,):
            raise ValueError("loss must have shape (num_nodes,)")
        if self.cost.shape != (num_nodes,):
            raise ValueError("cost must have shape (num_nodes,)")
        if self.info.shape != (num_nodes,):
            raise ValueError("info must have shape (num_nodes,)")


@dataclass(frozen=True)
class FluidSimulationConfig:
    """Numerical and physical parameters for the discrete simulation."""

    dt: float = 0.01
    steps: int = 6000
    c_s2: float = 0.5
    beta: float = 0.8
    nu: float = 0.15
    eta: float = 1.0
    lambda_L: float = 1.0
    lambda_C: float = 1.2
    lambda_I_phase_a: float = 0.9
    lambda_I_phase_b: float = 1.4
    phase_switch_step: int = 3000
    rho_floor: float = 1e-12


@dataclass(frozen=True)
class FluidSimulationResult:
    """Time-series outputs and terminal state.

    The ``floor_mass_inserted`` and ``renormalize_correction`` fields
    are populated by both the legacy and PR-A.0 sparse solvers as
    optional ledger signals (CR-2026-04-26 §2): every step that
    applied the ``rho_floor`` clip or the post-step renormalisation
    records the magnitude of the implicit mass injection / correction.
    Legacy callers can ignore these fields.
    """

    rho_history: np.ndarray
    flux_to_node_10: np.ndarray
    flux_to_node_14: np.ndarray
    dissipation: np.ndarray
    entropy: np.ndarray
    concentration_m2: np.ndarray
    mass_error: np.ndarray
    rho_final: np.ndarray
    u_final: np.ndarray
    phi_history: np.ndarray
    floor_mass_inserted: np.ndarray = None  # type: ignore[assignment]
    renormalize_correction: np.ndarray = None  # type: ignore[assignment]


def ring_distance(num_nodes: int, i: int, j: int) -> int:
    """Shortest ring distance between nodes i and j."""
    d = abs(i - j)
    return min(d, num_nodes - d)


def gaussian_bump_on_ring(num_nodes: int, center: int, sigma: float) -> np.ndarray:
    """Normalized Gaussian-like bump over ring distance."""
    x = np.array([ring_distance(num_nodes, i, center) for i in range(num_nodes)], dtype=float)
    rho = np.exp(-(x**2) / (2.0 * sigma**2))
    rho /= np.sum(rho)
    return rho


def make_twenty_node_reference_landscape(num_nodes: int = 20) -> PotentialLandscape:
    """Reference landscape used in the two-phase 20-node experiment."""
    if num_nodes != 20:
        raise ValueError("reference landscape is defined for num_nodes=20")
    dist_to_10 = np.array([ring_distance(num_nodes, i, 10) for i in range(num_nodes)], dtype=float)
    dist_to_14 = np.array([ring_distance(num_nodes, i, 14) for i in range(num_nodes)], dtype=float)
    loss = 0.02 * (dist_to_10**2)
    cost = np.zeros(num_nodes, dtype=float)
    cost[[13, 14, 15]] = 8.0
    info = 6.0 * np.exp(-(dist_to_14**2) / (2.0 * 1.2**2))
    return PotentialLandscape(loss=loss, cost=cost, info=info)


def _edge_laplacian_term(graph: FluidGraph, u: np.ndarray, i: int, j: int) -> float:
    """Simple edge-smoothing term for oriented edge i->j."""
    nbr_i = graph.adjacency[i]
    nbr_j = graph.adjacency[j]

    mean_out_i = float(np.mean([u[i, k] for k in nbr_i])) if nbr_i else 0.0
    mean_in_j = float(np.mean([u[k, j] for k in nbr_j])) if nbr_j else 0.0
    return mean_out_i + mean_in_j - 2.0 * u[i, j]


def _build_phi(
    landscape: PotentialLandscape,
    lambda_L: float,
    lambda_C: float,
    lambda_I: float,
) -> np.ndarray:
    return lambda_L * landscape.loss + lambda_C * landscape.cost - lambda_I * landscape.info


def _net_inflow_to_node(F: np.ndarray, target: int, neighbors: Iterable[int]) -> float:
    return float(np.sum([F[j, target] for j in neighbors]))


def simulate_kernel_fluid(
    graph: FluidGraph,
    landscape: PotentialLandscape,
    config: FluidSimulationConfig,
    rho0: np.ndarray | None = None,
    track_rho_history: bool = True,
) -> FluidSimulationResult:
    """Run discrete kernel-fluid simulation on a graph."""
    n = graph.num_nodes
    landscape.validate(n)

    if rho0 is None:
        rho = gaussian_bump_on_ring(n, center=0, sigma=1.5)
    else:
        rho = np.asarray(rho0, dtype=float).copy()
        if rho.shape != (n,):
            raise ValueError("rho0 must have shape (num_nodes,)")
        if np.any(rho < 0):
            raise ValueError("rho0 must be nonnegative")
        s = float(np.sum(rho))
        if s <= 0:
            raise ValueError("rho0 must have positive mass")
        rho /= s

    u = np.zeros((n, n), dtype=float)
    F = np.zeros((n, n), dtype=float)

    rho_hist = np.zeros((config.steps + 1, n), dtype=float) if track_rho_history else np.zeros((1, n), dtype=float)
    phi_hist = np.zeros((config.steps, n), dtype=float)
    flux10 = np.zeros(config.steps, dtype=float)
    flux14 = np.zeros(config.steps, dtype=float)
    diss = np.zeros(config.steps, dtype=float)
    entr = np.zeros(config.steps, dtype=float)
    m2 = np.zeros(config.steps, dtype=float)
    mass_err = np.zeros(config.steps, dtype=float)
    floor_inserted = np.zeros(config.steps, dtype=float)
    renorm_corr = np.zeros(config.steps, dtype=float)

    if track_rho_history:
        rho_hist[0] = rho

    for t in range(config.steps):
        lambda_I = config.lambda_I_phase_a if t < config.phase_switch_step else config.lambda_I_phase_b
        phi = _build_phi(landscape, config.lambda_L, config.lambda_C, lambda_I)
        phi_hist[t] = phi
        p = config.c_s2 * rho

        # momentum-like edge update
        for edge_idx, (i, j) in enumerate(graph.edges):
            ell = graph.edge_lengths[edge_idx]
            grad_p = (p[j] - p[i]) / ell
            grad_phi = (phi[j] - phi[i]) / ell
            lap_term = _edge_laplacian_term(graph, u, i, j)
            du = -grad_p - grad_phi - config.beta * u[i, j] + config.nu * lap_term
            uij_new = u[i, j] + config.dt * du
            u[i, j] = uij_new
            u[j, i] = -uij_new

        # compute edge fluxes
        F.fill(0.0)
        for i, j in graph.edges:
            rho_ij = 0.5 * (rho[i] + rho[j])
            fij = rho_ij * u[i, j]
            F[i, j] = fij
            F[j, i] = -fij

        # continuity update
        drho = np.zeros_like(rho)
        for i in range(n):
            drho[i] = -np.sum([F[i, j] for j in graph.adjacency[i]])
        rho = rho + config.dt * drho
        # PR-A.0 ledger signal: anything below the floor is implicitly
        # mass-inserted; expose the magnitude rather than absorbing it
        # silently.
        below_floor = np.maximum(config.rho_floor - rho, 0.0)
        floor_inserted[t] = float(np.sum(below_floor))
        rho = np.maximum(rho, config.rho_floor)
        # Post-floor renormalisation:  what we just changed by clipping
        # plus any continuity-step floating-point drift gets folded back
        # into the unit-mass simplex.  The correction's signed value is
        # also a ledger signal.
        mass_total = float(np.sum(rho))
        renorm_corr[t] = mass_total - 1.0
        rho /= mass_total

        # diagnostics
        flux10[t] = _net_inflow_to_node(F, 10, graph.adjacency[10]) if n > 10 else 0.0
        flux14[t] = _net_inflow_to_node(F, 14, graph.adjacency[14]) if n > 14 else 0.0
        diss[t] = config.eta * float(np.sum([u[i, j] ** 2 for i, j in graph.edges]))
        entr[t] = float(-np.sum(rho * np.log(np.clip(rho, config.rho_floor, None))))
        m2[t] = float(np.sum(rho**2))
        mass_err[t] = abs(float(np.sum(rho)) - 1.0)
        if track_rho_history:
            rho_hist[t + 1] = rho

    return FluidSimulationResult(
        rho_history=rho_hist,
        flux_to_node_10=flux10,
        flux_to_node_14=flux14,
        dissipation=diss,
        entropy=entr,
        concentration_m2=m2,
        mass_error=mass_err,
        rho_final=rho.copy(),
        u_final=u.copy(),
        phi_history=phi_hist,
        floor_mass_inserted=floor_inserted,
        renormalize_correction=renorm_corr,
    )


def save_timeseries_csv(result: FluidSimulationResult, path: str | Path) -> Path:
    """Save core time-series diagnostics as CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(result.flux_to_node_10.shape[0], dtype=int)
    stacked = np.column_stack(
        [
            t,
            result.flux_to_node_10,
            result.flux_to_node_14,
            result.dissipation,
            result.entropy,
            result.concentration_m2,
            result.mass_error,
        ]
    )
    header = "step,flux_to_10,flux_to_14,dissipation,entropy,m2,mass_error"
    np.savetxt(out, stacked, delimiter=",", header=header, comments="")
    return out

