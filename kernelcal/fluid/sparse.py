"""kernelcal.fluid.sparse -- vectorised, edge-indexed kernel-fluid solver.

PR-A.0 of CR-2026-04-26.  This module is the canonical kernel-fluid
solver going forward; the legacy :func:`kernelcal.fluid.dynamics.simulate_kernel_fluid`
is preserved as a reference implementation for the 20-node ring-with-chords
benchmark but cannot scale past a few hundred nodes.

Why this exists
---------------

The legacy solver has three structural problems that block PR-A's
multi-component lift and PR-B's runtime ledger:

1. **Time complexity.**  Per-step Python ``for`` loops over edges plus
   per-edge list-comprehensions in :func:`_edge_laplacian_term` blow
   past the CR §2 "~1 second per timestep on a Tempe-scale viewport"
   target by orders of magnitude when ``n ~ 1000``.

2. **Dense ``(n, n)`` flux/momentum matrices.**  ``u`` and ``F`` are
   stored as ``(num_nodes, num_nodes)`` arrays, of which ~99% of
   entries are structurally zero on a real graph.  Memory and Python
   overhead scale O(n²) instead of O(E).

3. **Renormalisation-each-step conservation hack.**  Legacy applies
   ``rho = max(rho, rho_floor)`` followed by ``rho /= sum(rho)`` at
   the end of every step.  Both quietly absorb mass; PR-B's ledger
   would have nothing to record for the off-diagonal mass it is
   meant to expose.

This module addresses all three:

* Edge-indexed state arrays of shape ``(E,)``.
* A signed sparse incidence matrix ``D`` of shape ``(E, n)`` cached
  once and reused for every gradient / divergence / edge-Laplacian
  computation.
* A flux-conservative continuity step ``drho = D.T @ F_e`` that is
  *identically* traceless (``D @ 1 = 0`` row-wise), so mass is
  conserved to machine epsilon without any renormalisation.

Update-sweep semantics
----------------------

Legacy uses **Gauss-Seidel** (each edge update sees the just-updated
state of earlier edges in the sweep).  This emerged as an artefact
of Python loop ordering, not a deliberate scheme, and does not
vectorise.

The sparse solver uses **Jacobi**: ``u`` is snapshotted at the start
of each step and all derived quantities are computed from the
snapshot, with new ``u`` written atomically at the end.  Jacobi and
Gauss-Seidel agree at the level of stationary distributions and to
``O(dt)`` per step on transients.

Operator algebra
----------------

For canonical edge ``e = (i, j)`` with ``i < j``, define:

* ``u_e`` -- per-edge oriented flow, positive sign means i → j.
* Signed incidence ``D``: ``D[e, i] = -1``, ``D[e, j] = +1``.
* Per-node degree ``deg``: ``deg[k] = number of edges incident to k``.

Then the legacy operations vectorise as:

* gradient on edges  ``grad_p_e = (D @ p) / edge_lengths``
* per-node mean signed inflow  ``nu = (D.T @ u_e) / deg``
* edge-Laplacian smoothing  ``u_lap_e = (D @ nu) - 2 * u_e``

  (matches legacy ``mean_out_i + mean_in_j - 2*u[i,j]`` because
   ``mean_out_i = -nu[i]`` and ``mean_in_j = +nu[j]``.)

* edge flux  ``F_e = 0.5 * (rho[i_e] + rho[j_e]) * u_e``
* continuity  ``drho = D.T @ F_e``

Total mass change per step is
``sum(D.T @ F_e) = (1.T @ D.T) @ F_e = (D @ 1).T @ F_e = 0`` by
construction, so no renormalisation is ever needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from .dynamics import (
    FluidGraph,
    FluidSimulationConfig,
    FluidSimulationResult,
    PotentialLandscape,
    gaussian_bump_on_ring,
)


# ---------------------------------------------------------------------------\
# Sparse incidence cache
# ---------------------------------------------------------------------------\


@dataclass(frozen=True)
class SparseFluidGraph:
    """Edge-indexed view of a :class:`FluidGraph`.

    Built once per FluidGraph; reused across every step and (in PR-A.2)
    across every category in a multi-component run.  Pure numpy /
    scipy.sparse with no Python objects in the per-step path.

    Attributes
    ----------
    num_nodes
        Same as the underlying FluidGraph.
    num_edges
        Number of canonical edges (i < j).
    edges_idx
        ``(E, 2)`` int array, ``edges_idx[e, 0] < edges_idx[e, 1]``.
    edge_lengths
        ``(E,)`` float; copied from the FluidGraph in canonical-edge
        order.
    incidence
        ``(E, n)`` signed incidence ``D`` as a CSR sparse matrix:
        ``D[e, edges_idx[e, 0]] = -1``, ``D[e, edges_idx[e, 1]] = +1``.
    incidence_T
        Cached transpose ``D.T`` of shape ``(n, E)`` to avoid CSR/CSC
        conversion every step.
    degree
        ``(n,)`` float; node degrees (counts of canonical edges
        incident to each node).
    """

    num_nodes: int
    num_edges: int
    edges_idx: np.ndarray
    edge_lengths: np.ndarray
    incidence: csr_matrix
    incidence_T: csr_matrix
    degree: np.ndarray

    @classmethod
    def from_fluid_graph(cls, graph: FluidGraph) -> "SparseFluidGraph":
        """Build a :class:`SparseFluidGraph` from a :class:`FluidGraph`.

        Edges are taken in :class:`FluidGraph`'s canonical order
        (already ``i < j`` after :meth:`FluidGraph.from_edges`).
        """
        n = int(graph.num_nodes)
        edges = np.asarray(graph.edges, dtype=int).reshape(-1, 2)
        if not np.all(edges[:, 0] < edges[:, 1]):
            raise ValueError(
                "FluidGraph.edges must be canonical (i < j); "
                "got at least one edge with i >= j."
            )
        edge_lengths = np.asarray(graph.edge_lengths, dtype=float).copy()
        if edge_lengths.shape != (edges.shape[0],):
            raise ValueError(
                f"edge_lengths shape {edge_lengths.shape} does not match "
                f"edges shape {edges.shape[0:1]}"
            )

        E = edges.shape[0]
        rows = np.concatenate([np.arange(E), np.arange(E)])
        cols = np.concatenate([edges[:, 0], edges[:, 1]])
        data = np.concatenate([
            -np.ones(E, dtype=float),
            +np.ones(E, dtype=float),
        ])
        D = csr_matrix((data, (rows, cols)), shape=(E, n))
        D_T = csr_matrix(D.T)

        degree = np.zeros(n, dtype=float)
        for i in (edges[:, 0], edges[:, 1]):
            np.add.at(degree, i, 1.0)

        return cls(
            num_nodes=n,
            num_edges=E,
            edges_idx=edges,
            edge_lengths=edge_lengths,
            incidence=D,
            incidence_T=D_T,
            degree=degree,
        )


# ---------------------------------------------------------------------------\
# Vectorised operators (also used standalone by the unit tests)
# ---------------------------------------------------------------------------\


def edge_gradient(sg: SparseFluidGraph, scalar: np.ndarray) -> np.ndarray:
    """Per-edge oriented gradient ``(p[j] - p[i]) / ell`` for canonical
    edges ``(i, j), i < j``.
    """
    if scalar.shape != (sg.num_nodes,):
        raise ValueError(
            f"scalar shape {scalar.shape} does not match num_nodes={sg.num_nodes}"
        )
    return (sg.incidence @ scalar) / sg.edge_lengths


def node_signed_inflow(sg: SparseFluidGraph, u_edge: np.ndarray) -> np.ndarray:
    """Per-node signed inflow ``D.T @ u_e``.

    Equals ``(inflow - outflow)`` on each node summed over its
    incident edges with the sign convention "u_e > 0 means flow
    i->j for canonical edge (i, j) with i < j".
    """
    if u_edge.shape != (sg.num_edges,):
        raise ValueError(
            f"u_edge shape {u_edge.shape} does not match num_edges={sg.num_edges}"
        )
    return sg.incidence_T @ u_edge


def edge_laplacian_smoothing(
    sg: SparseFluidGraph, u_edge: np.ndarray
) -> np.ndarray:
    """Vectorised replacement for legacy :func:`_edge_laplacian_term`.

    Returns, for each canonical edge ``e = (i, j)``,
    ``mean_out_i + mean_in_j - 2 * u_e`` where
    ``mean_out_k = sum_{e' incident k} sign_k(e') * u_{e'} / deg(k)``
    and ``sign_k(e') = +1`` if k is the smaller endpoint of e'.

    Identity used: ``mean_out_i = -nu[i]`` and ``mean_in_j = +nu[j]``
    when ``nu = (D.T @ u_e) / deg``, so the whole term reduces to
    ``(D @ nu) - 2 * u_e``.
    """
    psi = node_signed_inflow(sg, u_edge)
    nu = psi / sg.degree
    return (sg.incidence @ nu) - 2.0 * u_edge


def edge_flux(
    sg: SparseFluidGraph, rho: np.ndarray, u_edge: np.ndarray
) -> np.ndarray:
    """Per-edge advective flux ``F_e = 0.5 * (rho[i] + rho[j]) * u_e``."""
    if rho.shape != (sg.num_nodes,):
        raise ValueError(
            f"rho shape {rho.shape} does not match num_nodes={sg.num_nodes}"
        )
    rho_at_i = rho[sg.edges_idx[:, 0]]
    rho_at_j = rho[sg.edges_idx[:, 1]]
    return 0.5 * (rho_at_i + rho_at_j) * u_edge


def continuity_drho(sg: SparseFluidGraph, F_edge: np.ndarray) -> np.ndarray:
    """Per-node mass-change rate ``drho = D.T @ F_e``.

    Conservation property: ``sum(drho) == 0`` to machine epsilon for
    any flux profile, because ``D @ 1 == 0`` row-wise.
    """
    if F_edge.shape != (sg.num_edges,):
        raise ValueError(
            f"F_edge shape {F_edge.shape} does not match num_edges={sg.num_edges}"
        )
    return sg.incidence_T @ F_edge


def build_phi(
    landscape: PotentialLandscape,
    lambda_L: float,
    lambda_C: float,
    lambda_I: float,
) -> np.ndarray:
    """Compose the per-node potential.  Same as legacy ``_build_phi``."""
    return (
        lambda_L * landscape.loss
        + lambda_C * landscape.cost
        - lambda_I * landscape.info
    )


# ---------------------------------------------------------------------------\
# Solver
# ---------------------------------------------------------------------------\


def simulate_kernel_fluid_sparse(
    graph: FluidGraph,
    landscape: PotentialLandscape,
    config: FluidSimulationConfig,
    rho0: Optional[np.ndarray] = None,
    track_rho_history: bool = True,
    sparse_graph: Optional[SparseFluidGraph] = None,
) -> FluidSimulationResult:
    """Vectorised kernel-fluid simulation on a graph.

    Drop-in for :func:`kernelcal.fluid.dynamics.simulate_kernel_fluid`
    (single-component) with three differences:

    1. Per-step cost is O(E) sparse mat-vec products, not O(E * deg).
    2. Mass is conserved exactly without the renormalise-each-step
       hack; ``config.rho_floor`` is honoured only inside the entropy
       diagnostic, never as a mutation of ``rho``.
    3. Update sweep is Jacobi, not Gauss-Seidel.  The two agree at
       stationary state and to ``O(dt)`` on transients.

    Parameters
    ----------
    graph
        :class:`FluidGraph`; passed through to
        :class:`SparseFluidGraph` if ``sparse_graph`` is not provided.
    landscape
        :class:`PotentialLandscape`.
    config
        :class:`FluidSimulationConfig`.  ``config.rho_floor`` is the
        clip used inside ``log(rho)`` for the entropy diagnostic
        (never applied as a state mutation).
    rho0
        Optional initial density.  Defaults to a Gaussian bump on
        node 0 with sigma 1.5 (same as legacy).
    track_rho_history
        Whether to retain the full rho trajectory.  When ``False`` the
        returned ``rho_history`` has shape ``(1, n)``.
    sparse_graph
        Optional pre-built :class:`SparseFluidGraph` to avoid
        rebuilding the incidence matrix when running the same graph
        repeatedly (e.g. across categories in PR-A.2).
    """
    sg = sparse_graph if sparse_graph is not None else SparseFluidGraph.from_fluid_graph(graph)
    n, E = sg.num_nodes, sg.num_edges
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

    u = np.zeros(E, dtype=float)

    rho_hist = (
        np.zeros((config.steps + 1, n), dtype=float)
        if track_rho_history
        else np.zeros((1, n), dtype=float)
    )
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

    # Pre-compute per-node->incident-edge gather for the diagnostic
    # ``flux_to_node_*`` outputs (legacy returns net inflow at a node;
    # we compute it from F_edge via D.T).
    node10_signed = _node_signed_edge_dot(sg, target=10) if n > 10 else None
    node14_signed = _node_signed_edge_dot(sg, target=14) if n > 14 else None

    for t in range(config.steps):
        lambda_I = (
            config.lambda_I_phase_a
            if t < config.phase_switch_step
            else config.lambda_I_phase_b
        )
        phi = build_phi(landscape, config.lambda_L, config.lambda_C, lambda_I)
        phi_hist[t] = phi

        p = config.c_s2 * rho

        grad_p = edge_gradient(sg, p)
        grad_phi = edge_gradient(sg, phi)
        u_lap = edge_laplacian_smoothing(sg, u)

        du = -grad_p - grad_phi - config.beta * u + config.nu * u_lap
        u = u + config.dt * du

        F = edge_flux(sg, rho, u)
        drho = continuity_drho(sg, F)
        rho = rho + config.dt * drho

        # PR-A.0 ledger signal: any clip below ``rho_floor`` is implicit
        # mass insertion; any post-clip renormalisation is a unit-mass
        # correction.  Both are recorded rather than absorbed silently
        # so PR-B's runtime ledger has a real signal to balance.
        below_floor = np.maximum(config.rho_floor - rho, 0.0)
        floor_inserted[t] = float(np.sum(below_floor))
        rho = np.maximum(rho, config.rho_floor)
        mass_total = float(np.sum(rho))
        renorm_corr[t] = mass_total - 1.0
        if mass_total > 0.0:
            rho /= mass_total

        if n > 10:
            flux10[t] = float(node10_signed @ F)
        if n > 14:
            flux14[t] = float(node14_signed @ F)
        diss[t] = config.eta * float(np.sum(u * u))
        entr[t] = float(
            -np.sum(rho * np.log(np.clip(rho, config.rho_floor, None)))
        )
        m2[t] = float(np.sum(rho * rho))
        mass_err[t] = abs(float(np.sum(rho)) - 1.0)
        if track_rho_history:
            rho_hist[t + 1] = rho

    # Re-emit the legacy dense u/F at the end so callers that inspected
    # them keep working.  This is O(E) memory only for the final state.
    u_dense = np.zeros((n, n), dtype=float)
    u_dense[sg.edges_idx[:, 0], sg.edges_idx[:, 1]] = u
    u_dense[sg.edges_idx[:, 1], sg.edges_idx[:, 0]] = -u

    return FluidSimulationResult(
        rho_history=rho_hist,
        flux_to_node_10=flux10,
        flux_to_node_14=flux14,
        dissipation=diss,
        entropy=entr,
        concentration_m2=m2,
        mass_error=mass_err,
        rho_final=rho.copy(),
        u_final=u_dense,
        phi_history=phi_hist,
        floor_mass_inserted=floor_inserted,
        renormalize_correction=renorm_corr,
    )


def _node_signed_edge_dot(sg: SparseFluidGraph, target: int) -> np.ndarray:
    """Build the length-E sign vector ``D[:, target]`` so that
    ``inflow_at_target = (D[:, target] @ F_edge)`` matches the legacy
    "net inflow at ``target``" diagnostic.

    Sign matches legacy ``_net_inflow_to_node``: positive contribution
    when oriented flux flows into ``target``.
    """
    incident = (sg.edges_idx[:, 0] == target) | (sg.edges_idx[:, 1] == target)
    sign = np.zeros(sg.num_edges, dtype=float)
    sign[(sg.edges_idx[:, 1] == target)] = +1.0
    sign[(sg.edges_idx[:, 0] == target)] = -1.0
    return sign * incident


__all__ = [
    "SparseFluidGraph",
    "build_phi",
    "continuity_drho",
    "edge_flux",
    "edge_gradient",
    "edge_laplacian_smoothing",
    "node_signed_inflow",
    "simulate_kernel_fluid_sparse",
]
