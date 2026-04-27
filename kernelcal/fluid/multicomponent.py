"""kernelcal.fluid.multicomponent -- multi-component sparse-Laplacian
fluid solver.

PR-A.2 of CR-2026-04-26.  Lifts the single-component sparse solver
(:mod:`kernelcal.fluid.sparse`) to ``C + 1`` components: ``C`` real
categories that each flow with their own per-edge oriented velocity,
plus a passive ``unknown`` channel that does not flow but absorbs
the per-node simplex residual (the "Goedel-slot" of FN 102).

State shapes
------------

* ``rho`` of shape ``(C, n)`` -- per-category density at each node.
* ``rho_unknown`` of shape ``(n,)`` -- unknown-channel density.  Does
  not have its own per-edge velocity; evolves only through the
  per-node simplex projection.
* ``u`` of shape ``(C, E)`` -- per-category oriented edge velocity in
  canonical-edge order (positive sign means "flow from i to j" for
  ``e = (i, j), i < j``).
* ``phi`` of shape ``(C, n)`` -- per-category potential at each node.

Invariant
---------

For all ``t`` and all nodes ``n``,
``sum_c rho_c(t, n) + rho_unknown(t, n) == 1``.

The per-step continuity update preserves per-category mass to
machine epsilon (the same identity used in :mod:`kernelcal.fluid.sparse`:
``sum(D.T @ F_c) = (D @ 1).T @ F_c = 0`` for any category-specific
flux profile ``F_c``).  The simplex projection at the end of every
step reasserts the per-node sum-to-1 constraint by scaling all
components uniformly per node; the drift it introduces is logged as
``simplex_projection_drift`` so PR-B's runtime ledger has a real
signal to balance against.

Operator algebra (broadcast over the C-axis)
--------------------------------------------

Let ``D`` be the signed sparse incidence matrix of the underlying
:class:`SparseFluidGraph` (shape ``(E, n)``).  scipy.sparse supports
``dense @ sparse`` via ``__rmatmul__``, so:

* per-edge gradient    ``grad_p = (p @ D.T) / edge_lengths``  -- ``(C, E)``
* per-node signed inflow per category    ``psi = u @ D``  -- ``(C, n)``
* per-node mean signed inflow      ``nu = psi / degree``       -- ``(C, n)``
* per-edge Laplacian smoothing of u  ``u_lap = nu @ D.T - 2*u`` -- ``(C, E)``
* per-edge flux per category  ``F = 0.5 * (rho[:, i_e] + rho[:, j_e]) * u``
* per-node continuity per category  ``drho = F @ D``  -- ``(C, n)``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .dynamics import (
    FluidGraph,
    PotentialLandscape,
)
from .sparse import (
    SparseFluidGraph,
)


# ---------------------------------------------------------------------------\
# Config + result dataclasses
# ---------------------------------------------------------------------------\


@dataclass(frozen=True)
class MultiComponentFluidConfig:
    """Numerical and physical parameters for the multi-component
    discrete simulation.

    Most fields are per-category copies of
    :class:`FluidSimulationConfig`; categories share the same dt,
    pressure coefficient, drag, and viscosity.  Per-category
    landscape comes in via ``MultiComponentLandscape``.

    ``simplex_floor`` is the lower clip applied to ``rho`` and
    ``rho_unknown`` before the per-node KL projection.  Negative
    entries (from numerical drift in the continuity step) are
    floored, and the floor magnitude is logged.

    ``track_simplex_violation`` records the pre-projection per-node
    sum residual ``s(n) - 1`` so PR-B's ledger can balance against
    it.  Defaults to ``True`` because PR-B requires the signal; the
    cost is one O(n) reduction per step.
    """

    dt: float = 0.01
    steps: int = 1000
    c_s2: float = 0.5
    beta: float = 0.8
    nu: float = 0.15
    simplex_floor: float = 1e-12
    track_simplex_violation: bool = True


@dataclass(frozen=True)
class MultiComponentLandscape:
    """Bundle of per-category :class:`PotentialLandscape` instances.

    Stored in canonical category order matching the ``rho`` rows.
    """

    landscapes: Tuple[PotentialLandscape, ...]

    @property
    def num_categories(self) -> int:
        return len(self.landscapes)

    def to_phi_array(
        self,
        lambda_L: float,
        lambda_C: float,
        lambda_I: float,
    ) -> np.ndarray:
        """Compose per-category phi into a single ``(C, n)`` array."""
        rows = [
            lambda_L * lc.loss + lambda_C * lc.cost - lambda_I * lc.info
            for lc in self.landscapes
        ]
        return np.stack(rows, axis=0)

    def validate(self, num_nodes: int) -> None:
        for c, lc in enumerate(self.landscapes):
            try:
                lc.validate(num_nodes)
            except ValueError as e:
                raise ValueError(
                    f"landscape {c} failed validation: {e}"
                ) from e


@dataclass(frozen=True)
class MultiComponentResult:
    """Time-series outputs and terminal state.

    Shapes follow the underlying state: per-category arrays carry a
    leading ``C`` axis, history arrays prepend a time axis.

    Attributes
    ----------
    rho_history
        ``(T+1, C, n)`` per-category density, including initial
        condition.  ``T = config.steps``.
    rho_unknown_history
        ``(T+1, n)`` unknown-channel density, including initial
        condition.
    mass_per_component
        ``(T+1, C)`` per-category total mass at each time, conserved
        to within ``simplex_projection_drift`` magnitude.
    mass_unknown
        ``(T+1,)`` total unknown-channel mass at each time.
    simplex_projection_drift
        ``(T,)`` per-step magnitude of the simplex-projection
        correction:  ``sum_n |s(n) - 1|`` *before* the renormalise.
        For a well-posed run this stays at floating-point noise.
    floor_mass_inserted
        ``(T, C)`` per-category implicit mass injected by the
        ``simplex_floor`` clip on per-category densities.
    floor_mass_inserted_unknown
        ``(T,)`` mass injected by the ``simplex_floor`` clip on the
        unknown channel.
    projection_mass_transfer
        ``(T, C)`` per-category mass change induced by the per-node
        simplex projection (``sum_n rho_c_after_projection -
        sum_n rho_c_after_floor``).  PR-B's runtime ledger consumes
        this as the **simplex_projection** event signal.  Closure
        identity: ``mass_per_component[t+1] -
        mass_per_component[t] = floor_mass_inserted[t] +
        projection_mass_transfer[t]`` to floating-point.
    projection_mass_transfer_unknown
        ``(T,)`` unknown-channel counterpart to
        ``projection_mass_transfer``.
    rho_final
        ``(C, n)`` final per-category density.
    rho_unknown_final
        ``(n,)`` final unknown-channel density.
    u_final
        ``(C, E)`` final per-category, per-edge velocity.
    """

    rho_history: np.ndarray
    rho_unknown_history: np.ndarray
    mass_per_component: np.ndarray
    mass_unknown: np.ndarray
    simplex_projection_drift: np.ndarray
    floor_mass_inserted: np.ndarray
    floor_mass_inserted_unknown: np.ndarray
    projection_mass_transfer: np.ndarray
    projection_mass_transfer_unknown: np.ndarray
    rho_final: np.ndarray
    rho_unknown_final: np.ndarray
    u_final: np.ndarray


# ---------------------------------------------------------------------------\
# Solver
# ---------------------------------------------------------------------------\


def simulate_multicomponent_fluid(
    graph: FluidGraph,
    landscape: MultiComponentLandscape,
    config: MultiComponentFluidConfig,
    rho0: np.ndarray,
    rho0_unknown: np.ndarray,
    *,
    lambda_L: float = 1.0,
    lambda_C: float = 1.2,
    lambda_I: float = 0.9,
    track_history: bool = True,
    sparse_graph: Optional[SparseFluidGraph] = None,
) -> MultiComponentResult:
    """Multi-component sparse-Laplacian fluid simulation.

    Parameters
    ----------
    graph
        Underlying :class:`FluidGraph` (single, shared across
        categories).
    landscape
        :class:`MultiComponentLandscape` with one :class:`PotentialLandscape`
        per category in canonical order.
    config
        :class:`MultiComponentFluidConfig`.
    rho0
        ``(C, n)`` initial per-category density.  Must be
        nonnegative and respect ``sum_c rho0[c, n] + rho0_unknown[n]
        == 1`` per node within ``1e-9``.
    rho0_unknown
        ``(n,)`` initial unknown-channel density.
    lambda_L, lambda_C, lambda_I
        Phi composition coefficients applied uniformly across
        categories.  Per-category landscape encodes the per-category
        differences via its ``loss`` / ``cost`` / ``info`` channels.
    track_history
        Whether to retain the full ``rho_history`` /
        ``rho_unknown_history`` trajectories.  When ``False``, only
        the initial-condition row is kept.
    sparse_graph
        Optional pre-built :class:`SparseFluidGraph` (for callers
        that already paid the incidence-build cost via
        :func:`kernelcal.urban.adapter.to_fluid_graph`).

    Returns
    -------
    MultiComponentResult
    """
    sg = sparse_graph if sparse_graph is not None else SparseFluidGraph.from_fluid_graph(graph)
    n, E = sg.num_nodes, sg.num_edges
    C = landscape.num_categories
    landscape.validate(n)

    rho = np.asarray(rho0, dtype=float).copy()
    if rho.shape != (C, n):
        raise ValueError(
            f"rho0 must have shape (C, n) = ({C}, {n}); got {rho.shape}"
        )
    if np.any(rho < 0):
        raise ValueError("rho0 must be nonnegative")

    rho_unknown = np.asarray(rho0_unknown, dtype=float).copy()
    if rho_unknown.shape != (n,):
        raise ValueError(
            f"rho0_unknown must have shape (n,) = ({n},); got {rho_unknown.shape}"
        )
    if np.any(rho_unknown < 0):
        raise ValueError("rho0_unknown must be nonnegative")

    initial_sum = rho.sum(axis=0) + rho_unknown
    if not np.allclose(initial_sum, 1.0, atol=1e-9, rtol=0.0):
        raise ValueError(
            "Initial state violates simplex constraint: "
            f"max |sum_c rho_c(n) + rho_unknown(n) - 1| = "
            f"{float(np.max(np.abs(initial_sum - 1.0))):.3e}"
        )

    u = np.zeros((C, E), dtype=float)

    T = int(config.steps)
    if track_history:
        rho_hist = np.zeros((T + 1, C, n), dtype=float)
        ru_hist = np.zeros((T + 1, n), dtype=float)
        rho_hist[0] = rho
        ru_hist[0] = rho_unknown
    else:
        rho_hist = np.zeros((1, C, n), dtype=float)
        ru_hist = np.zeros((1, n), dtype=float)
        rho_hist[0] = rho
        ru_hist[0] = rho_unknown

    mass_per_component = np.zeros((T + 1, C), dtype=float)
    mass_unknown = np.zeros(T + 1, dtype=float)
    mass_per_component[0] = rho.sum(axis=1)
    mass_unknown[0] = float(rho_unknown.sum())

    simplex_drift = np.zeros(T, dtype=float)
    floor_inserted = np.zeros((T, C), dtype=float)
    floor_inserted_unknown = np.zeros(T, dtype=float)
    projection_transfer = np.zeros((T, C), dtype=float)
    projection_transfer_unknown = np.zeros(T, dtype=float)

    phi = landscape.to_phi_array(lambda_L, lambda_C, lambda_I)
    edges_idx = sg.edges_idx
    edge_lengths = sg.edge_lengths
    degree = sg.degree

    for t in range(T):
        p = config.c_s2 * rho

        grad_p = _per_edge_gradient_multi(sg, p, edge_lengths)
        grad_phi = _per_edge_gradient_multi(sg, phi, edge_lengths)
        u_lap = _per_edge_laplacian_multi(sg, u, degree)

        du = -grad_p - grad_phi - config.beta * u + config.nu * u_lap
        u = u + config.dt * du

        rho_at_i = rho[:, edges_idx[:, 0]]
        rho_at_j = rho[:, edges_idx[:, 1]]
        F = 0.5 * (rho_at_i + rho_at_j) * u

        drho = _per_node_continuity_multi(sg, F)
        rho = rho + config.dt * drho

        # Floor per category and on unknown channel; record injected mass.
        below = np.maximum(config.simplex_floor - rho, 0.0)
        floor_inserted[t] = below.sum(axis=1)
        rho = np.maximum(rho, config.simplex_floor)

        below_u = np.maximum(config.simplex_floor - rho_unknown, 0.0)
        floor_inserted_unknown[t] = float(below_u.sum())
        rho_unknown = np.maximum(rho_unknown, config.simplex_floor)

        # Per-node simplex projection: scale (rho_c, rho_unknown) by
        # 1/s(n) where s(n) = sum_c rho_c(n) + rho_unknown(n).  This
        # is the KL/Bregman projection onto the simplex once
        # positivity has been enforced by the floor step above.
        # Total mass per component is *not* preserved by this step:
        # if s(n) > 1 at nodes where category c is concentrated,
        # category c loses mass.  The per-category mass transfer is
        # logged as the simplex_projection ledger event.
        s = rho.sum(axis=0) + rho_unknown
        if config.track_simplex_violation:
            simplex_drift[t] = float(np.sum(np.abs(s - 1.0)))
        m_c_before = rho.sum(axis=1)
        m_u_before = float(rho_unknown.sum())
        rho = rho / s[None, :]
        rho_unknown = rho_unknown / s
        projection_transfer[t] = rho.sum(axis=1) - m_c_before
        projection_transfer_unknown[t] = float(rho_unknown.sum()) - m_u_before

        if track_history:
            rho_hist[t + 1] = rho
            ru_hist[t + 1] = rho_unknown
        mass_per_component[t + 1] = rho.sum(axis=1)
        mass_unknown[t + 1] = float(rho_unknown.sum())

    return MultiComponentResult(
        rho_history=rho_hist,
        rho_unknown_history=ru_hist,
        mass_per_component=mass_per_component,
        mass_unknown=mass_unknown,
        simplex_projection_drift=simplex_drift,
        floor_mass_inserted=floor_inserted,
        floor_mass_inserted_unknown=floor_inserted_unknown,
        projection_mass_transfer=projection_transfer,
        projection_mass_transfer_unknown=projection_transfer_unknown,
        rho_final=rho.copy(),
        rho_unknown_final=rho_unknown.copy(),
        u_final=u.copy(),
    )


# ---------------------------------------------------------------------------\
# Broadcast operators (kept private; tests invoke them via the solver)
# ---------------------------------------------------------------------------\


def _per_edge_gradient_multi(
    sg: SparseFluidGraph, scalar: np.ndarray, edge_lengths: np.ndarray
) -> np.ndarray:
    """``(C, n) -> (C, E)``: per-edge gradient ``(p[j] - p[i]) / ell``
    broadcast over the leading ``C`` axis."""
    if scalar.ndim != 2 or scalar.shape[1] != sg.num_nodes:
        raise ValueError(
            f"scalar must have shape (C, n); got {scalar.shape}"
        )
    grad_E_C = sg.incidence @ scalar.T  # (E, C)
    return grad_E_C.T / edge_lengths


def _per_edge_laplacian_multi(
    sg: SparseFluidGraph, u_edge: np.ndarray, degree: np.ndarray
) -> np.ndarray:
    """``(C, E) -> (C, E)``: per-edge Laplacian smoothing broadcast
    over the C-axis.  Same algebra as
    :func:`kernelcal.fluid.sparse.edge_laplacian_smoothing`,
    vectorised."""
    if u_edge.ndim != 2 or u_edge.shape[1] != sg.num_edges:
        raise ValueError(
            f"u_edge must have shape (C, E); got {u_edge.shape}"
        )
    psi = (sg.incidence_T @ u_edge.T).T  # (C, n)
    nu = psi / degree
    grad_E_C = sg.incidence @ nu.T  # (E, C)
    return grad_E_C.T - 2.0 * u_edge


def _per_node_continuity_multi(
    sg: SparseFluidGraph, F_edge: np.ndarray
) -> np.ndarray:
    """``(C, E) -> (C, n)``: per-node ``drho = D.T @ F`` broadcast
    over the C-axis."""
    if F_edge.ndim != 2 or F_edge.shape[1] != sg.num_edges:
        raise ValueError(
            f"F_edge must have shape (C, E); got {F_edge.shape}"
        )
    drho_n_C = sg.incidence_T @ F_edge.T  # (n, C)
    return drho_n_C.T


# ---------------------------------------------------------------------------\
# Convenience: simplex-feasible initial conditions
# ---------------------------------------------------------------------------\


def make_concentrated_initial_state(
    num_nodes: int,
    num_categories: int,
    *,
    category_centers: Optional[Sequence[int]] = None,
    sigma: float = 1.5,
    mass_unknown_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a simplex-feasible initial state ``(rho, rho_unknown)``.

    Each category gets a Gaussian bump on the ring around its
    designated center node; the unknown channel fills in to make
    ``sum_c rho_c(n) + rho_unknown(n) == 1`` per node.

    The total mass per category is set so that the *uniform fixed
    point* (the limit of pure diffusion with zero potential) is
    ``rho_c = (1 - mass_unknown_fraction) / num_categories`` per
    node, and ``rho_unknown = mass_unknown_fraction`` per node.
    This is the configuration used by acceptance criterion **A2-extra**.

    Parameters
    ----------
    num_nodes, num_categories
        Self-explanatory.
    category_centers
        One node index per category.  Defaults to evenly spaced
        across the ring.
    sigma
        Width of the per-category Gaussian bump.
    mass_unknown_fraction
        Fraction of the per-node mass reserved for the unknown
        channel at the uniform fixed point.  Must be in ``[0, 1]``.

    Returns
    -------
    rho, rho_unknown
        Shapes ``(C, n)`` and ``(n,)``.  Per-node sum is exactly 1
        within floating-point noise.
    """
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2")
    if num_categories < 1:
        raise ValueError("num_categories must be >= 1")
    if not (0.0 <= mass_unknown_fraction <= 1.0):
        raise ValueError("mass_unknown_fraction must be in [0, 1]")

    if category_centers is None:
        step = max(1, num_nodes // num_categories)
        category_centers = [(c * step) % num_nodes for c in range(num_categories)]
    if len(category_centers) != num_categories:
        raise ValueError(
            "category_centers must have length num_categories"
        )

    M_c = (1.0 - mass_unknown_fraction) * float(num_nodes) / float(num_categories)

    rho = np.zeros((num_categories, num_nodes), dtype=float)
    for c, ctr in enumerate(category_centers):
        x = np.array(
            [_ring_distance(num_nodes, i, int(ctr)) for i in range(num_nodes)],
            dtype=float,
        )
        bump = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
        bump *= M_c / float(np.sum(bump))
        rho[c] = bump

    per_node_sum = rho.sum(axis=0)
    rho_unknown = 1.0 - per_node_sum
    # If concentrated bumps overshoot 1.0 at a node, scale the
    # categories down on that node and zero the unknown there.
    over = rho_unknown < 0.0
    if np.any(over):
        scale = 1.0 / per_node_sum[over]
        rho[:, over] *= scale[None, :]
        rho_unknown[over] = 0.0

    return rho, rho_unknown


def _ring_distance(num_nodes: int, i: int, j: int) -> int:
    """Shortest ring distance.  Local copy to avoid pulling
    :mod:`kernelcal.fluid.dynamics` into the multi-component
    namespace."""
    d = abs(i - j)
    return min(d, num_nodes - d)


__all__ = [
    "MultiComponentFluidConfig",
    "MultiComponentLandscape",
    "MultiComponentResult",
    "make_concentrated_initial_state",
    "simulate_multicomponent_fluid",
]
