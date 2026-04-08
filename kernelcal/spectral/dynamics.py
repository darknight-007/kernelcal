"""
Spectral kernel dynamics: the MaxCal field equation on finite graphs.

Maps directly to Section 3 of the companion paper.  Given a SpectralGraph
and a source functional (e.g. GaussianMISource), this module implements:

  Proposition 1  — geometric functional  ℛ_l[h_t]
  Corollary 1    — fixed-point iteration and contraction check
  Corollary 2    — vacuum solution and log-linear geodesics
  Corollary 3    — full Hessian, Hessian gap, per-mode stability
  Remark 4       — heat-kernel critical-point verification
  Remark 8       — spectral entropy H[h_t]
  Q6             — Hessian gap Δ(h_t) and per-mode coupling entropy

All methods are pure functions of numpy arrays; no mutable state is kept
here.  The SpectralKernelDynamics class is a thin coordinator that holds a
SpectralGraph and a source functional and delegates to the module-level
functions below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


# ===========================================================================
# Module-level pure functions (can be used independently of the class)
# ===========================================================================

def geometric_functional(h: np.ndarray, h0: np.ndarray) -> np.ndarray:
    """ℛ_l[h] = −log(h(λ_l)/h₀(λ_l)) − 1.  (Proposition 1, Eq. 4)

    Parameters
    ----------
    h  : (N,) current spectral weights.
    h0 : (N,) reference spectral weights.

    Returns
    -------
    (N,) array of geometric functional values.
    """
    h = np.asarray(h, dtype=float)
    h0 = np.asarray(h0, dtype=float)
    return -np.log(h / h0) - 1.0


def field_equation_residual(
    h: np.ndarray,
    h0: np.ndarray,
    T_values: np.ndarray,
) -> np.ndarray:
    """ℛ_l[h] − 𝒯_l[h] per mode.  Zero at a self-consistent kernel. """
    return geometric_functional(h, h0) - T_values


def vacuum_solution(h0: np.ndarray) -> np.ndarray:
    """Source-free fixed point: h*(λ_l) = h₀(λ_l) · e⁻¹.  (Corollary 2)"""
    return np.asarray(h0, dtype=float) * np.exp(-1.0)


def geodesic(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Log-linear geodesic: h_t(λ_l) = exp(a_l + b_l · t).  (Corollary 2)

    Parameters
    ----------
    a : (N,) intercept vector.
    b : (N,) slope vector (b_l = −λ_l for the heat-kernel geodesic).
    t : scalar affine parameter.
    """
    return np.exp(np.asarray(a, dtype=float) + np.asarray(b, dtype=float) * t)


def spectral_entropy(h: np.ndarray) -> float:
    """Normalised spectral entropy H[h_t] = −∑_l h̄_l log h̄_l.  (Remark 8 / Eq. 14)

    h̄_l = h(λ_l) / Σ_{l'} h(λ_{l'}) is the normalised spectral weight.
    Entropy is maximal when all modes have equal weight and decreases as
    spectral mass concentrates, providing an early-warning signal for
    network fragmentation (λ_1 → 0).
    """
    h = np.asarray(h, dtype=float)
    h_bar = h / h.sum()
    h_bar = np.where(h_bar > 0, h_bar, 1.0)  # avoid log(0)
    return float(-np.sum(h_bar * np.log(h_bar)))


def hessian_matrix(
    h_star: np.ndarray,
    source_jacobian: np.ndarray,
) -> np.ndarray:
    """Full N×N Hessian of 𝒥 at critical point h*.  (Corollary 3)

    H_{lm} = −δ_{lm}/h*(λ_l) − ∂𝒯_l/∂h(λ_m)|_{h*}

    Parameters
    ----------
    h_star          : (N,) critical-point spectral weights.
    source_jacobian : (N, N) matrix ∂𝒯_l/∂h(λ_m) evaluated at h*.

    Returns
    -------
    (N, N) Hessian matrix H.
    """
    h_star = np.asarray(h_star, dtype=float)
    J = np.asarray(source_jacobian, dtype=float)
    return -np.diag(1.0 / h_star) - J


def hessian_gap(H: np.ndarray) -> float:
    """Δ(h*) = λ_min(−H): smallest eigenvalue of the negative Hessian.  (Q6)

    Δ > 0: strict local maximum, stable.
    Δ → 0: approach to a fold bifurcation (precursor to phase transition).
    Δ < 0: H has a positive eigenvalue; system has crossed a saddle.
    """
    eigvals = np.linalg.eigvalsh(-H)
    return float(eigvals.min())


def fiedler_gap(H: np.ndarray, eigenvalues: np.ndarray, tol: float = 1e-8) -> float:
    """Δ'(h*) = min_{l: λ_l > tol} diag(−H)_l — the Fiedler-mode gap.  (Q6)

    The zero-eigenvalue mode (λ_0 = 0) is always in the kernel of the
    Laplacian; its contribution to the Hessian is pinned to −1/h*_0 = −e
    regardless of graph topology.  Excluding it exposes the stability margin
    of the Fiedler mode (l=1), which contracts toward zero as the graph
    approaches disconnection and is the genuine early-warning signal for
    phase transitions described in Remark 8 and Q6.

    Parameters
    ----------
    H          : (N, N) Hessian evaluated at h*.
    eigenvalues: (N,) Laplacian eigenvalues λ_0 ≤ λ_1 ≤ … (from SpectralGraph).
    tol        : eigenvalues below this threshold are treated as zero.

    Returns
    -------
    Smallest diagonal entry of −H over all non-zero modes.  Returns the
    full hessian_gap if all eigenvalues are below tol (degenerate graph).
    """
    mask = np.asarray(eigenvalues) > tol
    if not np.any(mask):
        return hessian_gap(H)
    diag_neg_H = -np.diag(H)
    return float(diag_neg_H[mask].min())


def coupling_entropy(source_jacobian: np.ndarray) -> float:
    """Scalar coupling entropy 𝒮_coup.  (Q6)

    For each mode l, normalises the off-diagonal row |∂𝒯_l/∂h(λ_m)| (m≠l)
    to a probability distribution p_{lm} and computes its Shannon entropy.
    Returns the row-average coupling entropy.

    A decrease in 𝒮_coup signals the Fiedler mode concentrating its
    coupling partners — an early precursor to Δ → 0 and then H → 0.
    In the mode-separable (diagonal) case 𝒮_coup = log(N−1) (maximum),
    consistent with no preferential coupling.
    """
    J = np.asarray(source_jacobian, dtype=float)
    N = J.shape[0]
    if N < 2:
        return 0.0
    entropies = []
    for l in range(N):
        row = np.abs(J[l].copy())
        row[l] = 0.0          # exclude diagonal
        s = row.sum()
        if s < 1e-15:
            entropies.append(np.log(N - 1))  # uniform coupling = max entropy
        else:
            p = row / s
            p = np.where(p > 0, p, 1.0)
            entropies.append(-np.sum(p * np.log(p)))
    return float(np.mean(entropies))


def contraction_bound(
    h: np.ndarray,
    h0: np.ndarray,
    T_values: np.ndarray,
    dT_dh_diag: np.ndarray,
) -> float:
    """Sufficient contraction bound for uniqueness.  (Corollary 1)

    Returns  max_l F_l(h) · |∂𝒯_l/∂h(λ_l)|
    where F_l(h) = h₀(λ_l) exp(−1 − 𝒯_l[h]).  Value < 1 guarantees
    F is a local contraction in the ℓ∞ norm.
    """
    h0 = np.asarray(h0, dtype=float)
    T = np.asarray(T_values, dtype=float)
    dT = np.asarray(dT_dh_diag, dtype=float)
    F = h0 * np.exp(-1.0 - T)
    return float(np.max(F * np.abs(dT)))


# ===========================================================================
# Result dataclasses
# ===========================================================================

@dataclass
class FixedPointResult:
    """Outcome of a fixed-point iteration run."""
    h_star: np.ndarray                    # converged spectral weights
    iterations: int                       # steps taken
    converged: bool                       # True if tol was met
    history: List[np.ndarray]             # h at each iteration
    residual_history: List[float]         # ‖ℛ−𝒯‖ at each iteration
    contraction_value: float              # contraction bound at h*
    stable: bool                          # all per-mode stability margins > 0


@dataclass
class StabilityResult:
    """Full stability analysis at a critical point h*."""
    H: np.ndarray                         # (N,N) Hessian
    eigenvalues_H: np.ndarray             # eigenvalues of H (all < 0 ⟺ stable)
    gap: float                            # λ_min(−H)  — includes zero mode
    fiedler_gap: float                    # min_{l: λ_l>0} diag(−H)_l — excludes zero mode
    stable: bool                          # H ≺ 0
    per_mode_margin: np.ndarray           # ∂𝒯_l/∂h_l − (−1/h*_l)  (> 0 ⟺ stable)
    coupling_entropy_value: float         # 𝒮_coup (diagnostic for Q6)


# ===========================================================================
# Main coordinator class
# ===========================================================================

class SpectralKernelDynamics:
    """MaxCal spectral kernel dynamics on a finite graph.

    Coordinates a SpectralGraph and a source functional to implement every
    numerical claim in Section 3 of the paper.

    Parameters
    ----------
    graph : SpectralGraph
        The finite connected graph with Laplacian eigendecomposition.
    source : object with methods .T(h) and .jacobian(h)
        Source functional 𝒯_l[h].  Any object implementing the interface
        of GaussianMISource will work.
    h0 : (N,) array or None
        Reference spectral weights h₀.  Defaults to flat weights (all 1).
    """

    def __init__(self, graph, source, h0: Optional[np.ndarray] = None) -> None:
        self.graph = graph
        self.source = source
        self.h0 = (
            np.asarray(h0, dtype=float)
            if h0 is not None
            else graph.flat_weights()
        )
        if self.h0.shape != (graph.N,):
            raise ValueError(f"h0 must have shape ({graph.N},).")

    # ------------------------------------------------------------------
    # Proposition 1
    # ------------------------------------------------------------------

    def R(self, h: np.ndarray) -> np.ndarray:
        """Geometric functional ℛ_l[h] = −log(h/h₀) − 1.  (Proposition 1)"""
        return geometric_functional(h, self.h0)

    def R_finite_difference(self, h: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """Finite-difference estimate of ℛ_l, for numerical verification."""
        h = np.asarray(h, dtype=float)
        R_fd = np.zeros(self.graph.N)
        for l in range(self.graph.N):
            h_p = h.copy(); h_p[l] += delta
            h_m = h.copy(); h_m[l] -= delta

            def S(hh):
                return -np.sum(hh * np.log(hh / self.h0))

            R_fd[l] = (S(h_p) - S(h_m)) / (2 * delta)
        return R_fd

    # ------------------------------------------------------------------
    # Corollary 1: fixed-point iteration
    # ------------------------------------------------------------------

    def fixed_point_iteration(
        self,
        max_iter: int = 500,
        tol: float = 1e-10,
        h_init: Optional[np.ndarray] = None,
    ) -> FixedPointResult:
        """Iterate h^{n+1}_l = h₀_l · exp(−1 − 𝒯_l[h^n]) until convergence.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations.
        tol : float
            Convergence threshold on ‖ℛ[h] − 𝒯[h]‖_∞.
        h_init : (N,) array or None
            Starting point (defaults to h₀).

        Returns
        -------
        FixedPointResult
        """
        h = (h_init.copy() if h_init is not None else self.h0.copy())
        history = [h.copy()]
        residuals = []

        for i in range(max_iter):
            T_vals = self.source.T(h)
            res = np.max(np.abs(self.R(h) - T_vals))
            residuals.append(float(res))
            if res < tol:
                h_star = h
                # stability at convergence
                J = self.source.jacobian(h_star)
                H = hessian_matrix(h_star, J)
                stab = bool(np.all(np.linalg.eigvalsh(H) < 0))
                cb = contraction_bound(h_star, self.h0, T_vals, np.diag(J))
                return FixedPointResult(
                    h_star=h_star.copy(),
                    iterations=i,
                    converged=True,
                    history=history,
                    residual_history=residuals,
                    contraction_value=cb,
                    stable=stab,
                )
            h = self.h0 * np.exp(-1.0 - T_vals)
            history.append(h.copy())

        T_vals = self.source.T(h)
        J = self.source.jacobian(h)
        H = hessian_matrix(h, J)
        stab = bool(np.all(np.linalg.eigvalsh(H) < 0))
        cb = contraction_bound(h, self.h0, T_vals, np.diag(J))
        return FixedPointResult(
            h_star=h.copy(),
            iterations=max_iter,
            converged=False,
            history=history,
            residual_history=residuals,
            contraction_value=cb,
            stable=stab,
        )

    # ------------------------------------------------------------------
    # Corollary 2: vacuum and geodesics
    # ------------------------------------------------------------------

    def vacuum(self) -> np.ndarray:
        """Source-free solution h*(λ_l) = h₀(λ_l) · e⁻¹.  (Corollary 2)"""
        return vacuum_solution(self.h0)

    def geodesic_path(
        self, a: np.ndarray, b: np.ndarray, t_vals: np.ndarray
    ) -> np.ndarray:
        """Evaluate the geodesic h_t(λ_l) = exp(a_l + b_l t) at t_vals.

        Returns
        -------
        (len(t_vals), N) array of spectral weights along the geodesic.
        """
        t_vals = np.asarray(t_vals, dtype=float)
        return np.stack([geodesic(a, b, t) for t in t_vals])

    def heat_kernel_geodesic(self, tau_vals: np.ndarray) -> np.ndarray:
        """Heat-kernel family h_τ(λ_l) = exp(−λ_l τ) as a geodesic.

        a_l = 0,  b_l = −λ_l,  τ as affine parameter.  (Corollary 2)

        Returns
        -------
        (len(tau_vals), N) array.
        """
        lam = self.graph.eigenvalues
        return self.geodesic_path(
            a=np.zeros(self.graph.N),
            b=-lam,
            t_vals=tau_vals,
        )

    # ------------------------------------------------------------------
    # Remark 4: heat-kernel critical-point check
    # ------------------------------------------------------------------

    def heat_kernel_critical_point_check(
        self, tau: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Verify ℛ_l[h*_τ] = 𝒯_l[h*_τ] for the heat kernel.  (Remark 4)

        Returns
        -------
        R_vals  : (N,) geometric functional at h*_τ.
        T_vals  : (N,) source functional at h*_τ.
        residual: (N,) R_vals − T_vals  (should be near zero if source is
                  linear in eigenvalues with slope τ).
        """
        h_tau = self.graph.heat_kernel_weights(tau)
        R_vals = self.R(h_tau)
        T_vals = self.source.T(h_tau)
        return R_vals, T_vals, R_vals - T_vals

    # ------------------------------------------------------------------
    # Corollary 3 / Q6: stability analysis
    # ------------------------------------------------------------------

    def stability_analysis(self, h_star: np.ndarray) -> StabilityResult:
        """Full stability analysis at a candidate critical point h*.

        Computes: Hessian, its eigenvalues, Hessian gap Δ, per-mode
        stability margins, and coupling entropy 𝒮_coup.  (Corollary 3, Q6)
        """
        h_star = np.asarray(h_star, dtype=float)
        J = self.source.jacobian(h_star)
        H = hessian_matrix(h_star, J)
        eigvals = np.linalg.eigvalsh(H)
        gap = hessian_gap(H)
        f_gap = fiedler_gap(H, self.graph.eigenvalues)
        stable = bool(np.all(eigvals < 0))
        margin = self.source.dT_dh(h_star) - (-1.0 / h_star)
        s_coup = coupling_entropy(J)
        return StabilityResult(
            H=H,
            eigenvalues_H=eigvals,
            gap=gap,
            fiedler_gap=f_gap,
            stable=stable,
            per_mode_margin=margin,
            coupling_entropy_value=s_coup,
        )

    # ------------------------------------------------------------------
    # Remark 8 / Q6: early-warning diagnostics along a parameter sweep
    # ------------------------------------------------------------------

    def phase_transition_sweep(
        self,
        graph_sequence,
        max_iter: int = 500,
        tol: float = 1e-10,
        source_factory: Optional[Callable] = None,
    ) -> dict:
        """Compute early-warning diagnostics across a sequence of graphs.

        For each graph in graph_sequence (ordered by decreasing algebraic
        connectivity), runs the fixed-point iteration and records:

          - λ_1         : Fiedler value
          - H_entropy   : spectral entropy H[h*]  (Remark 8)
          - delta_gap   : Hessian gap Δ(h*)        (Q6)
          - s_coup      : coupling entropy          (Q6)

        Parameters
        ----------
        graph_sequence : iterable of SpectralGraph
            Graphs ordered by the perturbation parameter (e.g. ε ↘ 0).
        source_factory : callable or None
            Optional factory `source_factory(graph) -> source` used to build a
            graph-specific source at each sweep step. This is useful when the
            source itself should depend on spectral structure (e.g. w_l = λ_l).
            If None, reuses `self.source` for all graphs.

        Returns
        -------
        dict with keys 'fiedler', 'H_entropy', 'delta_gap', 'coup_entropy',
        each a list aligned with graph_sequence.
        """
        from .dynamics import SpectralKernelDynamics  # local import to avoid circular

        results: dict = {
            "fiedler": [],
            "H_entropy": [],
            "delta_gap": [],
            "coup_entropy": [],
        }
        for g in graph_sequence:
            src = source_factory(g) if source_factory is not None else self.source
            dyn = SpectralKernelDynamics(g, src, h0=None)
            fp = dyn.fixed_point_iteration(max_iter=max_iter, tol=tol)
            stab = dyn.stability_analysis(fp.h_star)
            results["fiedler"].append(g.fiedler_value)
            results["H_entropy"].append(spectral_entropy(fp.h_star))
            results["delta_gap"].append(stab.fiedler_gap)
            results["coup_entropy"].append(stab.coupling_entropy_value)
        return results

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SpectralKernelDynamics("
            f"graph={self.graph!r}, "
            f"source={self.source!r})"
        )
