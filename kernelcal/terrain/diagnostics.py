"""Spectral kernel diagnostics for terrain graphs.

Implements the core spectral diagnostics of P1, P2, and P4:

  - Spectral entropy H[h*]             (P1 Remark 7)
  - Fiedler-mode gap Δ'(h*)            (P1 Corollary 3)
  - Fixed-point kernel h*              (P1 Corollary 1)
  - Phase-transition sweep             (P1 Experiment 6)
  - Stability–conservation tradeoff    (P2 Proposition 1b)
  - Observability ratio R/İself        (P2 Table 2)
  - Bandwidth-optimal mode selection   (P2 Algorithm 1)

All functions accept a graph Laplacian L (dense numpy array) and return
computable scalar diagnostics suitable for real-time monitoring on rover or
drone platforms.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Spectral entropy
# ---------------------------------------------------------------------------

def spectral_entropy(h: np.ndarray) -> float:
    """Spectral entropy H[h] = -Σ h̄_l log h̄_l.

    h̄_l = h_l / Σ_l' h_l' is the normalised spectral weight.
    H[h] ∈ [0, log N]:
      - H = log N : maximally diffuse kernel (flat terrain, vacuum solution)
      - H → 0    : kernel mass concentrated at a single mode (phase transition)

    Parameters
    ----------
    h : (N,) float array — spectral transfer function values (positive)

    Returns
    -------
    float — spectral entropy in nats
    """
    h = np.asarray(h, dtype=float)
    pos = h[h > 0]
    if len(pos) == 0:
        return 0.0
    h_bar = pos / pos.sum()
    return float(-np.sum(h_bar * np.log(h_bar)))


def spectral_entropy_from_laplacian(
    L:     np.ndarray,
    tau:   float = 1.0,
    k:     int | None = None,
) -> float:
    """Spectral entropy of the heat kernel h(λ) = exp(-τλ) on graph Laplacian L.

    Parameters
    ----------
    L   : (N, N) graph Laplacian
    tau : heat diffusion time (scale)
    k   : restrict to first k eigenvalues (None = all)

    Returns
    -------
    float — spectral entropy
    """
    eigvals = np.maximum(np.linalg.eigvalsh(np.asarray(L, dtype=float)), 0.0)
    if k is not None:
        eigvals = eigvals[:k]
    h = np.exp(-tau * eigvals)
    return spectral_entropy(h)


# ---------------------------------------------------------------------------
# Fixed-point kernel (MaxCal Corollary 1 of P1)
# ---------------------------------------------------------------------------

def fixed_point_kernel(
    L:      np.ndarray,
    h0:     np.ndarray | None = None,
    mu2:    float = 2.0,
    sigma2: float = 1.0,
    w:      np.ndarray | None = None,
    n_iter: int = 200,
    tol:    float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """Compute the MaxCal fixed-point spectral kernel h* on graph Laplacian L.

    Implements the Gaussian MI source (P1 Eq. 15):
        T_l[h] = μ₂ w_l / (2(σ² + h_l))

    Fixed-point iteration (P1 Corollary 1):
        h^{(n+1)} = h0 * exp(-1 - T(h^{(n)}))

    Parameters
    ----------
    L      : (N, N) graph Laplacian
    h0     : (N,) reference spectral transfer function
             (default: heat kernel h0_l = exp(-λ_l))
    mu2    : information weight Lagrange multiplier
    sigma2 : observation noise variance
    w      : (N,) per-mode weights (default: eigenvalue-aware w_l = λ_l)
    n_iter : maximum iterations
    tol    : convergence tolerance ‖h^{n+1} - h^n‖∞

    Returns
    -------
    h_star : (N,) converged fixed-point spectral weights
    info   : dict with 'converged', 'n_iter', 'residual', 'contraction_ratio'
    """
    L = np.asarray(L, dtype=float)
    eigvals = np.maximum(np.linalg.eigvalsh(L), 0.0)
    n = len(eigvals)

    if h0 is None:
        h0 = np.exp(-eigvals)
        h0 = np.maximum(h0, 1e-10)
    else:
        h0 = np.asarray(h0, dtype=float)

    if w is None:
        w = eigvals.copy()
        w[0] = w[1] if n > 1 else 1.0   # avoid w_0 = 0

    h = h0.copy()
    rho = 0.0
    for it in range(n_iter):
        T = mu2 * w / (2.0 * (sigma2 + h))
        h_new = h0 * np.exp(-1.0 - T)
        diff = np.max(np.abs(h_new - h))
        if diff > 0 and h.max() > 0:
            rho = float(np.max(np.abs(h_new - h) / (np.abs(h) + 1e-12)))
        h = h_new
        if diff < tol:
            residual = float(np.max(np.abs((-np.log(h / h0) - 1.0) - T)))
            return h, {"converged": True, "n_iter": it + 1,
                       "residual": residual, "contraction_ratio": rho}

    residual = float(np.max(np.abs((-np.log(h / h0) - 1.0) - T)))
    return h, {"converged": False, "n_iter": n_iter,
               "residual": residual, "contraction_ratio": rho}


# ---------------------------------------------------------------------------
# Fiedler-mode gap Δ'(h*) and stability margin
# ---------------------------------------------------------------------------

def fiedler_mode_gap(
    h_star: np.ndarray,
    L:      np.ndarray,
    mu2:    float = 2.0,
    sigma2: float = 1.0,
    w:      np.ndarray | None = None,
) -> float:
    """Hessian gap Δ'(h*) = min_l (-H_ll) at the fixed point.

    From P1 Corollary 3 (mode-separable case):
        H_ll = -1/h*_l + μ₂ w_l / (2(σ²+h*_l)²)
        Δ' = min_l (-H_ll) = min_l (1/h*_l - μ₂ w_l / (2(σ²+h*_l)²))

    Higher Δ' = more stable fixed point = larger conservation law deficit
    (P2 Proposition 1b: D_m = -Δ').

    Parameters
    ----------
    h_star : (N,) fixed-point spectral weights
    L      : (N, N) graph Laplacian (to get eigenvalues for w)
    mu2, sigma2, w : source parameters (same as fixed_point_kernel)

    Returns
    -------
    float — Hessian gap Δ' > 0 means stable, = 0 means marginal
    """
    h = np.asarray(h_star, dtype=float)
    if w is None:
        eigvals = np.maximum(np.linalg.eigvalsh(np.asarray(L, dtype=float)), 0.0)
        w = eigvals.copy()
        if len(w) > 1:
            w[0] = w[1]
        else:
            w[0] = 1.0

    dT_dh = -mu2 * w / (2.0 * (sigma2 + h)**2)
    H_diag = -1.0 / h - dT_dh   # H_ll = -1/h_l* - ∂T_l/∂h_l|h*
    return float(np.min(-H_diag))


def stability_conservation_tradeoff(
    h_star: np.ndarray,
    L:      np.ndarray,
    mu2:    float = 2.0,
    sigma2: float = 1.0,
    w:      np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    """Compute the stability–conservation tradeoff (P2 Proposition 1b).

    D_m = Σ_l ∂(R_l - T_l)/∂h_m |_{h*} = H_mm = -Δ'_m

    Returns the deficit D_m per mode and the Hessian diagonal.
    D_m < 0 for all stable fixed points; |D_m| = Δ' is the conservation deficit.

    Returns
    -------
    dict with 'D_m', 'H_diag', 'Delta_prime', 'conservation_deficit'
    """
    h = np.asarray(h_star, dtype=float)
    if w is None:
        eigvals = np.maximum(np.linalg.eigvalsh(np.asarray(L, dtype=float)), 0.0)
        w = eigvals.copy()
        if len(w) > 1:
            w[0] = w[1]

    dR_dh = -1.0 / h
    dT_dh = -mu2 * w / (2.0 * (sigma2 + h)**2)
    D_m   = dR_dh - dT_dh          # conservation identity residual per mode
    H_diag = -1.0 / h - dT_dh      # Hessian diagonal

    return {
        "D_m":                   D_m,
        "H_diag":                H_diag,
        "Delta_prime":           float(np.min(-H_diag)),
        "conservation_deficit":  float(np.mean(np.abs(D_m))),
        "conservation_holds":    bool(np.allclose(D_m, 0, atol=1e-6)),
    }


# ---------------------------------------------------------------------------
# Phase-transition sweep
# ---------------------------------------------------------------------------

@dataclass
class PhaseSweepResult:
    """Result of a spectral entropy phase-transition sweep."""
    perturbation_values:  np.ndarray
    fiedler_values:       np.ndarray
    spectral_entropies:   np.ndarray
    hessian_gaps:         np.ndarray
    phase_transition_idx: int | None   # index of minimum Fiedler / entropy drop

    @property
    def phase_transition_value(self) -> float | None:
        if self.phase_transition_idx is None:
            return None
        return float(self.perturbation_values[self.phase_transition_idx])


def phase_transition_sweep(
    L_base:    np.ndarray,
    perturb_edge: tuple[int, int],
    weights:   np.ndarray | None = None,
    n_steps:   int = 20,
    mu2:       float = 2.0,
    sigma2:    float = 1.0,
) -> PhaseSweepResult:
    """Sweep edge weight from intact to removed and track diagnostics.

    Replicates P1 Experiment 6: weaken a single edge (i,j) from weight=1
    to weight≈0 and track H[h*], Δ'(h*), λ₁.

    Akin to monitoring a terrain graph as a geological boundary is approached
    (rover nearing a crater rim, channel approaching avulsion threshold).

    Parameters
    ----------
    L_base       : (N, N) baseline graph Laplacian
    perturb_edge : (i, j) node pair whose edge is weakened
    weights      : sweep of edge weight values (default: 1.0 → 0.02, 20 steps)
    n_steps      : number of steps if weights is None
    mu2, sigma2  : source parameters

    Returns
    -------
    PhaseSweepResult
    """
    L = np.asarray(L_base, dtype=float).copy()
    n = L.shape[0]
    i, j = int(perturb_edge[0]), int(perturb_edge[1])

    if weights is None:
        weights = np.linspace(1.0, 0.02, n_steps)
    weights = np.asarray(weights, dtype=float)

    fiedlers  = np.zeros(len(weights))
    entropies = np.zeros(len(weights))
    gaps      = np.zeros(len(weights))

    # Current edge weight
    w_current = float(L_base[i, j]) if L_base[i, j] > 0 else 1.0

    for k, eps in enumerate(weights):
        # Perturb the edge: reduce weight from w_current to eps.
        # Off-diagonal L[i,j] = -w → changes by -(eps - w_current) = w_current - eps
        # Diagonal L[i,i] = Σw  → changes by  (eps - w_current)
        L_k = L_base.copy()
        dw = eps - w_current          # change in weight (negative: weakening)
        L_k[i, j] += dw              # off-diag: -w → -(w + dw) = old + dw ... wait
        L_k[j, i] += dw
        # Correct sign: L[i,j] = -w, new L[i,j] = -eps = L[i,j] - dw
        # Revert and apply correctly:
        L_k[i, j] = L_base[i, j] - dw
        L_k[j, i] = L_base[j, i] - dw
        L_k[i, i] = L_base[i, i] + dw
        L_k[j, j] = L_base[j, j] + dw

        eigvals = np.maximum(np.linalg.eigvalsh(L_k), 0.0)
        fiedlers[k] = eigvals[1] if len(eigvals) > 1 else 0.0

        h_star, _ = fixed_point_kernel(L_k, mu2=mu2, sigma2=sigma2)
        entropies[k] = spectral_entropy(h_star)
        gaps[k]      = fiedler_mode_gap(h_star, L_k, mu2=mu2, sigma2=sigma2)

    # Detect phase transition: largest drop in Fiedler value
    d_fiedler = np.diff(fiedlers)
    if len(d_fiedler) > 0:
        pt_idx = int(np.argmin(d_fiedler)) + 1
    else:
        pt_idx = None

    return PhaseSweepResult(
        perturbation_values=weights,
        fiedler_values=fiedlers,
        spectral_entropies=entropies,
        hessian_gaps=gaps,
        phase_transition_idx=pt_idx,
    )


# ---------------------------------------------------------------------------
# Observability ratio (P2 Table 2)
# ---------------------------------------------------------------------------

def observability_ratio(
    R_bps:     float,
    P_phys_W:  float,
    T_K:       float = 300.0,
) -> dict[str, float]:
    """Compute R/İself — the fraction of scene information capturable by observation.

    From P2 Eq. (16) (Landauer–Shannon argument):
        İself = P_phys / (kB T ln 2)   [bits/s]
        R/İself = R_bps / İself

    Parameters
    ----------
    R_bps     : observational data rate [bits/s]
    P_phys_W  : physical power dissipated by the scene [W]
    T_K       : effective temperature [K]

    Returns
    -------
    dict with 'I_self_bps', 'R_over_I_self', 'log10_ratio', 'regime'
    """
    kB = 1.380649e-23  # J/K
    ln2 = np.log(2)
    I_self = P_phys_W / (kB * T_K * ln2)
    ratio  = R_bps / I_self
    log10r = float(np.log10(ratio)) if ratio > 0 else -np.inf

    if ratio == np.inf or P_phys_W == 0:
        regime = "static_topology"
    elif log10r > -15:
        regime = "swaplimited_dynamic"
    else:
        regime = "fundamentally_inaccessible"

    return {
        "I_self_bps":      I_self,
        "R_over_I_self":   ratio,
        "log10_ratio":     log10r,
        "regime":          regime,
    }


# ---------------------------------------------------------------------------
# Bandwidth-optimal mode selection (P2 Algorithm 1 / Eq. 13)
# ---------------------------------------------------------------------------

def bandwidth_optimal_modes(
    h_star:   np.ndarray,
    c:        np.ndarray,
    T_l:      np.ndarray,
    kmin:     int,
    k_budget: int,
) -> np.ndarray:
    """Select the k_budget most informative spectral modes beyond kmin.

    From P2 Eq. (13) and Algorithm 1:
        l* = argmax_{l ≥ kmin}  h*(λ_l) |c_l|² / T_l[h*]

    The first kmin modes are topologically obligate and always included.
    Remaining budget k_budget - kmin is allocated greedily.

    Parameters
    ----------
    h_star   : (N,) fixed-point spectral weights
    c        : (N,) spectral coefficients of the current scene signal
    T_l      : (N,) source functional values T_l[h*]
    kmin     : topologically obligate mode count (β₀ + β₁)
    k_budget : total mode budget (kmin ≤ k_budget ≤ N)

    Returns
    -------
    (k_budget,) int array of selected mode indices (always includes 0..kmin-1)
    """
    n = len(h_star)
    k_budget = min(k_budget, n)
    kmin     = min(kmin, k_budget)

    # Obligate modes
    selected = list(range(kmin))
    if k_budget <= kmin:
        return np.array(selected, dtype=int)

    # Information return per bit for remaining modes
    remaining = list(range(kmin, n))
    T_safe = np.maximum(np.asarray(T_l, dtype=float), 1e-12)
    score = (np.asarray(h_star, dtype=float) * np.asarray(c, dtype=float)**2) / T_safe

    remaining_sorted = sorted(remaining, key=lambda l: score[l], reverse=True)
    selected += remaining_sorted[: k_budget - kmin]
    return np.array(sorted(selected), dtype=int)
