"""Topological biosignature computations (P4 framework).

Implements the three biosignature diagnostics of Das (2026, P4):

  1. Topological biosignature  Δβ₁ = β₁^obs - β₁^abio  (Definition 1)
  2. Detection threshold        R_min ~ (kmin^abio + Δβ₁)·b / I_self  (Proposition 1)
  3. Cross-kernel factorization ‖k_cross‖_HS  (Proposition 2)
  4. Plume spectral entropy     bandpass spike at metabolic scale  (Proposition 3)

The module also provides a combined BiosignatureReport that aggregates
all diagnostics and flags anomalies against user-supplied null models.

Physical interpretation
-----------------------
  Δβ₁ > 0  → excess topological loops beyond abiotic expectation
             → consistent with an optimal controller building structure
             → NECESSARY but not sufficient condition for life

  ‖k_cross‖_HS >> 0  → chemistry and hydrology are coupled beyond
                        abiotic physics (weathering, dissolution)
                        → controller is coupling both domains
                        → additional evidence for biology

  Plume entropy drop + bandpass spike → chemistry is organised at a
                                        specific scale (metabolic network)
                                        → not consistent with equilibrium
                                        abiotic chemistry
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. Topological biosignature Δβ₁
# ---------------------------------------------------------------------------

@dataclass
class TopologicalBiosignature:
    """Result of the topological biosignature computation."""
    beta1_obs:   int     # observed β₁
    beta1_abio:  int     # abiotic null model β₁
    delta_beta1: int     # Δβ₁ = β₁_obs - β₁_abio
    beta2_obs:   int = 0
    beta2_abio:  int = 0
    delta_beta2: int = 0
    note:        str = ""

    @property
    def is_anomalous(self) -> bool:
        return self.delta_beta1 > 0 or self.delta_beta2 > 0

    @property
    def kmin_abio(self) -> int:
        return 1 + self.beta1_abio   # β₀ = 1 assumed connected terrain

    @property
    def kmin_obs(self) -> int:
        return 1 + self.beta1_obs


def topological_biosignature(
    beta1_obs:  int,
    beta1_abio: int,
    beta2_obs:  int = 0,
    beta2_abio: int = 0,
    note:       str = "",
) -> TopologicalBiosignature:
    """Compute the topological biosignature Δβ₁ = β₁^obs - β₁^abio.

    Parameters
    ----------
    beta1_obs   : observed number of independent cycles in the terrain graph
    beta1_abio  : predicted β₁ from known abiotic processes (impacts, channels)
    beta2_obs   : observed number of enclosed voids
    beta2_abio  : predicted β₂ from abiotic processes
    note        : descriptive label

    Returns
    -------
    TopologicalBiosignature
    """
    return TopologicalBiosignature(
        beta1_obs=beta1_obs,
        beta1_abio=beta1_abio,
        delta_beta1=beta1_obs - beta1_abio,
        beta2_obs=beta2_obs,
        beta2_abio=beta2_abio,
        delta_beta2=beta2_obs - beta2_abio,
        note=note,
    )


# ---------------------------------------------------------------------------
# 2. Detection threshold (Proposition 1, P4)
# ---------------------------------------------------------------------------

def detection_threshold(
    beta1_abio:     int,
    delta_beta1:    int,
    bits_per_coeff: float = 32.0,
    I_self_bps:     float | None = None,
) -> dict[str, float]:
    """Minimum observation requirements to detect Δβ₁ from orbit.

    From Proposition 1 of P4:
        k_obs ≥ kmin^abio + Δβ₁
        R_min ~ (kmin^abio + Δβ₁) · b / I_self

    Parameters
    ----------
    beta1_abio     : abiotic β₁ (defines kmin^abio = 1 + beta1_abio)
    delta_beta1    : biologically produced excess β₁ to be detected
    bits_per_coeff : bits per spectral coefficient (default 32 = float32)
    I_self_bps     : scene self-information rate in bits/s (from Landauer bound);
                     if None, R_min is returned in units of (total bits needed)

    Returns
    -------
    dict with:
      'k_min_abio'    minimum modes for abiotic topology
      'k_required'    minimum modes needed to detect Δβ₁
      'total_bits'    total bits required = k_required * bits_per_coeff
      'R_min'         minimum observability ratio (if I_self_bps provided)
      'detectable'    whether Δβ₁ ≥ 1 (i.e., there is a biosignature to detect)
    """
    kmin_abio   = 1 + beta1_abio
    k_required  = kmin_abio + max(0, delta_beta1)
    total_bits  = k_required * bits_per_coeff

    R_min = None
    if I_self_bps is not None and I_self_bps > 0:
        R_min = total_bits / I_self_bps

    return {
        "k_min_abio":   kmin_abio,
        "k_required":   k_required,
        "total_bits":   total_bits,
        "R_min":        R_min,
        "detectable":   delta_beta1 >= 1,
    }


# ---------------------------------------------------------------------------
# 3. Cross-kernel factorization test (Proposition 2, P4)
# ---------------------------------------------------------------------------

def cross_kernel(
    K_coupled: np.ndarray,
    K_A:       np.ndarray,
    K_B:       np.ndarray,
) -> np.ndarray:
    """Compute the cross-kernel k_cross = k_coupled - k_A ⊗ k_B.

    For two domains A (e.g., surface chemistry, nA nodes) and B (hydrology,
    nB nodes) on a product graph G_A × G_B, the Kronecker product kernel
    k_A ⊗ k_B encodes independent domain evolution.  Any residual
    k_cross = k_coupled - k_A ⊗ k_B signals coupling between the domains.

    Parameters
    ----------
    K_coupled : (nA*nB, nA*nB) joint kernel matrix on G_A × G_B
    K_A       : (nA, nA) marginal kernel on G_A
    K_B       : (nB, nB) marginal kernel on G_B

    Returns
    -------
    k_cross : (nA*nB, nA*nB) cross-kernel (non-factorizable remainder)
    """
    K_A = np.asarray(K_A, dtype=float)
    K_B = np.asarray(K_B, dtype=float)
    K_factorized = np.kron(K_A, K_B)     # Kronecker product = factorized joint
    return K_coupled - K_factorized


def cross_kernel_norm(
    K_coupled: np.ndarray,
    K_A:       np.ndarray,
    K_B:       np.ndarray,
) -> float:
    """Hilbert-Schmidt norm of the cross-kernel ‖k_cross‖_HS.

    ‖k_cross‖_HS = sqrt(tr(k_cross ᵀ k_cross)) = Frobenius norm of k_cross.

    Values near 0: domains evolve independently (no biological coupling).
    Large values: coupling agent present.
    """
    k_cross = cross_kernel(K_coupled, K_A, K_B)
    return float(np.linalg.norm(k_cross, "fro"))


def factorization_test(
    K_coupled: np.ndarray,
    K_A:       np.ndarray,
    K_B:       np.ndarray,
    significance_threshold: float = 0.05,
) -> dict[str, float | bool | np.ndarray]:
    """Test whether k_coupled factorizes as k_A ⊗ k_B.

    Uses the relative cross-kernel norm as the test statistic:
        r = ‖k_cross‖_HS / ‖k_coupled‖_HS

    r ≈ 0  → factorizable (abiotic, independent domains)
    r >> 0 → non-factorizable (coupled, potential biology)

    Parameters
    ----------
    K_coupled               : joint kernel matrix
    K_A, K_B               : marginal kernels
    significance_threshold  : r threshold above which coupling is flagged

    Returns
    -------
    dict with:
      'k_cross_norm'     ‖k_cross‖_HS
      'k_coupled_norm'   ‖k_coupled‖_HS
      'relative_norm'    r = k_cross_norm / k_coupled_norm
      'is_coupled'       bool — True if r > threshold
      'k_cross'          the cross-kernel matrix
    """
    k_cross = cross_kernel(K_coupled, K_A, K_B)
    hs_cross   = float(np.linalg.norm(k_cross, "fro"))
    hs_coupled = float(np.linalg.norm(np.asarray(K_coupled, dtype=float), "fro"))
    rel        = hs_cross / (hs_coupled + 1e-12)

    return {
        "k_cross_norm":   hs_cross,
        "k_coupled_norm": hs_coupled,
        "relative_norm":  rel,
        "is_coupled":     rel > significance_threshold,
        "k_cross":        k_cross,
    }


def spectral_kernel_from_laplacian(
    L: np.ndarray,
    tau: float = 1.0,
) -> np.ndarray:
    """Heat kernel K_h = exp(-τ L) on a graph Laplacian.

    Parameters
    ----------
    L   : (N, N) graph Laplacian
    tau : diffusion time (scale parameter)

    Returns
    -------
    (N, N) kernel matrix
    """
    L = np.asarray(L, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(L)
    h = np.exp(-tau * np.maximum(eigvals, 0.0))
    return (eigvecs * h) @ eigvecs.T


# ---------------------------------------------------------------------------
# 4. Plume spectral entropy biosignature (Proposition 3, P4)
# ---------------------------------------------------------------------------

def chemical_affinity_graph(
    species:      list[str],
    co_occurrence: np.ndarray,
    threshold:    float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a chemical-affinity graph from a co-occurrence / reaction matrix.

    Nodes = molecular species detected in plume.
    Edge weights = co-occurrence frequency or reaction-network proximity.

    Parameters
    ----------
    species       : list of N species names
    co_occurrence : (N, N) symmetric float matrix — co-occurrence or affinity
    threshold     : minimum edge weight to include

    Returns
    -------
    (W, L)  adjacency matrix W and Laplacian L, both (N, N)
    """
    W = np.asarray(co_occurrence, dtype=float)
    W = (W + W.T) / 2.0                  # symmetrize
    np.fill_diagonal(W, 0.0)
    W[W < threshold] = 0.0
    D = np.diag(W.sum(axis=1))
    L = D - W
    return W, L


def plume_spectral_entropy(
    L_chem:     np.ndarray,
    L_abio:     np.ndarray | None = None,
    tau:        float = 1.0,
    n_modes:    int | None = None,
) -> dict[str, float | np.ndarray | bool]:
    """Compute the spectral entropy biosignature for plume chemistry.

    Under abiotic (equilibrium) chemistry, h(λ) decays monotonically
    → high spectral entropy, no preferred chemical scale.

    A biologically maintained metabolism produces a bandpass spike at λ*
    corresponding to the metabolic network's characteristic scale.

    Biosignature criteria (Proposition 3, P4):
        h(λ*) / h_abio(λ*) >> 1   AND   H(ht) < H(h_abio)

    Parameters
    ----------
    L_chem   : (N, N) Laplacian of observed chemical-affinity graph
    L_abio   : (N, N) Laplacian of abiotic reference graph (if None:
               uses a uniform complete graph as abiotic reference)
    tau      : heat kernel diffusion time
    n_modes  : restrict to first n_modes eigenvalues (default: all)

    Returns
    -------
    dict with:
      'H_obs'            spectral entropy of observed plume chemistry
      'H_abio'           spectral entropy of abiotic reference
      'entropy_drop'     H_abio - H_obs  (positive = organized)
      'bandpass_spike'   max(h_obs / h_abio) across modes
      'spike_eigenvalue' λ at which the spike occurs
      'is_biosignature'  bool — True if entropy_drop > 0 and bandpass_spike > 2
      'h_obs'            observed spectral weights
      'h_abio'           abiotic spectral weights
      'eigenvalues'      eigenvalue vector
    """
    L = np.asarray(L_chem, dtype=float)
    n = L.shape[0]
    k = n if n_modes is None else min(n_modes, n)

    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals = np.maximum(eigvals[:k], 0.0)

    # Observed spectral weights (heat kernel)
    h_obs = np.exp(-tau * eigvals)

    # Abiotic reference
    if L_abio is None:
        # Uniform complete graph: all eigenvalues = n/(n-1) except λ₀ = 0
        lam_abio = np.zeros(k)
        if k > 1:
            lam_abio[1:] = float(n) / float(n - 1) if n > 1 else 1.0
        h_abio = np.exp(-tau * lam_abio)
    else:
        eigvals_a = np.maximum(np.linalg.eigvalsh(np.asarray(L_abio, dtype=float))[:k], 0.0)
        h_abio    = np.exp(-tau * eigvals_a)

    # Spectral entropy: H = -Σ h̄_l log h̄_l
    def _entropy(h: np.ndarray) -> float:
        pos = h[h > 1e-12]
        h_bar = pos / pos.sum()
        return float(-np.sum(h_bar * np.log(h_bar)))

    H_obs  = _entropy(h_obs)
    H_abio = _entropy(h_abio)
    entropy_drop = H_abio - H_obs

    # Bandpass spike: ratio h_obs / h_abio at each mode
    ratio = h_obs / (h_abio + 1e-12)
    spike_idx = int(np.argmax(ratio))
    bandpass_spike = float(ratio[spike_idx])
    spike_eigenvalue = float(eigvals[spike_idx])

    is_bio = (entropy_drop > 0) and (bandpass_spike > 2.0)

    return {
        "H_obs":             H_obs,
        "H_abio":            H_abio,
        "entropy_drop":      entropy_drop,
        "bandpass_spike":    bandpass_spike,
        "spike_eigenvalue":  spike_eigenvalue,
        "is_biosignature":   is_bio,
        "h_obs":             h_obs,
        "h_abio":            h_abio,
        "eigenvalues":       eigvals,
    }


# ---------------------------------------------------------------------------
# Combined biosignature report
# ---------------------------------------------------------------------------

@dataclass
class BiosignatureReport:
    """Full P4 biosignature assessment for a planetary terrain patch."""
    target:        str
    topological:   TopologicalBiosignature | None = None
    cross_kernel:  dict | None = None
    plume_entropy: dict | None = None
    detection:     dict | None = None
    notes:         list[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        """Count of diagnostic criteria flagged (0–3)."""
        s = 0
        if self.topological and self.topological.is_anomalous:
            s += 1
        if self.cross_kernel and self.cross_kernel.get("is_coupled", False):
            s += 1
        if self.plume_entropy and self.plume_entropy.get("is_biosignature", False):
            s += 1
        return s

    def summary(self) -> str:
        lines = [f"=== Biosignature Report: {self.target} ==="]
        if self.topological:
            tb = self.topological
            lines.append(f"  Δβ₁ = {tb.delta_beta1}  (obs={tb.beta1_obs}, abio={tb.beta1_abio})")
            lines.append(f"  Topological anomaly: {tb.is_anomalous}")
        if self.cross_kernel:
            ck = self.cross_kernel
            lines.append(f"  Cross-kernel ‖k_cross‖_HS = {ck['k_cross_norm']:.4f}")
            lines.append(f"  Relative norm r = {ck['relative_norm']:.4f}  coupled: {ck['is_coupled']}")
        if self.plume_entropy:
            pe = self.plume_entropy
            lines.append(f"  Entropy drop = {pe['entropy_drop']:.4f}")
            lines.append(f"  Bandpass spike = {pe['bandpass_spike']:.2f}× at λ={pe['spike_eigenvalue']:.4f}")
            lines.append(f"  Plume biosignature: {pe['is_biosignature']}")
        lines.append(f"  Total diagnostic score: {self.score}/3")
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)
