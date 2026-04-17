"""
Route 3: Direct Algebraic Test of ∇_K T_k = 0
===============================================
Tests whether the fixed-point condition R_l(h*) = T_l(h*) automatically
implies the conservation identity

    D_m ≡ Σ_l  ∂(R_l − T_l)/∂h_m  |_{h*}  =  0   for all m

If D_m = 0 for all m: the conservation law is automatic from the field equation
(GR-like: geometry forces matter conservation).
If D_m ≠ 0: the distinction dynamics equation dc/dt = G[c,h_t] is a genuinely
required additional ingredient — not derivable from R_l = T_l alone.

Setup: P8 path graph, Gaussian MI source, σ²=1, μ₂=2, w_l=1.
Same parameters as P1 Experiments 1–4.

References:
  P1 §3, Corollary 1 (fixed-point kernels), Corollary 3 (Hessian stability)
  P2 §3.1, Proposition 1 (Jacobian divergence), §8.1 (toward ∇_K T_k = 0)
  Field note 26 (three-paper assessment, highest-priority open problem)
"""

import numpy as np

# ─── P8 graph Laplacian ───────────────────────────────────────────────────────
N = 8
L = np.diag([1.0] + [2.0] * (N - 2) + [1.0]) - np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)
eigvals, _ = np.linalg.eigh(L)

# ─── Gaussian MI source (mode-separable, P1 eq. 15) ──────────────────────────
sigma2 = 1.0
mu2    = 2.0
w      = np.ones(N)
h0     = np.ones(N)

def T(h):
    """Source functional T_l[h] = μ₂ w_l / (2(σ²+h_l))."""
    return mu2 * w / (2.0 * (sigma2 + h))

def R(h):
    """Geometric functional R_l[h] = −log(h_l/h0_l) − 1."""
    return -np.log(h / h0) - 1.0

def dR_dh(h):
    """∂R_l/∂h_m|_h  →  diagonal vector (mode-separable)."""
    return -1.0 / h                       # dR_l/dh_l = -1/h_l, off-diag = 0

def dT_dh(h):
    """∂T_l/∂h_m|_h  →  diagonal vector (mode-separable Gaussian MI)."""
    return -mu2 * w / (2.0 * (sigma2 + h)**2)   # dT_l/dh_l, off-diag = 0

# ─── Fixed-point iteration (P1 Corollary 1) ──────────────────────────────────
h = h0.copy()
for it in range(500):
    h_new = h0 * np.exp(-1.0 - T(h))
    if np.max(np.abs(h_new - h)) < 1e-14:
        break
    h = h_new
h_star = h
converged_iter = it + 1

field_residual = np.max(np.abs(R(h_star) - T(h_star)))

# ─── Route 3 computation ─────────────────────────────────────────────────────
# In mode-separable case the N×N matrix ∂(R_l−T_l)/∂h_m is diagonal.
# Its diagonal entries are exactly the Hessian diagonal H_mm (P1 Corollary 3).
# Column sum = H_mm (since off-diagonal entries vanish).
# Conservation law ⟺ H_mm = 0 for all m.

dR = dR_dh(h_star)           # ∂R_l/∂h_l  (diagonal)
dT = dT_dh(h_star)           # ∂T_l/∂h_l  (diagonal)
D  = dR - dT                 # H_mm = D_m = column sum of ∂(R−T)/∂h_m

# ─── Cross-check against Hessian from P1 Corollary 3 ─────────────────────────
# H_lm = −δ_lm/h_l* − ∂T_l/∂h_m  →  H_mm = −1/h_m* + μ₂w_m/(2(σ²+h_m*)²)
H_diag = -1.0/h_star - dT_dh(h_star)

# ─── Vacuum case: μ₂ = 0, h_vac* = h0·e^{−1} ─────────────────────────────────
h_vac     = h0 * np.exp(-1.0)
D_vac     = -1.0/h_vac           # T=0 → dT/dh=0 → D_m = dR/dh = −1/h_vac

# ─── What source slope is required for conservation? ─────────────────────────
# Need dT_m/dh_m = dR_m/dh_m = −1/h_m*
# For Gaussian MI: −μ₂w_m/(2(σ²+h_m*)²) = −1/h_m*
# ⟺ μ₂ w_m h_m* = 2(σ²+h_m*)²  ⟺  2h² + (4σ²−μ₂w)h + 2σ⁴ = 0
a_q = 2.0
b_q = 4.0*sigma2 - mu2*w[0]
c_q = 2.0*sigma2**2
discriminant = b_q**2 - 4.0*a_q*c_q

# ─── Connection to P2 Proposition 1 (tr DF) ──────────────────────────────────
# P2 Prop 1: tr(DF)|_{h*} = Σ_l μ₂w_l h_l* / (2(σ²+h_l*)²)
tr_DF = np.sum(mu2 * w * h_star / (2.0*(sigma2 + h_star)**2))

# ─── Contraction ratio from P1 Exp 2 ─────────────────────────────────────────
rho = np.max(np.abs(dT_dh(h_star) * h_star))   # |∂F_l/∂h_l| = h_l* |∂T_l/∂h_l|

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

sep = "=" * 62

print(sep)
print("ROUTE 3 — Direct Algebraic Test of ∇_K T_k = 0")
print(sep)

print(f"\n[Setup]")
print(f"  Graph: P{N},  N = {N} modes")
print(f"  Source: Gaussian MI, σ² = {sigma2}, μ₂ = {mu2}, w_l = 1")
print(f"  Fixed-point converged: {converged_iter} iterations")
print(f"  h* = {h_star[0]:.6f}  (all modes equal: {np.allclose(h_star, h_star[0])})")
print(f"  Field equation residual ‖R[h*]−T[h*]‖∞ = {field_residual:.2e}")

print(f"\n[Analytical decomposition  D_m = ∂R_m/∂h_m − ∂T_m/∂h_m  at h*]")
print(f"  ∂R_m/∂h_m  = −1/h_m*               = {dR[0]:+.6f}")
print(f"  ∂T_m/∂h_m  = −μ₂w/(2(σ²+h*)²)    = {dT[0]:+.6f}")
print(f"  D_m                                 = {D[0]:+.6f}")

print(f"\n[Route 3 result: is D_m = 0 for all m?]")
print(f"  D vector: {np.round(D, 6)}")
print(f"  ‖D‖∞  =  {np.max(np.abs(D)):.6f}")
print(f"  Conservation law holds automatically?  {np.allclose(D, 0, atol=1e-8)}")

print(f"\n[Cross-check: D_m = H_mm (Hessian diagonal, P1 Exp 4)]")
print(f"  H_mm from Corollary 3:  {H_diag[0]:.6f}")
print(f"  D_m from Route 3:       {D[0]:.6f}")
print(f"  Match:  {np.allclose(D, H_diag, atol=1e-12)}")
print(f"  Hessian gap Δ' = −H_mm = {-D[0]:.6f}  (stability margin from Exp 4)")

print(f"\n[Required source slope for conservation law]")
print(f"  Need: ∂T_m/∂h_m|_h* = ∂R_m/∂h_m = −1/h_m* = {dR[0]:.6f}")
print(f"  Have: ∂T_m/∂h_m|_h*              =          {dT[0]:.6f}")
print(f"  Gap:  ∂T_m/∂h_m − ∂R_m/∂h_m     =          {dT[0]-dR[0]:+.6f}")
print(f"  Gaussian MI satisfies conservation law ⟺ 2h²+(4σ²−μ₂w)h+2σ⁴=0")
print(f"  Discriminant = {discriminant:.1f}  → real solution exists? {discriminant >= 0}")

print(f"\n[Vacuum check  (μ₂ = 0, h_vac* = h0·e⁻¹ = {h_vac[0]:.4f})]")
print(f"  D_m in vacuum = −1/h_vac* = {D_vac[0]:.6f}")
print(f"  Conservation law holds in vacuum?  {np.allclose(D_vac, 0, atol=1e-8)}")
print(f"  (Failure is structural — not caused by the source T_l)")

print(f"\n[Connection to P2 Proposition 1]")
print(f"  tr(DF)|_h*  = Σ_l μ₂w_l h_l*/(2(σ²+h_l*)²) = {tr_DF:.6f}")
print(f"  Contraction ratio ρ = max|∂F_l/∂h_l|        = {rho:.6f}")
print(f"  Note: tr(DF) > 0  (volume-expanding, P2 Prop 1)")
print(f"        D_m < 0      (conservation law violated)")
print(f"        Both → 0 as μ₂ → 0, but vacuum D_m = −e ≠ 0 (see above)")

print(f"\n{sep}")
print("CONCLUSION")
print(sep)
print(f"""
The conservation identity  Σ_l ∂(R_l−T_l)/∂h_m |_h* = 0
FAILS for the Gaussian MI source on P8.

  Violation per mode:  D_m = {D[0]:.4f}  (identical across all N={N} modes)
  This equals the Hessian diagonal H_mm = {H_diag[0]:.4f} from P1 Exp 4.

The failure is structural, not source-specific:
  • It persists in the vacuum (T_l = 0): D_m = −e ≈ {D_vac[0]:.4f} ≠ 0
  • The Gaussian MI source CANNOT satisfy the conservation constraint
    for any real h* (discriminant = {discriminant:.1f} < 0, no real root)
  • The "divergence" D_m equals exactly the stability margin Δ' = {-D[0]:.4f}:
    more stable fixed points have larger conservation-law violations

The conservation law ∇_K T_k = 0 is NOT automatic from R_l(h*) = T_l(h*).
The distinction dynamics equation dc/dt = G[c,h_t] (or some analog of the
Bianchi identity) is a genuinely required additional ingredient.

→ Route 1 (Q8): determine whether the JOINT (h,c) flow is symplectic even
  though the h-only flow has tr(DF) ≠ 0. The analog of the Bianchi identity
  — if it exists — must come from the combined system geometry, not from
  the field equation alone.
""")
