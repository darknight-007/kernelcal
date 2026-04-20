"""
Six-experiment verification suite for spectral kernel dynamics.

Each experiment makes one Section 3 claim of the companion paper
numerically visible on a path graph P_N with the Gaussian
mutual-information source (σ²=1, μ₂=1 by default).

    Exp 1  — Proposition 1:  ℛ_l formula vs finite-difference
    Exp 2  — Corollary 1:   fixed-point iteration and contraction
    Exp 3  — Corollary 2:   vacuum solution and log-linear geodesics
    Exp 4  — Corollary 3:   per-mode stability margins and Hessian
    Exp 5  — Remark 4:      heat kernel as a critical point
    Exp 6  — Remark 8 / Q6: spectral entropy and Hessian gap as
                             early-warning signals for fragmentation
    Exp 6b — Q6 supplemental: explicit coupling source sweep
    Exp 7  — Cross-topology robustness (river vs trunk-roots)

Usage
-----
    from kernelcal.spectral.experiments import run_all_experiments
    run_all_experiments(output_dir="figures/spectral")

    # or run from the command line:
    python -m kernelcal.spectral.experiments
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .graph import SpectralGraph
from .source import GaussianMISource, CoupledGaussianMISource
from .dynamics import (
    SpectralKernelDynamics,
    geometric_functional,
    spectral_entropy,
    geodesic,
)

# ---------------------------------------------------------------------------
# Shared style helpers are re-exported from :mod:`experiments_plots` so this
# module can share them with any future per-experiment plotting split.
# ---------------------------------------------------------------------------
from .experiments_plots import (  # noqa: E402, F401
    _draw_topology,
    _mode_colors,
    _mode_labels,
    _save,
    _save_topology_schematics,
)


def _spectral_adjacency_coupling_matrix(graph: SpectralGraph) -> np.ndarray:
    """Build a graph-dependent off-diagonal coupling matrix in spectral basis."""
    L = graph.laplacian
    d = np.diag(L)
    A = np.diag(d) - L  # weighted adjacency
    Phi = graph.eigenvectors
    C = np.abs(Phi.T @ A @ Phi)
    np.fill_diagonal(C, 0.0)
    row_sums = C.sum(axis=1, keepdims=True)
    C = np.divide(C, row_sums, out=np.zeros_like(C), where=row_sums > 1e-15)
    return C


def _laplacian_from_weighted_edges(n: int, edges: list[tuple[int, int, float]]) -> np.ndarray:
    """Build combinatorial Laplacian from weighted undirected edges."""
    A = np.zeros((n, n), dtype=float)
    for i, j, w in edges:
        A[i, j] += w
        A[j, i] += w
    D = np.diag(A.sum(axis=1))
    return D - A


def _river_graph(eps: float) -> SpectralGraph:
    """River-like topology: main stem with tributaries; choke edge has weight eps."""
    # nodes 0..9: stem, 10..15: tributaries
    edges: list[tuple[int, int, float]] = []
    for i in range(9):
        w = float(eps) if (i, i + 1) == (4, 5) else 1.0
        edges.append((i, i + 1, w))
    edges.extend(
        [
            (2, 10, 1.0),
            (4, 11, 1.0),
            (4, 12, 1.0),
            (6, 13, 1.0),
            (7, 14, 1.0),
            (8, 15, 1.0),
        ]
    )
    return SpectralGraph(_laplacian_from_weighted_edges(16, edges))


def _trunk_roots_graph(eps: float) -> SpectralGraph:
    """Tree-like topology: trunk with roots/crown branches; trunk choke has weight eps."""
    # nodes 0..7: trunk, 8..12: roots, 13..16: upper branches
    edges: list[tuple[int, int, float]] = []
    for i in range(7):
        w = float(eps) if (i, i + 1) == (3, 4) else 1.0
        edges.append((i, i + 1, w))
    edges.extend(
        [
            (0, 8, 1.0),
            (0, 9, 1.0),
            (0, 10, 1.0),
            (1, 11, 1.0),
            (1, 12, 1.0),
            (6, 13, 1.0),
            (6, 14, 1.0),
            (7, 15, 1.0),
            (7, 16, 1.0),
        ]
    )
    return SpectralGraph(_laplacian_from_weighted_edges(17, edges))


# ---------------------------------------------------------------------------
# Experiment 1 — Proposition 1: ℛ_l formula vs finite-difference
# ---------------------------------------------------------------------------

def experiment_1_geometric_functional(
    dyn: SpectralKernelDynamics, output_dir: Path
) -> None:
    """Verify ℛ_l[h] = −log(h/h₀)−1 against finite-difference estimate."""
    print("Exp 1: Proposition 1 — geometric functional")
    lam = dyn.graph.eigenvalues
    h0 = dyn.h0
    N = dyn.graph.N
    colors = _mode_colors(N)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    test_points = [
        h0 * 0.5,
        h0 * 1.0,
        dyn.graph.heat_kernel_weights(tau=1.0),
    ]
    titles = [r"$h = 0.5\,h_0$", r"$h = h_0$", r"$h = e^{-\lambda\tau},\;\tau=1$"]

    for ax, h_test, title in zip(axes, test_points, titles):
        R_analytic = dyn.R(h_test)
        R_fd = dyn.R_finite_difference(h_test)
        ax.plot(lam, R_analytic, "o-", color=colors[0], label=r"$\mathcal{R}_l$ (analytic)")
        ax.plot(lam, R_fd, "x--", color=colors[1], ms=8, label="finite-diff")
        ax.set_xlabel(r"$\lambda_l$")
        ax.set_ylabel(r"$\mathcal{R}_l$")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # annotate max error
        err = np.max(np.abs(R_analytic - R_fd))
        ax.text(0.05, 0.05, f"max err = {err:.2e}", transform=ax.transAxes,
                fontsize=8, color="gray")

    fig.suptitle("Exp 1 — Proposition 1: geometric functional ℛ_l", fontsize=11)
    fig.tight_layout()
    _save(fig, output_dir, "exp1_geometric_functional.pdf")


# ---------------------------------------------------------------------------
# Experiment 2 — Corollary 1: fixed-point iteration
# ---------------------------------------------------------------------------

def experiment_2_fixed_point(
    dyn: SpectralKernelDynamics, output_dir: Path
) -> None:
    """Show fixed-point iteration convergence and verify contraction bound."""
    print("Exp 2: Corollary 1 — fixed-point iteration")
    result = dyn.fixed_point_iteration(max_iter=200, tol=1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    lam = dyn.graph.eigenvalues
    N = dyn.graph.N
    colors = _mode_colors(N)
    labels = _mode_labels(N)

    # left: per-mode convergence
    ax = axes[0]
    history = np.array(result.history)
    for l in range(N):
        ax.plot(history[:, l], color=colors[l], label=labels[l])
    ax.axhline(0, color="k", lw=0.5, ls="--")
    for l in range(N):
        ax.axhline(result.h_star[l], color=colors[l], lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$h^{(n)}(\lambda_l)$")
    ax.set_title("Per-mode convergence to h*")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.6, 0.9,
            f"converged: {result.converged}\n"
            f"iterations: {result.iterations}\n"
            f"contraction: {result.contraction_value:.3f}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    # right: residual ‖ℛ − 𝒯‖_∞ log-scale
    ax = axes[1]
    ax.semilogy(result.residual_history, color="#2ca02c")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|\mathcal{R} - \mathcal{T}\|_\infty$")
    ax.set_title("Field-equation residual")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Exp 2 — Corollary 1: fixed-point iteration on P_{N}", fontsize=11)
    fig.tight_layout()
    _save(fig, output_dir, "exp2_fixed_point.pdf")

    # print table of h* vs eigenvalues
    print(f"  converged={result.converged}, iterations={result.iterations}, "
          f"contraction={result.contraction_value:.4f}")
    for l in range(dyn.graph.N):
        print(f"  l={l}: λ={lam[l]:.4f}  h*={result.h_star[l]:.6f}")


# ---------------------------------------------------------------------------
# Experiment 3 — Corollary 2: vacuum solution and geodesics
# ---------------------------------------------------------------------------

def experiment_3_geodesics(
    dyn: SpectralKernelDynamics, output_dir: Path
) -> None:
    """Plot the vacuum solution and several geodesic paths in spectral space."""
    print("Exp 3: Corollary 2 — vacuum solution and geodesics")
    lam = dyn.graph.eigenvalues
    N = dyn.graph.N
    colors = _mode_colors(N)
    t_vals = np.linspace(0, 3, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # left: bar chart h* (vacuum) vs h₀
    ax = axes[0]
    x = np.arange(N)
    w = 0.35
    ax.bar(x - w / 2, dyn.h0, w, label=r"$h_0$ (reference)", color="#aec6e8")
    ax.bar(x + w / 2, dyn.vacuum(), w, label=r"$h^*=h_0 e^{-1}$ (vacuum)", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([f"λ={v:.3f}" for v in lam], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Spectral weight")
    ax.set_title("Vacuum solution  h* = h₀ e⁻¹")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # right: geodesic paths for modes l=1,2,3 (skip l=0 λ=0), guarded for small N
    ax = axes[1]
    # heat-kernel geodesic: a_l=0, b_l=−λ_l
    paths = dyn.heat_kernel_geodesic(t_vals)
    geo_modes = [l for l in [1, 2, 3] if l < N]
    for l in geo_modes:
        ax.plot(t_vals, paths[:, l], color=colors[l],
                label=rf"heat kernel $b_l=-\lambda_{l}$")
    # custom geodesic: a_l = log(h0), b_l = +0.3 for all l (rising)
    a_custom = np.log(dyn.h0)
    b_custom = np.full(N, 0.3)
    paths_c = dyn.geodesic_path(a_custom, b_custom, t_vals)
    ax.plot(t_vals, paths_c[:, 1], color="gray", ls="--",
            label=r"custom $b_l=+0.3$")

    ax.set_xlabel(r"$t$ (affine parameter)")
    ax.set_ylabel(r"$h_t(\lambda_l)$")
    ax.set_title("Log-linear geodesics  $h_t=e^{a_l+b_l t}$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Exp 3 — Corollary 2: vacuum solution and geodesics", fontsize=11)
    fig.tight_layout()
    _save(fig, output_dir, "exp3_geodesics.pdf")


# ---------------------------------------------------------------------------
# Experiment 4 — Corollary 3: stability margins and Hessian
# ---------------------------------------------------------------------------

def experiment_4_stability(
    dyn: SpectralKernelDynamics, output_dir: Path
) -> None:
    """Compute and visualise the Hessian and per-mode stability margins at h*."""
    print("Exp 4: Corollary 3 — stability analysis at h*")
    fp = dyn.fixed_point_iteration(tol=1e-12)
    stab = dyn.stability_analysis(fp.h_star)
    lam = dyn.graph.eigenvalues
    N = dyn.graph.N

    fig = plt.figure(figsize=(12, 4.5))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # (a) per-mode stability margin
    ax_a = fig.add_subplot(gs[0])
    colors = ["#2ca02c" if m > 0 else "#d62728" for m in stab.per_mode_margin]
    ax_a.bar(range(N), stab.per_mode_margin, color=colors)
    ax_a.axhline(0, color="k", lw=1)
    ax_a.set_xticks(range(N))
    ax_a.set_xticklabels([f"l={l}" for l in range(N)])
    ax_a.set_ylabel(r"$\partial\mathcal{T}_l/\partial h_l + 1/h^*_l$")
    ax_a.set_title("Per-mode stability margin\n(must be > 0)")
    ax_a.grid(True, alpha=0.3, axis="y")

    # (b) Hessian heatmap
    ax_b = fig.add_subplot(gs[1])
    im = ax_b.imshow(stab.H, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax_b, shrink=0.8)
    ax_b.set_title(f"Hessian H at h*\n(all eigvals < 0 ⟺ stable)")
    ax_b.set_xlabel("mode m"); ax_b.set_ylabel("mode l")

    # (c) Hessian eigenvalues
    ax_c = fig.add_subplot(gs[2])
    eigvals = np.sort(stab.eigenvalues_H)
    ax_c.bar(range(N), eigvals,
             color=["#2ca02c" if v < 0 else "#d62728" for v in eigvals])
    ax_c.axhline(0, color="k", lw=1)
    ax_c.set_xticks(range(N))
    ax_c.set_xticklabels([f"e{i}" for i in range(N)], fontsize=8)
    ax_c.set_ylabel("Eigenvalue of H")
    ax_c.set_title(f"H eigenspectrum\nΔ = λ_min(−H) = {stab.gap:.4f}")
    ax_c.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Exp 4 -- Corollary 3: stability at h*  (stable={stab.stable})",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, output_dir, "exp4_stability.pdf")

    print(f"  stable={stab.stable}, gap Δ={stab.gap:.6f}, "
          f"𝒮_coup={stab.coupling_entropy_value:.4f}")


# ---------------------------------------------------------------------------
# Experiment 5 — Remark 4: heat kernel as a critical point
# ---------------------------------------------------------------------------

def experiment_5_heat_kernel(
    dyn: SpectralKernelDynamics, output_dir: Path
) -> None:
    """Verify ℛ_l[h*_τ] vs 𝒯_l[h*_τ] for the heat kernel at several τ."""
    print("Exp 5: Remark 4 — heat kernel as critical point")
    lam = dyn.graph.eigenvalues
    taus = [0.5, 1.0, 2.0]
    N = dyn.graph.N

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, tau in zip(axes, taus):
        R_vals, T_vals, resid = dyn.heat_kernel_critical_point_check(tau)
        ax.scatter(T_vals, R_vals, c=lam, cmap="viridis", s=80, zorder=3)
        # diagonal reference
        lo = min(T_vals.min(), R_vals.min()) - 0.1
        hi = max(T_vals.max(), R_vals.max()) + 0.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label=r"$\mathcal{R}=\mathcal{T}$")
        for l in range(N):
            ax.annotate(f"l={l}", (T_vals[l], R_vals[l]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel(r"$\mathcal{T}_l[h^*_\tau]$")
        ax.set_ylabel(r"$\mathcal{R}_l[h^*_\tau]$")
        ax.set_title(f"τ = {tau}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.88,
                f"max |resid| = {np.max(np.abs(resid)):.3f}",
                transform=ax.transAxes, fontsize=8, color="gray")

    fig.suptitle(
        r"Exp 5 -- Remark 4: heat kernel $\mathcal{R}$ vs $\mathcal{T}$"
        "  (diagonal = exact fixed point)",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, output_dir, "exp5_heat_kernel.pdf")


# ---------------------------------------------------------------------------
# Experiment 6 — Remark 8 / Q6: phase-transition early-warning signals
# ---------------------------------------------------------------------------

def experiment_6_phase_transition(
    dyn: SpectralKernelDynamics, output_dir: Path
) -> None:
    """Sweep edge weight ε ∈ [1, 0] on edge (2,3); track diagnostics.

    Uses an eigenvalue-aware source (w_l = λ_l) so that h* is non-uniform
    and the diagnostics H[h*], Δ', and Sc vary as the graph approaches
    disconnection.  The base source parameters (σ², μ₂) are inherited from
    the SpectralKernelDynamics object passed in.
    """
    print("Exp 6: Remark 8 / Q6 — phase-transition early-warning")
    N = dyn.graph.N
    if N < 4:
        raise ValueError("Experiment 6 requires N >= 4 to weaken edge (2,3).")
    eps_vals = np.linspace(1.0, 0.02, 60)  # avoid exact 0 (disconnected)
    weak_i, weak_j = 2, 3

    graph_seq = [
        SpectralGraph.path_graph_with_weak_edge(N, i=weak_i, j=weak_j, epsilon=float(eps))
        for eps in eps_vals
    ]

    # Build an eigenvalue-aware version of the base source for each graph
    # so that T_l = μ₂ λ_l / (2(σ²+h_l)) and h* depends on the spectrum.
    base = dyn.source

    def _aware_source(g: SpectralGraph) -> GaussianMISource:
        return GaussianMISource(
            sigma2=base.sigma2, mu2=base.mu2, eigenvalues=g.eigenvalues
        )

    sweep = dyn.phase_transition_sweep(
        graph_seq,
        tol=1e-10,
        source_factory=_aware_source,
    )

    fiedler = np.array(sweep["fiedler"])
    H_ent = np.array(sweep["H_entropy"])
    delta = np.array(sweep["delta_gap"])
    s_coup = np.array(sweep["coup_entropy"])

    # Normalise to [0,1] for overlay
    def _norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-12 else x

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # top: raw diagnostics
    ax = axes[0]
    ax.plot(eps_vals, fiedler, color="#1f77b4", label=r"$\lambda_1$ (Fiedler)")
    ax.set_ylabel(r"$\lambda_1(\varepsilon)$", color="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(eps_vals, H_ent, color="#ff7f0e",  lw=1.5, label=r"$\mathcal{H}[h^*]$")
    ax2.plot(eps_vals, delta, color="#2ca02c", lw=1.5, ls="--", label=r"$\Delta'(h^*)$")
    ax2.plot(eps_vals, s_coup, color="#9467bd", lw=1.2, ls=":", label=r"$\mathcal{S}_{\mathrm{coup}}$")
    ax2.set_ylabel(r"diagnostic value")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")
    ax.set_title(
        rf"Early-warning signals vs edge weight $\varepsilon$ on edge ({weak_i},{weak_j})"
    )
    ax.grid(True, alpha=0.3)

    # bottom: normalised overlay to show ordering of signals
    ax = axes[1]
    ax.plot(eps_vals, _norm(fiedler), color="#1f77b4", label=r"$\lambda_1$ (normalised)")
    ax.plot(eps_vals, _norm(H_ent),  color="#ff7f0e", lw=1.5,
            label=r"$\mathcal{H}[h^*]$ (Remark 8)")
    ax.plot(eps_vals, _norm(delta),  color="#2ca02c", lw=1.5, ls="--",
            label=r"$\Delta'(h^*)$ gap (Q6, fires earlier)")
    ax.plot(eps_vals, _norm(s_coup), color="#9467bd", lw=1.2, ls=":",
            label=r"$\mathcal{S}_{\mathrm{coup}}$ (mode-separable baseline)")
    ax.set_xlabel(r"Edge weight $\varepsilon$  (1 = intact, 0 = removed)")
    ax.set_ylabel("Normalised value")
    ax.set_title("Normalised overlay — Δ' drops before H")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()   # fragmentation increases left → right

    fig.tight_layout()
    _save(fig, output_dir, "exp6_phase_transition.pdf")


# ---------------------------------------------------------------------------
# Experiment 6b — Q6 supplemental: explicitly coupled source sweep
# ---------------------------------------------------------------------------

def experiment_6b_coupled_phase_transition(
    dyn: SpectralKernelDynamics, output_dir: Path
) -> None:
    """Repeat Exp 6 with explicit inter-modal coupling (non-diagonal Jacobian)."""
    print("Exp 6b: Q6 supplemental — coupled-source phase-transition")
    N = dyn.graph.N
    if N < 4:
        raise ValueError("Experiment 6b requires N >= 4 to weaken edge (2,3).")
    eps_vals = np.linspace(1.0, 0.02, 60)
    weak_i, weak_j = 2, 3
    eta = 0.05

    graph_seq = [
        SpectralGraph.path_graph_with_weak_edge(N, i=weak_i, j=weak_j, epsilon=float(eps))
        for eps in eps_vals
    ]

    base = dyn.source

    def _coupled_source(g: SpectralGraph) -> CoupledGaussianMISource:
        C = _spectral_adjacency_coupling_matrix(g)
        return CoupledGaussianMISource(
            sigma2=base.sigma2,
            mu2=base.mu2,
            eigenvalues=g.eigenvalues,
            coupling_matrix=C,
            eta=eta,
        )

    sweep = dyn.phase_transition_sweep(
        graph_seq,
        tol=1e-10,
        source_factory=_coupled_source,
    )

    fiedler = np.array(sweep["fiedler"])
    H_ent = np.array(sweep["H_entropy"])
    delta = np.array(sweep["delta_gap"])
    s_coup = np.array(sweep["coup_entropy"])

    def _norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-12 else x

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax = axes[0]
    ax.plot(eps_vals, fiedler, color="#1f77b4", label=r"$\lambda_1$ (Fiedler)")
    ax.set_ylabel(r"$\lambda_1(\varepsilon)$", color="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(eps_vals, H_ent, color="#ff7f0e", lw=1.5, label=r"$\mathcal{H}[h^*]$")
    ax2.plot(eps_vals, delta, color="#2ca02c", lw=1.5, ls="--", label=r"$\Delta'(h^*)$")
    ax2.plot(
        eps_vals,
        s_coup,
        color="#9467bd",
        lw=1.2,
        ls=":",
        label=r"$\mathcal{S}_{\mathrm{coup}}$",
    )
    ax2.set_ylabel(r"diagnostic value")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")
    ax.set_title(
        rf"Coupled-source early-warning vs edge weight $\varepsilon$ on edge ({weak_i},{weak_j})"
    )
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(eps_vals, _norm(fiedler), color="#1f77b4", label=r"$\lambda_1$ (normalised)")
    ax.plot(eps_vals, _norm(H_ent), color="#ff7f0e", lw=1.5, label=r"$\mathcal{H}[h^*]$")
    ax.plot(eps_vals, _norm(delta), color="#2ca02c", lw=1.5, ls="--", label=r"$\Delta'(h^*)$")
    ax.plot(
        eps_vals,
        _norm(s_coup),
        color="#9467bd",
        lw=1.2,
        ls=":",
        label=r"$\mathcal{S}_{\mathrm{coup}}$",
    )
    ax.set_xlabel(r"Edge weight $\varepsilon$  (1 = intact, 0 = removed)")
    ax.set_ylabel("Normalised value")
    ax.set_title(r"Coupled-source overlay (Q6): $\mathcal{S}_{\mathrm{coup}}$ now varies")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    fig.tight_layout()
    _save(fig, output_dir, "exp6b_coupled_phase_transition.pdf")
    print(
        f"  coupled sweep ranges: "
        f"H=[{H_ent.min():.4f},{H_ent.max():.4f}], "
        f"Δ'=[{delta.min():.4f},{delta.max():.4f}], "
        f"S_coup=[{s_coup.min():.4f},{s_coup.max():.4f}]"
    )


# ---------------------------------------------------------------------------
# Experiment 7 — Cross-topology robustness (river vs trunk-roots)
# ---------------------------------------------------------------------------

def experiment_7_cross_topology(
    dyn: SpectralKernelDynamics, output_dir: Path
) -> None:
    """Compare diagnostics on two non-path topologies under edge-constriction."""
    print("Exp 7: Cross-topology robustness — river vs trunk-roots")
    _save_topology_schematics(output_dir)
    eps_vals = np.array([1.0, 0.6, 0.3, 0.1, 0.03], dtype=float)
    base = dyn.source

    def _run_for_builder(builder):
        fiedler_b, H_b, D_b = [], [], []
        fiedler_c, H_c, D_c, S_c = [], [], [], []
        for eps in eps_vals:
            g = builder(float(eps))

            src_b = GaussianMISource(
                sigma2=base.sigma2, mu2=base.mu2, eigenvalues=g.eigenvalues
            )
            dyn_b = SpectralKernelDynamics(g, src_b)
            fp_b = dyn_b.fixed_point_iteration(tol=1e-10)
            st_b = dyn_b.stability_analysis(fp_b.h_star)
            fiedler_b.append(g.fiedler_value)
            H_b.append(spectral_entropy(fp_b.h_star))
            D_b.append(st_b.fiedler_gap)

            C = _spectral_adjacency_coupling_matrix(g)
            src_c = CoupledGaussianMISource(
                sigma2=base.sigma2,
                mu2=base.mu2,
                eigenvalues=g.eigenvalues,
                coupling_matrix=C,
                eta=0.05,
            )
            dyn_c = SpectralKernelDynamics(g, src_c)
            fp_c = dyn_c.fixed_point_iteration(tol=1e-10)
            st_c = dyn_c.stability_analysis(fp_c.h_star)
            fiedler_c.append(g.fiedler_value)
            H_c.append(spectral_entropy(fp_c.h_star))
            D_c.append(st_c.fiedler_gap)
            S_c.append(st_c.coupling_entropy_value)
        return (
            np.array(fiedler_b), np.array(H_b), np.array(D_b),
            np.array(fiedler_c), np.array(H_c), np.array(D_c), np.array(S_c)
        )

    rv = _run_for_builder(_river_graph)
    tr = _run_for_builder(_trunk_roots_graph)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    rows = [("River-channel", rv), ("Trunk+roots tree", tr)]
    for ax, (name, arrs) in zip(axes, rows):
        f_b, H_b, D_b, f_c, H_c, D_c, S_c = arrs
        ax.plot(eps_vals, f_b, "o-", color="#1f77b4", label=r"$\lambda_1$ (baseline)")
        ax.plot(eps_vals, H_b, "s-", color="#ff7f0e", label=r"$\mathcal{H}$ (baseline)")
        ax.plot(eps_vals, D_b, "^-", color="#2ca02c", label=r"$\Delta'$ (baseline)")
        ax.plot(eps_vals, S_c, "d--", color="#9467bd", label=r"$\mathcal{S}_{\mathrm{coup}}$ (coupled)")
        ax.set_ylabel("Diagnostic value")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    axes[1].set_xlabel(r"Edge weight $\varepsilon$  (1 = intact, 0 = severe constriction)")
    axes[1].invert_xaxis()
    fig.suptitle("Exp 7 — Cross-topology robustness (synthetic river and trunk-roots)", fontsize=11)
    fig.tight_layout()
    _save(fig, output_dir, "exp7_cross_topology.pdf")

    summary_path = output_dir / "exp7_cross_topology_summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("topology,eps,lambda1_baseline,H_baseline,Dprime_baseline,S_coup_coupled\n")
        for label, arrs in [("river", rv), ("trunk_roots", tr)]:
            f_b, H_b, D_b, _, _, _, S_c = arrs
            for i, eps in enumerate(eps_vals):
                f.write(
                    f"{label},{eps:.3f},{f_b[i]:.6f},{H_b[i]:.6f},{D_b[i]:.6f},{S_c[i]:.6f}\n"
                )
    print(f"  saved → {summary_path}")


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all_experiments(
    output_dir: str = "figures/spectral",
    sigma2: float = 1.0,
    mu2: float = 1.0,
    N: int = 5,
) -> None:
    """Run the full experiment suite and save figures to output_dir.

    Parameters
    ----------
    output_dir : str
        Directory for output PDF figures.
    sigma2 : float
        Observation noise variance for GaussianMISource.
    mu2 : float
        MI Lagrange multiplier μ₂.
    N : int
        Path graph size (default 5).
    """
    out = Path(output_dir)
    print(f"\n=== kernelcal spectral experiments  (P_{N}, σ²={sigma2}, μ₂={mu2}) ===")

    graph = SpectralGraph.path_graph(N)
    source = GaussianMISource(sigma2=sigma2, mu2=mu2)
    dyn = SpectralKernelDynamics(graph, source)

    print(f"Graph: {graph}")
    print(f"Source: {source}")
    print(f"Eigenvalues: {np.round(graph.eigenvalues, 4)}")

    experiment_1_geometric_functional(dyn, out)
    experiment_2_fixed_point(dyn, out)
    experiment_3_geodesics(dyn, out)
    experiment_4_stability(dyn, out)
    experiment_5_heat_kernel(dyn, out)
    experiment_6_phase_transition(dyn, out)
    experiment_6b_coupled_phase_transition(dyn, out)
    experiment_7_cross_topology(dyn, out)

    print(f"\nAll figures saved to: {out.resolve()}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Spectral kernel dynamics experiments (companion to paper Section 3)."
    )
    parser.add_argument("--output-dir", default="figures/spectral",
                        help="Output directory for figures (default: figures/spectral)")
    parser.add_argument("--sigma2", type=float, default=1.0,
                        help="Observation noise variance σ² (default: 1.0)")
    parser.add_argument("--mu2", type=float, default=1.0,
                        help="MI Lagrange multiplier μ₂ (default: 1.0)")
    parser.add_argument("--N", type=int, default=5,
                        help="Path graph size (default: 5)")
    args = parser.parse_args()

    run_all_experiments(
        output_dir=args.output_dir,
        sigma2=args.sigma2,
        mu2=args.mu2,
        N=args.N,
    )
