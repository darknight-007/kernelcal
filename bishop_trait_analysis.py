#!/usr/bin/env python3
"""
bishop_trait_analysis.py
========================
Full spectral kernel analysis of the Bishop fault scarp rock field using
per-rock traits (area, eccentricity, orientation, elevation) as vertex
signals in the distinction dynamics framework.

Analyses:
  A. Trait-weighted spectral kernels — area, eccentricity, orientation
     as independent vertex signals ψ_t, computing c_l = φ_l^T ψ and
     mode weights w_l = |c_l|² for each trait channel.
  B. Cross-kernel factorization test — spatial graph vs trait graph,
     measuring abiotic space-trait coupling (Prop. 5 of P4).
  C. Scarp vs off-scarp comparison — does the most abiotically
     processed part have different ΔH than the surrounding terrain?

Reference: Chen et al., arXiv:1909.12874 (IROS 2021)
"""

from __future__ import annotations
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh

KCAL_ROOT = Path(__file__).parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import (
    spectral_entropy_from_laplacian,
    fixed_point_kernel,
    fiedler_mode_gap,
    stability_conservation_tradeoff,
)

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent / 'datasets' / 'bishop_scarp'
FIG_DIR  = Path(__file__).parent / 'bishop_figures'
FIG_DIR.mkdir(exist_ok=True)

N_SCARP  = 2000   # subsample from scarp rocks (13,701)
N_OFF    = 2000   # subsample from off-scarp rocks
K_NN     = 8
SIGMA_M  = 1.0    # spatial RBF bandwidth [m]
MU2      = 2.0
SIGMA2   = 1.0

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'lines.linewidth': 1.2,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

COLORS = dict(
    area='#D55E00', ecc='#0072B2', orient='#009E73',
    spatial='#CC79A7', scarp='#E69F00', offscarp='#56B4E9',
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_scarp_traits():
    """Load scarp rock traits CSV (13,701 rocks with 8 columns)."""
    data = np.loadtxt(str(BASE / 'rock_traits_full.csv'), delimiter=',', skiprows=1)
    cols = dict(lon=0, lat=1, area=2, major=3, minor=4, ecc=5, orient=6, elev=7)
    return data, cols


def load_all_rocks():
    """Load all 82K rock centroids."""
    return np.loadtxt(str(BASE / 'rocks-coord-list.csv'), delimiter=',')


def lonlat_to_metres(lonlat):
    lon0, lat0 = lonlat[:, 0].mean(), lonlat[:, 1].mean()
    R = 6_371_000.0
    cos0 = math.cos(math.radians(lat0))
    E = (lonlat[:, 0] - lon0) * cos0 * (math.pi / 180.0) * R
    N = (lonlat[:, 1] - lat0) * (math.pi / 180.0) * R
    return np.column_stack([E, N])


def subsample(arr, n, seed=42):
    if len(arr) <= n:
        return arr, np.arange(len(arr))
    idx = np.random.default_rng(seed).choice(len(arr), size=n, replace=False)
    return arr[idx], idx


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_laplacian(xy, k, sigma):
    N = len(xy)
    tree = cKDTree(xy)
    dists, idxs = tree.query(xy, k=k + 1)
    A = np.zeros((N, N))
    for i in range(N):
        for r in range(1, k + 1):
            j = idxs[i, r]
            w = math.exp(-dists[i, r]**2 / (2.0 * sigma**2))
            A[i, j] += w
            A[j, i] += w
    A = np.minimum(A, 1.0)
    return np.diag(A.sum(axis=1)) - A


def build_trait_laplacian(traits, k):
    """k-NN graph in trait space (area, ecc, orient), normalised."""
    from sklearn.preprocessing import StandardScaler
    traits_norm = StandardScaler().fit_transform(traits)
    N = len(traits_norm)
    tree = cKDTree(traits_norm)
    dists, idxs = tree.query(traits_norm, k=k + 1)
    sigma = np.median(dists[:, 1])
    A = np.zeros((N, N))
    for i in range(N):
        for r in range(1, k + 1):
            j = idxs[i, r]
            w = math.exp(-dists[i, r]**2 / (2.0 * sigma**2))
            A[i, j] += w
            A[j, i] += w
    A = np.minimum(A, 1.0)
    return np.diag(A.sum(axis=1)) - A


def betti_from_laplacian(L):
    eigvals = np.linalg.eigvalsh(L)
    beta0 = int(np.sum(np.abs(eigvals) < 1e-6))
    V = L.shape[0]
    A_mat = np.diag(np.diag(L)) - L
    E = int(np.sum(A_mat > 1e-10)) // 2
    beta1 = max(0, E - V + beta0)
    return beta0, beta1


def kernelcal_diagnostics(L, label=''):
    """Run full kernelcal diagnostic suite. Returns dict."""
    eigvals = np.linalg.eigvalsh(L)
    N = L.shape[0]
    H = spectral_entropy_from_laplacian(L, tau=1.0)
    h_star, info = fixed_point_kernel(L, mu2=MU2, sigma2=SIGMA2)
    h0 = np.exp(-eigvals); h0[h0 < 1e-30] = 1e-30
    h_bar = h_star / h_star.sum()
    h0_bar = h0 / h0.sum()
    H_star = -np.sum(h_bar[h_bar > 0] * np.log(h_bar[h_bar > 0]))
    H_vac = -np.sum(h0_bar[h0_bar > 0] * np.log(h0_bar[h0_bar > 0]))
    delta_H = H_star - H_vac
    dp = fiedler_mode_gap(h_star, L, mu2=MU2, sigma2=SIGMA2)
    sct = stability_conservation_tradeoff(h_star, L, mu2=MU2, sigma2=SIGMA2)
    beta0, beta1 = betti_from_laplacian(L)
    if label:
        print(f'  [{label}] N={N}  ΔH={delta_H:.4f}  Δ\'={dp:.4f}  '
              f'β₀={beta0}  β₁={beta1}  conv={info["converged"]}')
    return dict(eigvals=eigvals, h_star=h_star, h0=h0, h_bar=h_bar, h0_bar=h0_bar,
                H_star=H_star, H_vac=H_vac, delta_H=delta_H, delta_prime=dp,
                beta0=beta0, beta1=beta1, deficit=sct['conservation_deficit'],
                converged=info['converged'], N=N)


# ══════════════════════════════════════════════════════════════════════════════
# SPECTRAL PROJECTION OF TRAITS
# ══════════════════════════════════════════════════════════════════════════════

def spectral_project(L, signal):
    """Project vertex signal onto Laplacian eigenbasis. Returns c_l, w_l."""
    eigvals, eigvecs = np.linalg.eigh(L)
    c_l = eigvecs.T @ signal            # spectral coefficients
    w_l = c_l**2                         # mode weights
    w_l_norm = w_l / (w_l.sum() + 1e-30)
    return eigvals, c_l, w_l, w_l_norm


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS A: TRAIT-WEIGHTED SPECTRAL KERNELS
# ══════════════════════════════════════════════════════════════════════════════

def analysis_A(data, cols, xy_scarp, idx_scarp):
    print('\n' + '='*70)
    print('ANALYSIS A: Trait-weighted spectral kernels')
    print('='*70)

    L = build_laplacian(xy_scarp, K_NN, SIGMA_M)
    base = kernelcal_diagnostics(L, label='spatial-only')

    traits = {
        'Rock area (m²)': data[idx_scarp, cols['area']],
        'Eccentricity': data[idx_scarp, cols['ecc']],
        'Orientation (°)': data[idx_scarp, cols['orient']],
    }
    trait_keys = list(traits.keys())
    trait_colors = [COLORS['area'], COLORS['ecc'], COLORS['orient']]

    projections = {}
    for name, signal in traits.items():
        sig_centered = signal - signal.mean()
        eigvals, c_l, w_l, w_l_norm = spectral_project(L, sig_centered)
        projections[name] = dict(eigvals=eigvals, c_l=c_l, w_l=w_l, w_l_norm=w_l_norm,
                                 signal=signal)
        top5 = np.argsort(w_l_norm)[-5:][::-1]
        print(f'  {name}: top-5 modes = {top5}, weight = {w_l_norm[top5].sum():.3f}')

    # ── Fig: Trait maps ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (name, col) in zip(axes, zip(trait_keys, trait_colors)):
        sig = traits[name]
        sc = ax.scatter(xy_scarp[:, 0], xy_scarp[:, 1], s=2,
                        c=sig, cmap='jet', alpha=0.7, edgecolors='none')
        ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]')
        ax.set_title(name, fontweight='bold')
        ax.set_aspect('equal')
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    fig.suptitle('Bishop Fault Scarp — Rock Traits as Vertex Signals\n'
                 '13,701 scarp rocks  |  Abiotic controllers: volcanic, tectonic, geomorphic',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_A1_trait_maps.png', dpi=200)
    plt.close(fig)
    print('  Saved fig_A1_trait_maps.png')

    # ── Fig: Spectral projections (mode weights) ──────────────────────────
    n_show = min(80, len(base['eigvals']))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, col) in zip(axes, zip(trait_keys, trait_colors)):
        p = projections[name]
        ax.bar(range(n_show), p['w_l_norm'][:n_show], color=col, alpha=0.7, width=1.0)
        ax.set_xlabel('Mode index l')
        ax.set_ylabel('Mode weight w_l = |c_l|²')
        ax.set_title(name, fontweight='bold', fontsize=9)
        ax.set_xlim(-1, n_show)
    fig.suptitle('Spectral Projection of Rock Traits onto Graph Laplacian Eigenbasis\n'
                 'w_l = |c_l|²  =  fraction of trait variance at spatial frequency l',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_A2_spectral_projections.png', dpi=200)
    plt.close(fig)
    print('  Saved fig_A2_spectral_projections.png')

    # ── Fig: Fixed-point kernels per trait ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, col) in zip(axes, zip(trait_keys, trait_colors)):
        p = projections[name]
        h_bar = base['h_bar']
        h0_bar = base['h0_bar']
        ax.semilogy(range(n_show), h_bar[:n_show], 'o-', color=col,
                    markersize=2, linewidth=1, label=f'h* (ΔH={base["delta_H"]:.3f})')
        ax.semilogy(range(n_show), h0_bar[:n_show], '--', color='#999999',
                    linewidth=0.8, label='vacuum h₀')
        # Overlay mode weights as shaded bars
        ax2 = ax.twinx()
        ax2.bar(range(n_show), p['w_l_norm'][:n_show], color=col,
                alpha=0.15, width=1.0, label='w_l (trait)')
        ax2.set_ylabel('w_l', color=col, fontsize=7)
        ax2.tick_params(axis='y', labelcolor=col, labelsize=7)
        ax.set_xlabel('Mode index l')
        ax.set_ylabel('Kernel weight')
        ax.set_title(name, fontweight='bold', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
    fig.suptitle('Fixed-Point Kernel h*(λ) with Trait Mode Weights Overlaid\n'
                 'Shaded bars = trait spectral projection;  line = MaxCal kernel',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_A3_kernel_with_traits.png', dpi=200)
    plt.close(fig)
    print('  Saved fig_A3_kernel_with_traits.png')

    return base, projections


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS B: CROSS-KERNEL FACTORIZATION TEST
# ══════════════════════════════════════════════════════════════════════════════

def analysis_B(data, cols, xy_scarp, idx_scarp):
    print('\n' + '='*70)
    print('ANALYSIS B: Cross-kernel factorization test (spatial vs traits)')
    print('='*70)

    N = len(xy_scarp)

    # Build spatial graph
    print('  Building spatial Laplacian...')
    L_space = build_laplacian(xy_scarp, K_NN, SIGMA_M)

    # Build trait graph (area, ecc, orient in normalised trait space)
    trait_arr = np.column_stack([
        data[idx_scarp, cols['area']],
        data[idx_scarp, cols['ecc']],
        data[idx_scarp, cols['orient']],
    ])
    print('  Building trait Laplacian...')
    L_trait = build_trait_laplacian(trait_arr, K_NN)

    # Diagnostics on each
    r_space = kernelcal_diagnostics(L_space, label='G_spatial')
    r_trait = kernelcal_diagnostics(L_trait, label='G_trait')

    # Cross-kernel: compute fixed-point kernels, then HS norm of residual
    # k_cross = k_coupled - k_space ⊗ k_trait
    # For spectral kernels: h_coupled vs h_space * h_trait (element-wise in shared basis)
    # Since bases differ, compute kernel matrices and compare
    K_space = np.diag(r_space['h_star'])  # diagonal in spatial eigenbasis
    K_trait = np.diag(r_trait['h_star'])  # diagonal in trait eigenbasis

    # Kronecker product of diagonal kernels
    k_sp_norm = r_space['h_star'] / (r_space['h_star'].sum() + 1e-30)
    k_tr_norm = r_trait['h_star'] / (r_trait['h_star'].sum() + 1e-30)

    # Build a coupled graph: spatial + trait edges (sum of adjacencies)
    print('  Building coupled Laplacian...')
    A_space = np.diag(np.diag(L_space)) - L_space
    A_trait = np.diag(np.diag(L_trait)) - L_trait
    # Normalise each adjacency to unit max before summing
    A_space_n = A_space / (A_space.max() + 1e-30)
    A_trait_n = A_trait / (A_trait.max() + 1e-30)
    A_coupled = A_space_n + A_trait_n
    L_coupled = np.diag(A_coupled.sum(axis=1)) - A_coupled

    r_coupled = kernelcal_diagnostics(L_coupled, label='G_coupled')

    # Cross-kernel norm: compare coupled kernel to product of marginals
    h_prod = k_sp_norm * k_tr_norm
    h_prod = h_prod / (h_prod.sum() + 1e-30)
    h_coup_norm = r_coupled['h_bar']

    # Pad to same length (coupled may have different spectrum)
    n_min = min(len(h_coup_norm), len(h_prod))
    residual = h_coup_norm[:n_min] - h_prod[:n_min]
    hs_norm = np.sqrt(np.sum(residual**2))

    print(f'\n  Cross-kernel ||k_cross||_HS = {hs_norm:.6f}')
    print(f'  ΔH_spatial  = {r_space["delta_H"]:.4f}')
    print(f'  ΔH_trait    = {r_trait["delta_H"]:.4f}')
    print(f'  ΔH_coupled  = {r_coupled["delta_H"]:.4f}')

    # ── Fig: Cross-kernel comparison ──────────────────────────────────────
    n_show = min(80, n_min)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.semilogy(range(n_show), r_space['h_bar'][:n_show], 'o-',
                color=COLORS['spatial'], markersize=2, label='spatial')
    ax.semilogy(range(n_show), r_trait['h_bar'][:n_show], 's-',
                color=COLORS['ecc'], markersize=2, label='trait')
    ax.semilogy(range(n_show), r_coupled['h_bar'][:n_show], 'D-',
                color=COLORS['scarp'], markersize=2, label='coupled')
    ax.set_xlabel('Mode index l')
    ax.set_ylabel('Normalised kernel weight')
    ax.set_title('Three kernels: spatial, trait, coupled', fontweight='bold', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(range(n_show), h_coup_norm[:n_show], color=COLORS['scarp'],
           alpha=0.5, width=1.0, label='coupled h*')
    ax.bar(range(n_show), h_prod[:n_show], color='#999999',
           alpha=0.5, width=1.0, label='spatial ⊗ trait')
    ax.set_xlabel('Mode index l')
    ax.set_ylabel('Kernel weight')
    ax.set_title('Coupled vs product of marginals', fontweight='bold', fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.bar(range(n_show), residual[:n_show], color=COLORS['area'], alpha=0.7, width=1.0)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Mode index l')
    ax.set_ylabel('Residual (coupled − product)')
    ax.set_title(f'Cross-kernel residual  ||k_cross||={hs_norm:.4f}',
                 fontweight='bold', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Cross-Kernel Factorization Test — Bishop Fault Scarp (Abiotic)\n'
                 'Non-zero residual = abiotic space-trait coupling from tectonic physics',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_B1_cross_kernel.png', dpi=200)
    plt.close(fig)
    print('  Saved fig_B1_cross_kernel.png')

    return dict(r_space=r_space, r_trait=r_trait, r_coupled=r_coupled,
                hs_norm=hs_norm, residual=residual)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS C: SCARP VS OFF-SCARP
# ══════════════════════════════════════════════════════════════════════════════

def analysis_C(data, cols, xy_scarp_full):
    print('\n' + '='*70)
    print('ANALYSIS C: Scarp vs off-scarp comparison')
    print('='*70)

    # Load all 82K rocks
    all_lonlat = load_all_rocks()
    all_xy = lonlat_to_metres(all_lonlat)

    # Scarp rocks (from traits file) — use their lon/lat to match
    scarp_lonlat = data[:, [cols['lon'], cols['lat']]]
    scarp_xy = lonlat_to_metres(scarp_lonlat)

    # Identify off-scarp rocks: all rocks NOT in scarp set
    # Match by nearest-neighbour distance < 0.5m
    tree_scarp = cKDTree(scarp_xy)
    dists_to_scarp, _ = tree_scarp.query(all_xy, k=1)
    off_mask = dists_to_scarp > 1.0  # more than 1m from any scarp rock
    off_xy = all_xy[off_mask]
    print(f'  All rocks: {len(all_xy):,}')
    print(f'  Scarp rocks: {len(scarp_xy):,}')
    print(f'  Off-scarp rocks: {len(off_xy):,}')

    # Subsample both to same N
    N_comp = min(N_SCARP, len(scarp_xy), len(off_xy))
    scarp_sub, _ = subsample(scarp_xy, N_comp, seed=42)
    off_sub, _ = subsample(off_xy, N_comp, seed=43)

    # Build graphs and run diagnostics
    print(f'  Subsampled to N={N_comp} each')
    L_scarp = build_laplacian(scarp_sub, K_NN, SIGMA_M)
    L_off = build_laplacian(off_sub, K_NN, SIGMA_M)

    r_scarp = kernelcal_diagnostics(L_scarp, label='SCARP')
    r_off = kernelcal_diagnostics(L_off, label='OFF-SCARP')

    # ── Fig: Scarp vs off-scarp ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    ax.scatter(off_sub[:, 0], off_sub[:, 1], s=1, c=COLORS['offscarp'],
               alpha=0.4, label='off-scarp')
    ax.scatter(scarp_sub[:, 0], scarp_sub[:, 1], s=1, c=COLORS['scarp'],
               alpha=0.6, label='scarp')
    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]')
    ax.set_title('Scarp vs off-scarp rock positions', fontweight='bold', fontsize=9)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, markerscale=5)

    n_show = min(80, len(r_scarp['eigvals']))
    ax = axes[1]
    ax.semilogy(range(n_show), r_scarp['h_bar'][:n_show], 'o-',
                color=COLORS['scarp'], markersize=2,
                label=f'scarp ΔH={r_scarp["delta_H"]:.3f}')
    ax.semilogy(range(n_show), r_off['h_bar'][:n_show], 's-',
                color=COLORS['offscarp'], markersize=2,
                label=f'off-scarp ΔH={r_off["delta_H"]:.3f}')
    ax.semilogy(range(n_show), r_scarp['h0_bar'][:n_show], '--',
                color='#999999', linewidth=0.8, label='vacuum')
    ax.set_xlabel('Mode index l')
    ax.set_ylabel('Kernel weight')
    ax.set_title('Fixed-point kernels', fontweight='bold', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    labels = ['Scarp\n(3 abiotic\ncontrollers)', 'Off-scarp\n(passive)']
    dH_vals = [abs(r_scarp['delta_H']), abs(r_off['delta_H'])]
    colors = [COLORS['scarp'], COLORS['offscarp']]
    bars = ax.bar(labels, dH_vals, color=colors, width=0.5)
    for bar, v in zip(bars, dH_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('|ΔH| (nats)')
    ax.set_title('Spectral concentration comparison', fontweight='bold', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Scarp vs Off-Scarp — Does Abiotic Processing Produce Spectral Concentration?\n'
                 'Scarp: volcanic fracture + tectonic strain + transport  |  Off-scarp: passive weathering',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_C1_scarp_vs_offscarp.png', dpi=200)
    plt.close(fig)
    print('  Saved fig_C1_scarp_vs_offscarp.png')

    return dict(r_scarp=r_scarp, r_off=r_off)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def summary_figure(base, proj, cross, comp):
    print('\n  Generating summary figure...')

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: Trait map (area)
    ax = fig.add_subplot(gs[0, 0])
    sig = proj['Rock area (m²)']['signal']
    sig_clip = np.clip(sig, 0, np.percentile(sig, 95))
    sc = ax.scatter(base['xy'][:, 0], base['xy'][:, 1], s=1.5,
                    c=sig_clip, cmap='jet', alpha=0.7, edgecolors='none')
    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]')
    ax.set_title('(a) Rock area on scarp', fontweight='bold', fontsize=9)
    ax.set_aspect('equal')
    fig.colorbar(sc, ax=ax, shrink=0.6, label='Area (m²)')

    # Panel 2: Spectral projection of 3 traits
    ax = fig.add_subplot(gs[0, 1])
    n_show = 60
    for name, col in zip(['Rock area (m²)', 'Eccentricity', 'Orientation (°)'],
                         [COLORS['area'], COLORS['ecc'], COLORS['orient']]):
        ax.plot(range(n_show), proj[name]['w_l_norm'][:n_show],
                color=col, linewidth=1.2, label=name.split('(')[0].strip())
    ax.set_xlabel('Mode index l')
    ax.set_ylabel('Mode weight w_l')
    ax.set_title('(b) Trait spectral projections', fontweight='bold', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Cross-kernel residual
    ax = fig.add_subplot(gs[0, 2])
    n_res = min(60, len(cross['residual']))
    ax.bar(range(n_res), cross['residual'][:n_res], color=COLORS['area'],
           alpha=0.7, width=1.0)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Mode index l')
    ax.set_ylabel('Cross-kernel residual')
    ax.set_title(f'(c) Cross-kernel ||k_cross||={cross["hs_norm"]:.4f}',
                 fontweight='bold', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Scarp vs off-scarp
    ax = fig.add_subplot(gs[1, 0])
    labels = ['Scarp', 'Off-scarp']
    vals = [abs(comp['r_scarp']['delta_H']), abs(comp['r_off']['delta_H'])]
    cols = [COLORS['scarp'], COLORS['offscarp']]
    bars = ax.bar(labels, vals, color=cols, width=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('|ΔH| (nats)')
    ax.set_title('(d) Scarp vs off-scarp |ΔH|', fontweight='bold', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    # Panel 5: Controller hierarchy bar chart
    ax = fig.add_subplot(gs[1, 1:])
    systems = [
        ('Bishop scarp\n(abiotic)', abs(comp['r_scarp']['delta_H']), COLORS['scarp']),
        ('Bishop off-scarp\n(passive)', abs(comp['r_off']['delta_H']), COLORS['offscarp']),
        ('Venice\n(organic city)', 0.237, '#009E73'),
        ('Marrakech\n(organic city)', 0.238, '#009E73'),
        ('Phoenix\n(planned grid)', 0.285, '#0072B2'),
        ('Houston\n(planned grid)', 0.301, '#0072B2'),
        ('Barcelona\n(planned grid)', 0.339, '#0072B2'),
    ]
    x_pos = range(len(systems))
    bars = ax.bar(x_pos, [s[1] for s in systems],
                  color=[s[2] for s in systems], width=0.6, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s[0] for s in systems], fontsize=7)
    for bar, s in zip(bars, systems):
        ax.text(bar.get_x() + bar.get_width()/2, s[1] + 0.005,
                f'{s[1]:.3f}', ha='center', fontsize=7, fontweight='bold')
    ax.set_ylabel('|ΔH| (nats)')
    ax.set_title('(e) Controller hierarchy: abiotic floor to active planning',
                 fontweight='bold', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(0.058, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(6.5, 0.065, 'abiotic ceiling', color='red', fontsize=7, ha='right')

    fig.suptitle('Bishop Fault Scarp — Full Spectral Kernel Analysis (Abiotic Null)\n'
                 '760 ka Bishop Tuff  |  3 abiotic controllers  |  No biological controller',
                 fontsize=12, fontweight='bold')
    fig.savefig(FIG_DIR / 'fig_summary_bishop_traits.png', dpi=200)
    plt.close(fig)
    print('  Saved fig_summary_bishop_traits.png')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('\n' + '='*70)
    print('BISHOP FAULT SCARP — FULL TRAIT ANALYSIS')
    print('='*70)

    data, cols = load_scarp_traits()
    print(f'Loaded {len(data)} scarp rocks with traits')

    # Project scarp rocks to local metres
    scarp_lonlat = data[:, [cols['lon'], cols['lat']]]
    scarp_xy_full = lonlat_to_metres(scarp_lonlat)

    # Subsample for analysis
    scarp_sub, idx_sub = subsample(scarp_xy_full, N_SCARP)
    print(f'Subsampled to N={len(scarp_sub)}')

    # Analysis A
    base, projections = analysis_A(data, cols, scarp_sub, idx_sub)
    # Attach xy for summary figure
    base['xy'] = scarp_sub

    # Analysis B
    cross = analysis_B(data, cols, scarp_sub, idx_sub)

    # Analysis C
    comp = analysis_C(data, cols, scarp_xy_full)

    # Summary figure
    summary_figure(base, projections, cross, comp)

    # ── Final summary table ───────────────────────────────────────────────
    print('\n' + '='*70)
    print('FINAL SUMMARY')
    print('='*70)
    print(f'  Scarp (spatial-only):   ΔH = {base["delta_H"]:.4f} nats')
    print(f'  Scarp (trait-coupled):  ΔH = {cross["r_coupled"]["delta_H"]:.4f} nats')
    print(f'  Off-scarp:             ΔH = {comp["r_off"]["delta_H"]:.4f} nats')
    print(f'  Cross-kernel norm:     ||k_cross|| = {cross["hs_norm"]:.4f}')
    print(f'  Cities (for reference): ΔH = -0.24 to -0.34 nats')
    print('='*70)
    print(f'\nAll figures → {FIG_DIR}/')


if __name__ == '__main__':
    main()
    print('\nDone.')
