#!/usr/bin/env python3
"""
bishop_kernelcal.py
===================
Kernelcal spectral diagnostics on the Bishop fault scarp rock field
(Volcanic Tablelands, CA) — abiotic null for the P4 biosignature framework.

Site: Bishop, CA  (~-118.442 W, 37.452 N)
      760 ka welded Bishop Tuff, tectonic fault scarp
      Abiotic controllers: volcanic fracturing, tectonic strain, geomorphic transport
      No biological controller present.

Data:
  rocks-coord-list.csv   82,122 rock centroids (lon, lat) from Mask R-CNN
  C3_dem.tif             Colormap-encoded DEM from UAS-SfM (2 cm/px)

Reference: Chen et al., "Geomorphological Analysis Using Unpiloted Aircraft
  Systems, Structure from Motion, and Deep Learning," arXiv:1909.12874, 2021.

Pipeline:
  1. Load rock centroids (lon, lat) from CSV.
  2. Sample elevation from colormap DEM at each centroid.
  3. Project to local metric frame.
  4. Deduplicate within 0.1 m.
  5. Subsample to N_MAX for tractable eigendecomposition.
  6. Build k-NN Gaussian-weighted graph Laplacian (2D + optional elev weighting).
  7. Run kernelcal diagnostics: H[h], h*, Delta', D_m, beta0, beta1.
  8. Generate publication figures (white background).

Usage:
  cd manuscripts/software-kernelcal-deepgis-integration
  python3 bishop_kernelcal.py
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ── kernelcal ──────────────────────────────────────────────────────────────
KCAL_ROOT = Path(__file__).parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import (
    spectral_entropy_from_laplacian,
    fixed_point_kernel,
    fiedler_mode_gap,
    stability_conservation_tradeoff,
)

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / 'datasets' / 'bishop_scarp'
FIG_DIR  = Path(__file__).parent / 'bishop_figures'
FIG_DIR.mkdir(exist_ok=True)

ROCK_CSV = DATA_DIR / 'rocks-coord-list.csv'
DEM_PATH = DATA_DIR / 'C3_dem.tif'

N_MAX    = 2000   # subsample size for dense eigendecomp
K_NN     = 8      # neighbours for graph construction
SIGMA_M  = 1.0    # RBF bandwidth [metres] — ~2x median NN distance
DEDUP_M  = 0.1    # deduplication radius [metres]
MU2      = 2.0    # kernelcal fixed-point parameter
SIGMA2   = 1.0    # kernelcal fixed-point parameter

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


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_rocks() -> np.ndarray:
    """Load rock centroids as (lon, lat) array."""
    return np.loadtxt(str(ROCK_CSV), delimiter=',')


def sample_elevation(lonlat: np.ndarray) -> np.ndarray:
    """Sample relative elevation from colormap DEM at each centroid."""
    import rasterio
    with rasterio.open(str(DEM_PATH)) as src:
        r = src.read(1).astype(float)
        g = src.read(2).astype(float)
        b = src.read(3).astype(float)
        elev_grid = (r + g + b) / 3.0

        elevs = np.full(len(lonlat), np.nan)
        for i, (lon, lat) in enumerate(lonlat):
            try:
                row, col = src.index(lon, lat)
                if 0 <= row < src.height and 0 <= col < src.width:
                    elevs[i] = elev_grid[row, col]
            except Exception:
                pass
    return elevs


# ══════════════════════════════════════════════════════════════════════════════
# COORDINATE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def lonlat_to_metres(lonlat: np.ndarray) -> np.ndarray:
    """Equirectangular projection. Returns (East, North) in metres."""
    lon0 = lonlat[:, 0].mean()
    lat0 = lonlat[:, 1].mean()
    R = 6_371_000.0
    cos0 = math.cos(math.radians(lat0))
    E = (lonlat[:, 0] - lon0) * cos0 * (math.pi / 180.0) * R
    N = (lonlat[:, 1] - lat0) * (math.pi / 180.0) * R
    return np.column_stack([E, N])


def deduplicate(xy: np.ndarray, radius: float) -> np.ndarray:
    """Remove centroids within radius of an already-selected point."""
    if len(xy) == 0:
        return xy
    tree = cKDTree(xy)
    kept = np.ones(len(xy), dtype=bool)
    for i in range(len(xy)):
        if not kept[i]:
            continue
        for j in tree.query_ball_point(xy[i], radius):
            if j != i:
                kept[j] = False
    return xy[kept]


def subsample(xy: np.ndarray, n_max: int, seed: int = 42):
    """Uniform random subsample to at most n_max points. Returns (xy_sub, indices)."""
    if len(xy) <= n_max:
        return xy, np.arange(len(xy))
    idx = np.random.default_rng(seed).choice(len(xy), size=n_max, replace=False)
    return xy[idx], idx


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_laplacian(xy: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """Symmetric k-NN Gaussian-weighted graph Laplacian (dense)."""
    N = len(xy)
    tree = cKDTree(xy)
    dists, idxs = tree.query(xy, k=k + 1)

    A = np.zeros((N, N))
    for i in range(N):
        for r in range(1, k + 1):
            j = idxs[i, r]
            w = math.exp(-dists[i, r] ** 2 / (2.0 * sigma ** 2))
            A[i, j] += w
            A[j, i] += w
    A = np.minimum(A, 1.0)
    return np.diag(A.sum(axis=1)) - A


def betti_from_laplacian(L: np.ndarray) -> tuple[int, int]:
    """Estimate beta0, beta1 from the Laplacian spectrum."""
    eigvals = np.linalg.eigvalsh(L)
    beta0 = int(np.sum(np.abs(eigvals) < 1e-6))
    V = L.shape[0]
    A = np.diag(np.diag(L)) - L
    E = int(np.sum(A > 1e-10)) // 2
    beta1 = max(0, E - V + beta0)
    return beta0, beta1


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis():
    print()
    print('=' * 70)
    print('KERNELCAL | Bishop Fault Scarp — Abiotic Null')
    print('Site: -118.442 W, 37.452 N  |  Volcanic Tablelands, CA')
    print('Geology: 760 ka Bishop Tuff, tectonic fault scarp')
    print('Controllers: volcanic fracturing, tectonic strain, geomorphic transport')
    print('Biological controller: NONE')
    print('=' * 70)

    # 1. Load rocks
    print('\n[1] Loading rock centroids...')
    lonlat = load_rocks()
    n_raw = len(lonlat)
    print(f'    Raw centroids: {n_raw:,}')

    # 2. Sample elevation
    print('\n[2] Sampling elevation from DEM...')
    elevs = sample_elevation(lonlat)
    n_valid_elev = int(np.sum(~np.isnan(elevs)))
    print(f'    Valid elevation samples: {n_valid_elev:,} / {n_raw:,}')
    if n_valid_elev > 0:
        ve = elevs[~np.isnan(elevs)]
        print(f'    Elevation range: [{ve.min():.1f}, {ve.max():.1f}] (relative)')
        print(f'    Elevation mean:  {ve.mean():.1f}, std: {ve.std():.1f}')

    # 3. Project to local metres
    print('\n[3] Projecting to local metric frame...')
    xy = lonlat_to_metres(lonlat)
    ew = xy[:, 0].max() - xy[:, 0].min()
    ns = xy[:, 1].max() - xy[:, 1].min()
    print(f'    Extent: {ew:.1f} m (E-W) x {ns:.1f} m (N-S)')

    # 3b. Sample elevation at each centroid in local frame
    print('    Sampling elevation at projected centroids...')
    elevs_all = sample_elevation(lonlat)
    valid_mask = ~np.isnan(elevs_all)
    xy_with_elev = xy[valid_mask]
    elevs_valid = elevs_all[valid_mask]
    print(f'    Rocks with valid elevation: {len(xy_with_elev):,}')

    # 4. Deduplicate
    print(f'\n[4] Deduplicating (r={DEDUP_M} m)...')
    xy_dd = deduplicate(xy_with_elev, DEDUP_M)
    # Re-match elevations after dedup via nearest-neighbour
    tree_dd = cKDTree(xy_with_elev)
    _, dd_idx = tree_dd.query(xy_dd, k=1)
    elevs_dd = elevs_valid[dd_idx]
    print(f'    After dedup: {len(xy_dd):,} centroids')

    # 5. Subsample
    print(f'\n[5] Subsampling to N={N_MAX}...')
    xy_sub, sub_idx = subsample(xy_dd, N_MAX)
    elevs_sub = elevs_dd[sub_idx]
    N = len(xy_sub)
    print(f'    Analysis graph: N={N}')

    # 6. Build graph
    print(f'\n[6] Building k-NN Laplacian (k={K_NN}, sigma={SIGMA_M} m)...')
    L = build_laplacian(xy_sub, K_NN, SIGMA_M)
    print(f'    Laplacian: {N}x{N}')

    # 7. Betti numbers
    print('\n[7] Computing Betti numbers...')
    beta0, beta1 = betti_from_laplacian(L)
    print(f'    beta0 (components): {beta0}')
    print(f'    beta1 (loops):      {beta1}')

    # 8. Kernelcal diagnostics
    print('\n[8] Running kernelcal diagnostics...')
    eigvals = np.linalg.eigvalsh(L)

    H = spectral_entropy_from_laplacian(L, tau=1.0)
    print(f'    Spectral entropy H[h]:    {H:.5f} nats')

    h_star, info = fixed_point_kernel(L, mu2=MU2, sigma2=SIGMA2)
    conv = 'yes' if info['converged'] else f'NO (r={info["residual"]:.2e})'
    print(f'    Fixed-point h*:           converged={conv} ({info["n_iter"]} iters)')

    h0 = np.exp(-eigvals)
    h0[h0 < 1e-30] = 1e-30
    h_bar = h_star / h_star.sum()
    h0_bar = h0 / h0.sum()
    H_star = -np.sum(h_bar[h_bar > 0] * np.log(h_bar[h_bar > 0]))
    H_vac = -np.sum(h0_bar[h0_bar > 0] * np.log(h0_bar[h0_bar > 0]))
    delta_H = H_star - H_vac
    print(f'    H[h*] = {H_star:.5f},  H[h0] = {H_vac:.5f},  DeltaH = {delta_H:.5f} nats')

    delta_prime = fiedler_mode_gap(h_star, L, mu2=MU2, sigma2=SIGMA2)
    print(f'    Hessian gap Delta\':       {delta_prime:.6f}')

    sct = stability_conservation_tradeoff(h_star, L, mu2=MU2, sigma2=SIGMA2)
    deficit = sct['conservation_deficit']
    print(f'    Conservation deficit:      {deficit:.3e}')

    # Also compute Δβ₁/N
    A_mat = np.diag(np.diag(L)) - L
    E_count = int(np.sum(A_mat > 1e-10)) // 2
    E_null = K_NN * N // 2
    delta_beta1 = beta1
    delta_beta1_per_n = delta_beta1 / N if N > 0 else 0

    # ── Summary ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('BISHOP FAULT SCARP — ABIOTIC NULL RESULT')
    print('=' * 70)
    print(f'  N_raw          = {n_raw:,}')
    print(f'  N_graph        = {N}')
    print(f'  k-NN           = {K_NN},  sigma = {SIGMA_M} m')
    print(f'  H[h*]          = {H_star:.4f} nats')
    print(f'  H[h0]          = {H_vac:.4f} nats')
    print(f'  DeltaH         = {delta_H:.4f} nats')
    print(f'  Delta\'         = {delta_prime:.4f}')
    print(f'  beta0          = {beta0}')
    print(f'  beta1          = {beta1}')
    print(f'  Dbeta1/N       = {delta_beta1_per_n:.4f}')
    print(f'  Edges          = {E_count}')
    print(f'  Converged      = {info["converged"]}')
    print('=' * 70)

    results = dict(
        n_raw=n_raw, N=N, H_star=H_star, H_vac=H_vac, delta_H=delta_H,
        delta_prime=delta_prime, deficit=deficit,
        beta0=beta0, beta1=beta1, delta_beta1_per_n=delta_beta1_per_n,
        eigvals=eigvals, h_star=h_star, h0=h0, xy=xy_sub, xy_all=xy,
        elevs=elevs_all, elevs_sub=elevs_sub,
        elevs_valid=elevs_valid, xy_with_elev=xy_with_elev,
        lonlat=lonlat, E_count=E_count,
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def make_figures(r: dict):
    print('\n[9] Generating figures...')
    xy_all = r['xy_all']
    xy = r['xy']
    eigvals = r['eigvals']
    h_star = r['h_star']
    h0 = r['h0']

    xy_elev_all = r['xy_with_elev']
    elevs_valid = r['elevs_valid']
    elevs_sub = r['elevs_sub']

    # ── Fig 1: Rock centroid map coloured by elevation ────────────────────
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    sc0 = ax.scatter(xy_elev_all[:, 0], xy_elev_all[:, 1], s=0.3,
                     c=elevs_valid, cmap='jet', alpha=0.5, rasterized=True)
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_title(f'All rocks with elevation (N={len(xy_elev_all):,})',
                 fontweight='bold')
    ax.set_aspect('equal')
    cb0 = fig1.colorbar(sc0, ax=ax, shrink=0.75, pad=0.02)
    cb0.set_label('Relative elevation', fontsize=8)

    ax = axes[1]
    sc1 = ax.scatter(xy[:, 0], xy[:, 1], s=4,
                     c=elevs_sub, cmap='jet', alpha=0.7, edgecolors='none')
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_title(f'Subsampled (N={len(xy):,})', fontweight='bold')
    ax.set_aspect('equal')
    cb1 = fig1.colorbar(sc1, ax=ax, shrink=0.75, pad=0.02)
    cb1.set_label('Relative elevation', fontsize=8)

    fig1.suptitle('Bishop Fault Scarp — Rock Centroids Coloured by Elevation\n'
                  '760 ka Bishop Tuff, Volcanic Tablelands, CA  |  Abiotic null',
                  fontsize=11, fontweight='bold')
    fig1.tight_layout()
    out1 = FIG_DIR / 'fig1_bishop_centroids.png'
    fig1.savefig(out1, dpi=200)
    plt.close(fig1)
    print(f'    Saved {out1.name}')

    # ── Fig 2: Eigenspectrum ──────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(7, 4))
    n_modes = min(100, len(eigvals))
    ax.semilogy(range(n_modes), eigvals[:n_modes] + 1e-12, 'o-',
                color='#D55E00', markersize=3, linewidth=1)
    ax.set_xlabel('Mode index l')
    ax.set_ylabel('Eigenvalue λ_l')
    ax.set_title('Laplacian eigenspectrum (first 100 modes)\n'
                 'Bishop Fault Scarp — Abiotic null', fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = FIG_DIR / 'fig2_bishop_eigenspectrum.png'
    fig2.savefig(out2, dpi=200)
    plt.close(fig2)
    print(f'    Saved {out2.name}')

    # ── Fig 3: Fixed-point kernel vs vacuum ───────────────────────────────
    fig3, ax = plt.subplots(figsize=(7, 4))
    n_modes = min(100, len(h_star))
    h_bar = h_star / h_star.sum()
    h0_bar = h0 / h0.sum()
    ax.semilogy(range(n_modes), h_bar[:n_modes], 'o-',
                color='#009E73', markersize=3, linewidth=1.2,
                label=f'h* (ΔH = {r["delta_H"]:.3f} nats)')
    ax.semilogy(range(n_modes), h0_bar[:n_modes], '--',
                color='#CC79A7', linewidth=1,
                label='vacuum h₀ = exp(−λ)')
    ax.fill_between(range(n_modes), h0_bar[:n_modes], h_bar[:n_modes],
                    where=h_bar[:n_modes] > h0_bar[:n_modes],
                    alpha=0.2, color='#009E73', label='amplified')
    ax.fill_between(range(n_modes), h0_bar[:n_modes], h_bar[:n_modes],
                    where=h_bar[:n_modes] < h0_bar[:n_modes],
                    alpha=0.2, color='#CC79A7', label='suppressed')
    ax.set_xlabel('Mode index l (sorted by eigenvalue)')
    ax.set_ylabel('Normalised spectral weight')
    ax.set_title('Fixed-point kernel h*(λ) vs vacuum\n'
                 'Bishop Fault Scarp — Abiotic null', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig3.tight_layout()
    out3 = FIG_DIR / 'fig3_bishop_kernel.png'
    fig3.savefig(out3, dpi=200)
    plt.close(fig3)
    print(f'    Saved {out3.name}')

    # ── Fig 4: Summary comparison card ────────────────────────────────────
    fig4, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    systems = [
        ('Bishop scarp\n(abiotic null)', r['delta_H'], r['delta_beta1_per_n'],
         '#D55E00', 'D'),
        ('Bobcat Fire\n(ctrl removed)', None, None, '#E69F00', 's'),
        ('Cities\n(active ctrl)', None, None, '#009E73', 'o'),
    ]

    # BF and city data from field notes
    bf_dH = 0.230  # temporal delta, not comparable to absolute
    city_dH_range = (-0.34, -0.24)
    city_db1n_range = (0.19, 0.61)

    ax.barh([2], [abs(r['delta_H'])], color='#D55E00', height=0.5,
            label=f'Bishop |ΔH| = {abs(r["delta_H"]):.3f}')
    ax.barh([1], [abs(city_dH_range[0])], color='#009E73', height=0.5,
            alpha=0.7, label=f'Cities |ΔH| = {abs(city_dH_range[0]):.2f}–{abs(city_dH_range[1]):.2f}')
    ax.barh([1], [abs(city_dH_range[1])], color='#009E73', height=0.5, alpha=0.3)

    ax.set_yticks([2, 1])
    ax.set_yticklabels(['Bishop scarp\n(abiotic null)', 'Cities\n(active controller)'],
                       fontsize=9)
    ax.set_xlabel('|ΔH| (nats)', fontsize=10)
    ax.set_title('Controller Hierarchy: Spectral Concentration |ΔH|\n'
                 'Bishop = abiotic floor  |  Cities = active controller tier',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0, 0.45)
    ax.grid(True, axis='x', alpha=0.3)

    fig4.tight_layout()
    out4 = FIG_DIR / 'fig4_bishop_comparison.png'
    fig4.savefig(out4, dpi=200)
    plt.close(fig4)
    print(f'    Saved {out4.name}')

    # ── Fig 5: Elevation distribution ─────────────────────────────────────
    elevs = r['elevs']
    valid_e = elevs[~np.isnan(elevs)]
    if len(valid_e) > 100:
        fig5, ax = plt.subplots(figsize=(7, 4))
        ax.hist(valid_e, bins=80, color='#0072B2', alpha=0.7, edgecolor='white',
                linewidth=0.3)
        ax.set_xlabel('Relative elevation (colormap units)')
        ax.set_ylabel('Count')
        ax.set_title('Rock elevation distribution\n'
                     'Bishop Fault Scarp — 760 ka Bishop Tuff', fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig5.tight_layout()
        out5 = FIG_DIR / 'fig5_bishop_elevation.png'
        fig5.savefig(out5, dpi=200)
        plt.close(fig5)
        print(f'    Saved {out5.name}')

    print(f'\nAll figures → {FIG_DIR}/')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results = run_analysis()
    make_figures(results)
    print('\nDone.')
