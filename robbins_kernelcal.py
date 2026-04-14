#!/usr/bin/env python3
"""
robbins_kernelcal.py
====================
Apply the kernelcal / MaxCal spectral-kernel framework to the Robbins (2018/2019)
global lunar crater catalog (1.3 M craters, D ≥ 1 km).

Five sub-analyses (each N ≈ 2 000 craters):
  A  Global random sample, D ≥ 5 km
  B  Northern highlands patch  (lat 30–80°, D ≥ 1 km)
  C  Southern highlands patch  (lat −80 – −30°, D ≥ 1 km)
  D  Near-side equatorial belt  (lat −15–15°, lon 280–360°, D ≥ 1 km)
  E  Large-crater global sample  (D ≥ 20 km, all ~7 000 craters)

Outputs (lunar_figures/ and P4 paper figures/):
  fig_robbins_globalmap.png          — global scatter coloured by diameter
  fig_robbins_sfd.png                — multi-panel SFD + spatial stats
  fig_robbins_kernelcal.png          — per-region eigenspectra + h*(λ) grid
  fig_robbins_phasespace.png         — Δ H × Δβ₁/N adding all Robbins regions
  fig_robbins_phasespace.pdf         — paper-quality version (→ P4 figures/)

Usage:
  python3 robbins_kernelcal.py [--csv /path/to/robbins.csv]
"""

from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import eigh as scipy_eigh
from scipy.stats import linregress, rayleigh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

KCAL_ROOT = Path(__file__).parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import (
    spectral_entropy, fixed_point_kernel, fiedler_mode_gap,
)

# ── paths ────────────────────────────────────────────────────────────────────
DEFAULT_CSV = Path.home() / 'Downloads' / 'lunar_crater_database_robbins_2018.csv'
FIG_DIR     = KCAL_ROOT / 'lunar_figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG   = (KCAL_ROOT.parent /
               'journal-spectral-kernel-biosignature-planetary-surfaces-p4' /
               'figures')

# ── constants ────────────────────────────────────────────────────────────────
R_MOON = 1_737_400.0          # metres
K_NN   = 8
MU2    = 2.0
SIGMA2 = 1.0
N_TARGET = 2000
RNG    = np.random.default_rng(42)

# Wong 2011 palette
C = dict(
    global5='#56B4E9', north='#009E73', south='#E69F00',
    equat='#CC79A7', large='#D55E00',
    abio='#0072B2', fossil='#E69F00', active='#009E73',
    lunar='#CC79A7', null='#BBBBBB',
)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'legend.fontsize': 7.5, 'legend.framealpha': 0.95,
    'legend.edgecolor': '#cccccc',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'lines.linewidth': 1.2,
})

# ── region definitions ───────────────────────────────────────────────────────
REGIONS = [
    dict(key='global5',  label='Global  D≥5 km',
         lat_min=-90, lat_max=90, lon_min=0, lon_max=360,
         diam_min=5,  n=N_TARGET, color=C['global5'], marker='o'),
    dict(key='north',    label='N. Highlands  D≥1 km',
         lat_min=30,  lat_max=80, lon_min=0, lon_max=360,
         diam_min=1,  n=N_TARGET, color=C['north'],   marker='s'),
    dict(key='south',    label='S. Highlands  D≥1 km',
         lat_min=-80, lat_max=-30, lon_min=0, lon_max=360,
         diam_min=1,  n=N_TARGET, color=C['south'],   marker='^'),
    dict(key='equat',    label='Near-side equatorial  D≥1 km',
         lat_min=-15, lat_max=15,  lon_min=280, lon_max=360,
         diam_min=1,  n=N_TARGET, color=C['equat'],   marker='D'),
    dict(key='large',    label='Global  D≥20 km  (full)',
         lat_min=-90, lat_max=90, lon_min=0, lon_max=360,
         diam_min=20, n=None,     color=C['large'],   marker='P'),
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_catalog(csv_path: Path) -> np.ndarray:
    """
    Read the Robbins CSV and return float32 array shaped (N, 3):
    columns = [lat_deg, lon_deg, diam_km].
    Uses stdlib csv to avoid pandas/numpy version conflicts.
    """
    print(f'  Reading {csv_path} …', flush=True)
    rows = []
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                lat  = float(r['LAT_CIRC_IMG'])
                lon  = float(r['LON_CIRC_IMG'])
                diam = float(r['DIAM_CIRC_IMG'])
                rows.append((lat, lon, diam))
            except (ValueError, KeyError):
                continue
    arr = np.array(rows, dtype=np.float32)
    print(f'  Loaded {len(arr):,} craters  '
          f'(D: {arr[:,2].min():.2f}–{arr[:,2].max():.0f} km)')
    return arr


def filter_region(catalog: np.ndarray, reg: dict) -> np.ndarray:
    """Return craters matching the region's lat/lon/diam bounds."""
    lat, lon, diam = catalog[:,0], catalog[:,1], catalog[:,2]
    mask = (
        (lat  >= reg['lat_min'])  & (lat  <= reg['lat_max']) &
        (lon  >= reg['lon_min'])  & (lon  <= reg['lon_max']) &
        (diam >= reg['diam_min'])
    )
    sub = catalog[mask]
    if reg['n'] is not None and len(sub) > reg['n']:
        idx = RNG.choice(len(sub), reg['n'], replace=False)
        sub = sub[idx]
    print(f'  [{reg["key"]:8s}]  {len(sub):,} craters  '
          f'(D≥{reg["diam_min"]} km, lat {reg["lat_min"]}–{reg["lat_max"]})')
    return sub


def latlon_to_xy(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """
    Simple equirectangular projection → lunar metres.
    Good for local patches; for global we use 3-D Cartesian.
    """
    lat_r = np.radians(lat_deg)
    lon_r = np.radians(lon_deg)
    x = R_MOON * lon_r * np.cos(lat_r)
    y = R_MOON * lat_r
    return np.column_stack([x, y])


def latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Unit-sphere Cartesian × R_MOON (for global k-NN with great-circle metric)."""
    lat_r = np.radians(lat_deg)
    lon_r = np.radians(lon_deg)
    x = R_MOON * np.cos(lat_r) * np.cos(lon_r)
    y = R_MOON * np.cos(lat_r) * np.sin(lon_r)
    z = R_MOON * np.sin(lat_r)
    return np.column_stack([x, y, z])


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH + KERNELCAL
# ══════════════════════════════════════════════════════════════════════════════

def run_kernelcal(sub: np.ndarray, global_knn: bool = False) -> dict:
    """Build k-NN graph and compute all kernelcal diagnostics."""
    lat, lon, diam = sub[:,0], sub[:,1], sub[:,2]
    n = len(sub)

    if global_knn:
        coords = latlon_to_xyz(lat, lon)
    else:
        coords = latlon_to_xy(lat, lon)

    tree   = cKDTree(coords)
    dists, inds = tree.query(coords, k=K_NN + 1)
    med_nn = float(np.median(dists[:, 1]))

    xrange = coords[:, 0].max() - coords[:, 0].min()
    yrange = coords[:, 1].max() - coords[:, 1].min()
    diag   = math.hypot(xrange, yrange)
    sigma  = max(0.05 * diag, 2 * max(med_nn, 1e3))

    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for ji in range(1, K_NN + 1):
            j = inds[i, ji]; d = dists[i, ji]
            w = math.exp(-d**2 / sigma**2)
            if w > W[i, j]:
                W[i, j] = w; W[j, i] = w

    L  = np.diag(W.sum(1)) - W
    ev, evec = scipy_eigh(L)
    ev = np.maximum(ev, 0.0)

    n_zero  = int(np.sum(ev < 1e-6))
    w_modes = ev.copy()
    w_modes[:n_zero] = ev[n_zero] if n_zero < n else 1e-3
    h0     = np.maximum(np.exp(-ev), 1e-10)
    h_star, _ = fixed_point_kernel(L, h0=h0, mu2=MU2, sigma2=SIGMA2, w=w_modes)
    h_star = np.maximum(h_star, 1e-8)

    H_obs = spectral_entropy(h_star)
    H_vac = spectral_entropy(h0)
    dH    = H_obs - H_vac
    dp    = fiedler_mode_gap(h_star, L, mu2=MU2, sigma2=SIGMA2, w=w_modes)
    lam_f = float(ev[n_zero]) if n_zero < n else 0.0
    n_edges = int((W > 0).sum()) // 2
    beta0   = n_zero
    beta1   = max(0, n_edges - (n - beta0))
    db1     = beta1 - max(0, K_NN * n // 2 - (n - 1))

    return dict(
        n=n, ev=ev, evec=evec, h0=h0, h_star=h_star,
        H_obs=H_obs, H_vac=H_vac, dH=dH, dp=dp,
        beta0=beta0, beta1=beta1, db1=db1, db1_N=db1/n,
        lam_f=lam_f, n_edges=n_edges,
        coords=coords, diam=diam, inds=inds, W=W,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Global map of all 1.3M craters
# ══════════════════════════════════════════════════════════════════════════════

def fig_global_map(catalog: np.ndarray):
    """Mollweide projection scatter of all craters coloured by diameter."""
    fig = plt.figure(figsize=(12, 6))
    ax  = fig.add_subplot(111, projection='mollweide')

    # Work with radians for mollweide
    lat_r = np.radians(catalog[:, 0])
    # Mollweide expects lon in [-π, π]; Robbins uses [0, 360]
    lon_deg = catalog[:, 1].copy()
    lon_deg[lon_deg > 180] -= 360
    lon_r = np.radians(lon_deg)
    diam  = catalog[:, 2]

    # Plot in size bins for visual clarity
    bins = [(1, 3), (3, 10), (10, 50), (50, 200), (200, 3000)]
    labels = ['1–3 km', '3–10 km', '10–50 km', '50–200 km', '>200 km']
    sizes  = [0.4, 1.5, 5, 15, 40]
    alphas = [0.12, 0.30, 0.55, 0.80, 1.0]
    cmap   = plt.cm.plasma_r
    norm   = LogNorm(vmin=1, vmax=2500)

    for (lo, hi), sz, al in zip(bins, sizes, alphas):
        m = (diam >= lo) & (diam < hi)
        if m.sum() == 0:
            continue
        ax.scatter(lon_r[m], lat_r[m],
                   s=sz, c=diam[m], cmap=cmap, norm=norm,
                   alpha=al, linewidths=0, rasterized=True)

    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                        pad=0.05, fraction=0.03, aspect=40)
    cbar.set_label('Crater diameter  $D$  (km)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel('Longitude  (°)', fontsize=9)
    ax.set_ylabel('Latitude  (°)', fontsize=9)
    ax.grid(True, color='white', linewidth=0.3, alpha=0.5)
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('white')

    fig.suptitle(
        f'Robbins (2018/2019) Lunar Crater Database  ·  {len(catalog):,} craters  ·  '
        f'$D \\geq 1$ km  ·  Mollweide projection',
        fontsize=11, fontweight='bold', y=1.01)

    out = FIG_DIR / 'fig_robbins_globalmap.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — SFD + per-region spatial statistics (2-row grid)
# ══════════════════════════════════════════════════════════════════════════════

def fig_sfd(catalog: np.ndarray, results: list):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    fig.subplots_adjust(wspace=0.38)

    # ── A: Global SFD log-log ────────────────────────────────────────────────
    ax = axes[0]
    diam_all = catalog[:, 2]
    bins = np.logspace(np.log10(1), np.log10(2500), 60)
    counts, edges = np.histogram(diam_all, bins=bins)
    centres = np.sqrt(edges[:-1] * edges[1:])
    nz = counts > 0
    ax.scatter(centres[nz], counts[nz], s=18, color='#0072B2',
               edgecolors='none', alpha=0.8, zorder=5, label='All craters')
    # cumulative count
    ax2t = ax.twinx()
    cumul = np.cumsum(counts[::-1])[::-1]
    ax2t.plot(centres, cumul, '-', color='#D55E00', lw=1.5,
              label='Cumulative N(>D)', alpha=0.7)
    ax2t.set_ylabel('Cumulative N($>$D)', fontsize=8, color='#D55E00')
    ax2t.tick_params(axis='y', labelcolor='#D55E00', labelsize=7)
    ax2t.set_yscale('log')

    # Power-law fit on differential (completeness range: 1–20 km)
    mask_fit = nz & (centres >= 1) & (centres <= 20)
    if mask_fit.sum() > 3:
        slope, intercept, r, *_ = linregress(
            np.log10(centres[mask_fit]), np.log10(counts[mask_fit]))
        xf = np.linspace(np.log10(1), np.log10(30), 100)
        ax.plot(10**xf, 10**(intercept + slope*xf),
                '--', color='#CC79A7', lw=2.0,
                label=f'SFD slope = {slope:.2f}')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Crater diameter  $D$  (km)')
    ax.set_ylabel('Count per bin')
    ax.set_title('(a)  Global size-frequency distribution\n'
                 '(differential + cumulative)',
                 fontweight='bold', pad=4)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2t.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=6.5, loc='upper right')

    # ── B: ΔH per region bar chart ───────────────────────────────────────────
    ax3 = axes[1]
    labels  = [r['region'] for r in results]
    dH_vals = [r['dH']    for r in results]
    cols    = [r['color']  for r in results]
    xb = np.arange(len(results))
    bars = ax3.barh(xb, dH_vals, color=cols, edgecolor='black',
                    linewidth=0.5, height=0.6)
    ax3.axvline(0, color='black', lw=0.8)
    for xi, v in enumerate(dH_vals):
        ax3.text(v - 0.005 if v < 0 else v + 0.005,
                 xi, f'{v:+.3f}', va='center',
                 ha='right' if v < 0 else 'left', fontsize=7)
    ax3.set_yticks(xb)
    ax3.set_yticklabels(labels, fontsize=7.5)
    ax3.set_xlabel(r'$\Delta H = H[h^*] - H[h_0]$  (nats)')
    ax3.set_title('(b)  Spectral entropy shift  $\\Delta H$\nper region / size stratum',
                  fontweight='bold', pad=4)
    # Reference lines from other systems
    ax3.axvline(-0.377, color='#E69F00', lw=1.2, ls='--', alpha=0.7)
    ax3.text(-0.377, len(results)-0.1, 'Jezero', color='#E69F00',
             fontsize=6.5, ha='center', va='top')
    ax3.axvline(-0.667, color='#CC79A7', lw=1.2, ls=':', alpha=0.7)
    ax3.text(-0.667, len(results)-0.1, 'Cities', color='#CC79A7',
             fontsize=6.5, ha='center', va='top')

    # ── C: Δβ₁/N per region ──────────────────────────────────────────────────
    ax4 = axes[2]
    db1_vals = [r['db1_N'] for r in results]
    bars2 = ax4.barh(xb, db1_vals, color=cols, edgecolor='black',
                     linewidth=0.5, height=0.6)
    ax4.axvline(0, color='black', lw=0.8)
    for xi, v in enumerate(db1_vals):
        ax4.text(v + 0.01, xi, f'{v:+.3f}', va='center', fontsize=7)
    ax4.set_yticks(xb)
    ax4.set_yticklabels(labels, fontsize=7.5)
    ax4.set_xlabel(r'$\Delta\beta_1 / N$')
    ax4.set_title('(c)  Normalised topological excess  $\\Delta\\beta_1/N$\nper region',
                  fontweight='bold', pad=4)
    ax4.axvline(0.561, color='#CC79A7', lw=1.2, ls=':', alpha=0.7)
    ax4.text(0.561, len(results)-0.1, 'Cities', color='#CC79A7',
             fontsize=6.5, ha='center', va='top')

    fig.suptitle(
        'Robbins Lunar Crater Database — Regional kernelcal Diagnostics',
        fontsize=10, fontweight='bold', y=1.02)

    out = FIG_DIR / 'fig_robbins_sfd.png'
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Per-region eigenspectrum + h*(λ) grid (5-region × 2-panel)
# ══════════════════════════════════════════════════════════════════════════════

def fig_kernelcal_grid(results: list):
    nreg = len(results)
    fig, axes = plt.subplots(nreg, 2, figsize=(8.5, 2.0 * nreg))
    fig.subplots_adjust(hspace=0.55, wspace=0.35)

    for ri, r in enumerate(results):
        ev = r['ev']; h0s = r['h0']; hs = r['h_star']
        col = r['color']
        n_show = min(80, len(ev))
        sort = np.argsort(ev)
        lam  = ev[sort][:n_show]
        h0_s = h0s[sort][:n_show]
        hs_s = hs[sort][:n_show]

        # Eigenspectrum
        axL = axes[ri, 0]
        axL.bar(range(n_show), lam, color=col, width=1.0,
                edgecolor='none', alpha=0.75)
        axL.set_ylabel('$\\lambda_l$', fontsize=7)
        axL.tick_params(labelsize=6.5)
        if ri == nreg - 1:
            axL.set_xlabel('Mode index  $l$', fontsize=8)
        axL.set_title(f'{r["region"]}\neigenspectrum', fontsize=7.5,
                      fontweight='bold', pad=2, color=col)
        axL.text(0.98, 0.95,
                 f'$\\lambda_f={r["lam_f"]:.4f}$\n$\\beta_0={r["beta0"]}$',
                 transform=axL.transAxes, ha='right', va='top', fontsize=6.5,
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc'))

        # h*(λ) vs h₀
        axR = axes[ri, 1]
        axR.fill_between(lam, h0_s, hs_s, where=(hs_s >= h0_s),
                         color='#009E73', alpha=0.25)
        axR.fill_between(lam, h0_s, hs_s, where=(hs_s < h0_s),
                         color='#D55E00', alpha=0.25)
        axR.plot(lam, h0_s, '--', color='#888888', lw=1.0, label='$h_0$')
        axR.plot(lam, hs_s, '-',  color=col,       lw=1.5, label='$h^*$')
        axR.set_ylabel('$h(\\lambda)$', fontsize=7)
        axR.tick_params(labelsize=6.5)
        if ri == nreg - 1:
            axR.set_xlabel('$\\lambda_l$', fontsize=8)
        axR.set_title(f'{r["region"]}\n$h^*$ vs $h_0$', fontsize=7.5,
                      fontweight='bold', pad=2, color=col)
        axR.text(0.04, 0.06,
                 f'$\\Delta H={r["dH"]:+.3f}$',
                 transform=axR.transAxes, va='bottom', fontsize=8,
                 color=col, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc'))
        axR.legend(fontsize=6, loc='upper right')

    fig.suptitle(
        'Robbins Catalog — kernelcal Diagnostics per Region/Stratum\n'
        '(k = 8 nearest-neighbour graph  ·  MaxCal fixed-point kernel)',
        fontsize=9.5, fontweight='bold')
    out = FIG_DIR / 'fig_robbins_kernelcal.png'
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Master phase space (all systems, Robbins regions added)
# ══════════════════════════════════════════════════════════════════════════════

def fig_phase_space(robbins_results: list):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    fig.subplots_adjust(wspace=0.42)

    # ── prior systems ────────────────────────────────────────────────────────
    PRIOR = [
        dict(label='AZ Plateau',    dH=-0.392, db1_N=221/1500,
             color=C['abio'],  marker='o', s=70,  tier='abiotic'),
        dict(label='Jezero delta',  dH=-0.377, db1_N=960/2000,
             color=C['fossil'], marker='s', s=70, tier='fossil'),
        dict(label='Barcelona',     dH=-0.670, db1_N=1060/678,
             color='#CC79A7',  marker='D', s=50,  tier='active'),
        dict(label='Phoenix',       dH=-0.691, db1_N=470/678,
             color='#56B4E9',  marker='D', s=50,  tier='active'),
        dict(label='Venice',        dH=-0.674, db1_N=1513/678,
             color='#D55E00',  marker='D', s=50,  tier='active'),
        dict(label='Marrakech',     dH=-0.619, db1_N=1490/678,
             color='#F0E442',  marker='D', s=50,  tier='active'),
        dict(label='Houston',       dH=-0.629, db1_N=840/678,
             color='#999999',  marker='D', s=50,  tier='active'),
        dict(label='LROC MaskRCNN', dH=-0.667, db1_N=0.561,
             color=C['lunar'],  marker='*', s=100, tier='lroc'),
    ]

    ax = axes[0]
    # Tier background bands
    ax.axhspan(-0.20, 0.35,  color='#EFF7FF', alpha=0.55, zorder=0)
    ax.axhspan(0.35,  0.65,  color='#FFF8E8', alpha=0.55, zorder=0)
    ax.axhspan(0.65,  2.50,  color='#EFFFEF', alpha=0.55, zorder=0)
    ax.text(-0.78, 0.02,  'Abiotic',         fontsize=7, color=C['abio'],  style='italic')
    ax.text(-0.78, 0.38,  'Fossil ctrl.',     fontsize=7, color=C['fossil'],style='italic')
    ax.text(-0.78, 1.80,  'Active ctrl.',     fontsize=7, color=C['active'],style='italic')

    # Poisson null
    ax.scatter(-0.718, 0.0, s=35, color=C['null'], marker='x',
               linewidths=1.5, zorder=4)
    ax.annotate('Poisson null', (-0.718, 0.04), fontsize=6.5, color='#888888')

    # Prior systems
    for p in PRIOR:
        ax.scatter(p['dH'], p['db1_N'], s=p['s'], color=p['color'],
                   marker=p['marker'], edgecolors='black', linewidths=0.6, zorder=5)

    # Robbins regions — clustered ellipse region
    for r in robbins_results:
        ax.scatter(r['dH'], r['db1_N'], s=90, color=r['color'],
                   marker='h', edgecolors='black', linewidths=0.8, zorder=7)

    # Annotation: Robbins centroid region
    rH  = np.mean([r['dH']    for r in robbins_results])
    rDB = np.mean([r['db1_N'] for r in robbins_results])
    ax.annotate(
        f'Robbins catalog\n(5 regions, mean ΔH={rH:+.3f})',
        (rH, rDB), xytext=(20, -30), textcoords='offset points',
        fontsize=7.5, color='black', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='black', lw=0.9))

    ax.axhline(0, color='#cccccc', lw=0.6, ls=':')
    ax.axvline(0, color='#cccccc', lw=0.6, ls=':')
    ax.set_xlim(-0.82, 0.10)
    ax.set_ylim(-0.20, 2.55)
    ax.set_xlabel(r'$\Delta H = H[h^*] - H[h_0]$  (nats)')
    ax.set_ylabel(r'$\Delta\beta_1 / N$')
    ax.set_title('(a)  Full Controller Phase Space\n'
                 '(hexagons = Robbins catalog regions)',
                 fontweight='bold', pad=5)

    # Legend
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0],[0], marker='h', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=9,
               label='Robbins — 5 regions'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C['abio'],
               markeredgecolor='black', markersize=7, label='AZ Plateau'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=C['fossil'],
               markeredgecolor='black', markersize=7, label='Jezero (fossil)'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#888888',
               markeredgecolor='black', markersize=7, label='Cities (active)'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor=C['lunar'],
               markeredgecolor='black', markersize=9, label='LROC MaskRCNN'),
        Line2D([0],[0], marker='x', color=C['null'], markersize=7,
               label='Poisson null'),
    ]
    ax.legend(handles=leg, fontsize=6.5, loc='lower left', framealpha=0.97)

    # ── Bar chart: all systems ranked by Δβ₁/N ──────────────────────────────
    ax2 = axes[1]
    all_sys = []
    for r in robbins_results:
        all_sys.append(dict(label=r['region'], dH=r['dH'],
                            db1_N=r['db1_N'], color=r['color'], tier='abiotic'))
    all_sys += PRIOR
    all_sys.append(dict(label='Poisson null', dH=-0.718, db1_N=0.0,
                        color=C['null'], tier='null'))
    all_sys.sort(key=lambda x: x['db1_N'])

    xb    = np.arange(len(all_sys))
    labs  = [s['label']  for s in all_sys]
    vals  = [s['db1_N']  for s in all_sys]
    cols2 = [s['color']  for s in all_sys]
    htch  = {'abiotic': '/', 'fossil': 'x', 'active': '', 'lroc': '.', 'null': ''}
    bars  = ax2.barh(xb, vals, color=cols2, edgecolor='black',
                     linewidth=0.5, height=0.72)
    for bar, s in zip(bars, all_sys):
        bar.set_hatch(htch.get(s.get('tier',''), ''))
    ax2.axvline(0, color='black', lw=0.7)
    for xi, v in enumerate(vals):
        ax2.text(max(v, 0) + 0.02, xi, f'{v:.3f}', va='center', fontsize=6.5)
    ax2.set_yticks(xb)
    ax2.set_yticklabels(labs, fontsize=7)
    ax2.set_xlabel(r'$\Delta\beta_1 / N$  (normalised topological excess)')
    ax2.set_title(r'(b)  $\Delta\beta_1/N$ ranked — all systems',
                  fontweight='bold', pad=5)

    fig.suptitle(
        'Multi-System Spectral Kernel Biosignature Survey\n'
        'Robbins Lunar Catalog  ·  LROC MaskRCNN  ·  AZ Plateau  ·  Jezero  ·  5 Cities',
        fontsize=10, fontweight='bold', y=1.03)

    for out_dir, name, dpi in [
        (FIG_DIR,   'fig_robbins_phasespace.png', 200),
        (PAPER_FIG, 'fig_robbins_phasespace.pdf', 300),
        (PAPER_FIG, 'fig_robbins_phasespace.png', 300),
    ]:
        fig.savefig(out_dir / name, dpi=dpi)
        print(f'  Saved {name}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=str(DEFAULT_CSV))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    print('=' * 65)
    print('Robbins (2018/2019) Lunar Crater Database — kernelcal analysis')
    print(f'  CSV: {csv_path}')
    print(f'  Out: {FIG_DIR}')
    print('=' * 65)

    print('\n[0] Loading catalog …')
    catalog = load_catalog(csv_path)

    print('\n[1] Global map …')
    fig_global_map(catalog)

    print('\n[2] Running kernelcal on 5 regions …')
    results = []
    for reg in REGIONS:
        sub  = filter_region(catalog, reg)
        # Use 3-D coords for global samples, 2-D for regional patches
        glob = reg['lat_min'] == -90 and reg['lat_max'] == 90
        gd   = run_kernelcal(sub, global_knn=glob)
        results.append(dict(
            region=reg['label'], color=reg['color'], marker=reg['marker'],
            **{k: gd[k] for k in [
               'n','ev','h0','h_star','dH','db1_N','beta0','beta1',
               'db1','lam_f','H_obs','H_vac','dp','coords','diam',
               'inds','W']},
        ))
        print(f'    ΔH={gd["dH"]:+.4f}  Δβ₁/N={gd["db1_N"]:+.4f}  '
              f'β₀={gd["beta0"]}  λ_f={gd["lam_f"]:.5f}')

    print('\n  Summary table:')
    print(f'  {"Region":35s}  {"N":>6}  {"ΔH":>8}  {"Δβ₁/N":>8}  {"β₀":>5}')
    print('  ' + '-'*72)
    for r in results:
        print(f'  {r["region"]:35s}  {r["n"]:6d}  {r["dH"]:+8.4f}  '
              f'{r["db1_N"]:+8.4f}  {r["beta0"]:5d}')

    print('\n[3] SFD + bar charts …')
    fig_sfd(catalog, results)

    print('\n[4] Per-region kernelcal grid …')
    fig_kernelcal_grid(results)

    print('\n[5] Master phase space …')
    fig_phase_space(results)

    print(f'\nAll figures → {FIG_DIR}')
    print('Paper figures → journal-spectral-kernel-biosignature-planetary-surfaces-p4/figures/')


if __name__ == '__main__':
    main()
