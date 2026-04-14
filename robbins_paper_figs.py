#!/usr/bin/env python3
"""
robbins_paper_figs.py
=====================
Publication-quality figures for the Robbins (2019) lunar crater
k-NN methodological null analysis.  White backgrounds throughout;
Wong 2011 colour palette; journal rcParams.

Figures produced (→ P4 paper figures/ and lunar_paper_figures/):
  fig_robbins_global_map.png     — global crater map, all 1.3 M craters
  fig_robbins_regions.png        — 5 regional sub-samples (spatial layout)
  fig_robbins_sfd.png            — size-frequency distribution
  fig_robbins_eigenspectra.png   — Laplacian eigenspectra, 5 regions
  fig_robbins_kernels.png        — fixed-point kernels h*(λ), 5 regions
  fig_robbins_phasespace.png     — phase space (Robbins null vs AZ + cities)
"""

from __future__ import annotations
import csv, math, sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import eigh as scipy_eigh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

KCAL_ROOT = Path(__file__).parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import spectral_entropy, fixed_point_kernel

CSV_PATH = Path.home() / 'Downloads' / 'lunar_crater_database_robbins_2018.csv'
OUT_DIR  = KCAL_ROOT / 'lunar_paper_figures'
PAPER    = (KCAL_ROOT.parent /
            'journal-spectral-kernel-biosignature-planetary-surfaces-p4' / 'figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

R_MOON  = 1_737_400.0
K_NN    = 8
MU2     = 2.0
SIGMA2  = 1.0
N_SAMP  = 2000
RNG     = np.random.default_rng(42)

# Wong 2011 palette
W = dict(
    global5='#56B4E9', north='#009E73', south='#E69F00',
    equat='#CC79A7',   large='#D55E00',
    az='#009E73',      city='#0072B2',   jez='#CC79A7',
)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'legend.framealpha': 0.97,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'figure.dpi': 150, 'savefig.dpi': 250, 'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

REGIONS = [
    dict(key='global5', label='Global D≥5 km',
         lat=(-90,90), lon=(0,360),   dmin=5,  n=N_SAMP, color=W['global5'], marker='o'),
    dict(key='north',   label='N. Highlands D≥1 km',
         lat=(30,80),  lon=(0,360),   dmin=1,  n=N_SAMP, color=W['north'],   marker='s'),
    dict(key='south',   label='S. Highlands D≥1 km',
         lat=(-80,-30),lon=(0,360),   dmin=1,  n=N_SAMP, color=W['south'],   marker='^'),
    dict(key='equat',   label='Near-side equatorial D≥1 km',
         lat=(-15,15), lon=(280,360), dmin=1,  n=N_SAMP, color=W['equat'],   marker='D'),
    dict(key='large',   label='Global D≥20 km',
         lat=(-90,90), lon=(0,360),   dmin=20, n=None,   color=W['large'],   marker='P'),
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_catalog() -> np.ndarray:
    print(f'  Reading Robbins catalog …', end='', flush=True)
    rows = []
    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row['LAT_CIRC_IMG'])
                lon = float(row['LON_CIRC_IMG'])
                d   = float(row['DIAM_CIRC_IMG'])
                if math.isfinite(lat) and math.isfinite(lon) and d > 0:
                    rows.append((lat, lon, d))
            except (ValueError, KeyError):
                pass
    arr = np.array(rows, dtype=np.float32)
    print(f' {len(arr):,} craters loaded')
    return arr  # (N, 3): lat, lon, diam_km

def filter_region(cat, reg) -> np.ndarray:
    lat_lo, lat_hi = reg['lat']
    lon_lo, lon_hi = reg['lon']
    mask = ((cat[:,0] >= lat_lo) & (cat[:,0] <= lat_hi) &
            (cat[:,1] >= lon_lo) & (cat[:,1] <= lon_hi) &
            (cat[:,2] >= reg['dmin']))
    sub = cat[mask]
    if reg['n'] is not None and len(sub) > reg['n']:
        idx = RNG.choice(len(sub), reg['n'], replace=False)
        sub = sub[idx]
    return sub

def latlon_to_xy(lat_deg, lon_deg) -> np.ndarray:
    """Equirectangular projection to metres centred on region."""
    lat_r = np.radians(lat_deg)
    lon_r = np.radians(lon_deg)
    lat0  = np.radians(np.mean(lat_deg))
    lon0  = np.radians(np.mean(lon_deg))
    x = R_MOON * (lon_r - lon0) * np.cos(lat0)
    y = R_MOON * (lat_r - lat0)
    return np.column_stack([x, y])


# ══════════════════════════════════════════════════════════════════════════════
# KERNELCAL
# ══════════════════════════════════════════════════════════════════════════════

def run_knn_kernelcal(xy, k=K_NN, label=''):
    n    = len(xy)
    tree = cKDTree(xy)
    dists, inds = tree.query(xy, k=k+1)
    med = float(np.median(dists[:,1]))
    xr  = xy[:,0].max()-xy[:,0].min(); yr = xy[:,1].max()-xy[:,1].min()
    sigma = max(0.05 * math.hypot(xr,yr), 2*max(med, 1e-3))

    W = np.zeros((n,n))
    for i in range(n):
        for ki in range(1, k+1):
            j = inds[i,ki]; d = dists[i,ki]
            w = math.exp(-d**2/sigma**2)
            if w > W[i,j]:
                W[i,j]=w; W[j,i]=w

    L      = np.diag(W.sum(1)) - W
    ev, _  = scipy_eigh(L)
    ev     = np.maximum(ev, 0.0)
    n_zero = int(np.sum(ev < 1e-6))
    wm     = ev.copy(); wm[:n_zero] = ev[n_zero] if n_zero < n else 1e-3
    h0     = np.maximum(np.exp(-ev), 1e-10)
    hs, _  = fixed_point_kernel(L, h0=h0, mu2=MU2, sigma2=SIGMA2, w=wm)
    hs     = np.maximum(hs, 1e-8)
    dH     = spectral_entropy(hs) - spectral_entropy(h0)
    n_edges = int((W>0).sum())//2
    beta0  = n_zero
    beta1  = max(0, n_edges - (n - beta0))
    print(f'  [{label:8s}]  ΔH={dH:+.3f}  β₀={beta0}  β₁={beta1}  '
          f'Δβ₁/N={beta1/n:+.4f}  n={n}')
    return dict(ev=ev, h0=h0, hs=hs, dH=dH, beta0=beta0,
                beta1=beta1, db1N=beta1/n, n=n, n_edges=n_edges)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Global lunar crater map
# ══════════════════════════════════════════════════════════════════════════════

def fig_global_map(cat):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.10)

    # All craters — tiny grey dots
    ax.scatter(cat[:,1], cat[:,0], s=0.02, c='#cccccc',
               linewidths=0, alpha=0.4, rasterized=True, zorder=1)

    # Highlight each region's sub-sample
    for reg in REGIONS:
        sub = filter_region(cat, reg)
        ax.scatter(sub[:,1], sub[:,0], s=2.5, c=reg['color'],
                   marker=reg['marker'], alpha=0.75, linewidths=0,
                   label=f'{reg["label"]}  (n={len(sub):,})', zorder=3)

    # Draw region bounding boxes
    bx_props = dict(linewidth=1.2, fill=False, zorder=4)
    for reg in REGIONS:
        lo, hi = reg['lon']
        la, lb = reg['lat']
        rect = plt.Rectangle((lo, la), hi-lo, lb-la,
                              edgecolor=reg['color'], **bx_props)
        ax.add_patch(rect)

    ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude (°E)', fontsize=10)
    ax.set_ylabel('Latitude (°N)', fontsize=10)
    ax.set_title('Robbins (2019) Lunar Crater Catalog — 1.3 M craters, $D \\geq 1$ km\n'
                 'Five sub-sample regions used for k-NN methodological null',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=7.5, markerscale=3, loc='lower right',
              framealpha=0.97, ncol=2)

    # Grid
    ax.set_xticks([0,60,120,180,240,300,360])
    ax.set_yticks([-90,-60,-30,0,30,60,90])
    ax.grid(True, color='#dddddd', lw=0.5, ls='--')
    ax.set_axisbelow(True)

    _save(fig, 'fig_robbins_global_map')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Regional sub-samples (spatial layout, projected)
# ══════════════════════════════════════════════════════════════════════════════

def fig_regions(cat):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8))
    fig.subplots_adjust(wspace=0.25, left=0.04, right=0.98,
                        top=0.87, bottom=0.14)

    for ax, reg in zip(axes, REGIONS):
        sub = filter_region(cat, reg)
        xy  = latlon_to_xy(sub[:,0], sub[:,1])
        d_km = sub[:,2]

        # marker area proportional to crater area (D²)
        s = np.clip((d_km / d_km.max()) * 60 + 3, 3, 80)
        sc = ax.scatter(xy[:,0]/1e3, xy[:,1]/1e3, s=s,
                        c=d_km, cmap='YlOrRd', norm=LogNorm(),
                        alpha=0.75, linewidths=0.2,
                        edgecolors='#555555', zorder=3)

        ax.set_title(reg['label'], fontweight='bold', fontsize=8,
                     color=reg['color'], pad=4)
        ax.set_xlabel('Δx (km)', fontsize=7.5)
        if ax is axes[0]:
            ax.set_ylabel('Δy (km)', fontsize=7.5)
        else:
            ax.set_yticklabels([])

        ax.text(0.03, 0.97, f'n = {len(sub):,}\nD ≥ {reg["dmin"]} km',
                transform=ax.transAxes, va='top', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc'))

        plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.03,
                     label='D (km)' if ax is axes[-1] else '')

    fig.suptitle('Robbins (2019) — Five Regional Sub-samples\n'
                 'Marker size ∝ crater area;  colour = diameter;  '
                 'no physical edge between craters (k-NN imposed by analyst)',
                 fontsize=9.5, fontweight='bold')
    _save(fig, 'fig_robbins_regions')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Size-frequency distribution
# ══════════════════════════════════════════════════════════════════════════════

def fig_sfd(cat):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.subplots_adjust(wspace=0.35)

    # Left: cumulative SFD for all regions
    ax = axes[0]
    for reg in REGIONS:
        sub = filter_region(cat, reg)
        d   = np.sort(sub[:,2])[::-1]
        cum = np.arange(1, len(d)+1)
        ax.loglog(d, cum/len(d), '-', color=reg['color'],
                  lw=1.5, label=reg['label'], alpha=0.85)

    # Power-law reference D^{-2}
    d_ref = np.logspace(np.log10(1), np.log10(200), 50)
    ax.loglog(d_ref, (d_ref/d_ref[0])**(-2) * 0.95,
              'k--', lw=1.0, label='$N(>D) \\propto D^{-2}$')
    ax.set_xlabel('Diameter $D$ (km)', fontsize=9)
    ax.set_ylabel('Cumulative fraction $N(>D)/N_\\mathrm{tot}$', fontsize=9)
    ax.set_title('(a) Cumulative SFD per region', fontweight='bold', pad=4)
    ax.legend(fontsize=7, loc='lower left')

    # Right: global SFD from full catalog (D ≥ 1 km)
    ax2 = axes[1]
    all_d = cat[cat[:,2] >= 1, 2]
    bins = np.logspace(np.log10(1), np.log10(3000), 60)
    counts, edges = np.histogram(all_d, bins=bins)
    mids = np.sqrt(edges[:-1]*edges[1:])
    ax2.loglog(mids[counts>0], counts[counts>0], 'o-',
               color='#333333', ms=3, lw=1.2, label='Full catalog')
    # Reference slope
    ref_x = np.array([1.0, 1000.0])
    ax2.loglog(ref_x, 2e6*(ref_x**(-2)), 'r--', lw=1.0,
               label='$N \\propto D^{-2}$')
    ax2.set_xlabel('Diameter $D$ (km)', fontsize=9)
    ax2.set_ylabel('Count per bin', fontsize=9)
    ax2.set_title('(b) Differential SFD — full 1.3 M catalog', fontweight='bold', pad=4)
    ax2.legend(fontsize=8)
    ax2.text(0.05, 0.05,
             'Power-law shape encodes impact physics,\nnot spatial organisation.',
             transform=ax2.transAxes, va='bottom', fontsize=8, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', fc='#FFFFEE', ec='#cccccc'))

    fig.suptitle('Robbins (2019) Lunar Crater Size-Frequency Distribution',
                 fontsize=10, fontweight='bold')
    _save(fig, 'fig_robbins_sfd')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Eigenspectra grid (5 regions)
# ══════════════════════════════════════════════════════════════════════════════

def fig_eigenspectra(results):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), sharey=False)
    fig.subplots_adjust(wspace=0.3, left=0.05, right=0.98,
                        top=0.87, bottom=0.14)

    for ax, (reg, gd) in zip(axes, zip(REGIONS, results)):
        ev = np.sort(gd['ev'])
        n_sh = min(80, len(ev))
        ax.bar(range(n_sh), ev[:n_sh], color=reg['color'],
               width=1.0, edgecolor='none', alpha=0.85)
        ax.set_xlabel('Mode $l$', fontsize=8)
        if ax is axes[0]: ax.set_ylabel('$\\lambda_l$', fontsize=9)
        ax.set_title(reg['label'].split('  ')[0], fontweight='bold',
                     color=reg['color'], fontsize=8.5, pad=3)
        ax.text(0.97, 0.97,
                f'$n={gd["n"]}$\n$\\beta_0={gd["beta0"]}$\n'
                f'$\\Delta H={gd["dH"]:+.2f}$',
                transform=ax.transAxes, ha='right', va='top', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc'))

    fig.suptitle('Robbins (2019) — k-NN Graph Eigenspectra (5 regions)\n'
                 'All five spectra are near-identical: k-NN is spectrally '
                 'invariant to the generating process',
                 fontsize=9.5, fontweight='bold')
    _save(fig, 'fig_robbins_eigenspectra')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Fixed-point kernels h*(λ) (5 regions)
# ══════════════════════════════════════════════════════════════════════════════

def fig_kernels(results):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), sharey=False)
    fig.subplots_adjust(wspace=0.32, left=0.05, right=0.98,
                        top=0.87, bottom=0.14)

    for ax, (reg, gd) in zip(axes, zip(REGIONS, results)):
        sort = np.argsort(gd['ev'])
        n_sh = min(80, len(gd['ev']))
        lam = gd['ev'][sort][:n_sh]
        h0s = gd['h0'][sort][:n_sh]
        hss = gd['hs'][sort][:n_sh]

        ax.fill_between(lam, h0s, hss, where=(hss >= h0s),
                        color='#009E73', alpha=0.25, label='amplified')
        ax.fill_between(lam, h0s, hss, where=(hss < h0s),
                        color='#D55E00', alpha=0.25, label='suppressed')
        ax.plot(lam, h0s, '--', color='#888888', lw=1.0, label='$h_0$')
        ax.plot(lam, hss, '-',  color=reg['color'], lw=1.8, label='$h^*$')

        ax.set_xlabel('$\\lambda_l$', fontsize=8)
        if ax is axes[0]: ax.set_ylabel('$h(\\lambda)$', fontsize=9)
        ax.set_title(reg['label'].split('  ')[0], fontweight='bold',
                     color=reg['color'], fontsize=8.5, pad=3)
        ax.text(0.05, 0.06,
                f'$\\Delta H = {gd["dH"]:+.3f}$',
                transform=ax.transAxes, va='bottom', fontsize=9,
                color=reg['color'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc'))
        if ax is axes[-1]:
            ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('Robbins (2019) — Fixed-Point Kernels $h^*(\\lambda)$ (5 regions)\n'
                 'All five show strong amplification of low modes: '
                 '$\\Delta H \\in [-0.52, -0.66]$ nats — entirely a k-NN artifact',
                 fontsize=9.5, fontweight='bold')
    _save(fig, 'fig_robbins_kernels')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Phase space: Robbins k-NN null vs AZ + Jezero + cities
# ══════════════════════════════════════════════════════════════════════════════

def fig_phasespace(results):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    # Background tier bands
    ax.axhspan(-0.02,  0.05, color='#EFF7FF', alpha=0.55, zorder=0)
    ax.axhspan( 0.05,  0.40, color='#FFFAEC', alpha=0.55, zorder=0)
    ax.axhspan( 0.40,  2.50, color='#EFFFEF', alpha=0.55, zorder=0)
    ax.text(-0.78, -0.008, 'Abiotic tier',      fontsize=8, color='#336699', style='italic')
    ax.text(-0.78,  0.06,  'Fossil/weak ctrl.',  fontsize=8, color='#888800', style='italic')
    ax.text(-0.78,  0.41,  'Active ctrl.',       fontsize=8, color='#006633', style='italic')

    # Reference systems
    ax.scatter(-0.027, 0.001, s=130, color=W['az'], marker='o',
               edgecolors='black', lw=1.0, zorder=6)
    ax.annotate('AZ Plateau\n(rook — abiotic)', (-0.027, 0.001),
                xytext=(-8, 14), textcoords='offset points',
                fontsize=8, color=W['az'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=W['az'], lw=0.9))

    ax.scatter(-0.263, 0.0023, s=180, color=W['jez'], marker='*',
               edgecolors='black', lw=0.8, zorder=6)
    ax.annotate('Jezero\n(rook — fossil)', (-0.263, 0.0023),
                xytext=(10, -18), textcoords='offset points',
                fontsize=8, color=W['jez'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=W['jez'], lw=0.9))

    CITIES = [
        ('Barcelona', -0.339, 0.608, '#0072B2'),
        ('Phoenix',   -0.285, 0.482, '#E69F00'),
        ('Venice',    -0.237, 0.190, '#009E73'),
        ('Marrakech', -0.238, 0.247, '#CC79A7'),
        ('Houston',   -0.301, 0.570, '#D55E00'),
    ]
    for name, dH, db1N, col in CITIES:
        ax.scatter(dH, db1N, s=60, color=col, marker='D',
                   edgecolors='black', lw=0.7, zorder=5)
        ax.annotate(name, (dH, db1N),
                    xytext=(4, 3), textcoords='offset points',
                    fontsize=7.5, color=col)

    # Robbins k-NN sub-samples
    for reg, gd in zip(REGIONS, results):
        ax.scatter(gd['dH'], gd['db1N'], s=70, color=reg['color'],
                   marker='h', edgecolors='black', lw=0.7,
                   zorder=4, alpha=0.85)
        ax.annotate(reg['label'].split('  ')[0],
                    (gd['dH'], gd['db1N']),
                    xytext=(4, -10), textcoords='offset points',
                    fontsize=7, color=reg['color'])

    ax.axhline(0, color='#bbbbbb', lw=0.6, ls=':')
    ax.axvline(0, color='#bbbbbb', lw=0.6, ls=':')

    # Shade k-NN artifact zone
    dHs = [gd['dH'] for gd in results]
    db1s = [gd['db1N'] for gd in results]
    ax.fill_between([min(dHs)-0.05, max(dHs)+0.05], -0.02, max(db1s)+0.05,
                    color='#ffeeee', alpha=0.35, zorder=0,
                    label='k-NN artifact zone')
    ax.text(np.mean(dHs), max(db1s)+0.03, 'k-NN artifact zone',
            ha='center', fontsize=7.5, color='#cc3333', style='italic')

    ax.set_xlim(-0.80, 0.05)
    ax.set_ylim(-0.025, 1.5)
    ax.set_xlabel(r'$\Delta H = H[h^*] - H[h_0]$  (nats)', fontsize=10)
    ax.set_ylabel(r'$\Delta\beta_1 / N$  (normalised topological excess)', fontsize=10)
    ax.set_title('Phase Space: Physically Motivated vs k-NN Graphs\n'
                 'Robbins crater k-NN null (hexagons) occupies a physically uninterpretable region',
                 fontweight='bold', pad=6, fontsize=11)

    leg = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=W['az'],
               markeredgecolor='black', ms=10, label='AZ Plateau (rook)'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor=W['jez'],
               markeredgecolor='black', ms=12, label='Jezero (rook — fossil)'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#888888',
               markeredgecolor='black', ms=8,  label='Cities (OSM road network)'),
        Line2D([0],[0], marker='h', color='w', markerfacecolor='#888888',
               markeredgecolor='black', ms=9,  label='Robbins craters (k-NN null)'),
    ]
    ax.legend(handles=leg, fontsize=8.5, loc='upper right', framealpha=0.97)
    _save(fig, 'fig_robbins_phasespace')


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, name):
    for d in [OUT_DIR, PAPER]:
        if d.exists():
            p = d / f'{name}.png'
            fig.savefig(p, dpi=250)
            print(f'  Saved {p.name}  →  {d.name}/')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 62)
    print('Robbins (2019) — Publication-quality figures')
    print('=' * 62)

    cat = load_catalog()

    print('\n[1] Running kernelcal on 5 regions …')
    results = []
    for reg in REGIONS:
        sub = filter_region(cat, reg)
        xy  = latlon_to_xy(sub[:,0], sub[:,1])
        gd  = run_knn_kernelcal(xy, label=reg['key'])
        results.append(gd)

    print('\n[2] Generating figures …')
    print('  [2a] Global crater map …')
    fig_global_map(cat)
    print('  [2b] Regional sub-samples …')
    fig_regions(cat)
    print('  [2c] Size-frequency distribution …')
    fig_sfd(cat)
    print('  [2d] Eigenspectra …')
    fig_eigenspectra(results)
    print('  [2e] Fixed-point kernels …')
    fig_kernels(results)
    print('  [2f] Phase space …')
    fig_phasespace(results)

    print(f'\nAll figures → {OUT_DIR}  +  {PAPER.name}/')

    print('\n  ── Results summary ────────────────────────────────')
    print(f'  {"Region":28s}  {"ΔH":>8}  {"β₁":>6}  {"Δβ₁/N":>8}')
    print('  ' + '─' * 55)
    for reg, gd in zip(REGIONS, results):
        print(f'  {reg["label"]:28s}  {gd["dH"]:+8.3f}  '
              f'{gd["beta1"]:6d}  {gd["db1N"]:+8.4f}')


if __name__ == '__main__':
    main()
