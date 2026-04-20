#!/usr/bin/env python3
"""
biosig_paper_figures.py — Publication-quality figures for the P4 biosignatures paper.

Produces white-background, journal-ready figures comparing three empirical systems:
  1. Arizona arid plateau (abiotic drainage controller)
  2. Jezero crater delta, Mars (fossil/candidate controller)
  3. Five world cities (active urban planning controller)

Output figures saved to:
  ../P4-journal-spectral-kernel-biosignature-planetary-surfaces/figures/

Figures:
  fig_controller_phasespace.pdf/.png  — Main: ΔH × Δβ₁/N phase space
  fig_delta_beta1_bar.pdf/.png        — Δβ₁/N bar chart by system (primary discriminant)
  fig_eigenspectra_comparison.pdf/.png — Laplacian eigenspectra
  fig_kernel_comparison.pdf/.png      — h*(λ) vs h₀(λ) per system
  fig_three_point_summary.pdf/.png    — Combined 4-panel paper figure

Journal style: two-column (7 in wide), 9-pt Helvetica-equivalent,
  colorblind-safe (Wong 2011 palette), prints in greyscale.
"""

from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

KCAL_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import (
    spectral_entropy,
    fixed_point_kernel,
    fiedler_mode_gap,
)

FIG_DIR = (KCAL_ROOT.parent /
           'P4-journal-spectral-kernel-biosignature-planetary-surfaces' /
           'figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Wong (2011) colorblind-safe palette ────────────────────────────────────
# black, orange, sky-blue, bluish-green, yellow, blue, vermillion, reddish-purple
C_ABIOTIC  = '#0072B2'   # blue
C_FOSSIL   = '#E69F00'   # orange/amber
C_ACTIVE   = '#009E73'   # green
C_CITY     = {
    'Barcelona': '#CC79A7',   # reddish-purple
    'Phoenix':   '#56B4E9',   # sky-blue
    'Venice':    '#D55E00',   # vermillion
    'Marrakech': '#F0E442',   # yellow
    'Houston':   '#999999',   # grey
}
C_NULL     = '#BBBBBB'   # light grey for Poisson null

# matplotlib journal settings
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size':          9,
    'axes.labelsize':     9,
    'axes.titlesize':     9,
    'axes.linewidth':     0.8,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'legend.fontsize':    7.5,
    'legend.framealpha':  0.9,
    'legend.edgecolor':   '#cccccc',
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'lines.linewidth':    1.2,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

# ── Empirical data (from production runs) ──────────────────────────────────

# Arizona plateau — badlands_kernelcal.py run (April 14 2026)
BADLANDS = {
    'label':       'AZ Plateau\n(abiotic)',
    'short':       'AZ Plateau',
    'color':       C_ABIOTIC,
    'marker':      'o',
    'dH':          -0.392,
    'dH_lo':       -0.392,  # single run, no bootstrap yet
    'dH_hi':       -0.392,
    'db1':         221,
    'N':           1500,
    'db1_N':       221 / 1500,
    'lam_fiedler': 0.0000,
    'H_obs':       4.943,
    'H_vac':       5.336,
    'beta0':       22,
    'beta1':       3222,
    'n_edges':     4700,
    'controller':  'Abiotic drainage',
    'system_type': 'abiotic',
}

# Jezero crater delta — jezero_kernelcal.py run (April 13 2026)
JEZERO = {
    'label':       'Jezero\n(fossil ctrl.)',
    'short':       'Jezero delta',
    'color':       C_FOSSIL,
    'marker':      's',
    'dH':          -0.3766,
    'dH_lo':       -0.3766,
    'dH_hi':       -0.3766,
    'db1':         960,
    'N':           2000,
    'db1_N':       960 / 2000,
    'lam_fiedler': 0.0040,
    'H_obs':       5.1270,
    'H_vac':       5.5036,
    'beta0':       36,    # 36 disconnected drainage basins
    'beta1':       None,  # not recorded directly
    'n_edges':     None,
    'controller':  'Ancient hydrology (fossil)',
    'system_type': 'fossil',
}

# Cities — osm_urban_kernelcal.py bootstrap N=678 medians (April 14 2026)
CITIES_BOOT = [
    {'label': 'Barcelona', 'short': 'Barcelona', 'color': C_CITY['Barcelona'],
     'marker': 'D',
     'dH': -0.670, 'dH_lo': -0.686, 'dH_hi': -0.660,
     'db1': 1060,  'N': 678, 'db1_N': 1060/678,
     'controller': 'Planned grid (Cerdà)'},
    {'label': 'Phoenix',   'short': 'Phoenix',   'color': C_CITY['Phoenix'],
     'marker': 'D',
     'dH': -0.691, 'dH_lo': -0.691, 'dH_hi': -0.691,
     'db1': 470,   'N': 678, 'db1_N': 470/678,
     'controller': 'Car grid (zoning)'},
    {'label': 'Venice',    'short': 'Venice',    'color': C_CITY['Venice'],
     'marker': 'D',
     'dH': -0.674, 'dH_lo': -0.686, 'dH_hi': -0.660,
     'db1': 1513,  'N': 678, 'db1_N': 1513/678,
     'controller': 'Medieval canal fabric'},
    {'label': 'Marrakech', 'short': 'Marrakech', 'color': C_CITY['Marrakech'],
     'marker': 'D',
     'dH': -0.619, 'dH_lo': -0.636, 'dH_hi': -0.610,
     'db1': 1490,  'N': 678, 'db1_N': 1490/678,
     'controller': 'Medina fabric'},
    {'label': 'Houston',   'short': 'Houston',   'color': C_CITY['Houston'],
     'marker': 'D',
     'dH': -0.629, 'dH_lo': -0.641, 'dH_hi': -0.617,
     'db1': 840,   'N': 678, 'db1_N': 840/678,
     'controller': 'Sprawl (no zoning)'},
]

# Poisson null (from bootstrap Poisson runs)
POISSON_NULL = {
    'dH': -0.718, 'dH_lo': -0.731, 'dH_hi': -0.704,
    'db1': 0, 'db1_N': 0.0,
}

# For representative eigenspectra synthesis: Laplacians generated analytically
# from known structural families (grid, organic, random)
MU2   = 2.0
SIGMA2 = 1.0


def _make_synthetic_laplacian(n: int, kind: str, k: int = 8,
                               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic (N,N) Laplacian for representative spectrum.

    kind: 'grid'    — regular lattice (like Phoenix / planned city)
          'organic' — clustered random (like Venice / Jezero delta)
          'sparse'  — tree-like k-NN on uniform (like arid drainage)
    Returns (eigvals, h_star).
    """
    rng = np.random.default_rng(seed)
    n_side = int(math.sqrt(n))
    n      = n_side * n_side

    if kind == 'grid':
        # Regular grid: connect each node to its 4 cardinal neighbours
        pos = np.array([[r, c] for r in range(n_side) for c in range(n_side)],
                       dtype=float)
    elif kind == 'organic':
        # Clustered positions (Gaussian clusters)
        n_clusters = max(3, n // 30)
        centres = rng.uniform(0, n_side, (n_clusters, 2))
        idx     = rng.integers(0, n_clusters, size=n)
        pos     = centres[idx] + rng.normal(0, 1.5, (n, 2))
    else:  # sparse / tree-like
        pos = np.column_stack([
            rng.uniform(0, n_side * 3, n),   # elongated domain
            rng.uniform(0, n_side, n),
        ])

    tree   = cKDTree(pos)
    dists, inds = tree.query(pos, k=k + 1)
    med_d  = float(np.median(dists[:, 1]))
    sigma  = max(2 * med_d, 1e-3)

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j_idx in range(1, k + 1):
            j = inds[i, j_idx]
            d = dists[i, j_idx]
            w = math.exp(-d**2 / sigma**2)
            if w > W[i, j]:
                W[i, j] = w
                W[j, i] = w

    L  = np.diag(W.sum(1)) - W
    ev, _ = scipy_eigh(L)
    ev    = np.maximum(ev, 0.0)
    n_zero = int(np.sum(ev < 1e-6))
    wm     = ev.copy(); wm[:n_zero] = ev[n_zero] if n_zero < n else 1e-3
    h0     = np.maximum(np.exp(-ev), 1e-10)
    h_star, _ = fixed_point_kernel(L, h0=h0, mu2=MU2, sigma2=SIGMA2, w=wm)
    h_star     = np.maximum(h_star, 1e-8)
    return ev, h_star, h0


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Controller Phase Space  (ΔH × Δβ₁/N)
# ══════════════════════════════════════════════════════════════════════════════

def fig_controller_phasespace():
    """Main paper figure: ΔH × Δβ₁/N with error bars and system ellipses."""
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    # ── Cities (bootstrap IQR as error bars) ──────────────────────────────
    city_dH    = [c['dH']    for c in CITIES_BOOT]
    city_db1N  = [c['db1_N'] for c in CITIES_BOOT]
    city_dH_lo = [c['dH'] - c['dH_lo'] for c in CITIES_BOOT]
    city_dH_hi = [c['dH_hi'] - c['dH']  for c in CITIES_BOOT]
    city_cols  = [c['color'] for c in CITIES_BOOT]
    city_labs  = [c['short'] for c in CITIES_BOOT]

    for i, c in enumerate(CITIES_BOOT):
        ax.errorbar(c['dH'], c['db1_N'],
                    xerr=[[c['dH'] - c['dH_lo']], [c['dH_hi'] - c['dH']]],
                    fmt='none', color=c['color'], capsize=3, lw=0.9)
        ax.scatter(c['dH'], c['db1_N'], s=55, color=c['color'],
                   marker='D', zorder=5, edgecolors='black', linewidths=0.5)
        ax.annotate(c['short'], (c['dH'], c['db1_N']),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=6.5, color=c['color'])

    # ── Jezero ────────────────────────────────────────────────────────────
    ax.scatter(JEZERO['dH'], JEZERO['db1_N'],
               s=90, color=JEZERO['color'], marker='s', zorder=6,
               edgecolors='black', linewidths=0.8)
    ax.annotate('Jezero\n(Mars)', (JEZERO['dH'], JEZERO['db1_N']),
                textcoords='offset points', xytext=(6, -12),
                fontsize=7.5, color=JEZERO['color'], fontweight='bold')

    # ── AZ Plateau ────────────────────────────────────────────────────────
    ax.scatter(BADLANDS['dH'], BADLANDS['db1_N'],
               s=90, color=BADLANDS['color'], marker='o', zorder=6,
               edgecolors='black', linewidths=0.8)
    ax.annotate('AZ Plateau\n(abiotic)', (BADLANDS['dH'], BADLANDS['db1_N']),
                textcoords='offset points', xytext=(6, 3),
                fontsize=7.5, color=BADLANDS['color'], fontweight='bold')

    # ── Poisson null band ─────────────────────────────────────────────────
    ax.axhspan(-0.05, 0.05, color=C_NULL, alpha=0.20, zorder=0)
    ax.axvspan(POISSON_NULL['dH_lo'], POISSON_NULL['dH_hi'],
               color=C_NULL, alpha=0.18, zorder=0)
    ax.scatter(POISSON_NULL['dH'], 0, s=35, color=C_NULL, marker='x',
               zorder=3, linewidths=1.2)
    ax.annotate('Poisson\nnull', (POISSON_NULL['dH'], 0.02),
                textcoords='data', fontsize=6.5, color='#888888')

    # ── Region labels ─────────────────────────────────────────────────────
    xlim = (-0.77, -0.32)
    ylim = (-0.15, 2.45)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.text(-0.75, 0.08, 'Abiotic\n(null)', fontsize=7, color=C_ABIOTIC,
            style='italic', va='bottom')
    ax.text(-0.75, 0.55, 'Fossil\ncontroller', fontsize=7, color=C_FOSSIL,
            style='italic', va='bottom')
    ax.text(-0.75, 1.40, 'Active\ncontroller', fontsize=7, color=C_ACTIVE,
            style='italic', va='bottom')

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xlabel(r'$\Delta H = H[h^*] - H[h_0]$  (nats)')
    ax.set_ylabel(r'$\Delta\beta_1 / N$  (normalised topological excess)')
    ax.set_title('Controller Phase Space', fontweight='bold', pad=6)

    # Legend
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_ABIOTIC,
               markeredgecolor='black', markersize=7, label='Abiotic drainage (AZ)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_FOSSIL,
               markeredgecolor='black', markersize=7, label='Fossil controller (Jezero)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#888888',
               markeredgecolor='black', markersize=7, label='Active controller (cities)'),
        Line2D([0], [0], marker='x', color=C_NULL, markersize=6,
               label='Poisson null'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=6.5,
              framealpha=0.95)

    ax.axhline(0, color='#cccccc', lw=0.7, ls=':')
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        out = FIG_DIR / f'fig_controller_phasespace.{ext}'
        fig.savefig(out, dpi=300)
        print(f'  Saved {out.name}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Δβ₁/N Bar Chart  (primary discriminant)
# ══════════════════════════════════════════════════════════════════════════════

def fig_delta_beta1_bar():
    """Δβ₁/N for all systems — the key discriminant figure."""
    systems = (
        [BADLANDS] +
        [JEZERO]   +
        CITIES_BOOT
    )
    labels   = [s['short'] for s in systems]
    db1N     = [s['db1_N'] for s in systems]
    colors   = [s['color'] for s in systems]
    types    = ['abiotic', 'fossil'] + ['active'] * len(CITIES_BOOT)

    hatch_map = {'abiotic': '/', 'fossil': 'x', 'active': ''}
    hatches   = [hatch_map[t] for t in types]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    x = np.arange(len(labels))
    bars = ax.bar(x, db1N, color=colors, edgecolor='black',
                  linewidth=0.6, width=0.65)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    # Annotate values
    for xi, v in zip(x, db1N):
        ax.text(xi, v + 0.04, f'{v:.2f}', ha='center', va='bottom',
                fontsize=6.5, color='black')

    ax.axhline(0, color='black', lw=0.8)

    # Tier separators
    ax.axvline(0.5, color='#bbbbbb', lw=0.8, ls='--', zorder=0)
    ax.axvline(1.5, color='#bbbbbb', lw=0.8, ls='--', zorder=0)

    # Tier labels at top
    ax.text(0,   max(db1N) * 1.08, 'Abiotic',  ha='center', fontsize=7,
            style='italic', color=C_ABIOTIC)
    ax.text(1,   max(db1N) * 1.08, 'Fossil',   ha='center', fontsize=7,
            style='italic', color=C_FOSSIL)
    ax.text(3.5, max(db1N) * 1.08, 'Active controller (cities)',
            ha='center', fontsize=7, style='italic', color=C_ACTIVE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel(r'$\Delta\beta_1 / N$  (normalised)')
    ax.set_title(r'Topological Excess $\Delta\beta_1 / N$  by System',
                 fontweight='bold', pad=6)
    ax.set_ylim(-0.1, max(db1N) * 1.20)

    # Hatch legend
    legend_handles = [
        mpatches.Patch(facecolor='white', edgecolor='black',
                       hatch='/', label='Abiotic'),
        mpatches.Patch(facecolor='white', edgecolor='black',
                       hatch='x', label='Fossil controller'),
        mpatches.Patch(facecolor='white', edgecolor='black',
                       hatch='',  label='Active controller'),
    ]
    ax.legend(handles=legend_handles, fontsize=6.5, loc='upper left',
              framealpha=0.95)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        out = FIG_DIR / f'fig_delta_beta1_bar.{ext}'
        fig.savefig(out, dpi=300)
        print(f'  Saved {out.name}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Eigenspectra  (three representative types)
# ══════════════════════════════════════════════════════════════════════════════

def fig_eigenspectra():
    """Synthetic eigenspectra for three structural types."""
    N_SYN = 400

    configs = [
        ('sparse',  C_ABIOTIC, 'Abiotic drainage\n(tree-like, sparse)',
         'AZ Plateau'),
        ('organic', C_FOSSIL,  'Fossil controller\n(clustered, reticulated)',
         'Jezero delta'),
        ('grid',    C_ACTIVE,  'Active controller\n(regular lattice)',
         'Planned city'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    for ax, (kind, color, title, example) in zip(axes, configs):
        ev, h_star, h0 = _make_synthetic_laplacian(N_SYN, kind)
        n_show = min(60, len(ev))
        ax.bar(range(n_show), ev[:n_show], color=color, width=1.0,
               edgecolor='none', alpha=0.75)
        ax.set_title(title, fontsize=8, fontweight='bold', pad=4)
        ax.set_xlabel('Mode index  $l$', fontsize=8)
        ax.set_ylabel('$\\lambda_l$', fontsize=8)
        ax.text(0.97, 0.97, f'e.g. {example}', transform=ax.transAxes,
                ha='right', va='top', fontsize=6.5, color=color,
                style='italic')
        ax.tick_params(labelsize=7)

        # Inset: h* vs h0
        inset = ax.inset_axes([0.40, 0.35, 0.58, 0.60])
        sort  = np.argsort(ev)
        inset.fill_between(ev[sort], h0[sort], h_star[sort],
                           where=(h_star[sort] >= h0[sort]),
                           color='green', alpha=0.25, label='amplified')
        inset.fill_between(ev[sort], h0[sort], h_star[sort],
                           where=(h_star[sort] < h0[sort]),
                           color='red', alpha=0.25, label='suppressed')
        inset.plot(ev[sort], h0[sort], '--', color='#999999', lw=0.8)
        inset.plot(ev[sort], h_star[sort], '-', color=color, lw=1.2)
        inset.tick_params(labelsize=5.5)
        inset.set_xlabel('$\\lambda$', fontsize=6)
        inset.set_ylabel('$h$',        fontsize=6)
        dH_syn = float(spectral_entropy(h_star) - spectral_entropy(h0))
        inset.set_title(f'$\\Delta H={dH_syn:+.2f}$', fontsize=6)

    fig.suptitle('Laplacian Eigenspectra  —  Three Structural Types',
                 fontweight='bold', y=1.03, fontsize=9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        out = FIG_DIR / f'fig_eigenspectra_comparison.{ext}'
        fig.savefig(out, dpi=300)
        print(f'  Saved {out.name}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Combined four-panel summary (the paper's main empirical figure)
# ══════════════════════════════════════════════════════════════════════════════

def fig_four_panel_summary():
    """Four-panel summary figure combining all empirical results.

    Panel A: Controller phase space (ΔH × Δβ₁/N)
    Panel B: Δβ₁/N bar chart
    Panel C: ΔH across systems with bootstrap error bars
    Panel D: Schematic of controller hierarchy + diagnostic mapping
    """
    fig = plt.figure(figsize=(7.0, 5.6))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.38)

    # ── Panel A: Phase space ─────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    for c in CITIES_BOOT:
        ax_a.errorbar(c['dH'], c['db1_N'],
                      xerr=[[c['dH'] - c['dH_lo']], [c['dH_hi'] - c['dH']]],
                      fmt='none', color=c['color'], capsize=2.5, lw=0.8)
        ax_a.scatter(c['dH'], c['db1_N'],
                     s=40, color=c['color'], marker='D', zorder=5,
                     edgecolors='black', linewidths=0.4)

    ax_a.scatter(JEZERO['dH'],   JEZERO['db1_N'],
                 s=70, color=JEZERO['color'],   marker='s', zorder=6,
                 edgecolors='black', linewidths=0.6)
    ax_a.scatter(BADLANDS['dH'], BADLANDS['db1_N'],
                 s=70, color=BADLANDS['color'], marker='o', zorder=6,
                 edgecolors='black', linewidths=0.6)

    ax_a.scatter(POISSON_NULL['dH'], 0,
                 s=30, color=C_NULL, marker='x', zorder=3, linewidths=1.0)

    # Annotate key points
    ax_a.annotate('Jezero', (JEZERO['dH'], JEZERO['db1_N']),
                  xytext=(4, 4), textcoords='offset points',
                  fontsize=6.5, color=JEZERO['color'])
    ax_a.annotate('AZ Plateau', (BADLANDS['dH'], BADLANDS['db1_N']),
                  xytext=(4, 4), textcoords='offset points',
                  fontsize=6.5, color=BADLANDS['color'])
    ax_a.annotate('Null', (POISSON_NULL['dH'], 0.03),
                  fontsize=6.0, color='#888888')

    ax_a.axhline(0, color='#cccccc', lw=0.6, ls=':')
    ax_a.set_xlabel(r'$\Delta H$  (nats)', fontsize=8)
    ax_a.set_ylabel(r'$\Delta\beta_1 / N$', fontsize=8)
    ax_a.set_title(r'(a) Controller Phase Space', fontsize=8.5, fontweight='bold',
                   loc='left', pad=4)

    handles_a = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_ABIOTIC,
               markeredgecolor='black', markersize=6, label='Abiotic drainage'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=C_FOSSIL,
               markeredgecolor='black', markersize=6, label='Jezero (Mars)'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#888888',
               markeredgecolor='black', markersize=6, label='Cities (active)'),
        Line2D([0],[0], marker='x', color=C_NULL, markersize=5,
               label='Poisson null'),
    ]
    ax_a.legend(handles=handles_a, fontsize=6.0, loc='lower left',
                framealpha=0.95)

    # ── Panel B: Δβ₁/N bar ──────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    systems = [BADLANDS, JEZERO] + CITIES_BOOT
    xlabs   = [s['short'] for s in systems]
    vals    = [s['db1_N'] for s in systems]
    cols    = [s['color'] for s in systems]
    types_b = ['abiotic', 'fossil'] + ['active'] * len(CITIES_BOOT)
    htch    = {'abiotic': '/', 'fossil': 'x', 'active': ''}

    xb = np.arange(len(xlabs))
    bars_b = ax_b.bar(xb, vals, color=cols, edgecolor='black',
                      linewidth=0.5, width=0.65)
    for bar, t in zip(bars_b, types_b):
        bar.set_hatch(htch[t])

    ax_b.axvline(0.5, color='#bbbbbb', lw=0.7, ls='--', zorder=0)
    ax_b.axvline(1.5, color='#bbbbbb', lw=0.7, ls='--', zorder=0)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(xlabs, rotation=35, ha='right', fontsize=6.5)
    ax_b.set_ylabel(r'$\Delta\beta_1 / N$', fontsize=8)
    ax_b.set_title(r'(b) Topological Excess $\Delta\beta_1/N$',
                   fontsize=8.5, fontweight='bold', loc='left', pad=4)

    # Tier brackets
    ymax = max(vals) * 1.15
    ax_b.set_ylim(-0.05, ymax * 1.1)
    for xi, v in enumerate(vals):
        ax_b.text(xi, v + ymax * 0.02, f'{v:.2f}',
                  ha='center', va='bottom', fontsize=5.5)

    # ── Panel C: ΔH with bootstrap bars ─────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    all_systems = [BADLANDS, JEZERO] + CITIES_BOOT
    xc    = np.arange(len(all_systems))
    dH_v  = [s['dH']    for s in all_systems]
    dH_lo = [s['dH'] - s['dH_lo'] for s in all_systems]
    dH_hi = [s['dH_hi'] - s['dH'] for s in all_systems]
    col_c = [s['color'] for s in all_systems]
    lab_c = [s['short'] for s in all_systems]
    types_c = ['abiotic', 'fossil'] + ['active'] * len(CITIES_BOOT)

    bars_c = ax_c.bar(xc, dH_v, color=col_c, edgecolor='black',
                      linewidth=0.5, width=0.65, zorder=2)
    for bar, t in zip(bars_c, types_c):
        bar.set_hatch(htch[t])
    ax_c.errorbar(xc, dH_v,
                  yerr=[np.clip(dH_lo, 0, None), np.clip(dH_hi, 0, None)],
                  fmt='none', color='black', capsize=2.5, lw=0.8, zorder=3)

    # Poisson null band
    ax_c.axhspan(POISSON_NULL['dH_lo'], POISSON_NULL['dH_hi'],
                 color=C_NULL, alpha=0.25, label='Poisson null IQR', zorder=0)
    ax_c.axhline(POISSON_NULL['dH'], color='#999999', lw=0.8, ls='--',
                 zorder=1, label='Poisson null median')

    ax_c.axhline(0, color='black', lw=0.7, ls=':')
    ax_c.axvline(0.5, color='#bbbbbb', lw=0.7, ls='--', zorder=0)
    ax_c.axvline(1.5, color='#bbbbbb', lw=0.7, ls='--', zorder=0)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(lab_c, rotation=35, ha='right', fontsize=6.5)
    ax_c.set_ylabel(r'$\Delta H = H[h^*] - H[h_0]$  (nats)', fontsize=8)
    ax_c.set_title(r'(c) Spectral Entropy Excess $\Delta H$',
                   fontsize=8.5, fontweight='bold', loc='left', pad=4)
    ax_c.legend(fontsize=6.0, loc='lower right', framealpha=0.95)

    # ── Panel D: Controller hierarchy schematic ───────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_xlim(0, 4)
    ax_d.set_ylim(0, 5)
    ax_d.axis('off')

    entries = [
        (C_NULL,    '/', 'Poisson random scatter',
         r'$\Delta H \approx -0.72$,  $\Delta\beta_1/N \approx 0$'),
        (C_ABIOTIC, '/', 'Abiotic drainage\n(AZ plateau / bare terrain)',
         r'$\Delta H \approx -0.39$,  $\Delta\beta_1/N \approx 0.15$'),
        (C_FOSSIL,  'x', 'Fossil controller\n(Jezero delta, Mars)',
         r'$\Delta H \approx -0.38$,  $\Delta\beta_1/N \approx 0.48$'),
        ('#dddddd', '',  'Active controller\n(cities — median)',
         r'$\Delta H \approx -0.65$,  $\Delta\beta_1/N \approx 1.2$'),
    ]

    y_start = 4.4
    dy      = 0.98
    arrow_x = 0.25

    ax_d.annotate('', xy=(arrow_x, 0.35), xytext=(arrow_x, 4.6),
                  arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax_d.text(arrow_x - 0.05, 4.75, 'Controller\nstrength',
              ha='center', fontsize=6.5, color='black', style='italic')

    for i, (col, hatch, name, vals_str) in enumerate(entries):
        y = y_start - i * dy
        rect = mpatches.FancyBboxPatch(
            (0.55, y - 0.30), 3.35, 0.80,
            boxstyle='round,pad=0.05',
            facecolor=col, edgecolor='black', linewidth=0.7,
            hatch=hatch if hatch else None,
        )
        ax_d.add_patch(rect)
        ax_d.text(0.72, y + 0.10, name,
                  fontsize=6.5, va='top', color='black', fontweight='bold')
        ax_d.text(0.72, y - 0.16, vals_str,
                  fontsize=5.8, va='top', color='#333333')

    ax_d.set_title('(d) Controller Hierarchy  —  Diagnostics',
                   fontsize=8.5, fontweight='bold', loc='left', pad=4)

    fig.suptitle(
        'Empirical Calibration of the Topological Biosignature Framework\n'
        r'Three systems spanning abiotic $\rightarrow$ fossil $\rightarrow$ active controllers',
        fontsize=9, fontweight='bold', y=1.02
    )
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        out = FIG_DIR / f'fig_three_point_summary.{ext}'
        fig.savefig(out, dpi=300)
        print(f'  Saved {out.name}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Single-column h*(λ) kernel comparison  (supplement)
# ══════════════════════════════════════════════════════════════════════════════

def fig_kernel_comparison():
    """h*(λ) vs h₀(λ) for three synthetic archetypes — single column."""
    N_SYN = 400
    configs = [
        ('sparse',  C_ABIOTIC, 'Abiotic drainage'),
        ('organic', C_FOSSIL,  'Fossil controller (Jezero)'),
        ('grid',    C_ACTIVE,  'Active controller (city)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=False)
    fig.subplots_adjust(wspace=0.40)

    for ax, (kind, color, title) in zip(axes, configs):
        ev, h_star, h0 = _make_synthetic_laplacian(N_SYN, kind)
        sort = np.argsort(ev)
        lam  = ev[sort]
        hs   = h_star[sort]
        h0s  = h0[sort]

        ax.fill_between(lam, h0s, hs, where=(hs >= h0s),
                        color='#009E73', alpha=0.25, label='amplified')
        ax.fill_between(lam, h0s, hs, where=(hs < h0s),
                        color='#D55E00', alpha=0.25, label='suppressed')
        ax.plot(lam, h0s, '--', color='#888888', lw=1.0, label=r'$h_0$ (vacuum)')
        ax.plot(lam, hs,   '-', color=color,      lw=1.5, label=r'$h^*$ (fixed-pt)')

        dH = float(spectral_entropy(h_star) - spectral_entropy(h0))
        ax.set_title(title, fontsize=8, fontweight='bold', pad=4)
        ax.set_xlabel(r'$\lambda_l$', fontsize=8)
        ax.set_ylabel(r'$h(\lambda)$', fontsize=8)
        ax.text(0.97, 0.97,
                f'$\\Delta H = {dH:+.2f}$\nnats',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=7.5, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='#cccccc', alpha=0.9))
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=6.5, loc='upper right', framealpha=0.95)
    fig.suptitle(r'MaxCal Fixed-Point Kernel $h^*(\lambda)$ vs Vacuum $h_0(\lambda)$',
                 fontweight='bold', y=1.03, fontsize=9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        out = FIG_DIR / f'fig_kernel_comparison.{ext}'
        fig.savefig(out, dpi=300)
        print(f'  Saved {out.name}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('P4 Biosignature Paper Figures  —  white-background, journal-ready')
    print(f'  Output: {FIG_DIR}')
    print('=' * 60)

    print('\n[1] Controller phase space  (ΔH × Δβ₁/N) …')
    fig_controller_phasespace()

    print('\n[2] Δβ₁/N bar chart …')
    fig_delta_beta1_bar()

    print('\n[3] Eigenspectra comparison …')
    fig_eigenspectra()

    print('\n[4] Kernel comparison …')
    fig_kernel_comparison()

    print('\n[5] Four-panel summary (main paper figure) …')
    fig_four_panel_summary()

    print('\nAll figures saved to:')
    for f in sorted(FIG_DIR.glob('fig_*.pdf')):
        print(f'  {f.name}')


if __name__ == '__main__':
    main()
