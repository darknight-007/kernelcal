#!/usr/bin/env python3
"""
osm_street_kernelcal.py
=======================
Re-run the urban controller analysis using OSM STREET NETWORK graphs
(nodes = intersections, edges = road segments) instead of k-NN proximity
graphs on building centroids.

The graph construction is now physically motivated: every edge is a road
that a controller (urban planner, historical growth, economic demand)
deliberately created or maintained. β₁ counts real city blocks.

Five cities:
  Barcelona Eixample  — Cerdà orthogonal grid
  Phoenix downtown    — American car-grid
  Venice              — canal-district labyrinth
  Marrakech medina    — organic medieval fabric
  Houston midtown     — mixed sprawl

Outputs (street_figures/):
  fig1_street_graphs.png      — node/edge map for each city
  fig2_street_eigenspectra.png — Laplacian eigenspectrum grid
  fig3_street_kernelcal.png   — h*(λ) vs h₀ grid
  fig4_street_phasespace.png  — full comparison phase space
  fig4_street_phasespace.pdf  — paper-quality (→ P4 figures/)
"""

from __future__ import annotations
import math, sys, warnings
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

KCAL_ROOT = Path(__file__).parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import (
    spectral_entropy, fixed_point_kernel, fiedler_mode_gap,
)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import osmnx as ox

# ── paths ────────────────────────────────────────────────────────────────────
FIG_DIR   = KCAL_ROOT / 'street_figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG = (KCAL_ROOT.parent /
             'P4-journal-spectral-kernel-biosignature-planetary-surfaces' /
             'figures')
CACHE_DIR = Path('/tmp/kernelcal_street_cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
MU2    = 2.0
SIGMA2 = 1.0
N_BOOT = 300        # bootstrap target N (min across cities)
N_ITER = 100        # bootstrap iterations
RNG    = np.random.default_rng(42)

# Wong 2011 colorblind-safe palette
CITY_COLORS = {
    'Barcelona': '#0072B2',
    'Phoenix':   '#E69F00',
    'Venice':    '#009E73',
    'Marrakech': '#CC79A7',
    'Houston':   '#D55E00',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'legend.fontsize': 7.5, 'legend.framealpha': 0.95,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'lines.linewidth': 1.2,
})

# City definitions — anchor + radius for graph_from_address
CITIES = [
    dict(name='Barcelona', anchor='Passeig de Gràcia 92, Barcelona, Spain',
         dist=700, color=CITY_COLORS['Barcelona']),
    dict(name='Phoenix',   anchor='100 N 1st Ave, Phoenix, AZ, USA',
         dist=800, color=CITY_COLORS['Phoenix']),
    dict(name='Venice',    anchor='Campo San Polo, Venice, Italy',
         dist=600, color=CITY_COLORS['Venice']),
    dict(name='Marrakech', anchor='Djemaa el-Fna, Marrakech, Morocco',
         dist=600, color=CITY_COLORS['Marrakech']),
    dict(name='Houston',   anchor='1000 Main St, Houston, TX, USA',
         dist=800, color=CITY_COLORS['Houston']),
]


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_street_graph(city: dict) -> tuple:
    """
    Download the undirected street network for *city* and return
    (G_undirected, node_xy_metres, edge_lengths).
    Uses 'all' network type to capture pedestrian paths (especially
    important for Venice and Marrakech where most connectivity is pedestrian).
    """
    cache = CACHE_DIR / f"{city['name'].replace(' ','_')}_street.graphml"
    if cache.exists():
        print(f'    [{city["name"]}] loading from cache …')
        G = ox.load_graphml(cache)
    else:
        print(f'    [{city["name"]}] downloading from OSM …')
        G = ox.graph_from_address(
            city['anchor'],
            dist=city['dist'],
            network_type='all',
            simplify=True,
            retain_all=False,
        )
        ox.save_graphml(G, cache)

    G = ox.convert.to_undirected(G)
    # Project to UTM for metric coordinates
    G_proj = ox.project_graph(G)

    nodes_proj = G_proj.nodes(data=True)
    node_list  = list(G_proj.nodes())
    xy = np.array([[G_proj.nodes[n]['x'], G_proj.nodes[n]['y']]
                   for n in node_list])

    # Edge lengths (metres, from road geometry)
    lengths = []
    edges_idx = []
    node_idx = {n: i for i, n in enumerate(node_list)}
    for u, v, data in G_proj.edges(data=True):
        if u not in node_idx or v not in node_idx:
            continue
        i, j = node_idx[u], node_idx[v]
        length = float(data.get('length', 1.0))
        lengths.append(length)
        edges_idx.append((i, j, length))

    print(f'    [{city["name"]}]  N={len(node_list)} nodes  '
          f'E={len(edges_idx)} edges  '
          f'median_len={np.median(lengths):.0f} m')
    return node_list, xy, edges_idx, lengths


def build_laplacian(xy: np.ndarray, edges_idx: list,
                    lengths: list) -> tuple:
    """
    Build the symmetric weighted Laplacian from road-segment edges.
    Weight: w_ij = exp(-l_ij / lambda) where lambda = median road length.
    (Exponential rather than Gaussian because road lengths are heavy-tailed.)
    """
    n = len(xy)
    lambda_len = float(np.median(lengths)) if lengths else 1.0

    W = np.zeros((n, n), dtype=np.float64)
    for i, j, length in edges_idx:
        if i == j:
            continue
        w = math.exp(-length / lambda_len)
        if w > W[i, j]:
            W[i, j] = w
            W[j, i] = w

    L = np.diag(W.sum(1)) - W
    return L, W


# ══════════════════════════════════════════════════════════════════════════════
# KERNELCAL
# ══════════════════════════════════════════════════════════════════════════════

def run_kernelcal_on_laplacian(L: np.ndarray, W: np.ndarray) -> dict:
    n = L.shape[0]
    ev, evec = scipy_eigh(L)
    ev = np.maximum(ev, 0.0)

    n_zero  = int(np.sum(ev < 1e-6))
    w_modes = ev.copy()
    w_modes[:n_zero] = ev[n_zero] if n_zero < n else 1e-3

    h0     = np.maximum(np.exp(-ev), 1e-10)
    h_star, info = fixed_point_kernel(L, h0=h0, mu2=MU2, sigma2=SIGMA2, w=w_modes)
    h_star = np.maximum(h_star, 1e-8)

    H_obs = spectral_entropy(h_star)
    H_vac = spectral_entropy(h0)
    dH    = H_obs - H_vac
    dp    = fiedler_mode_gap(h_star, L, mu2=MU2, sigma2=SIGMA2, w=w_modes)

    n_edges = int((W > 0).sum()) // 2
    beta0   = n_zero
    beta1   = max(0, n_edges - (n - beta0))
    # null expectation for a planar graph: E ~ 3N - 6 → β₁ = E - N + β₀ - null
    # use Euler characteristic null: E_null = N * mean_degree / 2
    mean_deg = W.sum(1).mean()
    e_null = int(mean_deg * n / 2)
    db1    = beta1 - max(0, e_null - (n - beta0))
    lam_f  = float(ev[n_zero]) if n_zero < n else 0.0

    return dict(
        n=n, ev=ev, h0=h0, h_star=h_star,
        H_obs=H_obs, H_vac=H_vac, dH=dH, dp=dp,
        beta0=beta0, beta1=beta1, db1=db1, db1_N=db1/n,
        lam_f=lam_f, n_edges=n_edges, converged=info.get('converged', True),
    )


def bootstrap_city(xy: np.ndarray, edges_idx: list, lengths: list,
                   n_boot: int, n_iter: int) -> dict | None:
    """
    Spatial bootstrap: for each iteration pick a random centre within the
    graph's bounding box, then take the n_boot nearest nodes (connected
    spatial patch rather than random node draw, which leaves too few edges
    in sparse road graphs).
    """
    n_full = len(xy)
    if n_full <= n_boot:
        return None

    from scipy.spatial import cKDTree as _KDT
    tree = _KDT(xy)
    dH_list, db1N_list = [], []

    x_lo, x_hi = xy[:, 0].min(), xy[:, 0].max()
    y_lo, y_hi = xy[:, 1].min(), xy[:, 1].max()

    attempts = 0
    while len(dH_list) < n_iter and attempts < n_iter * 5:
        attempts += 1
        # Random centre inside the bounding box
        cx = RNG.uniform(x_lo, x_hi)
        cy = RNG.uniform(y_lo, y_hi)
        _, idx = tree.query([cx, cy], k=n_boot)
        idx_set = set(idx)
        old2new = {old: new for new, old in enumerate(idx)}

        sub_edges  = []
        sub_lengths = []
        for i, j, length in edges_idx:
            if i in idx_set and j in idx_set:
                sub_edges.append((old2new[i], old2new[j], length))
                sub_lengths.append(length)

        if len(sub_edges) < n_boot // 2:   # patch too sparse, try again
            continue

        sub_xy = xy[idx]
        L_sub, W_sub = build_laplacian(sub_xy, sub_edges, sub_lengths)
        gd = run_kernelcal_on_laplacian(L_sub, W_sub)
        dH_list.append(gd['dH'])
        db1N_list.append(gd['db1_N'])

    if len(dH_list) < 5:
        return None

    dH_arr   = np.array(dH_list)
    db1N_arr = np.array(db1N_list)
    return dict(
        dH_med=float(np.median(dH_arr)),
        dH_iqr=(float(np.percentile(dH_arr, 25)),
                float(np.percentile(dH_arr, 75))),
        db1N_med=float(np.median(db1N_arr)),
        db1N_iqr=(float(np.percentile(db1N_arr, 25)),
                  float(np.percentile(db1N_arr, 75))),
        n_iter=len(dH_list),
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Street graph maps (5-panel)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_street_maps(city_data: list):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5))
    fig.subplots_adjust(wspace=0.08)

    for ax, cd in zip(axes, city_data):
        xy = cd['xy']; edges_idx = cd['edges_idx']
        col = cd['city']['color']

        # Edge collection
        segs = []
        for i, j, _ in edges_idx:
            segs.append([xy[i], xy[j]])
        lc = LineCollection(segs, linewidths=0.6, color=col, alpha=0.55)
        ax.add_collection(lc)

        # Degree-coloured nodes
        n = len(xy)
        W = cd['W']
        deg = (W > 0).sum(1)
        sc  = ax.scatter(xy[:, 0], xy[:, 1], c=deg,
                         cmap='YlOrRd', s=8, vmin=1, vmax=8,
                         edgecolors='none', zorder=5)

        ax.set_aspect('equal')
        ax.autoscale()
        ax.set_title(cd['city']['name'], fontweight='bold',
                     color=col, fontsize=9, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.02, 0.02, f"N={cd['gd']['n']}\nE={cd['gd']['n_edges']}\nβ₁={cd['gd']['beta1']}",
                transform=ax.transAxes, fontsize=6.5, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc'))

    fig.suptitle(
        'OSM Street Network Graphs — Nodes = Intersections, Edges = Road Segments\n'
        '(node colour = degree; β₁ = city blocks)',
        fontsize=10, fontweight='bold', y=1.02)

    out = FIG_DIR / 'fig1_street_graphs.png'
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Eigenspectrum grid (5-panel row)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_eigenspectra(city_data: list):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.0))
    fig.subplots_adjust(wspace=0.35)

    for ax, cd in zip(axes, city_data):
        gd  = cd['gd']
        col = cd['city']['color']
        ev  = np.sort(gd['ev'])
        n_show = min(100, len(ev))
        ax.bar(range(n_show), ev[:n_show], color=col, width=1.0,
               edgecolor='none', alpha=0.80)
        ax.set_title(cd['city']['name'], fontweight='bold',
                     color=col, fontsize=9, pad=3)
        ax.set_xlabel('Mode $l$', fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel('$\\lambda_l$', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(0.97, 0.97,
                f'$\\lambda_f={gd["lam_f"]:.4f}$\n$\\beta_0={gd["beta0"]}$',
                transform=ax.transAxes, ha='right', va='top', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc'))

    fig.suptitle('Laplacian Eigenspectra — OSM Street Network Graphs',
                 fontsize=10, fontweight='bold')
    out = FIG_DIR / 'fig2_street_eigenspectra.png'
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — h*(λ) vs h₀ grid
# ══════════════════════════════════════════════════════════════════════════════

def fig3_kernels(city_data: list):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.0))
    fig.subplots_adjust(wspace=0.38)

    for ax, cd in zip(axes, city_data):
        gd  = cd['gd']
        col = cd['city']['color']
        sort = np.argsort(gd['ev'])
        lam  = gd['ev'][sort]; h0s = gd['h0'][sort]; hs = gd['h_star'][sort]
        n_show = min(120, len(lam))
        lam = lam[:n_show]; h0s = h0s[:n_show]; hs = hs[:n_show]

        ax.fill_between(lam, h0s, hs, where=(hs >= h0s),
                        color='#009E73', alpha=0.25)
        ax.fill_between(lam, h0s, hs, where=(hs < h0s),
                        color='#D55E00', alpha=0.25)
        ax.plot(lam, h0s, '--', color='#888888', lw=1.2, label='$h_0$')
        ax.plot(lam, hs,  '-',  color=col,       lw=1.8, label='$h^*$')
        ax.set_title(cd['city']['name'], fontweight='bold',
                     color=col, fontsize=9, pad=3)
        ax.set_xlabel('$\\lambda_l$', fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel('$h(\\lambda)$', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(0.05, 0.05,
                f'$\\Delta H={gd["dH"]:+.3f}$\n$\\Delta\\beta_1/N={gd["db1_N"]:+.3f}$',
                transform=ax.transAxes, va='bottom', fontsize=7.5,
                color=col, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc'))
        if ax == axes[-1]:
            ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('MaxCal Fixed-Point Kernel $h^*(\\lambda)$ — OSM Street Network Graphs',
                 fontsize=10, fontweight='bold')
    out = FIG_DIR / 'fig3_street_kernelcal.png'
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Master phase space: ALL systems, graph provenance explicit
# ══════════════════════════════════════════════════════════════════════════════

def fig4_phase_space(city_data: list, boot_results: dict):
    fig = plt.figure(figsize=(13, 5.0))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[2, 0.05, 1.4],
                            wspace=0.40)
    ax   = fig.add_subplot(gs[0])
    ax2  = fig.add_subplot(gs[2])

    # ── Prior systems (D8 physically-motivated) ──────────────────────────────
    # Jezero EXCLUDED — DEM-derived D8 flow ≠ traced channel edges;
    # proper Mars analysis pending channel-network graph construction.
    PRIOR = [
        dict(label='AZ Plateau\n(D8 flow)',  dH=-0.392, db1_N=221/1500,
             color='#0072B2', marker='o', s=90, tier='abiotic'),
    ]
    # Robbins k-NN (old, physically ungrounded — shown as methodological null)
    ROBBINS = [
        dict(label='Robbins craters\n(k-NN — ungrounded)',
             dH=-0.590, db1_N=0.723,   # mean across 5 regions
             color='#BBBBBB', marker='h', s=60, tier='knnn'),
    ]
    POISSON = dict(label='Poisson null', dH=-0.718, db1_N=0.0,
                   color='#888888', marker='x', s=40)

    # Tier backgrounds (two confirmed tiers; Mars pending)
    ax.axhspan(-0.20, 0.30,  color='#EFF7FF', alpha=0.5, zorder=0)
    ax.axhspan(0.30,  5.50,  color='#EFFFEF', alpha=0.5, zorder=0)
    ax.text(-0.82, 0.04,  'Abiotic',       fontsize=7.5, color='#0072B2', style='italic')
    ax.text(-0.82, 1.50,  'Active ctrl.',  fontsize=7.5, color='#009E73', style='italic')

    # Poisson null
    ax.scatter(POISSON['dH'], POISSON['db1_N'], s=POISSON['s'],
               color=POISSON['color'], marker=POISSON['marker'],
               linewidths=1.5, zorder=4)
    ax.annotate('Poisson null\n(k-NN)', (POISSON['dH'], 0.04),
                fontsize=6.5, color='#888888', ha='center')

    # Robbins k-NN null (methodological demonstration)
    for r in ROBBINS:
        ax.scatter(r['dH'], r['db1_N'], s=r['s'], color=r['color'],
                   marker=r['marker'], edgecolors='black', linewidths=0.8,
                   zorder=4, alpha=0.7)
    ax.annotate('Robbins craters\n(k-NN — ungrounded)',
                (ROBBINS[0]['dH'], ROBBINS[0]['db1_N']),
                xytext=(18, 10), textcoords='offset points',
                fontsize=7, color='#888888',
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.7))

    # Mars placeholder arrow (pending channel graph)
    ax.annotate('Mars target\n(channel graph\npending)',
                xy=(-0.30, 0.45), xytext=(-0.05, 0.55),
                fontsize=6.5, color='#E69F00', style='italic',
                arrowprops=dict(arrowstyle='->', color='#E69F00',
                                lw=1.0, ls='dashed'),
                bbox=dict(boxstyle='round,pad=0.3', fc='#FFF8E8',
                          ec='#E69F00', alpha=0.9))

    # Prior D8 systems (AZ Plateau only)
    for p in PRIOR:
        ax.scatter(p['dH'], p['db1_N'], s=p['s'], color=p['color'],
                   marker=p['marker'], edgecolors='black', linewidths=0.7,
                   zorder=5)
        ax.annotate(p['label'], (p['dH'], p['db1_N']),
                    xytext=(7, -14), textcoords='offset points',
                    fontsize=7.5, color=p['color'], fontweight='bold')

    # NEW: street network city results (bootstrap medians + IQR)
    for cd in city_data:
        c   = cd['city']
        gd  = cd['gd']
        bst = cd.get('boot')
        if bst is not None:
            dH_c   = bst['dH_med']
            db1N_c = bst['db1N_med']
            xerr = [[dH_c - bst['dH_iqr'][0]], [bst['dH_iqr'][1] - dH_c]]
            yerr = [[db1N_c - bst['db1N_iqr'][0]], [bst['db1N_iqr'][1] - db1N_c]]
        else:
            dH_c   = gd['dH']
            db1N_c = gd['db1_N']
            xerr = yerr = None

        ax.errorbar(dH_c, db1N_c,
                    xerr=xerr, yerr=yerr,
                    fmt='D', color=c['color'], markeredgecolor='black',
                    markeredgewidth=0.7, markersize=9, capsize=3,
                    elinewidth=1.0, zorder=6)
        ax.annotate(c['name'], (dH_c, db1N_c),
                    xytext=(5, 4), textcoords='offset points',
                    fontsize=7, color=c['color'])

    ax.axhline(0, color='#cccccc', lw=0.6, ls=':')
    ax.axvline(0, color='#cccccc', lw=0.6, ls=':')
    ax.set_xlim(-0.92, 0.12)
    ax.set_ylim(-0.20, 5.60)
    ax.set_xlabel(r'$\Delta H = H[h^*] - H[h_0]$  (nats)')
    ax.set_ylabel(r'$\Delta\beta_1 / N$  (normalised topological excess)')
    ax.set_title('(a)  Controller Phase Space\n'
                 '(diamonds = street networks; hexagons = k-NN craters)',
                 fontweight='bold', pad=5)

    leg = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#0072B2',
               markeredgecolor='black', markersize=9, label='AZ Plateau (D8 flow, abiotic)'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#009E73',
               markeredgecolor='black', markersize=8, label='Grid cities (street network)'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#CC79A7',
               markeredgecolor='black', markersize=8, label='Organic cities (street network)'),
        Line2D([0],[0], marker='h', color='w', markerfacecolor='#BBBBBB',
               markeredgecolor='black', markersize=8, label='Robbins k-NN (methodological null)'),
        Line2D([0],[0], marker='x', color='#888888', markersize=7,
               label='Poisson null (k-NN)'),
    ]
    ax.legend(handles=leg, fontsize=6.5, loc='lower left', framealpha=0.97)

    # ── Bar chart: Δβ₁/N ranked ──────────────────────────────────────────────
    all_sys = []
    # D8 abiotic baseline
    all_sys.append(dict(label='AZ Plateau\n(D8)', dH=-0.392,
                        db1_N=221/1500, color='#0072B2', tier='abiotic', htch='/'))
    # Robbins k-NN (methodological null, shown for reference)
    all_sys.append(dict(label='Robbins craters\n(k-NN null)', dH=-0.590,
                        db1_N=0.723, color='#BBBBBB', tier='knnn', htch='.'))
    # Street cities
    for cd in city_data:
        bst = cd.get('boot')
        v   = bst['db1N_med'] if bst is not None else cd['gd']['db1_N']
        all_sys.append(dict(
            label=cd['city']['name'] + '\n(street)',
            dH=bst['dH_med'] if bst is not None else cd['gd']['dH'],
            db1_N=v, color=cd['city']['color'], tier='active', htch=''))

    all_sys.sort(key=lambda x: x['db1_N'])
    xb   = np.arange(len(all_sys))
    labs = [s['label'] for s in all_sys]
    vals = [s['db1_N'] for s in all_sys]
    cols = [s['color'] for s in all_sys]
    bars = ax2.barh(xb, vals, color=cols, edgecolor='black',
                    linewidth=0.5, height=0.70)
    for bar, s in zip(bars, all_sys):
        bar.set_hatch(s['htch'])
    ax2.axvline(0, color='black', lw=0.7)
    for xi, v in enumerate(vals):
        ax2.text(max(v, 0) + 0.05, xi, f'{v:.2f}',
                 va='center', fontsize=6.5)
    ax2.set_yticks(xb)
    ax2.set_yticklabels(labs, fontsize=7)
    ax2.set_xlabel(r'$\Delta\beta_1 / N$')
    ax2.set_title(r'(b)  $\Delta\beta_1/N$ ranked', fontweight='bold', pad=5)

    fig.suptitle(
        'Controller Phase Space — Physically-Motivated Graphs Only\n'
        'AZ Plateau (abiotic D8)  ·  5 cities (OSM road network)  ·  '
        'Mars target pending channel-graph construction',
        fontsize=10, fontweight='bold', y=1.03)

    for out_dir, name, dpi in [
        (FIG_DIR,   'fig4_street_phasespace.png', 200),
        (PAPER_FIG, 'fig_street_phasespace.pdf',  300),
        (PAPER_FIG, 'fig_street_phasespace.png',  300),
    ]:
        fig.savefig(out_dir / name, dpi=dpi)
        print(f'  Saved {name}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 65)
    print('OSM Street Network kernelcal analysis')
    print(f'  Output: {FIG_DIR}')
    print('=' * 65)

    print('\n[1] Fetching and analysing street networks …')
    city_data = []
    for city in CITIES:
        print(f'\n  ── {city["name"]} ──')
        node_list, xy, edges_idx, lengths = fetch_street_graph(city)
        L, W = build_laplacian(xy, edges_idx, lengths)
        gd   = run_kernelcal_on_laplacian(L, W)
        print(f'    ΔH={gd["dH"]:+.4f}  Δβ₁/N={gd["db1_N"]:+.4f}  '
              f'β₀={gd["beta0"]}  β₁={gd["beta1"]}  λ_f={gd["lam_f"]:.5f}')
        city_data.append(dict(
            city=city, node_list=node_list,
            xy=xy, edges_idx=edges_idx, lengths=lengths,
            L=L, W=W, gd=gd,
        ))

    # Determine bootstrap N (conservative: min network size)
    min_n = min(cd['gd']['n'] for cd in city_data)
    n_boot = max(100, min(N_BOOT, min_n - 1))
    print(f'\n[2] Bootstrap (N={n_boot}, {N_ITER} iterations each) …')
    boot_results = {}
    for cd in city_data:
        if cd['gd']['n'] > n_boot + 10:
            bst = bootstrap_city(cd['xy'], cd['edges_idx'], cd['lengths'],
                                 n_boot, N_ITER)
            cd['boot'] = bst
            boot_results[cd['city']['name']] = bst
            print(f'  [{cd["city"]["name"]:10s}]  '
                  f'ΔH={bst["dH_med"]:+.4f} '
                  f'[{bst["dH_iqr"][0]:+.4f},{bst["dH_iqr"][1]:+.4f}]  '
                  f'Δβ₁/N={bst["db1N_med"]:+.4f} '
                  f'[{bst["db1N_iqr"][0]:+.4f},{bst["db1N_iqr"][1]:+.4f}]  '
                  f'(n={bst["n_iter"]} valid)')
        else:
            cd['boot'] = None

    print('\n  Full-graph summary (no bootstrap):')
    print(f'  {"City":12s}  {"N":>6}  {"ΔH":>8}  {"Δβ₁/N":>8}  {"β₁":>7}')
    print('  ' + '-' * 52)
    for cd in city_data:
        gd = cd['gd']
        print(f'  {cd["city"]["name"]:12s}  {gd["n"]:6d}  '
              f'{gd["dH"]:+8.4f}  {gd["db1_N"]:+8.4f}  {gd["beta1"]:7d}')

    print('\n[3] Figures …')
    print('  [3a] Street graph maps …')
    fig1_street_maps(city_data)
    print('  [3b] Eigenspectrum grid …')
    fig2_eigenspectra(city_data)
    print('  [3c] Kernel grid …')
    fig3_kernels(city_data)
    print('  [3d] Phase space …')
    fig4_phase_space(city_data, boot_results)

    print(f'\nAll figures → {FIG_DIR}')
    return city_data, boot_results


if __name__ == '__main__':
    main()
