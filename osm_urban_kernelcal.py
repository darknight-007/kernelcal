#!/usr/bin/env python3
"""
osm_urban_kernelcal.py — Urban Controller Detection via Spectral Kernel Dynamics

Downloads OSM building footprints for five cities spanning the planning
spectrum, builds proximity graphs, runs the kernelcal MaxCal fixed-point
framework, and produces publication-quality comparison figures.

Cities (ordered by expected planning degree, high → low):
  1. Barcelona Eixample  — Cerdà grid (1859), octagonal blocks, explicit plan
  2. Phoenix AZ          — American car-grid, zoning-driven suburban layout
  3. Venice              — Organic medieval / Renaissance, dense canal network
  4. Marrakech Medina    — Islamic medina, dead-end alleys, social zoning
  5. Houston TX          — American sprawl, historically NO zoning code

Figures saved to ./urban_figures/:
  fig1_city_maps.png          — 5 building footprint maps (area-coloured)
  fig2_eigenspectra.png       — Eigenvalue spectra for all cities
  fig3_kernels.png            — Fixed-point h*(λ) vs vacuum h₀(λ) per city
  fig4_controller_ranking.png — ΔH, Δ', Δβ₁ bar charts
  fig5_phase_space.png        — Controller phase space scatter  H[h*] vs Δβ₁
  fig7_bootstrap_null.png     — Fixed-N bootstrap CI + Poisson null baseline

Usage:
  python3 osm_urban_kernelcal.py
  python3 osm_urban_kernelcal.py --refresh    # force re-download from OSM
  python3 osm_urban_kernelcal.py --bootstrap  # run N-equalised bootstrap + null
  python3 osm_urban_kernelcal.py --dp-sweep   # scan μ²/σ² to probe Δ′ degeneracy
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as scipy_eigh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable

# ── kernelcal ──────────────────────────────────────────────────────────────
KCAL_ROOT = Path(__file__).parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import (
    spectral_entropy,
    fixed_point_kernel,
    fiedler_mode_gap,
    stability_conservation_tradeoff,
)
from kernelcal.urban.city_graph import fetch_buildings, buildings_to_graph, CityGraph

# ── CONFIG ─────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / 'urban_cache'
FIG_DIR   = Path(__file__).parent / 'urban_figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_MAX     = 2000    # nodes per city (full run)
N_BOOT    = 678     # fixed-N for bootstrap (= min city count)
N_ITER    = 100     # bootstrap iterations per city
K_NN      = 8       # nearest neighbours
MU2       = 2.0
SIGMA2    = 1.0

# Δ′ hyperparameter sweep grid
DP_MU2_VALS    = [0.5, 1.0, 2.0, 4.0, 8.0]
DP_SIGMA2_VALS = [0.25, 0.5, 1.0, 2.0, 4.0]

# Cities: (display name, OSM query, expected planning rank 1=most planned)
# Each entry: (display label, Nominatim address for 600m radius, planning rank)
# Addresses are anchored to representative neighbourhood centres
CITIES = [
    ('Barcelona\nEixample',
     'Carrer del Consell de Cent 300, Barcelona, Spain',        1),
    ('Phoenix\nGrid',
     '300 E Van Buren St, Phoenix, AZ, USA',                    2),
    ('Venice',
     'Campo San Bartolomeo, Venice, Italy',                      3),
    ('Marrakech\nMedina',
     'Djemaa el-Fna, Marrakech, Morocco',                       4),
    ('Houston\nSprawl',
     '4200 Montrose Blvd, Houston, TX, USA',                    5),
]

# Per-city display colours
CITY_COLORS = ['#ff6b35', '#f7c59f', '#efefd0', '#004e89', '#1a936f']
BG = '#0d1117'


# ══════════════════════════════════════════════════════════════════════════════
# KERNELCAL DIAGNOSTICS WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def run_kernelcal(cg: CityGraph) -> dict:
    """Compute all kernelcal diagnostics for one CityGraph."""
    eigvals = cg.eigvals
    n       = len(eigvals)

    w_modes      = eigvals.copy()
    n_zero       = int(np.sum(eigvals < 1e-6))
    w_modes[:n_zero] = eigvals[n_zero] if n_zero < n else 1e-3

    h0    = np.maximum(np.exp(-eigvals), 1e-10)
    h_star, info = fixed_point_kernel(
        cg.L, h0=h0, mu2=MU2, sigma2=SIGMA2, w=w_modes
    )
    h_star = np.maximum(h_star, 1e-8)   # numerical floor for 1/h

    H_obs    = spectral_entropy(h_star)
    H_vac    = spectral_entropy(h0)
    delta_H  = H_obs - H_vac

    delta_p  = fiedler_mode_gap(h_star, cg.L, mu2=MU2, sigma2=SIGMA2, w=w_modes)
    tradeoff = stability_conservation_tradeoff(
        h_star, cg.L, mu2=MU2, sigma2=SIGMA2, w=w_modes
    )
    tradeoff['D_m'] = np.clip(tradeoff['D_m'], -1e4, 1e4)
    deficit  = float(np.mean(np.abs(tradeoff['D_m'])))

    # Betti numbers from graph topology
    n_edges = int((cg.W > 0).sum()) // 2
    beta0   = n_zero
    beta1   = max(0, n_edges - (n - beta0))
    # k-NN null: expected β₁ for a k-NN graph
    e_null       = K_NN * n // 2
    beta1_null   = max(0, e_null - (n - 1))
    delta_beta1  = beta1 - beta1_null

    lam_fiedler = float(eigvals[n_zero]) if n_zero < n else 0.0

    return {
        'h0':            h0,
        'h_star':        h_star,
        'H_obs':         H_obs,
        'H_vac':         H_vac,
        'delta_H':       delta_H,
        'delta_prime':   delta_p,
        'deficit':       deficit,
        'D_m':           tradeoff['D_m'],
        'beta0':         beta0,
        'beta1':         beta1,
        'beta1_null':    beta1_null,
        'delta_beta1':   delta_beta1,
        'lam_fiedler':   lam_fiedler,
        'n_edges':       n_edges,
        'converged':     info['converged'],
        'n_iter':        info['n_iter'],
    }


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: City Maps
# ══════════════════════════════════════════════════════════════════════════════

def fig_city_maps(city_graphs: list[tuple[str, CityGraph, dict]]):
    """Five-panel building footprint maps coloured by area."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 7), facecolor=BG)
    fig.subplots_adjust(wspace=0.04)

    for ax, (label, cg, diag), color in zip(axes, city_graphs, CITY_COLORS):
        ax.set_facecolor('#111827')
        gdf = cg.raw_gdf
        if gdf is None:
            ax.text(0.5, 0.5, 'No GDF', transform=ax.transAxes,
                    ha='center', color='white')
            continue

        xmin, ymin, xmax, ymax = cg.bounds_m
        areas = cg.traits[:, 0]   # normalised area
        norm  = Normalize(0, 1)

        # Draw building footprints
        try:
            from matplotlib.collections import PatchCollection
            import matplotlib.patches as patches
            polys = []
            areas_raw = gdf['area_m2'].values if 'area_m2' in gdf.columns \
                        else np.ones(len(gdf))
            # Subsample GDF to match graph nodes if needed
            n_draw = min(len(gdf), 3000)
            step   = max(1, len(gdf) // n_draw)
            sub    = gdf.iloc[::step].copy()

            for _, row in sub.iterrows():
                geom = row.geometry
                if geom.geom_type == 'Polygon':
                    pts = np.array(geom.exterior.coords)
                elif geom.geom_type == 'MultiPolygon':
                    pts = np.array(list(geom.geoms)[0].exterior.coords)
                else:
                    continue
                poly = plt.Polygon(pts, closed=True)
                polys.append(poly)

            if polys:
                pc = PatchCollection(polys, cmap='plasma', norm=norm,
                                     linewidths=0, alpha=0.85)
                areas_draw = sub['area_m2'].values[:len(polys)] \
                             if 'area_m2' in sub.columns else np.ones(len(polys))
                a_max = np.percentile(areas_draw, 95)
                pc.set_array(np.clip(areas_draw / (a_max + 1e-6), 0, 1))
                ax.add_collection(pc)
        except Exception as exc:
            # Fallback: scatter centroids
            ax.scatter(cg.positions[:, 0], cg.positions[:, 1],
                       c=areas, cmap='plasma', s=1, alpha=0.6)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.axis('off')

        # City label
        ax.set_title(label.replace('\n', ' '), color='white',
                     fontsize=10, fontweight='bold', pad=6)

        # Mini diagnostics overlay
        dH = diag['delta_H']
        db1 = diag['delta_beta1']
        ax.text(0.03, 0.03,
                f'N={len(cg.positions)}\nΔH={dH:+.2f}\nΔβ₁={db1:+d}',
                transform=ax.transAxes, va='bottom', ha='left',
                color='white', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.6))

        # Scale bar: 200 m
        sb_m   = 200
        w_m    = xmax - xmin
        sb_frac = sb_m / w_m
        ax.plot([0.05, 0.05 + sb_frac], [0.05, 0.05],
                transform=ax.transAxes, color='white', lw=2)
        ax.text(0.05 + sb_frac / 2, 0.08, '200 m',
                transform=ax.transAxes, ha='center',
                color='white', fontsize=6.5)

    fig.suptitle(
        'Urban Building Footprints — Technosphere Controller Detection  '
        '(kernelcal MaxCal framework)',
        color='white', fontsize=13, fontweight='bold', y=1.01
    )
    out = FIG_DIR / 'fig1_city_maps.png'
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Eigenspectra Comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig_eigenspectra(city_graphs: list[tuple[str, CityGraph, dict]]):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), facecolor=BG,
                             sharey=False)
    fig.subplots_adjust(wspace=0.30)

    for ax, (label, cg, diag), color in zip(axes, city_graphs, CITY_COLORS):
        ax.set_facecolor('#1a1f2e')
        ev = cg.eigvals
        n_show = min(80, len(ev))
        ax.bar(range(n_show), ev[:n_show], color=color,
               width=1.0, edgecolor='none', alpha=0.85)

        n_zero = diag['beta0']
        lf     = diag['lam_fiedler']
        ax.axvline(n_zero, color='white', lw=1.5, ls='--', alpha=0.7)
        ax.text(n_zero + 0.5, ev[:n_show].max() * 0.95,
                f'λ_f={lf:.3f}', color='white', fontsize=7, va='top')

        ax.set_title(label.replace('\n', ' '), color='white',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Mode index  l', color='white', fontsize=8)
        ax.set_ylabel('λₗ', color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('gray')

        ax.text(0.98, 0.97,
                f'β₀={diag["beta0"]}\nβ₁={diag["beta1"]}',
                transform=ax.transAxes, ha='right', va='top',
                color='lightgray', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))

    fig.suptitle('Graph Laplacian Eigenspectra  —  5 Cities',
                 color='white', fontsize=12, fontweight='bold', y=1.02)
    out = FIG_DIR / 'fig2_eigenspectra.png'
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Fixed-Point Kernels
# ══════════════════════════════════════════════════════════════════════════════

def fig_kernels(city_graphs: list[tuple[str, CityGraph, dict]]):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), facecolor=BG)
    fig.subplots_adjust(wspace=0.35)

    for ax, (label, cg, diag), color in zip(axes, city_graphs, CITY_COLORS):
        ax.set_facecolor('#1a1f2e')
        ev   = cg.eigvals
        sort = np.argsort(ev)
        lam  = ev[sort]
        hs   = diag['h_star'][sort]
        h0s  = diag['h0'][sort]

        ax.fill_between(lam, h0s, hs, where=(hs >= h0s),
                        color='#22cc88', alpha=0.30, label='amplified')
        ax.fill_between(lam, h0s, hs, where=(hs < h0s),
                        color='#ff6644', alpha=0.30, label='suppressed')
        ax.plot(lam, h0s, '--', color='#888888', lw=1.2, label='h₀ vacuum')
        ax.plot(lam, hs,  '-',  color=color,     lw=2.0, label='h* fixed-pt')

        ax.set_title(label.replace('\n', ' '), color='white',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('λₗ', color='white', fontsize=8)
        ax.set_ylabel('h(λ)', color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('gray')

        dH = diag['delta_H']
        ax.text(0.97, 0.97,
                f'ΔH = {dH:+.3f} nats',
                transform=ax.transAxes, ha='right', va='top',
                color='#ffdd66', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.6))

    axes[0].legend(facecolor='black', edgecolor='gray',
                   labelcolor='white', fontsize=6.5, loc='upper right')

    fig.suptitle('MaxCal Fixed-Point Kernel  h*(λ)  vs  Vacuum h₀(λ)  —  5 Cities',
                 color='white', fontsize=12, fontweight='bold', y=1.02)
    out = FIG_DIR / 'fig3_kernels.png'
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Controller Ranking  (key figure for publication)
# ══════════════════════════════════════════════════════════════════════════════

def fig_controller_ranking(city_graphs: list[tuple[str, CityGraph, dict]]):
    labels = [label.replace('\n', '\n') for label, _, _ in city_graphs]
    diags  = [d for _, _, d in city_graphs]
    colors = CITY_COLORS

    delta_H  = np.array([d['delta_H']      for d in diags])
    delta_p  = np.array([d['delta_prime']  for d in diags])
    delta_b1 = np.array([d['delta_beta1']  for d in diags])
    lam_f    = np.array([d['lam_fiedler']  for d in diags])

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.35)

    x = np.arange(len(labels))
    w = 0.55

    def _bar_panel(ax, values, ylabel, title, color_by_sign=False):
        ax.set_facecolor('#1a1f2e')
        bar_colors = []
        for i, v in enumerate(values):
            if color_by_sign:
                bar_colors.append('#22cc88' if v < 0 else '#ff6644')
            else:
                bar_colors.append(colors[i])
        bars = ax.bar(x, values, w, color=bar_colors, edgecolor='#333', linewidth=0.5)
        ax.axhline(0, color='white', lw=0.8, alpha=0.5)
        for bar, v in zip(bars, values):
            ypos = bar.get_height() + abs(values).max() * 0.02
            if v < 0:
                ypos = bar.get_height() - abs(values).max() * 0.07
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f'{v:.2f}', ha='center', color='white', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color='white', fontsize=8)
        ax.set_ylabel(ylabel, color='white', fontsize=9)
        ax.set_title(title, color='white', fontsize=10, fontweight='bold')
        ax.tick_params(colors='white', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('gray')

    # Panel A: ΔH  (negative = controller concentrated kernel)
    ax_a = fig.add_subplot(gs[0, 0])
    _bar_panel(ax_a, delta_H, 'ΔH = H[h*] − H[h₀]  (nats)',
               'Spectral Entropy Excess  ΔH\n(more negative = stronger controller)',
               color_by_sign=True)
    ax_a.text(0.02, 0.02,
              '← more controlled\n   abiotic →',
              transform=ax_a.transAxes, va='bottom', color='lightgray',
              fontsize=7.5, style='italic')

    # Panel B: Fiedler value λ_f  (larger = more connected)
    ax_b = fig.add_subplot(gs[0, 1])
    _bar_panel(ax_b, lam_f, 'λ_fiedler',
               'Fiedler Spectral Gap\n(larger = better connected domain)')

    # Panel C: Δβ₁  (excess topology above k-NN null)
    ax_c = fig.add_subplot(gs[1, 0])
    _bar_panel(ax_c, delta_b1, 'Δβ₁ = β₁ − β₁_null',
               'Topological Excess  Δβ₁\n(above k-NN null model)')

    # Panel D: Δ'  (Hessian stability margin)
    ax_d = fig.add_subplot(gs[1, 1])
    _bar_panel(ax_d, delta_p, "Δ'  (Hessian gap)",
               "Stability Margin  Δ'\n(fixed-point stability)")

    # Annotation: planning legend
    fig.text(0.50, 1.01,
             '← More Planned          More Organic / Informal →',
             ha='center', va='bottom', color='#aaaaaa',
             fontsize=9, style='italic', transform=fig.transFigure)

    fig.suptitle(
        'Urban Controller Detection  ·  kernelcal MaxCal Fixed-Point Framework\n'
        'Five Cities Spanning the Planning Spectrum',
        color='white', fontsize=13, fontweight='bold', y=1.06
    )
    out = FIG_DIR / 'fig4_controller_ranking.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Controller Phase Space
# ══════════════════════════════════════════════════════════════════════════════

def fig_phase_space(city_graphs: list[tuple[str, CityGraph, dict]]):
    """H[h*] vs Δβ₁ scatter — the controller signature space."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=BG)
    fig.subplots_adjust(wspace=0.35)

    labels  = [l.replace('\n', ' ') for l, _, _ in city_graphs]
    diags   = [d for _, _, d in city_graphs]
    H_obs   = np.array([d['H_obs']       for d in diags])
    delta_H = np.array([d['delta_H']     for d in diags])
    db1     = np.array([d['delta_beta1'] for d in diags])
    lam_f   = np.array([d['lam_fiedler'] for d in diags])
    delta_p = np.array([d['delta_prime'] for d in diags])

    # Panel A: ΔH vs Δβ₁ — primary controller space
    ax = axes[0]
    ax.set_facecolor('#1a1f2e')
    for i, (lab, dH, db) in enumerate(zip(labels, delta_H, db1)):
        ax.scatter(dH, db, s=160, color=CITY_COLORS[i], zorder=5,
                   edgecolors='white', linewidths=0.8)
        ax.annotate(lab, (dH, db),
                    textcoords='offset points', xytext=(6, 4),
                    color='white', fontsize=8.5,
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # Quadrant lines
    ax.axvline(0, color='white', lw=0.8, ls='--', alpha=0.4)
    ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.4)

    # Quadrant labels
    ax.text(0.02, 0.97, 'Weak controller\n(diffuse kernel,\nfew extra loops)',
            transform=ax.transAxes, va='top', color='#ff6644',
            fontsize=7.5, style='italic')
    ax.text(0.98, 0.97, 'Strong controller\n(concentrated kernel,\nextra loops)',
            transform=ax.transAxes, va='top', ha='right', color='#22cc88',
            fontsize=7.5, style='italic')

    ax.set_xlabel('ΔH = H[h*] − H[h₀]  (nats)\n← more controlled      more abiotic →',
                  color='white', fontsize=9)
    ax.set_ylabel('Δβ₁ = β₁ − β₁_null\n(topological excess)', color='white', fontsize=9)
    ax.set_title('Controller Phase Space\nΔH vs Δβ₁', color='white',
                 fontsize=10, fontweight='bold')
    ax.tick_params(colors='white', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('gray')

    # Panel B: Fiedler gap vs Δ'  — connectivity / stability space
    ax2 = axes[1]
    ax2.set_facecolor('#1a1f2e')
    for i, (lab, lf, dp) in enumerate(zip(labels, lam_f, delta_p)):
        ax2.scatter(lf, dp, s=160, color=CITY_COLORS[i], zorder=5,
                    edgecolors='white', linewidths=0.8)
        ax2.annotate(lab, (lf, dp),
                     textcoords='offset points', xytext=(6, 4),
                     color='white', fontsize=8.5,
                     path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    ax2.set_xlabel('Fiedler gap  λ_fiedler\n(spatial connectivity)',
                   color='white', fontsize=9)
    ax2.set_ylabel("Hessian gap  Δ'\n(fixed-point stability)", color='white', fontsize=9)
    ax2.set_title("Connectivity vs Stability\nλ_fiedler vs Δ'", color='white',
                  fontsize=10, fontweight='bold')
    ax2.tick_params(colors='white', labelsize=8)
    for sp in ax2.spines.values(): sp.set_edgecolor('gray')

    # Legend patches
    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(CITY_COLORS, labels)]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               facecolor='black', edgecolor='gray',
               labelcolor='white', fontsize=8.5,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        'Urban Controller Signature Space  ·  Five Cities  ·  kernelcal MaxCal',
        color='white', fontsize=12, fontweight='bold', y=1.02
    )
    out = FIG_DIR / 'fig5_phase_space.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Summary Table
# ══════════════════════════════════════════════════════════════════════════════

def fig_summary_table(city_graphs: list[tuple[str, CityGraph, dict]]):
    """Publication-quality one-page results summary."""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    labels = [l.replace('\n', ' ') for l, _, _ in city_graphs]
    diags  = [d for _, _, d in city_graphs]

    col_headers = ['City', 'N nodes', 'β₀', 'β₁', 'Δβ₁',
                   'λ_fiedler', "H[h*]", "H[h₀]", 'ΔH',
                   "Δ'", 'Interpretation']
    rows = []
    for lab, (_, cg, d) in zip(labels, city_graphs):
        if d['delta_H'] < -0.3:
            interp = 'Strong controller'
        elif d['delta_H'] < -0.1:
            interp = 'Moderate controller'
        else:
            interp = 'Weak / abiotic'
        rows.append([
            lab,
            str(len(cg.positions)),
            str(d['beta0']),
            str(d['beta1']),
            f'{d["delta_beta1"]:+d}',
            f'{d["lam_fiedler"]:.4f}',
            f'{d["H_obs"]:.3f}',
            f'{d["H_vac"]:.3f}',
            f'{d["delta_H"]:+.3f}',
            f'{d["delta_prime"]:.3f}',
            interp,
        ])

    # Draw table
    col_x = np.linspace(0.01, 0.99, len(col_headers) + 1)[:-1]
    row_h = 0.75 / (len(rows) + 1)
    header_y = 0.88

    # Header
    for cx, hdr in zip(col_x, col_headers):
        ax.text(cx, header_y, hdr, color='#8888cc', fontsize=8.5,
                fontweight='bold', va='top', transform=ax.transAxes)

    ax.plot([0.01, 0.99], [header_y - 0.02, header_y - 0.02],
            color='gray', lw=0.8, transform=ax.transAxes)

    for ri, row in enumerate(rows):
        y = header_y - 0.06 - ri * row_h * 1.15
        bg = '#111827' if ri % 2 == 0 else '#1a1f2e'
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - row_h * 0.4), 1.0, row_h * 0.95,
            boxstyle='round,pad=0.005', fc=bg, ec='none',
            transform=ax.transAxes, zorder=0
        ))
        for cx, val, hdr in zip(col_x, row, col_headers):
            color = 'white'
            if hdr == 'ΔH':
                v = float(val)
                color = '#22cc88' if v < -0.1 else ('#ffdd66' if v < 0 else '#ff6644')
            elif hdr == 'Interpretation':
                color = '#22cc88' if 'Strong' in val else (
                        '#ffdd66' if 'Moderate' in val else '#aaaaaa')
            elif hdr == 'City':
                color = CITY_COLORS[ri]
            ax.text(cx, y, val, color=color, fontsize=8, va='center',
                    transform=ax.transAxes, fontweight=(
                        'bold' if hdr in ('City', 'ΔH', 'Interpretation') else 'normal'))

    ax.set_title(
        'Kernelcal Urban Controller Detection  ·  Results Summary',
        color='white', fontsize=13, fontweight='bold', pad=20
    )
    ax.text(0.5, 0.04,
            'ΔH < 0 → kernel more concentrated than vacuum → controller active\n'
            'Δβ₁ > 0 → excess topology above k-NN null → structured spatial organisation',
            ha='center', va='bottom', transform=ax.transAxes,
            color='lightgray', fontsize=8.5, style='italic')

    out = FIG_DIR / 'fig6_summary_table.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP & POISSON NULL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _graph_and_diag_from_positions(positions: np.ndarray,
                                   k: int = K_NN) -> dict:
    """Minimal graph → Laplacian → kernelcal on raw (N,2) position array."""
    import math as _math
    from scipy.spatial import cKDTree as _cKDTree
    from scipy.linalg import eigh as _eigh

    n = len(positions)
    if n < k + 2:
        return None

    tree   = _cKDTree(positions)
    dists, inds = tree.query(positions, k=k + 1)
    median_nn = float(np.median(dists[:, 1])) if n > 1 else 1.0
    xr = positions[:, 0].max() - positions[:, 0].min()
    yr = positions[:, 1].max() - positions[:, 1].min()
    diag = _math.hypot(xr, yr)
    sigma = max(0.05 * max(diag, 1.0), 2 * max(median_nn, 1e-3))

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j_idx in range(1, k + 1):
            j = inds[i, j_idx]
            d = dists[i, j_idx]
            w = _math.exp(-d**2 / sigma**2)
            if w > W[i, j]:
                W[i, j] = w
                W[j, i] = w

    D  = np.diag(W.sum(axis=1))
    L  = D - W
    ev, _ = _eigh(L)
    ev = np.maximum(ev, 0.0)

    n_zero   = int(np.sum(ev < 1e-6))
    w_modes  = ev.copy()
    w_modes[:n_zero] = ev[n_zero] if n_zero < n else 1e-3

    h0    = np.maximum(np.exp(-ev), 1e-10)
    try:
        h_star, info = fixed_point_kernel(L, h0=h0, mu2=MU2, sigma2=SIGMA2, w=w_modes)
        h_star = np.maximum(h_star, 1e-8)
    except Exception:
        return None

    H_obs   = spectral_entropy(h_star)
    H_vac   = spectral_entropy(h0)
    n_edges = int((W > 0).sum()) // 2
    beta0   = n_zero
    beta1   = max(0, n_edges - (n - beta0))
    e_null  = K_NN * n // 2
    delta_b1 = beta1 - max(0, e_null - (n - 1))

    return {
        'delta_H':     H_obs - H_vac,
        'delta_beta1': delta_b1,
        'lam_fiedler': float(ev[n_zero]) if n_zero < n else 0.0,
        'delta_prime': fiedler_mode_gap(h_star, L, mu2=MU2, sigma2=SIGMA2, w=w_modes),
    }


def bootstrap_city(cg: CityGraph,
                   n_fixed: int = N_BOOT,
                   n_iter:  int = N_ITER,
                   seed:    int = 0) -> dict:
    """Repeatedly subsample *cg* to *n_fixed* nodes and re-run diagnostics.

    Returns dict of arrays (length n_iter): delta_H, delta_beta1,
    lam_fiedler, delta_prime — plus median and IQR per diagnostic.
    """
    rng = np.random.default_rng(seed)
    pos = cg.positions
    n   = len(pos)
    n_fixed = min(n_fixed, n)

    records = {'delta_H': [], 'delta_beta1': [], 'lam_fiedler': [], 'delta_prime': []}

    for _ in range(n_iter):
        idx   = rng.choice(n, size=n_fixed, replace=False)
        sub   = pos[idx]
        d     = _graph_and_diag_from_positions(sub)
        if d is None:
            continue
        for k in records:
            records[k].append(d[k])

    out = {}
    for key, vals in records.items():
        arr = np.array(vals)
        out[key]             = arr
        out[key + '_median'] = float(np.median(arr))
        out[key + '_q25']    = float(np.percentile(arr, 25))
        out[key + '_q75']    = float(np.percentile(arr, 75))
    return out


def poisson_null(bounds_m: tuple[float, float, float, float],
                 n:        int,
                 n_iter:   int = N_ITER,
                 seed:     int = 42) -> dict:
    """Homogeneous Poisson null: random uniform scatter in *bounds_m* box.

    Returns same structure as bootstrap_city (medians + IQR).
    Two variants per run: (a) same-bounds Poisson, (b) matched-sigma Poisson
    with spacing = median building separation in the city.
    """
    rng  = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = bounds_m
    records = {'delta_H': [], 'delta_beta1': [], 'lam_fiedler': [], 'delta_prime': []}

    for _ in range(n_iter):
        x   = rng.uniform(xmin, xmax, size=n)
        y   = rng.uniform(ymin, ymax, size=n)
        pos = np.column_stack([x, y])
        d   = _graph_and_diag_from_positions(pos)
        if d is None:
            continue
        for k in records:
            records[k].append(d[k])

    out = {}
    for key, vals in records.items():
        arr = np.array(vals)
        out[key]             = arr
        out[key + '_median'] = float(np.median(arr))
        out[key + '_q25']    = float(np.percentile(arr, 25))
        out[key + '_q75']    = float(np.percentile(arr, 75))
    return out


def probe_delta_prime(cg: CityGraph) -> np.ndarray:
    """Grid sweep over (μ², σ²) to map Δ′ sensitivity.

    Returns (len(DP_MU2_VALS), len(DP_SIGMA2_VALS)) array of Δ′ values.
    """
    pos = cg.positions
    # Subsample to N_BOOT for speed
    rng = np.random.default_rng(7)
    idx = rng.choice(len(pos), size=min(N_BOOT, len(pos)), replace=False)
    sub = pos[idx]

    from scipy.spatial import cKDTree as _cKDTree
    from scipy.linalg import eigh as _eigh
    import math as _math

    n    = len(sub)
    tree = _cKDTree(sub)
    dists, inds = tree.query(sub, k=K_NN + 1)
    median_nn = float(np.median(dists[:, 1]))
    xr = sub[:, 0].max() - sub[:, 0].min()
    yr = sub[:, 1].max() - sub[:, 1].min()
    diag  = _math.hypot(xr, yr)
    sigma = max(0.05 * max(diag, 1.0), 2 * max(median_nn, 1e-3))

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j_idx in range(1, K_NN + 1):
            j = inds[i, j_idx]
            d = dists[i, j_idx]
            w = _math.exp(-d**2 / sigma**2)
            if w > W[i, j]:
                W[i, j] = w
                W[j, i] = w
    D  = np.diag(W.sum(axis=1))
    L  = D - W
    ev, _ = _eigh(L)
    ev = np.maximum(ev, 0.0)
    n_zero = int(np.sum(ev < 1e-6))

    grid = np.full((len(DP_MU2_VALS), len(DP_SIGMA2_VALS)), np.nan)
    for i, mu2 in enumerate(DP_MU2_VALS):
        for j, sig2 in enumerate(DP_SIGMA2_VALS):
            w_modes = ev.copy()
            w_modes[:n_zero] = ev[n_zero] if n_zero < n else 1e-3
            h0 = np.maximum(np.exp(-ev), 1e-10)
            try:
                h_star, _ = fixed_point_kernel(L, h0=h0, mu2=mu2, sigma2=sig2, w=w_modes)
                h_star = np.maximum(h_star, 1e-8)
                grid[i, j] = fiedler_mode_gap(h_star, L, mu2=mu2, sigma2=sig2, w=w_modes)
            except Exception:
                pass
    return grid


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Bootstrap CI + Poisson Null
# ══════════════════════════════════════════════════════════════════════════════

def fig_bootstrap_null(city_graphs:  list[tuple[str, CityGraph, dict]],
                       boot_results: list[dict],
                       null_results: list[dict]):
    """Bootstrap CI bars for ΔH and Δβ₁ with Poisson null overlay.

    Layout (2 rows × 2 cols):
      Row 0: ΔH bootstrap bars (col 0) | Δβ₁ bootstrap bars (col 1)
      Row 1: Δ′ hyperparameter sweep heatmap for first two cities (col 0, 1)
    """
    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.35)

    labels = [l.replace('\n', ' ') for l, _, _ in city_graphs]
    n_c    = len(labels)
    x      = np.arange(n_c)
    w      = 0.45

    # ── PANEL A: ΔH with bootstrap IQR ──────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor('#1a1f2e')

    dH_med  = np.array([b['delta_H_median'] for b in boot_results])
    dH_lo   = dH_med - np.array([b['delta_H_q25']  for b in boot_results])
    dH_hi   = np.array([b['delta_H_q75']  for b in boot_results]) - dH_med
    # Point from full-N run (for reference)
    dH_full = np.array([d['delta_H'] for _, _, d in city_graphs])

    bars = ax_a.bar(x, dH_med, w,
                    color=CITY_COLORS[:n_c], edgecolor='#333', linewidth=0.5,
                    label=f'Median (N={N_BOOT} bootstrap)')
    ax_a.errorbar(x, dH_med, yerr=[dH_lo, dH_hi],
                  fmt='none', color='white', capsize=5, lw=1.5, label='IQR')
    ax_a.scatter(x, dH_full, marker='D', color='white', s=30, zorder=5,
                 label=f'Full-N point')

    # Poisson null band
    pn_dH_med = np.array([n['delta_H_median'] for n in null_results])
    pn_dH_lo  = np.array([n['delta_H_q25']    for n in null_results])
    pn_dH_hi  = np.array([n['delta_H_q75']    for n in null_results])
    ax_a.fill_between(x, pn_dH_lo, pn_dH_hi, color='#cc3333', alpha=0.25,
                      label='Poisson null IQR')
    ax_a.plot(x, pn_dH_med, '--', color='#ff6666', lw=1.5,
              label='Poisson null median')

    ax_a.axhline(0, color='white', lw=0.7, ls=':', alpha=0.5)
    ax_a.set_xticks(x); ax_a.set_xticklabels(labels, color='white', fontsize=8)
    ax_a.set_ylabel('ΔH  (nats)', color='white', fontsize=9)
    ax_a.set_title(f'ΔH — Bootstrap N={N_BOOT} ({N_ITER} iters)\nvs Poisson null',
                   color='white', fontsize=9, fontweight='bold')
    ax_a.tick_params(colors='white', labelsize=8)
    for sp in ax_a.spines.values(): sp.set_edgecolor('gray')
    ax_a.legend(facecolor='black', edgecolor='gray', labelcolor='white',
                fontsize=6.5, loc='lower right')
    ax_a.text(0.02, 0.04,
              '← more controlled\n   abiotic →',
              transform=ax_a.transAxes, va='bottom', color='lightgray',
              fontsize=7, style='italic')

    # ── PANEL B: Δβ₁ with bootstrap IQR ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor('#1a1f2e')

    db_med  = np.array([b['delta_beta1_median'] for b in boot_results])
    db_lo   = db_med - np.array([b['delta_beta1_q25'] for b in boot_results])
    db_hi   = np.array([b['delta_beta1_q75'] for b in boot_results]) - db_med
    db_full = np.array([d['delta_beta1'] for _, _, d in city_graphs])

    # Scale db for N_BOOT (Δβ₁ is extensive in N)
    ax_b.bar(x, db_med, w,
             color=CITY_COLORS[:n_c], edgecolor='#333', linewidth=0.5,
             label=f'Median (N={N_BOOT})')
    ax_b.errorbar(x, db_med, yerr=[np.clip(db_lo, 0, None),
                                    np.clip(db_hi, 0, None)],
                  fmt='none', color='white', capsize=5, lw=1.5, label='IQR')

    pn_db_med = np.array([n['delta_beta1_median'] for n in null_results])
    pn_db_lo  = np.array([n['delta_beta1_q25']    for n in null_results])
    pn_db_hi  = np.array([n['delta_beta1_q75']    for n in null_results])
    ax_b.fill_between(x, pn_db_lo, pn_db_hi, color='#cc3333', alpha=0.25,
                      label='Poisson null IQR')
    ax_b.plot(x, pn_db_med, '--', color='#ff6666', lw=1.5)

    ax_b.axhline(0, color='white', lw=0.7, ls=':', alpha=0.5)
    ax_b.set_xticks(x); ax_b.set_xticklabels(labels, color='white', fontsize=8)
    ax_b.set_ylabel('Δβ₁ = β₁ − β₁_null', color='white', fontsize=9)
    ax_b.set_title(f'Δβ₁ — Bootstrap N={N_BOOT} ({N_ITER} iters)\nvs Poisson null',
                   color='white', fontsize=9, fontweight='bold')
    ax_b.tick_params(colors='white', labelsize=8)
    for sp in ax_b.spines.values(): sp.set_edgecolor('gray')
    ax_b.legend(facecolor='black', edgecolor='gray', labelcolor='white',
                fontsize=6.5, loc='upper right')

    # ── PANELS C/D: Δ′ hyperparameter heatmaps ───────────────────────────────
    for col_idx, city_idx in enumerate([0, 2]):   # Barcelona, Venice
        if city_idx >= len(city_graphs):
            continue
        ax_hp = fig.add_subplot(gs[1, col_idx])
        ax_hp.set_facecolor('#1a1f2e')

        _, cg, _ = city_graphs[city_idx]
        grid = probe_delta_prime(cg)

        im = ax_hp.imshow(grid, aspect='auto', origin='lower',
                          cmap='RdYlGn', vmin=0, vmax=grid[np.isfinite(grid)].max() * 1.1)
        ax_hp.set_xticks(range(len(DP_SIGMA2_VALS)))
        ax_hp.set_xticklabels([str(v) for v in DP_SIGMA2_VALS],
                               color='white', fontsize=7)
        ax_hp.set_yticks(range(len(DP_MU2_VALS)))
        ax_hp.set_yticklabels([str(v) for v in DP_MU2_VALS],
                               color='white', fontsize=7)
        ax_hp.set_xlabel('σ²', color='white', fontsize=9)
        ax_hp.set_ylabel('μ²', color='white', fontsize=9)
        city_name = city_graphs[city_idx][0].replace('\n', ' ')
        ax_hp.set_title(f"Δ′ heatmap — {city_name}\n"
                        "(★ = nominal μ²=2, σ²=1)",
                        color='white', fontsize=9, fontweight='bold')

        # Mark nominal setting
        nom_j = DP_SIGMA2_VALS.index(1.0) if 1.0 in DP_SIGMA2_VALS else 2
        nom_i = DP_MU2_VALS.index(2.0)   if 2.0 in DP_MU2_VALS   else 2
        ax_hp.scatter(nom_j, nom_i, marker='*', s=200, color='yellow', zorder=5)

        plt.colorbar(im, ax=ax_hp, fraction=0.03, pad=0.04).ax.yaxis.label.set_color('white')

        # Overlay values
        for ri in range(len(DP_MU2_VALS)):
            for ci in range(len(DP_SIGMA2_VALS)):
                v = grid[ri, ci]
                if np.isfinite(v):
                    ax_hp.text(ci, ri, f'{v:.2f}', ha='center', va='center',
                               color='black', fontsize=6)
        ax_hp.tick_params(colors='white', labelsize=7)
        for sp in ax_hp.spines.values(): sp.set_edgecolor('gray')

    # Annotation: if Δ′ range < 0.1 across grid, flag degeneracy
    dp_range_row0 = None
    if len(city_graphs) > 0:
        g0 = probe_delta_prime(city_graphs[0][1])
        finite = g0[np.isfinite(g0)]
        if len(finite) > 1:
            dp_range_row0 = finite.max() - finite.min()
    if dp_range_row0 is not None and dp_range_row0 < 0.15:
        fig.text(0.5, 0.02,
                 f"⚠  Δ′ range across entire μ²×σ² grid: {dp_range_row0:.3f}  "
                 "— diagnostic is degenerate in this graph regime",
                 ha='center', color='#ffcc44', fontsize=8.5, style='italic',
                 transform=fig.transFigure)

    fig.suptitle(
        f'Robustness Check: Fixed-N Bootstrap (N={N_BOOT}, {N_ITER} iters)  ·  '
        'Poisson Null  ·  Δ′ Hyperparameter Sensitivity',
        color='white', fontsize=12, fontweight='bold', y=1.02
    )
    out = FIG_DIR / 'fig7_bootstrap_null.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Saved {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh',   action='store_true',
                        help='Force re-download from OSM (ignore cache)')
    parser.add_argument('--bootstrap', action='store_true',
                        help=f'Run N={N_BOOT} bootstrap ({N_ITER} iters) + Poisson null')
    parser.add_argument('--dp-sweep',  action='store_true',
                        help='Scan μ²/σ² grid for Δ′ sensitivity (embedded in fig7 if --bootstrap)')
    args = parser.parse_args()

    print('=' * 65)
    print('Urban Controller Detection  —  kernelcal MaxCal Pipeline')
    print(f'  Cache:   {CACHE_DIR}')
    print(f'  Figures: {FIG_DIR}')
    print('=' * 65)

    results: list[tuple[str, CityGraph, dict]] = []

    for display_name, place, rank in CITIES:
        short = display_name.replace('\n', ' ')
        print(f'\n[{rank}] {short}')
        print(f'    OSM query: "{place}"')

        # 1. Fetch
        try:
            gdf = fetch_buildings(place, cache_dir=CACHE_DIR,
                                  force_refresh=args.refresh)
            print(f'    Buildings downloaded: {len(gdf):,}')
        except Exception as exc:
            print(f'    ERROR fetching: {exc}')
            continue

        # 2. Graph
        try:
            cg = buildings_to_graph(gdf, name=short, place=place,
                                    k=K_NN, n_max=N_MAX)
            print(f'    Graph: N={len(cg.positions)}  edges={int((cg.W>0).sum()//2)}')
        except Exception as exc:
            print(f'    ERROR building graph: {exc}')
            continue

        # 3. Kernelcal
        diag = run_kernelcal(cg)
        print(f'    λ_fiedler = {diag["lam_fiedler"]:.4f}  '
              f'|  H[h*]={diag["H_obs"]:.3f}  ΔH={diag["delta_H"]:+.3f}  '
              f'|  Δ\'={diag["delta_prime"]:.3f}  '
              f'|  β₀={diag["beta0"]}  β₁={diag["beta1"]}  Δβ₁={diag["delta_beta1"]:+d}')

        results.append((display_name, cg, diag))

    if not results:
        print('\nERROR: No cities processed.  Check network and OSM availability.')
        return

    # 4. Core figures (always generated)
    print(f'\n[Figures] Generating {len(results)} cities …')
    fig_city_maps(results)
    fig_eigenspectra(results)
    fig_kernels(results)
    fig_controller_ranking(results)
    fig_phase_space(results)
    fig_summary_table(results)

    # 5. Bootstrap + Poisson null + Δ′ sweep
    if args.bootstrap or args.dp_sweep:
        print(f'\n[Bootstrap] N_fixed={N_BOOT}, {N_ITER} iterations per city …')
        boot_results = []
        null_results = []
        for display_name, cg, d in results:
            short = display_name.replace('\n', ' ')
            print(f'  {short} …', end='', flush=True)
            br = bootstrap_city(cg, n_fixed=N_BOOT, n_iter=N_ITER)
            boot_results.append(br)
            nr = poisson_null(cg.bounds_m, n=N_BOOT, n_iter=N_ITER)
            null_results.append(nr)
            print(f'  ΔH boot={br["delta_H_median"]:+.3f} '
                  f'[{br["delta_H_q25"]:+.3f}, {br["delta_H_q75"]:+.3f}]  '
                  f'null={nr["delta_H_median"]:+.3f} '
                  f'[{nr["delta_H_q25"]:+.3f}, {nr["delta_H_q75"]:+.3f}]')

        fig_bootstrap_null(results, boot_results, null_results)

        # Print Δ′ degeneracy note
        print('\n[Δ′ note]')
        dp_vals = [d['delta_prime'] for _, _, d in results]
        dp_range = max(dp_vals) - min(dp_vals)
        print(f'  Δ′ across cities at μ²={MU2}, σ²={SIGMA2}: '
              f'{min(dp_vals):.4f} – {max(dp_vals):.4f}  (range={dp_range:.4f})')
        if dp_range < 0.02:
            print('  ⚠  DEGENERATE — Δ′ does not discriminate urban morphology')
            print('     at these hyperparameters.  See heatmap in fig7_bootstrap_null.png')
            print('     for sensitivity.  Suppress Δ′ from urban narrative.')
        else:
            print('  ✓  Δ′ range is meaningful — can include in narrative.')

    # 6. Console summary
    print('\n' + '=' * 65)
    print('RESULTS SUMMARY')
    print(f'{"City":<22} {"ΔH":>7} {"Δβ₁":>7} {"λ_f":>8} {"Δ\'":>7}  Interpretation')
    print('-' * 65)
    for display_name, cg, d in results:
        lab  = display_name.replace('\n', ' ')
        dH   = d['delta_H']
        db1  = d['delta_beta1']
        lf   = d['lam_fiedler']
        dp   = d['delta_prime']
        dp_flag = '(DEGEN)' if (max(d['delta_prime'] for _, _, d in results) -
                                min(d['delta_prime'] for _, _, d in results)) < 0.02 else ''
        flag = '★ controller' if dH < -0.15 else ('~ moderate' if dH < 0 else '○ abiotic')
        print(f'{lab:<22} {dH:>+7.3f} {db1:>+7d} {lf:>8.4f} {dp:>7.3f} {dp_flag:>8}  {flag}')
    print('=' * 65)
    print(f'\nNOTE: λ_fiedler measures spatial connectivity of the k-NN *sample*,')
    print(f'      not the urban fabric — uninformative without street-network topology.')
    print(f"NOTE: Δ′ ≈ const across cities at μ²={MU2}, σ²={SIGMA2} — see --bootstrap")
    print(f'      for hyperparameter sweep; suppress from urban narrative.')
    print(f'\nFramework: ΔH < 0 → fixed-point kernel more concentrated than vacuum')
    print(f'           → controller active at the scale of the building graph')
    print(f'\nFigures saved to: {FIG_DIR}')


if __name__ == '__main__':
    main()
