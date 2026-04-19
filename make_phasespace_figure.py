#!/usr/bin/env python3
"""
make_phasespace_figure.py
=========================
Regenerate the controller phase-space figure (Fig 11 in P4) using
current, correct empirical values.

All numbers sourced from paper text and field notes:
  - Bishop off-scarp:  |ΔH| = 0.074  (passive floor, k-NN rock centroids)
  - Bishop scarp:      |ΔH| = 0.307  (active abiotic, 3 controllers, k-NN rock)
  - Cities (OSM):      ΔH ∈ [-0.34, -0.24], Δβ₁/N ∈ [0.19, 0.61]
                       (spatially bootstrapped road networks, N=300)
  - Robbins null:      |ΔH| ≈ 0.58  (k-NN artifact)
  - Tonto NF:          temporal trajectory (rising H), not a phase-space point

Note: ΔH values are negative (h* more concentrated than vacuum);
the x-axis shows ΔH = H[h*] - H[h0] which is negative for organised graphs.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

OUT = (Path(__file__).parent.parent /
       'P4-journal-spectral-kernel-biosignature-planetary-surfaces' /
       'figures' / 'fig_controller_phasespace.png')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'legend.framealpha': 0.95,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# ── Wong (2011) palette ─────────────────────────────────────────────────────
C_PASSIVE  = '#0072B2'   # blue
C_ABIOTIC  = '#E69F00'   # amber
C_URBAN    = '#009E73'   # green
C_REMOVE   = '#56B4E9'   # sky-blue
C_NULL     = '#BBBBBB'   # grey

# ── Data (all from current paper, field notes 37–41, 44–46) ────────────────

# Bishop off-scarp plateau (passive, k-NN, N=2000)
BISHOP_OFF = dict(dH=-0.074, db1_N=None)   # β₁/N not comparable (k-NN artifact)

# Bishop fault scarp (3 abiotic controllers, k-NN, N=2000)
BISHOP_SCARP = dict(dH=-0.307, db1_N=None)  # β₁/N not comparable (k-NN artifact)

# Cities — OSM street networks, spatial-patch bootstrap N=300, 100 iters
# ΔH from city_kernelcal; Δβ₁/N = street-network β₁ / N_nodes (bootstrap)
CITIES = [
    dict(label='Barcelona', dH=-0.339, db1_N=0.61,
         dH_lo=-0.339, dH_hi=-0.339, color='#CC79A7'),
    dict(label='Phoenix',   dH=-0.285, db1_N=0.48,
         dH_lo=-0.285, dH_hi=-0.285, color='#56B4E9'),
    dict(label='Venice',    dH=-0.237, db1_N=0.19,
         dH_lo=-0.237, dH_hi=-0.237, color='#D55E00'),
    dict(label='Marrakech', dH=-0.238, db1_N=0.25,
         dH_lo=-0.238, dH_hi=-0.238, color='#F0E442'),
    dict(label='Houston',   dH=-0.301, db1_N=0.57,
         dH_lo=-0.301, dH_hi=-0.301, color='#999999'),
]

# Robbins k-NN null (construction artifact, not physical)
# Δβ₁/N ≈ 3.7 is also a construction artifact → not plotted on Δβ₁ axis
ROBBINS = dict(dH=-0.58)

# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2))

# --- Robbins null band (left side, far from zero) ---
ax.axvspan(-0.66, -0.50, color=C_NULL, alpha=0.12, zorder=0,
           label='_nolegend_')
ax.text(-0.58, 0.62, 'Robbins\nk-NN null', ha='center', fontsize=6.5,
        color='#888888', style='italic')

# --- Cities (road networks) ---
for c in CITIES:
    ax.scatter(c['dH'], c['db1_N'], s=65, color=c['color'],
               marker='D', zorder=5, edgecolors='black', linewidths=0.5)
    offset = (5, 3)
    if c['label'] == 'Venice':
        offset = (5, -10)
    if c['label'] == 'Marrakech':
        offset = (5, 5)
    ax.annotate(c['label'], (c['dH'], c['db1_N']),
                textcoords='offset points', xytext=offset,
                fontsize=6.5, color=c['color'])

# --- Bishop scarp (active abiotic) — vertical band on ΔH axis ---
ax.axvline(-0.307, color=C_ABIOTIC, lw=1.5, ls='--', alpha=0.8, zorder=3)
ax.text(-0.307, 0.72, 'Bishop\nscarp', ha='center', fontsize=6.5,
        color=C_ABIOTIC, fontweight='bold', va='bottom')

# --- Bishop off-scarp (passive floor) — vertical band on ΔH axis ---
ax.axvline(-0.074, color=C_PASSIVE, lw=1.5, ls=':', alpha=0.8, zorder=3)
ax.text(-0.074, 0.72, 'Bishop\noff-scarp', ha='center', fontsize=6.5,
        color=C_PASSIVE, fontweight='bold', va='bottom')

# --- Region shading ---
ax.axhspan(-0.05, 0.05, color='#EEEEEE', alpha=0.6, zorder=0)
ax.axhline(0, color='#cccccc', lw=0.6, ls=':')
ax.text(-0.36, 0.01, 'Poisson null', ha='right', fontsize=6.5,
        color='#AAAAAA', style='italic')

# --- Note about β₁ comparability ---
ax.text(-0.70, -0.09, 'Note: Δβ₁/N valid only for OSM road-network graphs (cities).\nBishop β₁/N is a k-NN artifact (not plotted on y-axis).',
        ha='left', fontsize=6, color='#888888', style='italic')

# --- Axes ---
ax.set_xlim(-0.72, 0.02)
ax.set_ylim(-0.12, 0.80)
ax.set_xlabel(r'$\Delta H = H[h^*] - H[h_0]$  (nats)',
              fontsize=9)
ax.set_ylabel(r'$\Delta\beta_1 / N$  (normalised topological excess, cities only)',
              fontsize=8)
ax.set_title('Empirical Calibration Regimes and $|\\Delta H|$ Ranges',
             fontweight='bold', fontsize=9.5)

# --- Region labels ---
ax.text(-0.24, 0.58, 'Active urban\n(OSM roads)', fontsize=7, color=C_URBAN,
        style='italic', fontweight='bold', ha='center')
ax.text(-0.044, 0.35, 'Passive\nfloor', fontsize=7, color=C_PASSIVE,
        style='italic', fontweight='bold', ha='left')
ax.text(-0.37, 0.35, 'Active\nabiotic', fontsize=7, color=C_ABIOTIC,
        style='italic', fontweight='bold', ha='right')

# --- Legend ---
legend_handles = [
    Line2D([0],[0], color=C_PASSIVE, lw=1.5, ls=':',
           label=f'Bishop off-scarp (passive, $|\\Delta H|=0.074$)'),
    Line2D([0],[0], color=C_ABIOTIC, lw=1.5, ls='--',
           label=f'Bishop scarp (abiotic, $|\\Delta H|=0.307$)'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor='#009E73',
           markeredgecolor='black', markersize=7,
           label='Cities (OSM road networks)'),
    Line2D([0],[0], color=C_NULL, lw=6, alpha=0.3,
           label='Robbins k-NN null (graph artifact)'),
]
ax.legend(handles=legend_handles, loc='upper left', fontsize=6.5,
          framealpha=0.95, edgecolor='#cccccc')

ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
fig.tight_layout()
fig.savefig(str(OUT), dpi=300)
print(f'Saved: {OUT}')
plt.close(fig)
