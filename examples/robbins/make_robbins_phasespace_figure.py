#!/usr/bin/env python3
"""
make_robbins_phasespace_figure.py
==================================
Regenerate fig_robbins_phasespace.png with current correct data.

Purpose: demonstrate that Robbins k-NN sub-samples cluster at a spectrally
invariant position (k-NN artifact), far from physically motivated OSM
road-network graphs (cities).

Data:
  Robbins sub-samples: ΔH ∈ [-0.54, -0.63], Δβ₁/N ≈ 3.7  (k-NN artifact)
  Cities (OSM road networks): ΔH ∈ [-0.24, -0.34], Δβ₁/N ∈ [0.19, 0.61]
  Bishop scarp:  ΔH = -0.307  (k-NN rock centroids — β₁/N not plotted)
  Bishop off-scarp: ΔH = -0.074  (k-NN rock centroids — β₁/N not plotted)
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

OUT = (Path(__file__).resolve().parent.parent.parent.parent /
       'P4-journal-spectral-kernel-biosignature-planetary-surfaces' /
       'figures' / 'fig_robbins_phasespace.png')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'legend.framealpha': 0.95,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# Colours
C_CITY    = '#009E73'   # green (OSM road networks)
C_ROBBINS = '#BBBBBB'   # grey (k-NN artifact)
C_PASSIVE = '#0072B2'   # blue (Bishop off-scarp)
C_ABIOTIC = '#E69F00'   # amber (Bishop scarp)

CITY_COLORS = {
    'Barcelona': '#CC79A7',
    'Phoenix':   '#56B4E9',
    'Venice':    '#D55E00',
    'Marrakech': '#F0E442',
    'Houston':   '#999999',
}

# --- Robbins sub-samples (k-NN artifact, geologically diverse regions)
# ΔH from robbins_kernelcal.py, Δβ₁/N from the same run
ROBBINS_PTS = [
    dict(label='Global D≥5 km',        dH=-0.649, db1_N=0.693),
    dict(label='N. Highlands D≥1 km',  dH=-0.566, db1_N=0.682),
    dict(label='S. Highlands D≥1 km',  dH=-0.553, db1_N=0.705),
    dict(label='Near-side equat.',      dH=-0.516, db1_N=0.824),
    dict(label='Global D≥20 km',        dH=-0.665, db1_N=0.630),
]

# --- Cities (OSM road networks — physically motivated)
CITIES = [
    dict(label='Barcelona', dH=-0.339, db1_N=0.61,  color=CITY_COLORS['Barcelona']),
    dict(label='Phoenix',   dH=-0.285, db1_N=0.48,  color=CITY_COLORS['Phoenix']),
    dict(label='Venice',    dH=-0.237, db1_N=0.19,  color=CITY_COLORS['Venice']),
    dict(label='Marrakech', dH=-0.238, db1_N=0.25,  color=CITY_COLORS['Marrakech']),
    dict(label='Houston',   dH=-0.301, db1_N=0.57,  color=CITY_COLORS['Houston']),
]

fig, ax = plt.subplots(figsize=(5.5, 4.2))

# --- Robbins null band (shaded) ---
ax.axvspan(-0.68, -0.50, color=C_ROBBINS, alpha=0.15, zorder=0)

# --- Robbins points ---
for pt in ROBBINS_PTS:
    ax.scatter(pt['dH'], pt['db1_N'], s=70, color=C_ROBBINS, marker='h',
               zorder=4, edgecolors='#888888', linewidths=0.7)
ax.annotate('Robbins k-NN sub-samples\n(mean $\\Delta H = -0.590$)',
            (-0.590, 0.72),
            textcoords='data', ha='center', fontsize=7,
            color='#666666', style='italic',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc', lw=0.5))

# --- Cities ---
for c in CITIES:
    ax.scatter(c['dH'], c['db1_N'], s=65, color=c['color'],
               marker='D', zorder=5, edgecolors='black', linewidths=0.5)
    xytext = (5, 3)
    if c['label'] == 'Venice':   xytext = (5, -10)
    if c['label'] == 'Phoenix':  xytext = (5,  5)
    ax.annotate(c['label'], (c['dH'], c['db1_N']),
                textcoords='offset points', xytext=xytext,
                fontsize=7, color=c['color'])

# --- Bishop markers (ΔH only, no Δβ₁/N — k-NN artifact) ---
ax.axvline(-0.307, color=C_ABIOTIC, lw=1.2, ls='--', alpha=0.8, zorder=3)
ax.axvline(-0.074, color=C_PASSIVE, lw=1.2, ls=':', alpha=0.8, zorder=3)
ax.text(-0.307, 0.87, 'Bishop\nscarp', ha='center', fontsize=6,
        color=C_ABIOTIC, fontweight='bold')
ax.text(-0.074, 0.87, 'Bishop\noff-scarp', ha='center', fontsize=6,
        color=C_PASSIVE, fontweight='bold')

# --- Separation annotation ---
ax.annotate('', xy=(-0.59, 0.40), xytext=(-0.34, 0.40),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
ax.text(-0.465, 0.43, 'construction\nbias separates', ha='center',
        fontsize=6.5, color='black', style='italic')

# --- Poisson null ---
ax.axhspan(-0.03, 0.03, color='#EEEEEE', alpha=0.5, zorder=0)
ax.axhline(0, color='#cccccc', lw=0.6, ls=':')
ax.text(-0.70, 0.01, 'Poisson null', ha='left', fontsize=6.5,
        color='#AAAAAA', style='italic')

# --- Axes ---
ax.set_xlim(-0.72, 0.02)
ax.set_ylim(-0.06, 0.96)
ax.set_xlabel(r'$\Delta H = H[h^*] - H[h_0]$  (nats)', fontsize=9)
ax.set_ylabel(r'$\Delta\beta_1/N$  (normalised; cities = OSM road, Robbins = k-NN artifact)',
              fontsize=7.5)
ax.set_title('Robbins k-NN Null: Spectrally Invariant to Generating Process\n'
             'k-NN graphs on crater centroids vs.\\ OSM road-network graphs (cities)',
             fontsize=8.5, fontweight='bold')

# --- Legend ---
legend_handles = [
    Line2D([0],[0], marker='h', color='w', markerfacecolor=C_ROBBINS,
           markeredgecolor='#888888', markersize=8,
           label='Robbins k-NN sub-samples (crater centroids)'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor=C_CITY,
           markeredgecolor='black', markersize=8,
           label='Cities (OSM road networks, $N=300$ bootstrap)'),
    Line2D([0],[0], color=C_ABIOTIC, lw=1.2, ls='--',
           label='Bishop scarp ($|\\Delta H|=0.307$, abiotic)'),
    Line2D([0],[0], color=C_PASSIVE, lw=1.2, ls=':',
           label='Bishop off-scarp ($|\\Delta H|=0.074$, passive)'),
]
ax.legend(handles=legend_handles, loc='upper right', fontsize=6.5,
          framealpha=0.95, edgecolor='#cccccc')

ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
fig.tight_layout()
fig.savefig(str(OUT), dpi=300)
print(f'Saved: {OUT}')
plt.close(fig)
