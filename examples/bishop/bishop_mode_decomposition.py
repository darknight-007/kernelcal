#!/usr/bin/env python3
"""
bishop_mode_decomposition.py
=============================
Spectral mode decomposition of the Bishop fault scarp rock field.
Maps dominant Laplacian eigenvectors back to physical space and
connects them to the three abiotic controller length scales.

Goal: identify which spectral modes carry which trait variance,
and whether those modes correspond to volcanic fracturing (~1m),
geomorphic transport (~10-50m), or tectonic strain (~200-500m).
"""

from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

KCAL_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(KCAL_ROOT))
from kernelcal.terrain.diagnostics import fixed_point_kernel

BASE    = Path(__file__).resolve().parent.parent.parent / 'datasets' / 'bishop_scarp'
FIG_DIR = Path(__file__).resolve().parent.parent.parent / 'bishop_figures'
FIG_DIR.mkdir(exist_ok=True)

N_SUB   = 2000
K_NN    = 8
SIGMA_M = 1.0
MU2, SIGMA2 = 2.0, 1.0

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def load_and_prepare():
    data = np.loadtxt(str(BASE / 'rock_traits_full.csv'), delimiter=',', skiprows=1)
    lonlat = data[:, :2]
    lon0, lat0 = lonlat[:, 0].mean(), lonlat[:, 1].mean()
    R = 6_371_000.0
    cos0 = math.cos(math.radians(lat0))
    xy = np.column_stack([
        (lonlat[:, 0] - lon0) * cos0 * (math.pi / 180) * R,
        (lonlat[:, 1] - lat0) * (math.pi / 180) * R,
    ])
    rng = np.random.default_rng(42)
    idx = rng.choice(len(xy), size=min(N_SUB, len(xy)), replace=False)
    return xy[idx], data[idx], idx


def build_laplacian(xy):
    N = len(xy)
    tree = cKDTree(xy)
    dists, idxs = tree.query(xy, k=K_NN + 1)
    A = np.zeros((N, N))
    for i in range(N):
        for r in range(1, K_NN + 1):
            j = idxs[i, r]
            w = math.exp(-dists[i, r]**2 / (2 * SIGMA_M**2))
            A[i, j] += w; A[j, i] += w
    A = np.minimum(A, 1.0)
    return np.diag(A.sum(axis=1)) - A


def eigenvalue_to_wavelength(eigvals, median_nn):
    """Approximate spatial wavelength for each eigenvalue."""
    wavelengths = np.zeros_like(eigvals)
    mask = eigvals > 1e-10
    wavelengths[mask] = 2 * np.pi * median_nn / np.sqrt(eigvals[mask])
    wavelengths[~mask] = np.inf
    return wavelengths


def main():
    print('Loading and subsampling...')
    xy, data, idx = load_and_prepare()
    N = len(xy)
    print(f'  N = {N} scarp rocks')

    # Median nearest-neighbour distance for wavelength estimation
    tree = cKDTree(xy)
    nn_dists, _ = tree.query(xy, k=2)
    median_nn = np.median(nn_dists[:, 1])
    print(f'  Median NN distance: {median_nn:.3f} m')

    print('Building Laplacian and computing eigenpairs...')
    L = build_laplacian(xy)
    eigvals, eigvecs = np.linalg.eigh(L)

    wavelengths = eigenvalue_to_wavelength(eigvals, median_nn)
    print(f'  Eigenvalue range: [{eigvals[0]:.4f}, {eigvals[-1]:.2f}]')
    print(f'  Wavelength range: [{wavelengths[1]:.1f} m, {wavelengths[-1]:.3f} m]')
    print(f'  (mode 0 = DC, wavelength = inf)')

    # Trait signals
    traits = {
        'Rock area': data[:, 2],
        'Eccentricity': data[:, 5],
        'Orientation': data[:, 6],
        'Elevation': data[:, 7],
    }

    print('\nSpectral projection per trait:')
    projections = {}
    for name, signal in traits.items():
        valid = ~np.isnan(signal)
        sig = signal.copy()
        sig[~valid] = np.nanmean(signal)
        sig_c = sig - sig.mean()
        c_l = eigvecs.T @ sig_c
        w_l = c_l**2
        w_l_norm = w_l / (w_l.sum() + 1e-30)

        top10 = np.argsort(w_l_norm)[-10:][::-1]
        projections[name] = dict(c_l=c_l, w_l_norm=w_l_norm, top10=top10, signal=sig)

        print(f'\n  {name}:')
        print(f'    Top-10 modes: {top10}')
        print(f'    Top-10 wavelengths: {[f"{wavelengths[m]:.1f}m" for m in top10]}')
        print(f'    Top-10 weights: {[f"{w_l_norm[m]:.3f}" for m in top10]}')
        print(f'    Cumulative top-10: {w_l_norm[top10].sum():.3f}')

    # Controller scale bands
    bands = {
        'Tectonic (>100 m)': wavelengths > 100,
        'Transport (10-100 m)': (wavelengths >= 10) & (wavelengths <= 100),
        'Fracture (<10 m)': (wavelengths > 0) & (wavelengths < 10),
    }

    print('\n\nMode weight by controller scale band:')
    print(f'  {"Trait":<16} {"Tectonic >100m":>16} {"Transport 10-100m":>18} {"Fracture <10m":>16}')
    print(f'  {"-"*16} {"-"*16} {"-"*18} {"-"*16}')
    for name in traits:
        w = projections[name]['w_l_norm']
        t = w[bands['Tectonic (>100 m)']].sum()
        tr = w[bands['Transport (10-100 m)']].sum()
        f = w[bands['Fracture (<10 m)']].sum()
        print(f'  {name:<16} {t:>16.3f} {tr:>18.3f} {f:>16.3f}')

    # ══════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════

    # Fig 1: Top eigenvectors in physical space
    print('\nGenerating figures...')
    modes_to_show = [1, 2, 3, 5, 7, 10, 20, 50]
    modes_to_show = [m for m in modes_to_show if m < N]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, m in zip(axes.flat, modes_to_show):
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=2,
                        c=eigvecs[:, m], cmap='RdBu_r', alpha=0.7,
                        edgecolors='none', vmin=-0.06, vmax=0.06)
        ax.set_aspect('equal')
        ax.set_title(f'Mode {m}  (λ={eigvals[m]:.3f},  ~{wavelengths[m]:.0f} m)',
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('East [m]', fontsize=7)
        ax.set_ylabel('North [m]', fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle('Laplacian Eigenvectors in Physical Space — Bishop Fault Scarp\n'
                 'Red/blue = positive/negative eigenvector amplitude at each rock',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_D1_eigenvectors_physical.png', dpi=200)
    plt.close(fig)
    print('  Saved fig_D1_eigenvectors_physical.png')

    # Fig 2: Mode weights by trait with wavelength axis
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {'Rock area': '#D55E00', 'Eccentricity': '#0072B2',
              'Orientation': '#009E73', 'Elevation': '#CC79A7'}

    for ax, (name, col) in zip(axes.flat, colors.items()):
        w = projections[name]['w_l_norm']
        n_show = min(100, len(w))
        ax.bar(range(n_show), w[:n_show], color=col, alpha=0.7, width=1.0)

        # Annotate controller bands
        ax.axvspan(0, np.searchsorted(eigvals, (2*np.pi*median_nn/100)**2),
                   alpha=0.08, color='red', label='>100 m (tectonic)')
        trans_start = np.searchsorted(eigvals, (2*np.pi*median_nn/100)**2)
        trans_end = np.searchsorted(eigvals, (2*np.pi*median_nn/10)**2)
        ax.axvspan(trans_start, min(trans_end, n_show),
                   alpha=0.08, color='orange', label='10-100 m (transport)')

        # Mark top mode
        top = projections[name]['top10'][0]
        if top < n_show:
            ax.annotate(f'mode {top}\n~{wavelengths[top]:.0f} m',
                        xy=(top, w[top]), xytext=(top+5, w[top]*1.3),
                        fontsize=7, fontweight='bold', color=col,
                        arrowprops=dict(arrowstyle='->', color=col, lw=0.8))

        ax.set_xlabel('Mode index l')
        ax.set_ylabel('Mode weight w_l')
        ax.set_title(name, fontweight='bold')
        ax.set_xlim(-1, n_show)
        ax.legend(fontsize=6, loc='upper right')

    fig.suptitle('Trait Spectral Projections with Controller Scale Bands\n'
                 'Red band = tectonic (>100m)  |  Orange band = transport (10-100m)  |  '
                 'Unshaded = fracture (<10m)',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_D2_trait_mode_weights_bands.png', dpi=200)
    plt.close(fig)
    print('  Saved fig_D2_trait_mode_weights_bands.png')

    # Fig 3: Controller scale decomposition bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    trait_names = list(traits.keys())
    x = np.arange(len(trait_names))
    width = 0.25

    tectonic_w = []
    transport_w = []
    fracture_w = []
    for name in trait_names:
        w = projections[name]['w_l_norm']
        tectonic_w.append(w[bands['Tectonic (>100 m)']].sum())
        transport_w.append(w[bands['Transport (10-100 m)']].sum())
        fracture_w.append(w[bands['Fracture (<10 m)']].sum())

    bars1 = ax.bar(x - width, tectonic_w, width, color='#E63946',
                   alpha=0.8, label='Tectonic (>100 m)')
    bars2 = ax.bar(x, transport_w, width, color='#F4A261',
                   alpha=0.8, label='Transport (10–100 m)')
    bars3 = ax.bar(x + width, fracture_w, width, color='#2A9D8F',
                   alpha=0.8, label='Fracture (<10 m)')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f'{h:.2f}', ha='center', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(trait_names, fontsize=9)
    ax.set_ylabel('Fraction of trait variance in scale band')
    ax.set_title('Controller Scale Decomposition of Rock Traits\n'
                 'Which abiotic controller dominates each trait\'s spatial pattern?',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_D3_controller_scale_decomposition.png', dpi=200)
    plt.close(fig)
    print('  Saved fig_D3_controller_scale_decomposition.png')

    # Fig 4: Dominant mode for rock area visualized in space
    top_area_mode = projections['Rock area']['top10'][0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    sig = projections['Rock area']['signal']
    sig_clip = np.clip(sig, 0, np.percentile(sig, 95))
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=3, c=sig_clip, cmap='jet',
                    alpha=0.7, edgecolors='none')
    ax.set_aspect('equal')
    ax.set_title('Rock area (m²)', fontweight='bold')
    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]')
    fig.colorbar(sc, ax=ax, shrink=0.7)

    ax = axes[1]
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=3,
                    c=eigvecs[:, top_area_mode], cmap='RdBu_r',
                    alpha=0.7, edgecolors='none')
    ax.set_aspect('equal')
    ax.set_title(f'Eigenvector φ_{top_area_mode} (~{wavelengths[top_area_mode]:.0f} m)',
                 fontweight='bold')
    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]')
    fig.colorbar(sc, ax=ax, shrink=0.7)

    ax = axes[2]
    elev = projections['Elevation']['signal']
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=3, c=elev, cmap='terrain',
                    alpha=0.7, edgecolors='none')
    ax.set_aspect('equal')
    ax.set_title('Elevation (relative)', fontweight='bold')
    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]')
    fig.colorbar(sc, ax=ax, shrink=0.7)

    fig.suptitle(f'Rock Area Concentrates at Mode {top_area_mode} '
                 f'(~{wavelengths[top_area_mode]:.0f} m wavelength)\n'
                 'The dominant spatial pattern of rock size matches the scarp-scale '
                 'tectonic structure',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_D4_area_dominant_mode.png', dpi=200)
    plt.close(fig)
    print('  Saved fig_D4_area_dominant_mode.png')

    print(f'\nAll figures → {FIG_DIR}/')
    print('\nDone.')


if __name__ == '__main__':
    main()
