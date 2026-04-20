"""
kernelcal.attention.landauer_plots
==================================
Figure generation for the Landauer bound experiment.

This module owns all matplotlib code used by
:mod:`kernelcal.attention.landauer`.  Keeping the numerics and plotting
split lets the training sweep stay importable on headless / no-matplotlib
installs, and makes it easier to regenerate figures from saved JSON without
re-running the experiment.

Back-compat
-----------
The public names here (``_generate_landauer_figures`` and
``_generate_landauer_figures_wall``) are also re-exported from
``kernelcal.attention.landauer`` under their original names, so any
``from kernelcal.attention.landauer import _generate_landauer_figures``
continues to work.  See ``add_wall_power.py`` for the canonical consumer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


__all__ = [
    "generate_landauer_figures",
    "generate_landauer_figures_wall",
    # Back-compat underscore names — historically exported as private.
    "_generate_landauer_figures",
    "_generate_landauer_figures_wall",
]


# Shared dark-theme palette used across both figures.
_BG, _SURF, _GOLD, _SILVER = '#0d1117', '#161b22', '#e2b44d', '#b0bec5'


def _style_axes(axes) -> None:
    for ax in axes:
        ax.set_facecolor(_SURF)
        ax.tick_params(colors='#78909c')
        for sp in ax.spines.values():
            sp.set_color('#263238')


def generate_landauer_figures(results: list, out: Path) -> None:
    """Render the three-panel summary figure for a completed Landauer sweep.

    Produces ``fig_landauer_results.{pdf,png}`` in ``out``.
    Silently returns on matplotlib import / plotting failure so headless
    experiment drivers do not crash on figure generation.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        lrs   = sorted(set(r['lr'] for r in results))
        widths = sorted(set(r['d_model'] for r in results))

        fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), dpi=150)
        fig.patch.set_facecolor(_BG)
        fig.suptitle('Landauer Bound Experiment: W_total / ΔI vs Learning Rate',
                     color=_GOLD, fontsize=12, y=1.01, fontweight='bold')

        cmap = cm.get_cmap('plasma', len(widths))
        _style_axes(axes)

        ax = axes[0]
        for wi, d in enumerate(widths):
            dr = [r for r in results if r['d_model'] == d]
            ax.scatter([r['delta_I'] for r in dr],
                       [r['watt_hours'] for r in dr],
                       c=[cmap(wi)]*len(dr), s=20, alpha=0.7, label=f'd={d}')
        ax.set_xlabel('ΔI (1 - CKA)', color=_SILVER, fontsize=9)
        ax.set_ylabel('W_total (Wh)', color=_SILVER, fontsize=9)
        ax.set_title('W vs ΔI', color='#e0e0e0', fontsize=9)
        ax.legend(fontsize=7, labelcolor=_SILVER, facecolor='#1a1a2e', edgecolor='#37474f')

        ax = axes[1]
        for wi, d in enumerate(widths):
            ratios_per_lr = []
            for lr in lrs:
                dr = [r for r in results if r['d_model'] == d and abs(r['lr']-lr) < 1e-10]
                if dr:
                    ratios_per_lr.append(np.mean([r['ratio_Wh_per_I'] for r in dr]))
                else:
                    ratios_per_lr.append(np.nan)
            ax.plot([str(lr) for lr in lrs], ratios_per_lr,
                    color=cmap(wi), marker='o', ms=5, lw=1.5, label=f'd={d}')
        ax.set_xlabel('Learning rate', color=_SILVER, fontsize=9)
        ax.set_ylabel('W_total / ΔI  (Wh/nat)', color=_SILVER, fontsize=9)
        ax.set_title('Speed limit: should decrease as lr→0', color='#e0e0e0', fontsize=9)
        ax.legend(fontsize=7, labelcolor=_SILVER, facecolor='#1a1a2e', edgecolor='#37474f')

        ax = axes[2]
        for wi, d in enumerate(widths):
            vels = []
            for lr in lrs:
                dr = [r for r in results if r['d_model'] == d and abs(r['lr']-lr) < 1e-10]
                if dr:
                    vels.append(np.mean([r['kernel_velocity_mean'] for r in dr]))
                else:
                    vels.append(np.nan)
            ax.plot([str(lr) for lr in lrs], vels,
                    color=cmap(wi), marker='s', ms=5, lw=1.5, label=f'd={d}')
        ax.set_xlabel('Learning rate', color=_SILVER, fontsize=9)
        ax.set_ylabel('Mean kernel velocity ‖Δk‖', color=_SILVER, fontsize=9)
        ax.set_title('Kernel speed limit test', color='#e0e0e0', fontsize=9)
        ax.legend(fontsize=7, labelcolor=_SILVER, facecolor='#1a1a2e', edgecolor='#37474f')

        plt.tight_layout()
        plt.savefig(out / 'fig_landauer_results.pdf', bbox_inches='tight', facecolor=_BG)
        plt.savefig(out / 'fig_landauer_results.png', bbox_inches='tight', facecolor=_BG)
        plt.close()
        print(f'[landauer] figures saved → {out}')
    except Exception as e:
        print(f'[landauer] figure generation failed: {e}')


def generate_landauer_figures_wall(results: list, out: Path) -> None:
    """Render the extended four-panel figure that includes wall-plug energy.

    If ``wall_kwh_per_run_estimate`` is absent from the result dicts, this
    delegates to :func:`generate_landauer_figures` (the GPU-only summary).
    """
    has_wall = any('wall_kwh_per_run_estimate' in r for r in results)
    if not has_wall:
        return generate_landauer_figures(results, out)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        lrs    = sorted(set(r['lr'] for r in results))
        widths = sorted(set(r['d_model'] for r in results))
        cmap   = cm.get_cmap('plasma', len(widths))

        fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), dpi=150)
        fig.patch.set_facecolor(_BG)
        fig.suptitle(
            'Landauer Bound — GPU energy vs Wall-plug energy comparison',
            color=_GOLD, fontsize=11, y=1.01, fontweight='bold',
        )
        _style_axes(axes)

        for wi, d in enumerate(widths):
            dr = [r for r in results if r['d_model'] == d]
            axes[0].scatter([r['delta_I'] for r in dr],
                            [r['watt_hours'] for r in dr],
                            c=[cmap(wi)]*len(dr), s=20, alpha=0.7, label=f'd={d}')
        axes[0].set_xlabel('ΔI (1-CKA)', color=_SILVER, fontsize=9)
        axes[0].set_ylabel('W_gpu (Wh)', color=_SILVER, fontsize=9)
        axes[0].set_title('GPU energy vs ΔI', color='#e0e0e0', fontsize=9)
        axes[0].legend(fontsize=7, labelcolor=_SILVER, facecolor='#1a1a2e', edgecolor='#37474f')

        for wi, d in enumerate(widths):
            vals = [np.mean([r['ratio_gpu_per_I'] for r in results
                             if r['d_model'] == d and abs(r['lr']-lr) < 1e-10])
                    if any(r['d_model'] == d and abs(r['lr']-lr) < 1e-10 for r in results)
                    else np.nan for lr in lrs]
            axes[1].plot([str(lr) for lr in lrs], vals, color=cmap(wi),
                         marker='o', ms=5, lw=1.5, label=f'd={d}')
        axes[1].set_xlabel('Learning rate', color=_SILVER, fontsize=9)
        axes[1].set_ylabel('W_gpu / ΔI', color=_SILVER, fontsize=9)
        axes[1].set_title('GPU energy ratio (should decrease as lr→0)', color='#e0e0e0', fontsize=9)
        axes[1].legend(fontsize=7, labelcolor=_SILVER, facecolor='#1a1a2e', edgecolor='#37474f')

        for wi, d in enumerate(widths):
            vals = [np.mean([r['ratio_wall_per_I'] for r in results
                             if r['d_model'] == d and abs(r['lr']-lr) < 1e-10
                             and 'ratio_wall_per_I' in r])
                    if any(r['d_model'] == d and abs(r['lr']-lr) < 1e-10
                           and 'ratio_wall_per_I' in r for r in results)
                    else np.nan for lr in lrs]
            axes[2].plot([str(lr) for lr in lrs], vals, color=cmap(wi),
                         marker='s', ms=5, lw=1.5, label=f'd={d}')
        axes[2].set_xlabel('Learning rate', color=_SILVER, fontsize=9)
        axes[2].set_ylabel('W_wall / ΔI', color=_SILVER, fontsize=9)
        axes[2].set_title('Wall-plug energy ratio (true thermodynamic cost)', color='#e0e0e0', fontsize=9)
        axes[2].legend(fontsize=7, labelcolor=_SILVER, facecolor='#1a1a2e', edgecolor='#37474f')

        overheads = [r['wall_gpu_overhead'] for r in results
                     if r.get('wall_gpu_overhead') is not None]
        if overheads:
            axes[3].hist(overheads, bins=15, color='#4fc3f7', alpha=0.8, edgecolor='white', lw=0.5)
            axes[3].axvline(np.mean(overheads), color=_GOLD, lw=2, ls='--',
                            label=f'mean={np.mean(overheads):.2f}x')
            axes[3].set_xlabel('Wall/GPU overhead factor', color=_SILVER, fontsize=9)
            axes[3].set_ylabel('Count', color=_SILVER, fontsize=9)
            axes[3].set_title('System overhead (>1 = CPU/cooling not in GPU reading)',
                              color='#e0e0e0', fontsize=9)
            axes[3].legend(fontsize=8, labelcolor=_SILVER, facecolor='#1a1a2e', edgecolor='#37474f')

        plt.tight_layout()
        plt.savefig(out / 'fig_landauer_wall.pdf', bbox_inches='tight', facecolor=_BG)
        plt.savefig(out / 'fig_landauer_wall.png', bbox_inches='tight', facecolor=_BG)
        plt.close()
        print(f'[landauer] wall-power figure saved → {out}/fig_landauer_wall.pdf')

    except Exception as e:
        print(f'[landauer] wall figure failed: {e}')


# Back-compat aliases for existing imports that used the underscore names.
_generate_landauer_figures = generate_landauer_figures
_generate_landauer_figures_wall = generate_landauer_figures_wall
