"""
kernelcal.attention.training_plots
==================================
Figure and JSON writers for :mod:`kernelcal.attention.training`.

This module owns the matplotlib code used by the training-dynamics (Fig. 5)
and ensemble-trajectories / final-distributions (Figs. 6, 7) outputs.
Keeping it separate from the experiment driver lets the numerical training
loop stay importable on headless installs and makes figures easy to
regenerate from saved JSON.

All public names are re-exported from ``kernelcal.attention.training``
under their historical underscore-prefixed names for back-compat.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .training import TrainingRecord


__all__ = [
    "save_training_results",
    "save_ensemble_results",
    "_save_results",
    "_save_ensemble_results",
]


# Shared dark-theme palette.
_BG, _SURF, _GOLD, _SILVER = '#0d1117', '#161b22', '#e2b44d', '#b0bec5'


def save_training_results(records: "List[TrainingRecord]", out: Path, verbose: bool) -> None:
    """Persist per-step diagnostics JSON and Fig. 5 training-dynamics PDF.

    Silently skips plotting if matplotlib is unavailable.
    """
    out.mkdir(parents=True, exist_ok=True)
    data = [
        dict(step=r.step, train_acc=r.train_acc, val_acc=r.val_acc,
             loss=r.loss, residual=r.residuals[0],
             H=r.h_entropies[0], delta_prime=r.fiedler_gaps[0],
             lambda1=r.fiedler_values[0], velocity=r.kernel_velocity[0])
        for r in records
    ]
    (out / 'training_diagnostics.json').write_text(json.dumps(data, indent=2))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        steps  = [r.step for r in records]
        acc_tr = [r.train_acc for r in records]
        acc_v  = [r.val_acc for r in records]
        loss   = [r.loss for r in records]
        H      = [r.h_entropies[0] for r in records]
        delta  = [r.fiedler_gaps[0] for r in records]
        resid  = [r.residuals[0] for r in records]
        vel    = [r.kernel_velocity[0] for r in records]

        fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)
        fig.patch.set_facecolor(_BG)
        fig.suptitle('MaxCal Diagnostics During Transformer Training (Modular Addition)',
                     color=_GOLD, fontsize=12, y=1.01)

        panels = [
            (axes[0, 0], steps, [acc_tr, acc_v], ['#a5d6a7', '#ffb74d'],
             ['Train acc', 'Val acc'],  'Accuracy', 'Grokking transition'),
            (axes[0, 1], steps, [loss], ['#ef5350'], ['Loss'],
             'Cross-entropy loss', 'Training loss'),
            (axes[0, 2], steps, [H], ['#4fc3f7'], ['H[h_t]'],
             'Spectral entropy', 'MaxCal path entropy'),
            (axes[1, 0], steps, [delta], ['#a5d6a7'], ["Δ'(h_t)"],
             "Fiedler gap Δ'", 'Stability margin'),
            (axes[1, 1], steps, [resid], ['#ce93d8'], ['||R-T||_∞'],
             'Field-eq residual',
             'Self-consistency: → 0 = converging to MaxCal fixed point'),
            (axes[1, 2], steps[1:], [vel[1:]], ['#ff8a65'], ['||Δh_t||'],
             'Kernel velocity',
             'Speed limit test: should decrease at convergence'),
        ]

        for ax, xs, ys, cs, labs, ylabel, title in panels:
            ax.set_facecolor(_SURF)
            for y, c, lab in zip(ys, cs, labs):
                ax.plot(xs[:len(y)], y, color=c, lw=1.6, label=lab)
            ax.set_xlabel('Step', color='#b0bec5', fontsize=9)
            ax.set_ylabel(ylabel, color='#b0bec5', fontsize=9)
            ax.set_title(title, color='#e0e0e0', fontsize=9)
            ax.tick_params(colors='#78909c')
            for sp in ax.spines.values():
                sp.set_color('#263238')
            if len(labs) > 1:
                ax.legend(fontsize=8, labelcolor='#b0bec5',
                          facecolor='#1a1a2e', edgecolor='#37474f')

        plt.tight_layout()
        plt.savefig(out / 'fig5_training_dynamics.pdf',
                    bbox_inches='tight', facecolor=_BG)
        plt.close()
        if verbose:
            print(f"[training] saved → {out}/fig5_training_dynamics.pdf")
    except Exception as e:
        if verbose:
            print(f"[training] plot skipped: {e}")


def save_ensemble_results(result: dict, out: Path, verbose: bool) -> None:
    """Persist ensemble summary JSON and Figs. 6/7 (trajectories + distributions)."""
    steps = result['steps']
    mean  = result['mean']
    std   = result['std']
    final = result['final']
    n_runs = len(result['raw'])

    summary = {
        'n_runs': n_runs, 'primes': result['primes'], 'seeds': result['seeds'],
        'final_distributions': {k: {'mean': float(v.mean()), 'std': float(v.std()),
                                    'min': float(v.min()), 'max': float(v.max())}
                                for k, v in final.items()},
    }
    (out / 'ensemble_summary.json').write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        METRICS = [
            ('residual',    '#ce93d8', 'Field-eq residual ||R-T||_inf',
             'Self-consistency → 0 = MaxCal fixed point'),
            ('H',           '#4fc3f7', 'Spectral entropy H[h_t]',
             'Path entropy — measures breadth of attention'),
            ('delta_prime', '#a5d6a7', "Fiedler gap Delta'(h_t)",
             'Stability margin — should increase with training'),
            ('velocity',    '#ff8a65', 'Kernel velocity ||Δh_t||',
             'Speed limit — should decrease with training'),
        ]
        colors_p = ['#80cbc4', '#ffcc02', '#f48fb1']

        # ── Fig 6: Trajectories with error bands ──────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
        fig.patch.set_facecolor(_BG)
        fig.suptitle(
            f'MaxCal Kernel Dynamics — Ensemble ({n_runs} runs, '
            f'primes={result["primes"]}, {result["seeds"]} seeds each)',
            color=_GOLD, fontsize=11, y=1.01,
        )

        metric_to_attr = {
            'residual': 'residuals',
            'H': 'h_entropies',
            'delta_prime': 'fiedler_gaps',
            'lambda1': 'fiedler_values',
            'velocity': 'kernel_velocity',
        }

        for ax, (key, col, ylabel, title) in zip(axes.ravel(), METRICS):
            ax.set_facecolor(_SURF)
            ax.tick_params(colors='#78909c')
            for sp in ax.spines.values():
                sp.set_color('#263238')
            mu, sg = mean[key], std[key]
            xs = steps[:len(mu)]
            if key == 'velocity':
                xs = steps[1:len(mu) + 1]
            ax.fill_between(xs, mu - sg, mu + sg,
                            alpha=0.3, color=col, label='mean ± 1 std')
            ax.plot(xs, mu, color=col, lw=2, label='mean')

            attr = metric_to_attr[key]
            for prime, c_p in zip(result['primes'], colors_p):
                runs_p = [i for i, m in enumerate(result['meta'])
                          if m['prime'] == prime]
                arr_p  = [result['raw'][i] for i in runs_p]
                vals_p = np.array([[getattr(r, attr)[0] for r in recs]
                                   for recs in arr_p])
                ax.plot(xs[:vals_p.shape[1]], vals_p.mean(axis=0),
                        color=c_p, lw=1, ls='--', alpha=0.7, label=f'p={prime}')
            ax.set_xlabel('Step', color=_SILVER, fontsize=9)
            ax.set_ylabel(ylabel, color=_SILVER, fontsize=9)
            ax.set_title(title, color='#e0e0e0', fontsize=9)
            ax.legend(fontsize=7, labelcolor=_SILVER,
                      facecolor='#1a1a2e', edgecolor='#37474f', ncol=2)

        plt.tight_layout()
        plt.savefig(out / 'fig6_ensemble_trajectories.pdf',
                    bbox_inches='tight', facecolor=_BG)
        plt.close()

        # ── Fig 7: Final-step distributions ───────────────────────────
        fig, axes = plt.subplots(1, 4, figsize=(14, 5), dpi=150)
        fig.patch.set_facecolor(_BG)
        fig.suptitle('Distribution of MaxCal Diagnostics at Final Training Step',
                     color=_GOLD, fontsize=11, y=1.01)

        for ax, (key, col, ylabel, _title) in zip(axes, METRICS):
            ax.set_facecolor(_SURF)
            ax.tick_params(colors='#78909c')
            for sp in ax.spines.values():
                sp.set_color('#263238')
            vals = final[key]
            vp = ax.violinplot([vals], positions=[0], showmedians=True,
                               showextrema=True)
            for pc in vp['bodies']:
                pc.set_facecolor(col)
                pc.set_alpha(0.7)
            vp['cmedians'].set_color('white')
            vp['cmaxes'].set_color('#37474f')
            vp['cmins'].set_color('#37474f')
            vp['cbars'].set_color('#37474f')
            for prime, c_p in zip(result['primes'], colors_p):
                runs_p = [i for i, m in enumerate(result['meta'])
                          if m['prime'] == prime]
                v_p = final[key][runs_p]
                xs_jit = np.random.uniform(-0.15, 0.15, len(v_p))
                ax.scatter(xs_jit, v_p, c=c_p, s=25, alpha=0.8,
                           label=f'p={prime}', zorder=5)
            ax.set_xticks([])
            ax.set_ylabel(ylabel, color=_SILVER, fontsize=9)
            ax.set_title(f'μ={vals.mean():.3f}\nσ={vals.std():.3f}',
                         color='#e0e0e0', fontsize=8)
            ax.legend(fontsize=7, labelcolor=_SILVER,
                      facecolor='#1a1a2e', edgecolor='#37474f')

        plt.tight_layout()
        plt.savefig(out / 'fig7_final_distributions.pdf',
                    bbox_inches='tight', facecolor=_BG)
        plt.close()

        if verbose:
            print(f"[ensemble] saved → {out}/fig6_ensemble_trajectories.pdf")
            print(f"[ensemble] saved → {out}/fig7_final_distributions.pdf")

    except Exception as e:
        if verbose:
            print(f"[ensemble] plots failed: {e}")
        import traceback
        traceback.print_exc()


# Back-compat aliases matching the historical private names in training.py.
_save_results = save_training_results
_save_ensemble_results = save_ensemble_results
