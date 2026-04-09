"""
Perturbation-and-relaxation experiment for attention kernels.

Tests whether converged attention kernels are dynamical attractors:
perturb a head's Q/K weights, resume training, measure return dynamics.

MaxCal prediction: displacement decays as δ_t ~ δ_0 exp(-α t)
where α ≈ Δ'(h*), the Fiedler gap of the pre-perturbation fixed point.

Usage:
    python -m kernelcal.attention.perturbation \\
        --checkpoint path/to/step_050000.pt \\
        --sigmas 0.01 0.05 0.1 0.2 \\
        --relax-steps 2000 \\
        --output-dir figures/perturbation
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PerturbationResult:
    """Result of perturbing one head and measuring relaxation."""
    layer: int
    head: int
    sigma: float
    pre_fiedler_gap: float
    pre_spectral_entropy: float
    displacements: List[float]       # δ_t = ||k_t - k*||_F at each logged step
    steps: List[int]
    fitted_alpha: float              # exponential decay rate
    fitted_r2: float                 # fit quality
    predicted_alpha: float           # Δ'(h*) from pre-perturbation
    spectral_entropies: List[float]
    fiedler_gaps: List[float]


@dataclass
class RelaxationSummary:
    """Aggregate results across heads and sigma values."""
    results: List[PerturbationResult]
    alpha_vs_delta_prime_corr: float  # Pearson r between fitted α and Δ'


def _fit_exponential_decay(steps: np.ndarray, displacements: np.ndarray
                           ) -> Tuple[float, float]:
    """Fit δ_t ~ δ_0 exp(-α t) via log-linear regression. Returns (alpha, r2)."""
    mask = displacements > 1e-12
    if mask.sum() < 3:
        return 0.0, 0.0
    t = steps[mask].astype(float)
    y = np.log(displacements[mask])
    t_mean, y_mean = t.mean(), y.mean()
    ss_xy = ((t - t_mean) * (y - y_mean)).sum()
    ss_xx = ((t - t_mean) ** 2).sum()
    if ss_xx < 1e-15:
        return 0.0, 0.0
    slope = ss_xy / ss_xx
    alpha = -slope
    y_pred = y_mean + slope * (t - t_mean)
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1.0 - ss_res / (ss_tot + 1e-15)
    return float(alpha), float(r2)


def perturb_and_relax(
    checkpoint_path: str,
    target_heads: Optional[List[Tuple[int, int]]] = None,
    sigmas: List[float] = (0.01, 0.05, 0.1, 0.2),
    relax_steps: int = 2000,
    log_every: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    batch_size: int = 128,
    sigma2: float = 1.0,
    mu2: float = 2.0,
    device_str: Optional[str] = None,
    verbose: bool = True,
) -> RelaxationSummary:
    """
    Load a converged checkpoint, perturb target heads, resume training,
    and measure return-to-fixed-point dynamics.

    Parameters
    ----------
    checkpoint_path : str
        Path to a checkpoint saved by training.py (contains model state,
        optimizer state, and config).
    target_heads : list of (layer, head) tuples
        Which heads to perturb. None = sample 10 heads across layers.
    sigmas : list of float
        Perturbation magnitudes to test.
    relax_steps : int
        Number of training steps after perturbation.
    """
    import torch
    import torch.nn.functional as F
    from .device import best_device
    from .kernel import AttentionKernel

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    prime = ckpt['prime']
    d_model = ckpt['d_model']
    n_heads = ckpt['n_heads']
    n_layers = ckpt['n_layers']

    device = best_device(device_str)
    from .training import _build_model, _make_dataset
    base_model = _build_model(prime, d_model, n_heads, n_layers, device)
    base_model.load_state_dict(ckpt['model'])

    train_d, val_d = _make_dataset(prime, device)

    if target_heads is None:
        all_heads = [(li, hi) for li in range(n_layers) for hi in range(n_heads)]
        rng = np.random.default_rng(42)
        n_sample = min(10, len(all_heads))
        idxs = rng.choice(len(all_heads), n_sample, replace=False)
        target_heads = [all_heads[i] for i in sorted(idxs)]

    # Capture pre-perturbation kernels and diagnostics
    base_model.eval()
    with torch.inference_mode():
        probe = train_d[:8, :-1]
        base_model(probe)
        attn_ws = base_model.attn_weights()

    pre_diagnostics: Dict[Tuple[int,int], dict] = {}
    pre_kernels: Dict[Tuple[int,int], np.ndarray] = {}
    for li, hi in target_heads:
        if attn_ws[li] is None:
            continue
        A = attn_ws[li][0, hi].float().cpu().numpy()
        ak = AttentionKernel(A, layer=li, head=hi, sigma2=sigma2, mu2=mu2)
        res = ak.analyse()
        pre_diagnostics[(li, hi)] = {
            'fiedler_gap': res.fiedler_gap,
            'spectral_entropy': res.spectral_entropy,
            'h_star': res.h_star.copy(),
        }
        pre_kernels[(li, hi)] = A.copy()

    if verbose:
        print(f"[perturbation] {len(target_heads)} heads × {len(sigmas)} sigmas "
              f"× {relax_steps} steps  prime={prime} d={d_model}")

    all_results: List[PerturbationResult] = []

    for li, hi in target_heads:
        if (li, hi) not in pre_diagnostics:
            continue
        pre = pre_diagnostics[(li, hi)]
        K_star = pre_kernels[(li, hi)]

        for sigma in sigmas:
            # Rebuild model from checkpoint
            model = _build_model(prime, d_model, n_heads, n_layers, device)
            model.load_state_dict(ckpt['model'])

            # Perturb W_Q and W_K for the target head
            block = model.blocks[li]
            with torch.no_grad():
                qkv_w = block.attn.qkv.weight  # (3*d_model, d_model)
                d_head = d_model // n_heads
                q_start = hi * d_head
                q_end = q_start + d_head
                k_start = d_model + hi * d_head
                k_end = k_start + d_head
                noise_q = torch.randn_like(qkv_w[q_start:q_end]) * sigma
                noise_k = torch.randn_like(qkv_w[k_start:k_end]) * sigma
                qkv_w[q_start:q_end] += noise_q
                qkv_w[k_start:k_end] += noise_k

            opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                    weight_decay=weight_decay)

            displacements = []
            step_list = []
            h_ents = []
            f_gaps = []

            model.train()
            for step in range(relax_steps + 1):
                idx = torch.randint(len(train_d), (min(batch_size, len(train_d)),))
                seqs = train_d[idx]
                x, y = seqs[:, :-1], seqs[:, 1:]
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, prime + 3), y.reshape(-1))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if step % log_every == 0:
                    model.eval()
                    with torch.inference_mode():
                        model(probe)
                        ws = model.attn_weights()
                    A_now = ws[li][0, hi].float().cpu().numpy()
                    delta = float(np.linalg.norm(A_now - K_star))
                    displacements.append(delta)
                    step_list.append(step)

                    ak_now = AttentionKernel(A_now, layer=li, head=hi,
                                            step=step, sigma2=sigma2, mu2=mu2)
                    res_now = ak_now.analyse(fp_max_iter=100, fp_tol=1e-7)
                    h_ents.append(res_now.spectral_entropy)
                    f_gaps.append(res_now.fiedler_gap)
                    model.train()

            steps_arr = np.array(step_list)
            disp_arr = np.array(displacements)
            fitted_alpha, r2 = _fit_exponential_decay(steps_arr, disp_arr)

            result = PerturbationResult(
                layer=li, head=hi, sigma=sigma,
                pre_fiedler_gap=pre['fiedler_gap'],
                pre_spectral_entropy=pre['spectral_entropy'],
                displacements=displacements,
                steps=step_list,
                fitted_alpha=fitted_alpha,
                fitted_r2=r2,
                predicted_alpha=pre['fiedler_gap'],
                spectral_entropies=h_ents,
                fiedler_gaps=f_gaps,
            )
            all_results.append(result)

            if verbose:
                print(f"  L{li}H{hi} σ={sigma:.2f}: δ₀={displacements[0]:.4f} → "
                      f"δ_end={displacements[-1]:.4f}  "
                      f"α_fit={fitted_alpha:.4f}  α_pred={pre['fiedler_gap']:.4f}  "
                      f"R²={r2:.3f}")

    # Correlation between fitted α and predicted Δ'
    if len(all_results) >= 3:
        alphas = np.array([r.fitted_alpha for r in all_results])
        deltas = np.array([r.pre_fiedler_gap for r in all_results])
        if alphas.std() > 0 and deltas.std() > 0:
            corr = float(np.corrcoef(alphas, deltas)[0, 1])
        else:
            corr = 0.0
    else:
        corr = 0.0

    return RelaxationSummary(results=all_results,
                             alpha_vs_delta_prime_corr=corr)


def save_perturbation_results(summary: RelaxationSummary, output_dir: str,
                              verbose: bool = True) -> None:
    """Save JSON and figures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = {
        'alpha_vs_delta_prime_corr': summary.alpha_vs_delta_prime_corr,
        'n_experiments': len(summary.results),
        'results': [
            {
                'layer': r.layer, 'head': r.head, 'sigma': r.sigma,
                'pre_fiedler_gap': r.pre_fiedler_gap,
                'fitted_alpha': r.fitted_alpha, 'fitted_r2': r.fitted_r2,
                'predicted_alpha': r.predicted_alpha,
                'initial_displacement': r.displacements[0] if r.displacements else 0,
                'final_displacement': r.displacements[-1] if r.displacements else 0,
            }
            for r in summary.results
        ],
    }
    (out / 'perturbation_results.json').write_text(json.dumps(data, indent=2))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        BG, SURF, GOLD = '#0d1117', '#161b22', '#e2b44d'

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
        fig.patch.set_facecolor(BG)
        fig.suptitle('Perturbation–Relaxation Dynamics', color=GOLD,
                     fontsize=12, y=1.01)

        for ax in axes:
            ax.set_facecolor(SURF)
            ax.tick_params(colors='#78909c')
            for sp in ax.spines.values():
                sp.set_color('#263238')

        # Panel 1: displacement curves
        for r in summary.results:
            label = f"L{r.layer}H{r.head} σ={r.sigma:.2f}"
            axes[0].semilogy(r.steps, r.displacements, lw=1, alpha=0.7, label=label)
        axes[0].set_xlabel('Step', color='#b0bec5', fontsize=9)
        axes[0].set_ylabel('||k_t - k*||_F', color='#b0bec5', fontsize=9)
        axes[0].set_title('Displacement from fixed point', color='#e0e0e0', fontsize=9)

        # Panel 2: fitted α vs predicted Δ'
        alphas = [r.fitted_alpha for r in summary.results]
        deltas = [r.predicted_alpha for r in summary.results]
        axes[1].scatter(deltas, alphas, c='#4fc3f7', s=30, alpha=0.8)
        lim = max(max(alphas, default=1), max(deltas, default=1)) * 1.1
        axes[1].plot([0, lim], [0, lim], '--', color=GOLD, lw=1, label='α = Δ\'')
        axes[1].set_xlabel("Predicted α (Δ')", color='#b0bec5', fontsize=9)
        axes[1].set_ylabel('Fitted α', color='#b0bec5', fontsize=9)
        axes[1].set_title(f'α vs Δ\' (r={summary.alpha_vs_delta_prime_corr:.3f})',
                          color='#e0e0e0', fontsize=9)
        axes[1].legend(fontsize=8, labelcolor='#b0bec5',
                       facecolor='#1a1a2e', edgecolor='#37474f')

        # Panel 3: R² distribution
        r2s = [r.fitted_r2 for r in summary.results]
        axes[2].hist(r2s, bins=15, color='#a5d6a7', alpha=0.8, edgecolor='white', lw=0.5)
        axes[2].set_xlabel('R² of exponential fit', color='#b0bec5', fontsize=9)
        axes[2].set_ylabel('Count', color='#b0bec5', fontsize=9)
        axes[2].set_title('Fit quality distribution', color='#e0e0e0', fontsize=9)

        plt.tight_layout()
        plt.savefig(out / 'fig_perturbation.pdf', bbox_inches='tight', facecolor=BG)
        plt.close()
        if verbose:
            print(f"[perturbation] saved → {out}/fig_perturbation.pdf")
    except Exception as e:
        if verbose:
            print(f"[perturbation] plot failed: {e}")


def _main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Perturbation-relaxation experiment for attention kernels.')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.2])
    parser.add_argument('--relax-steps', type=int, default=2000)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output-dir', default='figures/perturbation')
    args = parser.parse_args()

    summary = perturb_and_relax(
        checkpoint_path=args.checkpoint,
        sigmas=args.sigmas,
        relax_steps=args.relax_steps,
        log_every=args.log_every,
        device_str=args.device,
    )
    save_perturbation_results(summary, args.output_dir)
    print(f"\n[perturbation] α vs Δ' correlation: {summary.alpha_vs_delta_prime_corr:.4f}")


if __name__ == '__main__':
    _main()
