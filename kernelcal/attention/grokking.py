"""
Extended grokking experiment: spectral diagnostics across phase transition.

Trains modular-arithmetic transformers for 50K+ steps with per-head
spectral diagnostics, phase-transition detection, and optional
path-entropy estimation over the ensemble.

MaxCal prediction: the grokking transition appears as a spectral phase
transition — sharp drop in H[h_t] and Fiedler discontinuity BEFORE
accuracy changes. The transition epoch is predictable from H < H_c.

Usage:
    python -m kernelcal.attention.grokking \\
        --primes 23 53 97 \\
        --widths 64 128 256 \\
        --seeds 50 --steps 50000 \\
        --output-dir figures/grokking

    # Quick test (1 seed, 1 width, 5K steps)
    python -m kernelcal.attention.grokking --quick
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GrokkingRecord:
    """Per-step record with per-head diagnostics."""
    step: int
    train_acc: float
    val_acc: float
    loss: float
    per_head_entropy: List[List[float]]    # [layer][head] → H
    per_head_fiedler: List[List[float]]    # [layer][head] → λ₁
    per_head_gap: List[List[float]]        # [layer][head] → Δ'
    per_head_velocity: List[List[float]]   # [layer][head] → ||Δh||
    mean_entropy: float
    mean_fiedler: float
    mean_gap: float
    mean_velocity: float


@dataclass
class PhaseTransition:
    """Detected phase transition from spectral diagnostics."""
    spectral_epoch: int          # first step where H < H_c
    accuracy_epoch: int          # first step where val_acc > 0.9
    spectral_leads_by: int       # spectral_epoch - accuracy_epoch (neg = spectral first)
    H_at_transition: float
    lambda1_at_transition: float


def detect_phase_transition(
    records: List[GrokkingRecord],
    H_threshold_quantile: float = 0.1,
    acc_threshold: float = 0.9,
) -> Optional[PhaseTransition]:
    """Detect grokking transition from spectral and accuracy signals.

    H_threshold is set as the bottom quantile of the pre-transition
    spectral entropy distribution (first 20% of training).
    """
    if len(records) < 10:
        return None

    entropies = np.array([r.mean_entropy for r in records])
    val_accs = np.array([r.val_acc for r in records])
    steps = np.array([r.step for r in records])

    n_early = max(5, len(records) // 5)
    H_c = np.quantile(entropies[:n_early], H_threshold_quantile)

    spectral_idx = np.where(entropies < H_c)[0]
    spectral_epoch = int(steps[spectral_idx[0]]) if len(spectral_idx) > 0 else -1

    acc_idx = np.where(val_accs > acc_threshold)[0]
    accuracy_epoch = int(steps[acc_idx[0]]) if len(acc_idx) > 0 else -1

    if spectral_epoch < 0 and accuracy_epoch < 0:
        return None

    leads_by = (spectral_epoch - accuracy_epoch) if spectral_epoch > 0 and accuracy_epoch > 0 else 0

    h_at = float(entropies[spectral_idx[0]]) if len(spectral_idx) > 0 else float(entropies[-1])
    fiedlers = np.array([r.mean_fiedler for r in records])
    l1_at = float(fiedlers[spectral_idx[0]]) if len(spectral_idx) > 0 else float(fiedlers[-1])

    return PhaseTransition(
        spectral_epoch=spectral_epoch,
        accuracy_epoch=accuracy_epoch,
        spectral_leads_by=leads_by,
        H_at_transition=h_at,
        lambda1_at_transition=l1_at,
    )


def run_grokking_experiment(
    prime: int = 97,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    n_steps: int = 50000,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    log_every: int = 100,
    checkpoint_every: int = 5000,
    sigma2: float = 1.0,
    mu2: float = 2.0,
    seed: int = 0,
    device_str: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[List[GrokkingRecord], Optional[PhaseTransition]]:
    """Run extended grokking with per-head spectral diagnostics."""
    import torch
    import torch.nn.functional as F
    from .device import best_device
    from .kernel import AttentionKernel
    from .energy import EnergyMonitor
    from .training import _build_model, _make_dataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = best_device(device_str)
    model = _build_model(prime, d_model, n_heads, n_layers, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_d, val_d = _make_dataset(prime, device)
    n_params = sum(p.numel() for p in model.parameters())

    energy = EnergyMonitor.auto_detect()
    energy.start()

    ckpt_dir = None
    if output_dir:
        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[grokking] prime={prime} d={d_model} h={n_heads} L={n_layers} "
              f"steps={n_steps} seed={seed}  energy={energy._sources}")

    prev_h: Dict[Tuple[int,int], np.ndarray] = {}
    records: List[GrokkingRecord] = []
    probe = train_d[:8, :-1]
    t0 = time.time()

    for step in range(n_steps + 1):
        idx = torch.randint(len(train_d), (min(batch_size, len(train_d)),))
        seqs = train_d[idx]
        x, y = seqs[:, :-1], seqs[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, prime + 3), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        energy.add_training_step(n_params, seqs[:, :-1].numel())

        if ckpt_dir and checkpoint_every > 0 and step > 0 and step % checkpoint_every == 0:
            torch.save({
                'step': step, 'model': model.state_dict(),
                'optimizer': opt.state_dict(),
                'prime': prime, 'd_model': d_model,
                'n_heads': n_heads, 'n_layers': n_layers,
            }, ckpt_dir / f"step_{step:06d}.pt")

        if step % log_every != 0:
            continue

        model.eval()
        with torch.inference_mode():
            logits_tr = model(train_d[:, :-1])
            pred_tr = logits_tr[:, -2, :prime].argmax(-1)
            acc_tr = (pred_tr == train_d[:, 4]).float().mean().item()
            logits_v = model(val_d[:, :-1])
            pred_v = logits_v[:, -2, :prime].argmax(-1)
            acc_v = (pred_v == val_d[:, 4]).float().mean().item()
            model(probe)
            attn_ws = model.attn_weights()

        per_head_H = []
        per_head_lam = []
        per_head_gap = []
        per_head_vel = []

        for li, attn_w in enumerate(attn_ws):
            if attn_w is None:
                continue
            attn_arr = attn_w[0].float().cpu().numpy()
            layer_H, layer_lam, layer_gap, layer_vel = [], [], [], []

            for hi in range(attn_arr.shape[0]):
                A = attn_arr[hi]
                ak = AttentionKernel(A, layer=li, head=hi, step=step,
                                     sigma2=sigma2, mu2=mu2, eigenvalue_aware=True)
                res = ak.analyse(fp_max_iter=100, fp_tol=1e-7)

                key = (li, hi)
                vel = 0.0
                if key in prev_h:
                    vel = float(np.linalg.norm(res.h_star - prev_h[key]))
                prev_h[key] = res.h_star.copy()

                layer_H.append(res.spectral_entropy)
                layer_lam.append(res.fiedler_value)
                layer_gap.append(res.fiedler_gap)
                layer_vel.append(vel)

            per_head_H.append(layer_H)
            per_head_lam.append(layer_lam)
            per_head_gap.append(layer_gap)
            per_head_vel.append(layer_vel)

        all_H = [v for layer in per_head_H for v in layer]
        all_lam = [v for layer in per_head_lam for v in layer]
        all_gap = [v for layer in per_head_gap for v in layer]
        all_vel = [v for layer in per_head_vel for v in layer]

        rec = GrokkingRecord(
            step=step, train_acc=acc_tr, val_acc=acc_v, loss=float(loss.item()),
            per_head_entropy=per_head_H, per_head_fiedler=per_head_lam,
            per_head_gap=per_head_gap, per_head_velocity=per_head_vel,
            mean_entropy=float(np.mean(all_H)) if all_H else 0,
            mean_fiedler=float(np.mean(all_lam)) if all_lam else 0,
            mean_gap=float(np.mean(all_gap)) if all_gap else 0,
            mean_velocity=float(np.mean(all_vel)) if all_vel else 0,
        )
        records.append(rec)

        if verbose and step % (log_every * 10) == 0:
            print(f"  step={step:6d}  loss={rec.loss:.3f}  "
                  f"tr={rec.train_acc:.3f}  val={rec.val_acc:.3f}  "
                  f"H={rec.mean_entropy:.3f}  Δ'={rec.mean_gap:.3f}  "
                  f"vel={rec.mean_velocity:.4f}")

        model.train()

    energy_report = energy.stop()
    elapsed = time.time() - t0

    transition = detect_phase_transition(records)

    if verbose:
        print(f"[grokking] done  {elapsed:.1f}s  energy={energy_report.total_joules:.1f}J")
        if transition:
            print(f"[grokking] phase transition: spectral@{transition.spectral_epoch}  "
                  f"accuracy@{transition.accuracy_epoch}  "
                  f"spectral leads by {-transition.spectral_leads_by} steps")
        else:
            print("[grokking] no phase transition detected in this run")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary = {
            'prime': prime, 'd_model': d_model, 'n_heads': n_heads,
            'n_layers': n_layers, 'n_steps': n_steps, 'seed': seed,
            'elapsed_s': elapsed,
            'energy': energy_report.summary(),
            'transition': {
                'spectral_epoch': transition.spectral_epoch,
                'accuracy_epoch': transition.accuracy_epoch,
                'spectral_leads_by': transition.spectral_leads_by,
            } if transition else None,
            'records': [
                {'step': r.step, 'train_acc': r.train_acc, 'val_acc': r.val_acc,
                 'loss': r.loss, 'H': r.mean_entropy, 'lambda1': r.mean_fiedler,
                 'delta_prime': r.mean_gap, 'velocity': r.mean_velocity}
                for r in records
            ],
        }
        (out / f'grokking_p{prime}_d{d_model}_s{seed}.json').write_text(
            json.dumps(summary, indent=2))

    return records, transition


def _main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Extended grokking experiment with spectral phase detection.')
    parser.add_argument('--primes', type=int, nargs='+', default=[97])
    parser.add_argument('--widths', type=int, nargs='+', default=[64])
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--checkpoint-every', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output-dir', default='figures/grokking')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 1 prime, 1 width, 5K steps')
    args = parser.parse_args()

    if args.quick:
        args.primes = [23]
        args.widths = [64]
        args.seeds = 1
        args.steps = 5000

    for prime in args.primes:
        for d in args.widths:
            n_heads = max(2, d // 32)
            for seed in range(args.seeds):
                print(f"\n{'='*60}")
                print(f"prime={prime}  d={d}  h={n_heads}  seed={seed}")
                print(f"{'='*60}")
                run_grokking_experiment(
                    prime=prime, d_model=d, n_heads=n_heads,
                    n_steps=args.steps, seed=seed,
                    log_every=args.log_every,
                    checkpoint_every=args.checkpoint_every,
                    lr=args.lr, device_str=args.device,
                    output_dir=args.output_dir,
                )


if __name__ == '__main__':
    _main()
