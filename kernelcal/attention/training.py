"""
Toy training experiment: MaxCal diagnostics during transformer training.

Tests whether SGD on a small transformer drives attention kernels toward
MaxCal self-consistent fixed points.  Uses the modular-addition (grokking)
task — a canonical phase-transition benchmark — so we can observe kernel
dynamics through a known regime change.

Key measurements at each logged step:
  1. Field-equation residual  ||R[h*] - T[h*]||_inf  → tests fixed-point convergence
  2. Spectral entropy         H[h_t]                  → tracks path entropy
  3. Fiedler gap              Delta'(h_t)              → tracks stability margin
  4. Kernel velocity          ||h_t - h_{t-1}||_2     → weak speed-limit test
  5. Train / val accuracy                              → grokking phase reference

Quick usage:
    python -m kernelcal.attention.training \\
        --prime 97 --steps 3000 --log-every 20 \\
        --output-dir figures/attention_training
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Tiny transformer (pure PyTorch — no HuggingFace)
# ──────────────────────────────────────────────────────────────────────

def _build_model(prime: int, d_model: int, n_heads: int, n_layers: int,
                 device):
    """Build a tiny GPT-style transformer for modular addition."""
    import torch
    import torch.nn as nn

    vocab = prime + 3   # 0..p-1, plus tokens for '=', '+', EOS

    class MultiHeadSelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.proj = nn.Linear(d_model, d_model, bias=False)
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self._attn_weights: Optional[torch.Tensor] = None

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            scale = math.sqrt(self.d_head)
            scores = (q @ k.transpose(-2, -1)) / scale
            attn = scores.softmax(dim=-1)
            self._attn_weights = attn.detach()
            out = (attn @ v).transpose(1, 2).reshape(B, T, C)
            return self.proj(out)

    class TransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = MultiHeadSelfAttn()
            self.ln1  = nn.LayerNorm(d_model)
            self.ff   = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
            self.ln2  = nn.LayerNorm(d_model)

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
            return x

    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, d_model)
            self.pos   = nn.Embedding(16, d_model)
            self.blocks = nn.ModuleList([TransformerBlock() for _ in range(n_layers)])
            self.ln_f  = nn.LayerNorm(d_model)
            self.head  = nn.Linear(d_model, vocab, bias=False)

        def forward(self, idx):
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device)
            x   = self.embed(idx) + self.pos(pos)
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.ln_f(x))

        def attn_weights(self) -> List[torch.Tensor]:
            """Return list of (n_heads, T, T) attention weight tensors per layer."""
            return [blk.attn._attn_weights for blk in self.blocks]

    return TinyGPT().to(device)


def _make_dataset(prime: int, device):
    """Modular addition dataset: all (a, b, a+b mod p) triples."""
    import torch
    EQ = prime; PLUS = prime + 1; EOS = prime + 2
    pairs = [(a, b) for a in range(prime) for b in range(prime)]
    np.random.shuffle(pairs)
    split = int(0.8 * len(pairs))
    train_pairs, val_pairs = pairs[:split], pairs[split:]

    def to_tensors(ps):
        seqs = torch.tensor(
            [[a, PLUS, b, EQ, (a + b) % prime] for a, b in ps],
            dtype=torch.long, device=device,
        )
        return seqs

    return to_tensors(train_pairs), to_tensors(val_pairs)


# ──────────────────────────────────────────────────────────────────────
# Training loop with MaxCal kernel diagnostics
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrainingRecord:
    step: int
    train_acc: float
    val_acc: float
    loss: float
    # Per-layer/head lists
    residuals: List[float]      # mean ||R - T||_inf across heads
    h_entropies: List[float]    # mean H[h_t] across heads
    fiedler_gaps: List[float]   # mean Δ' across heads
    fiedler_values: List[float] # mean λ₁ across heads
    kernel_velocity: List[float]# mean ||h_t - h_{t-1}||_2 across heads


def run_training_experiment(
    prime: int = 97,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    n_steps: int = 3000,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    log_every: int = 25,
    sigma2: float = 1.0,
    mu2: float = 2.0,
    device_str: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_every: int = 0,
    monitor_energy: bool = True,
    verbose: bool = True,
) -> List[TrainingRecord]:
    """
    Train a tiny transformer on modular addition and log MaxCal diagnostics.

    Parameters
    ----------
    checkpoint_every : int
        Save model checkpoint every N steps (0 = no checkpoints).
    monitor_energy : bool
        Auto-detect and log energy consumption (GPU/RAPL/FLOPs).

    Returns a list of TrainingRecord snapshots.
    """
    import torch
    import torch.nn.functional as F
    from .device import best_device

    device = best_device(device_str)
    if verbose:
        from .device import device_info
        print(f"[training] device={device_info(device)}  prime={prime}  "
              f"d={d_model}  h={n_heads}  L={n_layers}  steps={n_steps}")

    model   = _build_model(prime, d_model, n_heads, n_layers, device)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_d, val_d = _make_dataset(prime, device)
    n_params = sum(p.numel() for p in model.parameters())

    energy_monitor = None
    if monitor_energy:
        from .energy import EnergyMonitor
        energy_monitor = EnergyMonitor.auto_detect()
        energy_monitor.start()
        if verbose:
            print(f"[training] energy sources: {energy_monitor._sources}")

    ckpt_dir = None
    if checkpoint_every > 0 and output_dir:
        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Previous kernel snapshots for velocity
    prev_h: Dict[Tuple[int,int], np.ndarray] = {}
    records: List[TrainingRecord] = []
    t0 = time.time()

    for step in range(n_steps + 1):
        # ── forward / backward ────────────────────────────────────────
        idx  = torch.randint(len(train_d), (min(batch_size, len(train_d)),))
        seqs = train_d[idx]
        x, y = seqs[:, :-1], seqs[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, prime + 3), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if energy_monitor is not None:
            batch_tokens = seqs[:, :-1].numel()
            energy_monitor.add_training_step(n_params, batch_tokens)

        if ckpt_dir and checkpoint_every > 0 and step > 0 and step % checkpoint_every == 0:
            torch.save({
                'step': step, 'model': model.state_dict(),
                'optimizer': opt.state_dict(), 'prime': prime,
                'd_model': d_model, 'n_heads': n_heads, 'n_layers': n_layers,
            }, ckpt_dir / f"step_{step:06d}.pt")

        # ── logging ───────────────────────────────────────────────────
        if step % log_every != 0:
            continue

        model.eval()
        with torch.inference_mode():
            # Train accuracy
            logits_tr = model(train_d[:, :-1])
            pred_tr   = logits_tr[:, -2, :prime].argmax(-1)
            acc_tr    = (pred_tr == train_d[:, 4]).float().mean().item()
            # Val accuracy
            logits_v  = model(val_d[:, :-1])
            pred_v    = logits_v[:, -2, :prime].argmax(-1)
            acc_v     = (pred_v == val_d[:, 4]).float().mean().item()

            attn_ws = model.attn_weights()

        # ── MaxCal diagnostics for each layer/head ────────────────────
        from .kernel import AttentionKernel

        residuals_step = []
        entropies_step = []
        delta_step     = []
        lam1_step      = []
        velocity_step  = []

        for li, attn_w in enumerate(attn_ws):
            if attn_w is None:
                continue
            attn_arr = attn_w[0].float().cpu().numpy()  # (n_heads, T, T)
            for hi in range(attn_arr.shape[0]):
                A = attn_arr[hi]
                ak = AttentionKernel(A, layer=li, head=hi, step=step,
                                     sigma2=sigma2, mu2=mu2,
                                     eigenvalue_aware=True)
                res = ak.analyse(fp_max_iter=100, fp_tol=1e-7)

                key = (li, hi)
                vel = 0.0
                if key in prev_h:
                    vel = float(np.linalg.norm(res.h_star - prev_h[key]))
                prev_h[key] = res.h_star.copy()

                residuals_step.append(res.residual_inf_norm)
                entropies_step.append(res.spectral_entropy)
                delta_step.append(res.fiedler_gap)
                lam1_step.append(res.fiedler_value)
                velocity_step.append(vel)

        rec = TrainingRecord(
            step=step,
            train_acc=acc_tr, val_acc=acc_v, loss=float(loss.item()),
            residuals=[float(np.mean(residuals_step))],
            h_entropies=[float(np.mean(entropies_step))],
            fiedler_gaps=[float(np.mean(delta_step))],
            fiedler_values=[float(np.mean(lam1_step))],
            kernel_velocity=[float(np.mean(velocity_step)) if velocity_step else 0.0],
        )
        records.append(rec)

        if verbose and step % (log_every * 10) == 0:
            print(f"  step={step:4d}  loss={rec.loss:.3f}  "
                  f"tr_acc={rec.train_acc:.3f}  val_acc={rec.val_acc:.3f}  "
                  f"H={rec.h_entropies[0]:.3f}  "
                  f"Δ'={rec.fiedler_gaps[0]:.3f}  "
                  f"resid={rec.residuals[0]:.2e}  "
                  f"vel={rec.kernel_velocity[0]:.4f}")

        model.train()

    elapsed = time.time() - t0

    energy_report = None
    if energy_monitor is not None:
        energy_report = energy_monitor.stop()
        if verbose:
            print(f"[training] energy: {energy_report.total_joules:.2f} J "
                  f"({energy_report.total_wh:.6f} Wh)  "
                  f"mean_power={energy_report.mean_power_watts:.1f} W  "
                  f"sources={energy_report.sources_used}")

    if verbose:
        print(f"[training] done  {elapsed:.1f}s  {n_steps/elapsed:.0f} steps/s")

    if output_dir:
        out = Path(output_dir)
        _save_results(records, out, verbose)
        if energy_report is not None:
            import json
            (out / 'energy_report.json').write_text(
                json.dumps(energy_report.summary(), indent=2))

    return records


# Plotting + JSON writing lives in ``training_plots`` so this module can be
# imported on headless installs.  The underscore names remain available as
# re-exports for any external callers that reach into the private API.
from .training_plots import (  # noqa: E402, F401  (re-export)
    _save_results,
    _save_ensemble_results,
)


def run_ensemble_experiment(
    primes: List[int] = (23, 53, 97),
    seeds: int = 10,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    n_steps: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    log_every: int = 50,
    sigma2: float = 1.0,
    mu2: float = 2.0,
    device_str: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Run multiple training experiments across different seeds and problem
    structures (primes) and compute distributions of MaxCal kernel behavior.

    Returns a dict with:
      'raw'    : list of all TrainingRecord lists
      'steps'  : common step indices
      'mean'   : {metric: array(n_steps)} mean across all runs
      'std'    : {metric: array(n_steps)} std across all runs
      'final'  : {metric: array(n_runs)} values at last step (for distributions)
      'meta'   : run metadata (prime, seed per run)
    """
    import torch

    from .device import best_device, device_info
    device = best_device(device_str)
    if verbose:
        n_total = len(primes) * seeds
        print(f"[ensemble] {n_total} runs  primes={list(primes)}  seeds={seeds}"
              f"  device={device_info(device)}")

    all_records: List[List[TrainingRecord]] = []
    all_meta: List[dict] = []

    run_idx = 0
    for prime in primes:
        for seed in range(seeds):
            import torch
            torch.manual_seed(seed)
            np.random.seed(seed)
            if verbose:
                print(f"  [{run_idx+1}/{len(primes)*seeds}] prime={prime} seed={seed}")
            recs = run_training_experiment(
                prime=prime, d_model=d_model, n_heads=n_heads,
                n_layers=n_layers, n_steps=n_steps, batch_size=batch_size,
                lr=lr, weight_decay=weight_decay, log_every=log_every,
                sigma2=sigma2, mu2=mu2, device_str=device_str,
                output_dir=None, verbose=False,
            )
            all_records.append(recs)
            all_meta.append({'prime': prime, 'seed': seed})
            run_idx += 1

    # Align on common steps (all runs share same log_every)
    steps = [r.step for r in all_records[0]]
    metrics = ['residuals', 'h_entropies', 'fiedler_gaps', 'fiedler_values', 'kernel_velocity']
    short   = ['residual', 'H', 'delta_prime', 'lambda1', 'velocity']

    arrays = {m: np.array([[getattr(r, mk)[0] for r in recs]
                            for recs in all_records])
              for m, mk in zip(short, metrics)}
    # shape: (n_runs, n_steps)

    mean_d = {m: arrays[m].mean(axis=0) for m in short}
    std_d  = {m: arrays[m].std(axis=0)  for m in short}
    final  = {m: arrays[m][:, -1]       for m in short}

    result = {
        'raw': all_records, 'steps': steps,
        'mean': mean_d, 'std': std_d, 'final': final, 'meta': all_meta,
        'primes': list(primes), 'seeds': seeds,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _save_ensemble_results(result, out, verbose)

    return result


def main() -> None:
    """Console-script entry point for the transformer-training MaxCal experiment.

    Wired to ``kernelcal-attention-training`` in ``pyproject.toml``.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="MaxCal diagnostics during toy transformer training.")
    parser.add_argument('--primes', type=int, nargs='+', default=[23, 53, 97])
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--output-dir',
        default=os.environ.get(
            'KERNELCAL_ATTENTION_FIG_DIR',
            '/home/jdas/Documents/manuscripts/attention-kernel-maxcal/figures',
        ),
        help='directory to write training figures + JSON into '
             '(env: KERNELCAL_ATTENTION_FIG_DIR)')
    parser.add_argument('--single', action='store_true',
        help='Run a single seed (prime=97) for quick testing')
    args = parser.parse_args()

    if args.single:
        run_training_experiment(
            prime=97, d_model=args.d_model,
            n_heads=args.n_heads, n_layers=args.n_layers,
            n_steps=args.steps, log_every=args.log_every,
            device_str=args.device, output_dir=args.output_dir,
            verbose=True,
        )
    else:
        run_ensemble_experiment(
            primes=args.primes, seeds=args.seeds,
            d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, n_steps=args.steps,
            log_every=args.log_every,
            device_str=args.device, output_dir=args.output_dir,
            verbose=True,
        )


# Back-compat alias — earlier revisions exported this CLI as the private
# ``_main``.  Kept so any external caller that reaches in keeps working.
_main = main


if __name__ == '__main__':
    main()
