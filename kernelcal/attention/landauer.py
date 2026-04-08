"""
kernelcal.attention.landauer
============================
Landauer bound experiment for transformer training.

Tests Hypothesis 3: W_total >= k_B T * delta_I_total

Runs width-scaled GPT-2 family at multiple learning rates,
logging GPU watt-hours (via pynvml) and estimating MI gain
from CKA between initial and final attention kernels.

Designed for server with 2×Titan RTX (NVIDIA).  Falls back
to CPU watt-estimate if pynvml is unavailable.

Quick usage:
    python -m kernelcal.attention.landauer \\
        --widths 128 256 512 1024 \\
        --lrs 1e-2 1e-3 1e-4 1e-5 \\
        --steps 2000 \\
        --output-dir /results/landauer

Full server run (expected ~3-4h on 2×Titan RTX):
    python -m kernelcal.attention.landauer \\
        --widths 128 256 512 1024 \\
        --lrs 1e-2 1e-3 1e-4 1e-5 \\
        --steps 5000 \\
        --n-seeds 3 \\
        --output-dir /results/landauer
"""

from __future__ import annotations

import json
import math
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── GPU power monitoring ─────────────────────────────────────────────────

class GPUPowerMonitor:
    """Background thread that polls nvidia-smi / pynvml for GPU power draw."""

    def __init__(self, device_id: int = 0, poll_interval_s: float = 0.5):
        self._device_id = device_id
        self._poll_interval = poll_interval_s
        self._readings: List[Tuple[float, float]] = []  # (timestamp, watts)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._backend = self._init_backend()

    def _init_backend(self):
        # nvidia-ml-py installs as 'pynvml' module; both old pynvml and
        # new nvidia-ml-py expose the same API under that module name.
        try:
            import pynvml as nvml  # works for both pynvml and nvidia-ml-py
            nvml.nvmlInit()
            self._handle = nvml.nvmlDeviceGetHandleByIndex(self._device_id)
            self._nvml = nvml
            return 'nvml'
        except Exception:
            self._nvml = None
            return 'cpu_estimate'

    def _poll(self):
        while self._running:
            w = self._read_watts()
            self._readings.append((time.time(), w))
            time.sleep(self._poll_interval)

    def _read_watts(self) -> float:
        if self._backend == 'nvml' and self._nvml is not None:
            try:
                mw = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
                return mw / 1000.0
            except Exception:
                return 0.0
        # CPU fallback: use /sys/class/power_supply if available, else 0
        try:
            p = Path('/sys/class/hwmon')
            for d in p.iterdir():
                pf = d / 'power1_input'
                if pf.exists():
                    return float(pf.read_text()) / 1e6
        except Exception:
            pass
        return 0.0

    def start(self) -> None:
        self._readings.clear()
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        """Stop monitoring and return total energy in watt-hours."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self._readings:
            return 0.0
        # Trapezoidal integration in watt-seconds → watt-hours
        t = np.array([r[0] for r in self._readings])
        w = np.array([r[1] for r in self._readings])
        ws = float(np.trapz(w, t))
        return ws / 3600.0  # watt-hours

    @property
    def backend(self) -> str:
        return self._backend  # 'nvml' or 'cpu_estimate'


# ── CKA for mutual information proxy ─────────────────────────────────────

def centered_kernel_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
    """Linear CKA as a proxy for representational similarity.

    CKA ∈ [0, 1]; higher = more similar. Used to estimate ΔI
    between initial and final representations.
    """
    def center(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    Kc1 = center(K1)
    Kc2 = center(K2)
    num   = np.trace(Kc1 @ Kc2)
    denom = np.sqrt(np.trace(Kc1 @ Kc1) * np.trace(Kc2 @ Kc2))
    return float(num / (denom + 1e-12))


# ── Width-scaled tiny GPT-2 ──────────────────────────────────────────────

def _build_width_scaled_model(d_model: int, n_layers: int, n_heads: int,
                               vocab: int, device, dtype):
    import torch
    import torch.nn as nn

    class MultiHeadAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.proj = nn.Linear(d_model, d_model, bias=False)
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self._attn: Optional[torch.Tensor] = None

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            s = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
            a = s.softmax(-1)
            self._attn = a.detach()
            out = (a @ v).transpose(1, 2).reshape(B, T, C)
            return self.proj(out)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = MultiHeadAttn()
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

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, d_model)
            self.pos   = nn.Embedding(32, d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
            self.ln_f  = nn.LayerNorm(d_model)
            self.head  = nn.Linear(d_model, vocab, bias=False)

        def forward(self, idx):
            B, T = idx.shape
            pos  = torch.arange(T, device=idx.device)
            x    = self.embed(idx) + self.pos(pos)
            for blk in self.blocks: x = blk(x)
            return self.head(self.ln_f(x))

        def attn_kernel_flat(self, seq_len: int) -> np.ndarray:
            """Return mean attention matrix averaged across heads/layers."""
            mats = []
            for blk in self.blocks:
                if blk.attn._attn is not None:
                    A = blk.attn._attn[0].mean(0).float().cpu().numpy()
                    # symmetrize
                    K = (A + A.T) / 2
                    K -= K.min(); np.fill_diagonal(K, 0)
                    mats.append(K)
            if not mats:
                return np.eye(seq_len)
            return np.stack(mats).mean(0)

    return GPT().to(device=device, dtype=dtype)


# ── Single-run experiment ─────────────────────────────────────────────────

@dataclass
class LandauerRunResult:
    d_model: int
    lr: float
    seed: int
    n_steps: int
    watt_hours: float
    delta_I: float              # 1 - CKA(initial, final), proxy for MI gain
    kernel_velocity_mean: float # mean ‖Δk‖ per step
    train_acc: float
    backend: str                # 'pynvml' or 'cpu_estimate'
    step_velocities: List[float] = field(default_factory=list)


def run_single_landauer(
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    prime: int = 53,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    n_steps: int = 2000,
    batch_size: int = 64,
    seed: int = 0,
    device_id: int = 0,
    sigma2: float = 1.0,
    mu2: float = 2.0,
    verbose: bool = True,
) -> LandauerRunResult:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)

    from .device import best_device
    device = best_device(f'cuda:{device_id}' if torch.cuda.is_available() else None)
    dtype  = torch.float32  # use float32 for power accuracy

    # Dataset
    vocab = prime + 3
    pairs = [(a, b) for a in range(prime) for b in range(prime)]
    np.random.shuffle(pairs)
    train_data = torch.tensor(
        [[a, prime+1, b, prime, (a+b)%prime] for a,b in pairs],
        dtype=torch.long, device=device,
    )

    model = _build_width_scaled_model(d_model, n_layers, n_heads, vocab, device, dtype)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Capture initial kernel
    with torch.inference_mode():
        probe = train_data[:8, :-1]
        model(probe)
        K_init = model.attn_kernel_flat(probe.shape[1])

    # Start power monitoring
    monitor = GPUPowerMonitor(device_id=device_id if torch.cuda.is_available() else 0)
    monitor.start()

    prev_K = K_init.copy()
    velocities = []
    t0 = time.time()

    model.train()
    for step in range(n_steps):
        idx  = torch.randint(len(train_data), (min(batch_size, len(train_data)),))
        seqs = train_data[idx]
        x, y = seqs[:, :-1], seqs[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step+1) % 100 == 0:
            with torch.inference_mode():
                model(probe)
                K_new = model.attn_kernel_flat(probe.shape[1])
            vel = float(np.linalg.norm(K_new - prev_K))
            velocities.append(vel)
            prev_K = K_new.copy()

    watt_hours = monitor.stop()

    # Capture final kernel and compute ΔI = 1 - CKA(initial, final)
    model.eval()
    with torch.inference_mode():
        model(probe)
        K_final = model.attn_kernel_flat(probe.shape[1])
    cka = centered_kernel_alignment(K_init, K_final)
    delta_I = 1.0 - cka   # 0 = no change, 1 = maximally different

    # Train accuracy
    with torch.inference_mode():
        logits_all = model(train_data[:, :-1])
        pred = logits_all[:, -2, :prime].argmax(-1)
        acc  = (pred == train_data[:, 4]).float().mean().item()

    elapsed = time.time() - t0
    if verbose:
        print(f"  d={d_model:4d} lr={lr:.0e} seed={seed}  "
              f"loss->{loss.item():.3f}  acc={acc:.3f}  "
              f"W={watt_hours:.4f} Wh  ΔI={delta_I:.4f}  "
              f"ratio={watt_hours/(delta_I+1e-9):.4f}  {elapsed:.1f}s  [{monitor.backend}]")

    return LandauerRunResult(
        d_model=d_model, lr=lr, seed=seed, n_steps=n_steps,
        watt_hours=watt_hours, delta_I=delta_I,
        kernel_velocity_mean=float(np.mean(velocities)) if velocities else 0.0,
        train_acc=acc, backend=monitor.backend,
        step_velocities=velocities,
    )


# ── Full experiment sweep ─────────────────────────────────────────────────

def run_landauer_experiment(
    widths: List[int] = (128, 256, 512, 1024),
    lrs: List[float] = (1e-2, 1e-3, 1e-4, 1e-5),
    n_steps: int = 2000,
    n_seeds: int = 3,
    prime: int = 53,
    device_id: int = 0,
    output_dir: str = '/results/landauer',
    verbose: bool = True,
) -> dict:
    """Run full Landauer bound sweep across widths and learning rates."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results = []
    total = len(widths) * len(lrs) * n_seeds
    run_n = 0

    for d in widths:
        n_heads  = max(1, d // 32)   # heads scale with width
        n_layers = 2
        for lr in lrs:
            for seed in range(n_seeds):
                run_n += 1
                if verbose:
                    print(f"[{run_n}/{total}] d={d} lr={lr:.0e} seed={seed}")
                try:
                    r = run_single_landauer(
                        d_model=d, n_layers=n_layers, n_heads=n_heads,
                        prime=prime, lr=lr, n_steps=n_steps, seed=seed,
                        device_id=device_id, verbose=verbose,
                    )
                    all_results.append({
                        'd_model': r.d_model, 'lr': r.lr, 'seed': r.seed,
                        'watt_hours': r.watt_hours, 'delta_I': r.delta_I,
                        'ratio_Wh_per_I': r.watt_hours / (r.delta_I + 1e-9),
                        'kernel_velocity_mean': r.kernel_velocity_mean,
                        'train_acc': r.train_acc, 'backend': r.backend,
                    })
                except Exception as e:
                    print(f"  ERROR: {e}")

    summary = {
        'config': {
            'widths': list(widths), 'lrs': list(lrs),
            'n_steps': n_steps, 'n_seeds': n_seeds, 'prime': prime,
        },
        'results': all_results,
    }
    (out / 'landauer_results.json').write_text(json.dumps(summary, indent=2))

    _generate_landauer_figures(all_results, out)

    return summary


def _generate_landauer_figures(results: list, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        BG, SURF, GOLD, SILVER = '#0d1117', '#161b22', '#e2b44d', '#b0bec5'
        lrs   = sorted(set(r['lr'] for r in results))
        widths = sorted(set(r['d_model'] for r in results))

        fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), dpi=150)
        fig.patch.set_facecolor(BG)
        fig.suptitle('Landauer Bound Experiment: W_total / ΔI vs Learning Rate',
                     color=GOLD, fontsize=12, y=1.01, fontweight='bold')

        import matplotlib.cm as cm
        cmap = cm.get_cmap('plasma', len(widths))

        for ax in axes:
            ax.set_facecolor(SURF)
            ax.tick_params(colors='#78909c')
            for sp in ax.spines.values(): sp.set_color('#263238')

        # Panel 1: W_total vs ΔI scatter
        ax = axes[0]
        for wi, d in enumerate(widths):
            dr = [r for r in results if r['d_model'] == d]
            ax.scatter([r['delta_I'] for r in dr],
                       [r['watt_hours'] for r in dr],
                       c=[cmap(wi)]*len(dr), s=20, alpha=0.7, label=f'd={d}')
        ax.set_xlabel('ΔI (1 - CKA)', color=SILVER, fontsize=9)
        ax.set_ylabel('W_total (Wh)', color=SILVER, fontsize=9)
        ax.set_title('W vs ΔI', color='#e0e0e0', fontsize=9)
        ax.legend(fontsize=7, labelcolor=SILVER, facecolor='#1a1a2e', edgecolor='#37474f')

        # Panel 2: Ratio W/ΔI vs lr per width
        ax = axes[1]
        for wi, d in enumerate(widths):
            ratios_per_lr = []
            for lr in lrs:
                dr = [r for r in results if r['d_model'] == d and abs(r['lr']-lr)<1e-10]
                if dr:
                    ratios_per_lr.append(np.mean([r['ratio_Wh_per_I'] for r in dr]))
                else:
                    ratios_per_lr.append(np.nan)
            ax.plot([str(lr) for lr in lrs], ratios_per_lr,
                    color=cmap(wi), marker='o', ms=5, lw=1.5, label=f'd={d}')
        ax.set_xlabel('Learning rate', color=SILVER, fontsize=9)
        ax.set_ylabel('W_total / ΔI  (Wh/nat)', color=SILVER, fontsize=9)
        ax.set_title('Speed limit: should decrease as lr→0', color='#e0e0e0', fontsize=9)
        ax.legend(fontsize=7, labelcolor=SILVER, facecolor='#1a1a2e', edgecolor='#37474f')

        # Panel 3: Kernel velocity mean vs lr
        ax = axes[2]
        for wi, d in enumerate(widths):
            vels = []
            for lr in lrs:
                dr = [r for r in results if r['d_model'] == d and abs(r['lr']-lr)<1e-10]
                if dr:
                    vels.append(np.mean([r['kernel_velocity_mean'] for r in dr]))
                else:
                    vels.append(np.nan)
            ax.plot([str(lr) for lr in lrs], vels,
                    color=cmap(wi), marker='s', ms=5, lw=1.5, label=f'd={d}')
        ax.set_xlabel('Learning rate', color=SILVER, fontsize=9)
        ax.set_ylabel('Mean kernel velocity ‖Δk‖', color=SILVER, fontsize=9)
        ax.set_title('Kernel speed limit test', color='#e0e0e0', fontsize=9)
        ax.legend(fontsize=7, labelcolor=SILVER, facecolor='#1a1a2e', edgecolor='#37474f')

        plt.tight_layout()
        plt.savefig(out / 'fig_landauer_results.pdf', bbox_inches='tight', facecolor=BG)
        plt.savefig(out / 'fig_landauer_results.png', bbox_inches='tight', facecolor=BG)
        plt.close()
        print(f'[landauer] figures saved → {out}')
    except Exception as e:
        print(f'[landauer] figure generation failed: {e}')


# ── CLI ──────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Landauer bound experiment for transformer training.')
    parser.add_argument('--widths', type=int, nargs='+', default=[128, 256, 512, 1024])
    parser.add_argument('--lrs', type=float, nargs='+', default=[1e-2, 1e-3, 1e-4, 1e-5])
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--n-seeds', type=int, default=3)
    parser.add_argument('--prime', type=int, default=53)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--output-dir', default='/results/landauer')
    args = parser.parse_args()

    print(f'[landauer] widths={args.widths}  lrs={args.lrs}  '
          f'steps={args.steps}  seeds={args.n_seeds}  prime={args.prime}')
    run_landauer_experiment(
        widths=args.widths, lrs=args.lrs,
        n_steps=args.steps, n_seeds=args.n_seeds,
        prime=args.prime, device_id=args.device_id,
        output_dir=args.output_dir, verbose=True,
    )


if __name__ == '__main__':
    _main()
