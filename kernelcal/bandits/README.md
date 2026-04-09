# kernelcal.bandits — DDK-GPUCB Simulation Suite

Decentralised Dynamic-Kernel GP-UCB (DDK-GPUCB) for spatiotemporal
adaptive sampling.  Implements and benchmarks the algorithm from:

> "Decentralised Gaussian Process Bandits with Dynamic Kernels under MaxCal"  
> Das, 2026.  `manuscripts/decentralized-dynamic-kernel-gpucb/`

## Quick start (local or server)

```bash
# from repo root
python3 -m kernelcal.bandits.run_server --seeds 10 --T 500 --out results/
```

Or directly:

```python
from kernelcal.bandits import run_experiment, ExperimentConfig
results = run_experiment(ExperimentConfig(T=300, N=4, seed=42))
results.plot("figures/")
```

## Scenario

Arms are `(x, t)` pairs — spatial position × normalised time.

| Region | x | True kernel class | What happens without it |
|--------|---|-------------------|------------------------|
| Periodic | x < 0.5 | SE(x) × Periodic(t) | SE cannot represent cosine modulation at any lengthscale |
| Smooth   | x ≥ 0.5 | Anisotropic SE(x,t) | Periodic kernel overfits |

The **mixture kernel** `k = (1−w)·k_SE + w·k_SE×Per` learns the mixing
weight `w` per agent from local data.  The primary metric is whether
agents in the periodic region converge to `w → 1` and agents in the
smooth region converge to `w → 0`.

## Module map

| File | Key classes / functions |
|------|------------------------|
| `field.py` | `SpatiotemporalField` — deterministic (x,t) field |
| `kernels.py` | `AnisotropicSEKernel`, `SEPeriodicKernel`, `MixtureKernel`, `hs_distance` |
| `network.py` | `GossipNetwork` — Metropolis gossip matrix, Chebyshev mixing |
| `agents.py` | `MixtureKernelAgent`, `DDKGPUCBAgent`, `StaticGPUCBAgent`, `DDUCBAgent` |
| `experiment.py` | `run_experiment(ExperimentConfig)` |

## Server run (recommended for T ≥ 500, seeds ≥ 10)

```bash
# run_server.py handles parallelism and checkpointing
python3 run_server.py --T 1000 --seeds 20 --n_arms_x 5 --n_arms_t 6 --n_jobs 4
```

## Key hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `kernel_step` | 0.03 | LML gradient step; 0.08 causes w oscillation |
| `landauer_pen` | 0.005 | Thermodynamic cost weight (L2 in log-space) |
| `consensus_rho` | 0.3 | Kernel consensus weight; 0 = no consensus |
| `adapt_every` | 3 | Kernel update frequency (rounds) |
| `T` | 500 | Minimum for reliable w convergence (~167 adapt steps) |

## Validity tests

```bash
python3 -c "
from kernelcal.bandits.field import SpatiotemporalField
from kernelcal.bandits.kernels import AnisotropicSEKernel, SEPeriodicKernel
f = SpatiotemporalField(seed=42)
X_per = f.arm_locations[[f.arm_region(k)=='periodic' for k in range(f.K_arms)]]
y_per = f.f_true[[f.arm_region(k)=='periodic' for k in range(f.K_arms)]]
import numpy as np
k_se  = AnisotropicSEKernel(log_sigma_n=np.log(0.08))
k_per = SEPeriodicKernel(log_sigma_n=np.log(0.08))
print('LML SE on periodic:', k_se.log_marginal_likelihood(X_per, y_per))
print('LML Per on periodic:', k_per.log_marginal_likelihood(X_per, y_per))
# Expected: SE ~ -111, Per ~ -9
"
```
