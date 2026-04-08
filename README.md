# kernelcal

**Kernel dynamics under Maximum Caliber — variational tools for representational change, thermodynamic bounds, and adaptive sampling.**

[![arXiv](https://img.shields.io/badge/arXiv-2603.27880-b31b1b.svg)](https://arxiv.org/abs/2603.27880)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

> **Status:** research companion library, v0.1.0 — pre-publication, API subject to change.

Companion library to:

> **Kernel Dynamics under Path Entropy Maximization**
> Jnaneshwar Das — School of Earth and Space Exploration, Arizona State University / Earth Innovation Hub
> [arXiv:2603.27880](https://arxiv.org/abs/2603.27880)

Integration reference for [DeepGIS-XR](https://github.com/Earth-Innovation-Hub/deepgis-xr), an AI-powered geospatial platform.

---

## What it does

The paper treats the kernel $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ — the object encoding what distinctions an agent can represent — as a dynamical variable governed by Maximum Caliber (path entropy maximization). `kernelcal` implements the full framework:

| Subpackage | Implements |
|---|---|
| `kernelcal.kernel` | Hilbert-Schmidt geometry, kernel trajectories, fixed-point detection |
| `kernelcal.maxcal` | Path entropy functional, Lagrange dual, MaxCal sampler |
| `kernelcal.ntk` | NTK drift tracking, Hellinger kernel, Conjecture 3 test |
| `kernelcal.assembly` | RKHS complexity maps, assembly-theory reward signal |
| `kernelcal.thermodynamics` | Landauer bound $\delta W \geq k_B T \, \delta I_k$, GPU power logging |
| `kernelcal.models` | MaxCal multi-model selector (SAM / YOLOv8 / Grounding DINO / ...) |
| `kernelcal.prompts` | Self-consistent Grounding DINO prompt iteration |
| `kernelcal.spectral` | Spectral kernel dynamics on finite graphs: fixed points, geodesics, stability, phase-transition diagnostics |
| `kernelcal.attention` | MaxCal diagnostics on transformer attention kernels: GPT-2 probing, toy training, Landauer bound experiment |

---

## Installation

```bash
git clone https://github.com/darknight-007/kernelcal.git
cd kernelcal
pip install -e .
```

PyTorch is optional — required only for `kernelcal.ntk.compute_empirical_ntk` on live models:

```bash
pip install -e ".[torch]"
```

---

## Quick start

> The code snippets below are schematic — variables such as `reward_scores`,
> `kernel_snapshots`, and `embeddings_per_tile` represent arrays you supply
> from your own pipeline.

### MaxCal adaptive sampler

Replaces the heuristic update in the [DeepGIS World Sampler](https://github.com/Earth-Innovation-Hub/deepgis-xr) with a principled MaxCal rule:

```python
import numpy as np
from kernelcal.maxcal import MaxCalSampler

# (N, 2) array of (lon, lat) candidate locations
locations = np.array([...])

sampler = MaxCalSampler(locations)

# Update from reward feedback — e.g. anomaly scores from SAM detections
sampler.update(feedback=reward_scores)

# Sample next survey locations
next_locations = sampler.sample(n=10)

print(sampler.statistics())
# {'entropy_nats': 4.31, 'is_fixed_point': False, 'classification': 'transient', ...}
```

### Kernel trajectory and fixed-point detection

```python
from kernelcal.kernel import KernelTrajectory, FixedPointDetector

traj = KernelTrajectory(name="NTK during fine-tuning")
fp   = FixedPointDetector(tol=1e-3, window=5)

for step, K in enumerate(kernel_snapshots):
    traj.add(step, K)
    fp.update(K)

print(traj.summary())
print(f"Fixed point: {fp.is_fixed_point()}  stability: {fp.stability_score():.3f}")
print(f"Landscape phase: {fp.classify()}")   # 'transient' | 'stable_fp' | 'oscillating'
```

### Landauer bound verification

```python
from kernelcal.thermodynamics import PowerMonitor, check_landauer_bound

with PowerMonitor(gpu_id=0, interval_s=0.1) as pm:
    fine_tune_step(model, batch)

result = check_landauer_bound(
    measured_work_joules=pm.total_energy_joules(),
    K1=ntk_before,
    K2=ntk_after,
)
print(result)
# {'delta_I_nats': 0.42, 'landauer_bound_joules': 1.73e-21, 'bound_satisfied': True, ...}
```

### NTK–Hellinger comparison (Conjecture 3)

```python
from kernelcal.ntk import NTKTracker, compare_ntk_to_hellinger

tracker = NTKTracker(probe_inputs=X_probe)

for step, batch in enumerate(loader):
    train_step(model, batch)
    if step % 100 == 0:
        tracker.record(step, model)

softmax_outputs = model(X_probe).softmax(dim=-1).detach().numpy()
result = compare_ntk_to_hellinger(tracker.final_kernel(), softmax_outputs)
print(f"HS distance to Hellinger kernel: {result['hs_distance']:.4f}")
```

### Assembly complexity reward for geospatial sampling

```python
from kernelcal.assembly import complexity_map, assembly_reward_signal

# embeddings_per_tile: list of (M_i, D) feature arrays from SAM / Grounding DINO
scores  = complexity_map(embeddings_per_tile)
rewards = assembly_reward_signal(scores, coverage_counts=visit_counts)

sampler.update(feedback=rewards)
```

### MaxCal diagnostics on transformer attention (GPU-accelerated)

```python
from kernelcal.attention import run_attention_experiment

# Synthetic demo — no GPU or pretrained model required (~0.2s)
result = run_attention_experiment(model_name="synthetic", seq_len=32)
print(result.summary())

# GPT-2 small on GPU — float16, ~1.5 GB VRAM (~20s)
result = run_attention_experiment(model_name="gpt2", seq_len=64)
```

Run the 30-run ensemble (3 primes × 10 seeds) for statistical validation:

```bash
python -m kernelcal.attention.training \
    --primes 23 53 97 --seeds 10 --steps 2000 \
    --output-dir figures/attention
```

Run the Landauer bound experiment on a 2×GPU server:

```bash
# One-command Docker launcher (splits widths across 2 GPUs)
bash run_landauer_server.sh

# Or directly:
python -m kernelcal.attention.landauer \
    --widths 128 256 512 1024 \
    --lrs 1e-2 1e-3 1e-4 1e-5 \
    --steps 2000 --n-seeds 3 \
    --output-dir /results/landauer
```

### Spectral kernel dynamics on a graph

```python
from kernelcal.spectral import SpectralGraph, GaussianMISource, SpectralKernelDynamics

# Build a path graph and run MaxCal spectral dynamics
g   = SpectralGraph.path_graph(8)
src = GaussianMISource(sigma2=1.0, mu2=2.0, eigenvalues=g.eigenvalues)
dyn = SpectralKernelDynamics(g, src)

fp   = dyn.fixed_point_iteration()
stab = dyn.stability_analysis(fp.h_star)

print(f"Converged: {fp.converged}  iterations: {fp.iterations}")
print(f"Stable: {stab.stable}  Fiedler gap: {stab.fiedler_gap:.4f}")
print(f"Spectral entropy: {dyn.spectral_entropy(fp.h_star):.4f}")
```

Run the standard procedural examples (path, weak-path, cycle):

```bash
python -m kernelcal.spectral.procedural_examples --output-dir figures/spectral
```

Run the full 7-experiment verification suite:

```bash
python -m kernelcal.spectral.experiments --N 8 --mu2 2.0
```

### Self-consistent Grounding DINO prompt

```python
from kernelcal.prompts import PromptKernelIterator

iterator = PromptKernelIterator(
    detector_fn=grounding_dino_detect,   # callable: (image, prompt) → list[dict]
    max_steps=15,
    tol=1e-2,
)

final_prompt = iterator.iterate(image, initial_prompt="rock . debris . crater")
print(iterator.summary())
# {'converged': True, 'n_steps': 7, 'final_prompt': 'boulder . crater . ejecta', ...}
```

---

## Package structure

```
kernelcal/
├── kernel/
│   ├── space.py          # HS distance, PSD projection, kernel algebra
│   ├── trajectory.py     # KernelTrajectory: path length, velocity, interpolation
│   └── fixed_points.py   # FixedPointDetector: stability score, landscape classifier
├── maxcal/
│   ├── functional.py     # Path entropy, Lagrange dual, fit_lagrange_multipliers
│   └── sampler.py        # MaxCalSampler: drop-in for DeepGIS World Sampler
├── ntk/
│   ├── tracker.py        # NTKTracker: empirical NTK, HS drift, convergence rate
│   └── hellinger.py      # Hellinger kernel, NTK–Hellinger distance (Conjecture 3)
├── assembly/
│   └── complexity.py     # RKHS norm, spectral complexity, per-tile complexity map
├── thermodynamics/
│   └── bounds.py         # Landauer bound, PowerMonitor, ThermodynamicEfficiency
├── models/
│   └── selector.py       # ModelKernelSelector: MaxCal over SAM/YOLOv8/DINO/...
├── prompts/
│   └── grounding.py      # PromptKernelIterator: fixed-point prompt search
├── attention/
│   ├── device.py          # GPU auto-selection (CUDA/MPS/CPU), float16 on GPU
│   ├── kernel.py          # AttentionKernel: spectral MaxCal on attention matrices
│   ├── tracker.py         # AttentionKernelTracker: forward-hook training logger
│   ├── experiment.py      # Frozen GPT-2 probing, synthetic mode, null-model check
│   ├── training.py        # Toy training loop + 30-run ensemble with MaxCal diagnostics
│   └── landauer.py        # pynvml power logging + CKA ΔI + Landauer bound sweep
└── spectral/
    ├── graph.py           # SpectralGraph: Laplacian eigendecomposition, factory methods
    ├── source.py          # GaussianMISource, CoupledGaussianMISource
    ├── dynamics.py        # SpectralKernelDynamics: R_l, fixed points, geodesics, stability
    ├── experiments.py     # 7-experiment verification suite (Exps 1–7)
    ├── procedural.py      # Procedural graph diagnostics pipeline
    ├── procedural_examples.py  # Standard examples runner + CLI
    ├── channel_image.py   # (dormant) Image-to-graph extraction pipeline
    └── pipeline.py        # (dormant) Image-to-spectral diagnostics pipeline
```

### Server deployment (Landauer experiment)

For the Landauer bound experiment on a 2×GPU server:

```
Dockerfile.landauer          # CUDA 12.1 + PyTorch 2.2 container
docker-compose.landauer.yml  # 2-GPU parallel sweep (widths split per GPU)
run_landauer_server.sh       # One-command: build → run → merge → figure
requirements.landauer.txt    # pynvml, transformers, matplotlib
```

```bash
# Build once, run across 2 GPUs in parallel
bash run_landauer_server.sh

# Results → /results/landauer/landauer_results_merged.json
#           /results/landauer/fig_landauer_results.pdf
```

---

## Relation to the paper

| Paper element | Library location |
|---|---|
| Kernel space $\mathcal{K}$, HS metric (§3) | `kernelcal.kernel.space` |
| MaxCal path entropy $\mathcal{S}[p]$ (Eq. 1) | `kernelcal.maxcal.functional` |
| Self-consistent fixed-point condition (Def. 3) | `kernelcal.kernel.fixed_points` |
| Thermodynamic bound $\delta W \geq k_B T \delta I_k$ (Thm. 1) | `kernelcal.thermodynamics.bounds` |
| RG flow as MaxCal (Prop. 1) | `kernelcal.kernel.trajectory` (decay rate) |
| NTK–Hellinger conjecture (Conj. 3) | `kernelcal.ntk.hellinger` |
| Assembly theory interface (§6) | `kernelcal.assembly.complexity` |
| Adaptive sample-return planning (§5) | `kernelcal.maxcal.sampler` |
| Attention as kernel (Prop. 1--2) | `kernelcal.attention.kernel` |
| Endogenous landscape (Obs.~1) | `kernelcal.attention.kernel` |
| Fixed-point structural probe (GPT-2) | `kernelcal.attention.experiment` |
| Landauer bound experiment (H3) | `kernelcal.attention.landauer` |
| Spectral geometric functional $\mathcal{R}_l$ (Prop. 1†) | `kernelcal.spectral.dynamics` |
| Self-consistent kernels via exponential tilting (Cor. 1†) | `kernelcal.spectral.dynamics` |
| Log-linear Fisher–Rao geodesics (Cor. 2†) | `kernelcal.spectral.dynamics` |
| Hessian stability, Fiedler gap $\Delta'$ (Cor. 3†, Q6†) | `kernelcal.spectral.dynamics` |
| Spectral entropy early-warning (Rem. 8†) | `kernelcal.spectral.dynamics` |

*† from the companion spectral paper (in preparation)*

---

## DeepGIS-XR integration

Full integration analysis: [`INTEGRATION.md`](INTEGRATION.md)

Seven integration threads mapped to DeepGIS-XR components:

1. **World Sampler → MaxCalSampler** — replace heuristic update with MaxCal rule
2. **Multi-model inference → kernel switching** — MaxCal distribution over AI backends
3. **Multi-scale CesiumJS → RG flow** — entropy-maximising coarse-graining across zoom levels
4. **Fine-tuning → NTK tracking** — representational drift + power draw logging
5. **AI detections → assembly complexity** — RKHS norm as geospatial sampling reward
6. **Change detection → fixed-point stability** — stable vs. transitional landscape classification
7. **Grounding DINO prompts → self-consistent iteration** — fixed-point prompt search

---

## Dependencies

- `numpy >= 1.24`
- `scipy >= 1.11`
- `torch >= 2.0` *(optional — NTK computation on live models)*

---

## License

[MIT](LICENSE)

---

## Citation

```bibtex
@article{das2026kerneldynamics,
  title   = {Kernel Dynamics under Path Entropy Maximization},
  author  = {Das, Jnaneshwar},
  journal = {arXiv preprint arXiv:2603.27880},
  year    = {2026}
}
```
