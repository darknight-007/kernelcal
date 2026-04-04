# kernelcal

**Kernel dynamics under Maximum Caliber — variational tools for representational change, thermodynamic bounds, and adaptive sampling.**

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

---

## Installation

```bash
git clone https://github.com/darknight-007/kernelcal.git
cd kernelcal
pip install numpy scipy
```

PyTorch is optional — required only for `kernelcal.ntk.compute_empirical_ntk` on live models.

---

## Quick start

### MaxCal adaptive sampler

Replaces the heuristic update in the [DeepGIS World Sampler](https://github.com/Earth-Innovation-Hub/deepgis-xr) with a principled MaxCal rule:

```python
import numpy as np
from kernelcal.maxcal import MaxCalSampler

# Grid of (lon, lat) candidate locations
locations = np.random.rand(200, 2)

sampler = MaxCalSampler(locations)

# Update from reward feedback (e.g. anomaly scores from SAM)
sampler.update(feedback=reward_scores)

# Sample next survey locations
next_locations = sampler.sample(n=10)

print(sampler.statistics())
# {'entropy_nats': ..., 'is_fixed_point': False, 'classification': 'transient', ...}
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
# {'delta_I_nats': ..., 'landauer_bound_joules': ..., 'bound_satisfied': True, ...}
```

### NTK–Hellinger comparison (Conjecture 3)

```python
from kernelcal.ntk import NTKTracker, compare_ntk_to_hellinger

tracker = NTKTracker(probe_inputs=X_probe)

for step, batch in enumerate(loader):
    train_step(model, batch)
    if step % 100 == 0:
        tracker.record(step, model)

# Test whether NTK converges to Hellinger kernel
softmax_outputs = model(X_probe).softmax(dim=-1).detach().numpy()
result = compare_ntk_to_hellinger(tracker.final_kernel(), softmax_outputs)
print(f"HS distance to Hellinger kernel: {result['hs_distance']:.4f}")
```

### Assembly complexity reward for geospatial sampling

```python
from kernelcal.assembly import complexity_map, assembly_reward_signal

# embeddings_per_tile: list of (M_i, D) SAM / Grounding DINO feature arrays
scores  = complexity_map(embeddings_per_tile)
rewards = assembly_reward_signal(scores, coverage_counts=visit_counts)

sampler.update(feedback=rewards)
```

### Self-consistent Grounding DINO prompt

```python
from kernelcal.prompts import PromptKernelIterator

iterator = PromptKernelIterator(
    detector_fn=grounding_dino_detect,   # (image, prompt) → list[dict]
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
└── prompts/
    └── grounding.py      # PromptKernelIterator: fixed-point prompt search
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

---

## DeepGIS-XR integration

Full integration analysis: [`deepgis_maxcal_integration.md`](deepgis_maxcal_integration.md)

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

[MIT](LICENSE) — see the [philosophical justification](deepgis_maxcal_integration.md) in the integration notes.

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
