# kernelcal

**Kernel dynamics under Maximum Caliber — variational tools for representational change, thermodynamic bounds, and adaptive sampling.**

[![arXiv](https://img.shields.io/badge/arXiv-2603.27880-b31b1b.svg)](https://arxiv.org/abs/2603.27880)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

> **Status:** research companion library, v0.5.0 — pre-publication, API subject to change.

Companion library to the **four-paper kernel dynamics series**:

> **P1 — Kernel Dynamics under Path Entropy Maximization**
> Jnaneshwar Das — ASU School of Earth and Space Exploration
> [arXiv:2603.27880](https://arxiv.org/abs/2603.27880)
>
> **P2 — Spectral Kernel Dynamics for Planetary Twins**
> Topological Conservation, the Stability–Conservation Tradeoff, and Early Warning from Dust Devils to Rapid Intensification
>
> **P3 — Spectral Kernel Dynamics for Terrestrial Environmental Networks**
> Flow Conservation, Biogeomorphic Coupling, and Cyber-Physical Twins
>
> **P4 — Spectral Kernel Dynamics as a Biosignature Framework**
> Topological Detection of Optimal Controllers on Planetary Surfaces

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
| `kernelcal.attention` | MaxCal diagnostics on transformer attention kernels: GPT-2 probing, toy training, Landauer bound, perturbation-relaxation, grokking phase detection |
| `kernelcal.fluid` | Fluid learning dynamics under MaxCal; kernel-trajectory experiments for flow-based systems |
| `kernelcal.bandits` | **Decentralised Dynamic-Kernel GP-UCB (DDK-GPUCB):** spatiotemporal bandit simulation with learnable mixture kernels, gossip consensus, and Chebyshev-accelerated mixing |
| `kernelcal.geo3d` | **Spectral compression for 3D geometry:** point clouds, triangle meshes (DAE/OBJ), and temporal LiDAR sequences. Hodge Laplacian complex (L₀/L₁/L₂), persistent homology (0D/1D), compression ratio bounds, Nyström large-mesh path. **`score_compression()`** self-introspection: four-channel quality report (geometry / spectral / kernel / topology) with composite loss and grade |
| `kernelcal.terrain` | **Planetary terrain analysis and topological biosignature detection** (P2, P3, P4). DEM→graph pipeline (D8 flow routing, slope/curvature), crater rim detection and Betti numbers, drainage network graphs (Strahler ordering, max-flow/min-cut), the triple spectral diagnostic for channel detection (Proposition 3, P2), topological biosignature Δβ₁, cross-kernel factorization test, plume spectral entropy biosignature, fixed-point kernel, stability–conservation tradeoff (Route 3), bandwidth-optimal mode selection, observability ratio. **66 tests, stdlib-only (numpy + scipy).** |

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

### Planetary terrain and biosignature analysis

```python
import numpy as np
from kernelcal.terrain import (
    synthetic_crater_dem, synthetic_channel_dem,
    dem_to_graph, terrain_graph_laplacian,
    d8_flow_direction, flow_accumulation,
    drainage_network_graph, triple_spectral_diagnostic, topology_budget,
    crater_rim_graph, crater_betti_numbers, abiotic_beta1_craters,
    topological_biosignature, detection_threshold,
    cross_kernel_norm, factorization_test,
    plume_spectral_entropy, chemical_affinity_graph,
    fixed_point_kernel, spectral_entropy, observability_ratio,
    stability_conservation_tradeoff, phase_transition_sweep,
)

# ── Crater analysis (Moon / Mars) ──────────────────────────────────────────
from kernelcal.terrain.craters import CraterCandidate

dem_moon  = synthetic_crater_dem(nrows=64, ncols=64, radius=12.0, depth=5.0)
crater    = CraterCandidate(row=32, col=32, radius=12.0,
                            rim_completeness=1.0, curvature_contrast=1.0)
tg_rim    = crater_rim_graph(dem_moon, [crater], rim_width=3)
betti     = crater_betti_numbers(tg_rim)          # {'beta0':1, 'beta1':1, ...}
null      = abiotic_beta1_craters(n_intact=1)     # {'beta1_abio':1, 'kmin_abio':2}
bio       = topological_biosignature(betti["beta1"], null["beta1_abio"])
print(f"Δβ₁ = {bio.delta_beta1}  anomalous: {bio.is_anomalous}")

# ── River channel detection (Mars / Titan) ─────────────────────────────────
dem_chan  = synthetic_channel_dem(nrows=64, ncols=64, n_tributaries=4)
dg       = drainage_network_graph(dem_chan, threshold=6)
budget   = topology_budget(dg)                    # {'beta0':1,'beta1':3,'kmin':4}
diag     = triple_spectral_diagnostic(dg)         # P2 Proposition 3
print(f"H={diag.H_spectral:.3f}  E_curl={diag.E_curl:.4f}  β₁={diag.beta1}")
print(f"Triple diagnostic (channeled?): {diag.is_channeled}")

# ── Topological biosignature detection threshold (P4 Proposition 1) ────────
thresh = detection_threshold(beta1_abio=2, delta_beta1=1,
                             bits_per_coeff=32.0, I_self_bps=1e9)
print(f"Modes needed: {thresh['k_required']}  R_min: {thresh['R_min']:.2e}")

# ── Cross-kernel factorization test (P4 Proposition 2 / Titan/Dragonfly) ──
K_chem  = np.eye(4)              # marginal chemistry kernel
K_hydro = np.eye(4)              # marginal hydrology kernel
K_joint = np.kron(K_chem, K_hydro) + 0.3 * np.ones((16, 16))  # biological coupling
result  = factorization_test(K_joint, K_chem, K_hydro)
print(f"Coupled: {result['is_coupled']}  r={result['relative_norm']:.3f}")

# ── Plume spectral entropy biosignature (P4 Proposition 3 / Enceladus) ────
species = ["H2", "CH4", "CO2", "NH3", "C2H6", "HCN"]
co = np.array([[0,3,1,0,2,0],[3,0,1,2,0,1],[1,1,0,1,0,0],
               [0,2,1,0,1,2],[2,0,0,1,0,1],[0,1,0,2,1,0]], dtype=float)
_, L_chem = chemical_affinity_graph(species, co)
plume = plume_spectral_entropy(L_chem)
print(f"Entropy drop: {plume['entropy_drop']:.3f}  "
      f"Bandpass spike: {plume['bandpass_spike']:.2f}×  "
      f"Biosignature: {plume['is_biosignature']}")

# ── Spectral kernel diagnostics (P1 / P2) ─────────────────────────────────
L      = terrain_graph_laplacian(dem_to_graph(dem_chan))
h0     = np.ones(L.shape[0])
h_star, info = fixed_point_kernel(L, h0=h0, mu2=2.0, sigma2=1.0,
                                   w=np.ones(L.shape[0]))
H = spectral_entropy(h_star)
print(f"Fixed point converged: {info['converged']}  "
      f"H[h*]={H:.3f}  ρ={info['contraction_ratio']:.3f}")

# ── Stability–conservation tradeoff (Route 3, P2 Prop 1b) ─────────────────
sc = stability_conservation_tradeoff(h_star, L, mu2=2.0, sigma2=1.0,
                                      w=np.ones(L.shape[0]))
print(f"Conservation holds: {sc['conservation_holds']}  "
      f"Δ'={sc['Delta_prime']:.3f}  deficit={sc['conservation_deficit']:.3f}")

# ── Observability ratio — which measurement regime? (P2 Table 2) ───────────
obs = observability_ratio(R_bps=2e5, P_phys_W=1e7, T_K=250.0)  # Mars dust devil
print(f"log₁₀(R/İself) = {obs['log10_ratio']:.1f}  regime: {obs['regime']}")
```

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

### 3D spectral compression (v0.4.2) — `kernelcal.geo3d`

Compress meshes, point clouds, and LiDAR sequences into truncated Laplacian spectral basis:

```python
from kernelcal.geo3d import compress_obj, decompress_obj, large_mesh_bounds

# Compress an OBJ terrain mesh (no trimesh required)
c = compress_obj("terrain.obj", n_modes=128, heat_tau=1.0,
                 payload_path="terrain.kcmesh")

# Inspect compression ratio and distortion
bounds = large_mesh_bounds(c, vertices_original)
print(bounds)
# {'compression_ratio (coeff_only)': 4.5, 'bits_per_vertex': 10.4,
#  'relative_distortion': 0.0012, 'rms_vertex_error': 0.0031, ...}

# Reconstruct
decompress_obj("terrain.kcmesh", "terrain_reconstructed.obj")
```

```python
from kernelcal.geo3d import compress_dae, decompress_dae   # DAE round-trip (needs trimesh)
from kernelcal.geo3d import compress_point_cloud           # point clouds
from kernelcal.geo3d import compress_temporal_clouds       # LiDAR sequences (stable_subsample=True)
```

**Self-introspection — `score_compression()` (v0.4.2):**

The package can assess its own compression quality across four orthogonal loss channels:
geometry (vertex drift), spectral (frequency structure retention), kernel (HS norm retention),
and topology (handles and components preserved).

```python
from kernelcal.geo3d import compress_mesh_roundtrip, score_compression, betti_numbers
from kernelcal.geo3d.mesh import mesh_combinatorial_laplacian
from kernelcal.spectral import SpectralGraph

c = compress_mesh_roundtrip(vertices, faces, n_modes=64)

# Optional: pass full Laplacian spectrum for kernel retention + spectral gap ratio
L  = mesh_combinatorial_laplacian(len(vertices), faces)
sg = SpectralGraph(L)

score = score_compression(
    c,
    vertices_original=vertices,
    eigenvalues_full=sg.eigenvalues,        # enables kernel_hs_relative, spectral_gap_ratio
    betti=betti_numbers(len(vertices), faces),
)
print(score.summary())
print(score.grade())        # "Excellent" / "Good" / "Fair" / "Poor"
print(score.bottleneck)     # "geometry" / "spectral" / "topology"
```

Sample output:
```
── Compression Score ──────────────────────────────────
  Grade:             Good  (loss=0.1243)
  Bottleneck:        spectral

  Modes / vertices:  k=64  /  V=10000  F=20000
  Compression ratio: 4.62×  (10.4 bpv)

  ── Geometry ──
  Relative distortion:     0.1270
  RMS vertex error:        0.003561

  ── Spectral ──
  Entropy (compressed): 3.8271
  Entropy (max k modes):4.1589
  Entropy retention:    0.9202
  Kernel HS retention:  0.9714
  Spectral gap at k:    0.2340  ✗ mid-cluster

  ── Topology ──
  Topology:             ✓ preserved  (margin=+61)
────────────────────────────────────────────────────
```

`overall_loss` = 0.5 × geometry + 0.3 × spectral + 0.2 × topology.
`spectral_gap_ratio` diagnoses whether the truncation at k falls on a natural cluster boundary
(large gap → principled cut) or mid-cluster (gap near 0 → increase k or adjust τ).

**Hodge topology layer:**

```python
from kernelcal.geo3d import betti_numbers, build_hodge_basis, mesh_persistence

b0, b1, b2 = betti_numbers(n_vertices, faces)   # β₀ components, β₁ loops, β₂ voids
basis = build_hodge_basis(n_vertices, faces, n_modes_0=64)

result = mesh_persistence(n_vertices, faces, vertices_xyz)
print(result.betti_at_inf)   # {0: 1, 1: 0}
```

**Compression bounds:**

```python
from kernelcal.geo3d import compression_ratio_formula, mode_count_for_topology

ratio = compression_ratio_formula(n_vertices=10000, n_faces=20000, n_modes=64)
# → 4.6× (coeff_only: eigenvectors recomputed at decode from faces)

k_min = mode_count_for_topology(betti=(1, 2, 0))
# → 3  (β₀ + β₁: minimum modes to preserve connected components + loops)
```

Storage model (`coeff_only=True`, eigenvectors recomputed at decode):

| k modes | Ratio (V=10K, F=20K) | Bits / vertex |
|---------|----------------------|---------------|
| 32 | 9.1× | 5.2 bpv |
| 64 | 4.6× | 10.4 bpv |
| 128 | 2.3× | 20.8 bpv |

Distortion = spectral tail energy ‖V − Φ_k Φ_kᵀ V‖²_F. Topology preserved iff k ≥ β₀ + β₁.

---

### Decentralised Dynamic-Kernel GP-UCB (v0.3.0) — `kernelcal.bandits`

Simulation suite for the manuscript
*"Decentralised Gaussian Process Bandits with Dynamic Kernels under MaxCal"* (Das, 2026).

Arms are `(x, t)` pairs (space × time).  The field has two structurally distinct regions
requiring **different kernel classes** — not just different hyperparameters:

| Region | True kernel | What SE alone cannot do |
|--------|-------------|------------------------|
| Left (`x < 0.5`) | SE(x) × Periodic(t) | Represent cosine temporal modulation at any lengthscale |
| Right (`x ≥ 0.5`) | Anisotropic SE(x,t) | — (SE is correct here) |

The **mixture kernel** `k = (1−w)·k_SE + w·k_SE×Per` learns the mixing weight `w` per agent.
Agents in the periodic region converge to `w → 1`; agents in the smooth region to `w → 0`.

```python
from kernelcal.bandits import run_experiment, ExperimentConfig

results = run_experiment(ExperimentConfig(T=300, N=4, seed=42))
results.plot("figures/")
```

**Server run** (parallelised across seeds — recommended for T ≥ 500):

```bash
python3 kernelcal/bandits/run_server.py --T 1000 --seeds 20 --n_jobs 4 --out results/
```

**Validity tests** (all pass):
- `LML(SE×Per, periodic data) = −9.4`  vs  `LML(SE, periodic data) = −111.7`  ✓
- `w_per: 0.5 → 0.63` (increases toward 1 on periodic region)  ✓
- `w_smo: 0.5 → 0.12` (decreases toward 0 on smooth region)  ✓

---

### Fluid learning dynamics — `kernelcal.fluid`

```python
from kernelcal.fluid import FluidKernelDynamics

dyn = FluidKernelDynamics(viscosity=0.1, diffusion_coeff=0.5)
trajectory = dyn.evolve(initial_kernel=K0, n_steps=200)
print(trajectory.summary())
```

---

### ROS2 bloom simulation — `ros2_ws/bloom_maxcal_sim`

MaxCal-guided rover following a spatiotemporal algal bloom field.
Works as a standalone Python demo (no ROS2 installation required):

```bash
python3 ros2_ws/src/bloom_maxcal_sim/demo_bloom_maxcal.py
```

Or with a full ROS2 Humble installation:

```bash
cd ros2_ws
colcon build --packages-select bloom_maxcal_sim
source install/setup.bash
ros2 launch bloom_maxcal_sim bloom_sim.launch.py
```

---

### New experiments (v0.2.0)

**Perturbation-relaxation** — test whether converged kernels are dynamical attractors:

```bash
# Requires a checkpoint from a grokking or training run
python -m kernelcal.attention.perturbation \
    --checkpoint figures/grokking/checkpoints/step_050000.pt \
    --sigmas 0.01 0.05 0.1 0.2 \
    --relax-steps 2000 \
    --output-dir figures/perturbation
```

**Extended grokking** — spectral diagnostics across the memorization→generalization phase transition:

```bash
# Full experiment: 50K steps, width scaling, 50 seeds
python -m kernelcal.attention.grokking \
    --primes 23 53 97 --widths 64 128 256 \
    --seeds 50 --steps 50000 \
    --output-dir figures/grokking

# Quick test (~2 min)
python -m kernelcal.attention.grokking --quick
```

**Landauer bound** — energy is auto-detected from GPU/RAPL/FLOPs (no manual kWh entry):

```bash
python -m kernelcal.attention.landauer \
    --widths 128 256 512 1024 \
    --lrs 1e-2 1e-3 1e-4 1e-5 \
    --steps 2000 --n-seeds 3 \
    --output-dir ~/landauer_results
```

When wifi wall-power meters are available, pass a callback:

```python
from kernelcal.attention.energy import EnergyMonitor

monitor = EnergyMonitor.auto_detect(
    wall_watts_callback=lambda: my_wifi_meter.read_watts()
)
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
│   ├── training.py        # Training loop + ensemble with MaxCal diagnostics + checkpoints
│   ├── energy.py          # EnergyMonitor: auto-detect GPU/RAPL/FLOPs, wall-meter callback
│   ├── perturbation.py    # Perturbation-relaxation: perturb head, resume, measure return
│   ├── grokking.py        # Extended grokking: 50K steps, per-head diagnostics, phase detection
│   └── landauer.py        # CKA ΔI + auto energy monitoring + Landauer bound sweep
├── spectral/
│   ├── graph.py           # SpectralGraph: Laplacian eigendecomposition, factory methods
│   ├── source.py          # GaussianMISource, CoupledGaussianMISource
│   ├── dynamics.py        # SpectralKernelDynamics: R_l, fixed points, geodesics, stability
│   ├── experiments.py     # 7-experiment verification suite (Exps 1–7)
│   ├── procedural.py      # Procedural graph diagnostics pipeline
│   ├── procedural_examples.py  # Standard examples runner + CLI
│   ├── channel_image.py   # (dormant) Image-to-graph extraction pipeline
│   └── pipeline.py        # (dormant) Image-to-spectral diagnostics pipeline
├── fluid/                 # NEW v0.3.0
│   ├── dynamics.py        # FluidKernelDynamics: MaxCal-governed flow learning
│   └── experiments.py     # Experiment runners for fluid kernel trajectories
├── geo3d/                 # NEW v0.4.0 — spectral 3D compression
│   ├── graph3d.py         # k-NN adjacency, combinatorial Laplacian, subsampling
│   ├── spectral_codec.py  # CompressedSpectralKernel, compress_point_cloud, heat-kernel weights
│   ├── mesh.py            # CompressedMeshGeometry, compress/decompress_mesh_roundtrip, DAE IO
│   ├── temporal.py        # compress_temporal_clouds: LiDAR sequences + HS path geometry; stable_subsample
│   ├── hodge.py           # Hodge complex: B₁/B₂, L₀/L₁/L₂, Betti numbers, hodge_decompose
│   ├── topology.py        # Persistent homology: 0D (union-find), 1D (matrix reduction), VR
│   ├── bounds.py          # CompressionBounds, CompressionScore, score_compression(); ratio, distortion, topology
│   └── large_mesh.py      # LargeMeshCompressed, Nyström extension, LOBPCG, load_obj, compress_obj
└── bandits/               # NEW v0.3.0 — DDK-GPUCB simulation suite
    ├── field.py           # SpatiotemporalField: (x,t) arms, SE×Per vs SE regions
    ├── kernels.py         # AnisotropicSEKernel, SEPeriodicKernel, MixtureKernel
    ├── network.py         # GossipNetwork: Metropolis P, Chebyshev mixing
    ├── agents.py          # MixtureKernelAgent, DDKGPUCBAgent, StaticGPUCBAgent, DDUCBAgent
    ├── experiment.py      # run_experiment(ExperimentConfig): 4-way comparison + plots
    ├── run_server.py      # CLI runner with joblib parallelism + JSON checkpoint
    └── README.md          # Subpackage docs and server usage

ros2_ws/
└── src/bloom_maxcal_sim/  # NEW v0.3.0 — ROS2 package
    ├── bloom_field.py           # Spatiotemporal bloom field model
    ├── maxcal_bloom_follower.py # MaxCal-guided rover trajectory
    ├── rover_model.py           # Rover kinematics
    ├── nodes/                   # ROS2 node wrappers
    │   ├── bloom_field_node.py
    │   ├── maxcal_controller_node.py
    │   ├── rover_sim_node.py
    │   └── visualizer_node.py
    ├── demo_bloom_maxcal.py     # Standalone demo (no ROS2 needed)
    ├── launch/bloom_sim.launch.py
    └── config/default.yaml
```

### Energy monitoring

`EnergyMonitor` auto-detects all available power sources — no manual entry needed:

| Source | How | Priority |
|---|---|---|
| GPU hardware counter | `pynvml` `nvmlDeviceGetTotalEnergyConsumption` | Highest (most accurate) |
| GPU power polling | `pynvml` 500ms polls, trapezoidal integration | Fallback |
| Intel RAPL | `/sys/class/powercap/intel-rapl/*/energy_uj` | CPU + DRAM |
| FLOPs estimate | `6 × n_params × batch_tokens` at ~0.5 pJ/FLOP | Always available |
| Wall-power meter | User-supplied callback (wifi smart plug) | Authoritative when present |

```python
from kernelcal.attention.energy import EnergyMonitor

monitor = EnergyMonitor.auto_detect()
monitor.start()
# ... training ...
report = monitor.stop()
print(f"{report.total_joules:.2f} J  sources={report.sources_used}")
```

### Server deployment (Landauer experiment)

For the Landauer bound experiment on a 2×GPU server:

```
Dockerfile.landauer          # CUDA 12.1 + PyTorch 2.2 container
docker-compose.landauer.yml  # 2-GPU parallel sweep (widths split per GPU)
run_landauer_server.sh       # One-command: build → run → merge → figure
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
| Perturbation-relaxation (H4) | `kernelcal.attention.perturbation` |
| Grokking as spectral phase transition (H5/OQ3) | `kernelcal.attention.grokking` |
| Auto energy monitoring | `kernelcal.attention.energy` |
| Spectral geometric functional $\mathcal{R}_l$ (Prop. 1†) | `kernelcal.spectral.dynamics` |
| Self-consistent kernels via exponential tilting (Cor. 1†) | `kernelcal.spectral.dynamics` |
| Log-linear Fisher–Rao geodesics (Cor. 2†) | `kernelcal.spectral.dynamics` |
| Hessian stability, Fiedler gap $\Delta'$ (Cor. 3†, Q6†) | `kernelcal.spectral.dynamics` |
| Spectral entropy early-warning (Rem. 8†) | `kernelcal.spectral.dynamics` |

| DDK-GPUCB regret decomposition (Thm. 1‡) | `kernelcal.bandits.experiment` |
| Mixture kernel w/ logit-weight adaptation (‡) | `kernelcal.bandits.kernels.MixtureKernel` |
| SE×Periodic product kernel (‡) | `kernelcal.bandits.kernels.SEPeriodicKernel` |
| Chebyshev-accelerated gossip (Lem. 3.1‡) | `kernelcal.bandits.network.GossipNetwork` |
| Fluid learning dynamics under MaxCal (§) | `kernelcal.fluid.dynamics` |
| Bloom field MaxCal rover (§) | `ros2_ws/bloom_maxcal_sim` |

| Stability–conservation tradeoff $D_m = H_{mm} = -\Delta'$ (P2 Prop. 1b) | `kernelcal.terrain.diagnostics.stability_conservation_tradeoff` |
| Route 3 numerical verification (P2 Exp. 4) | `kernelcal/route3_conservation_test.py`, `tests/test_terrain.py::TestDiagnostics` |
| Topological Conservation Theorem $k_{\min} = \beta_0 + \beta_1$ (P2 Thm. 1) | `kernelcal.terrain.craters.abiotic_beta1_craters`, `kernelcal.terrain.channels.topology_budget` |
| Triple spectral diagnostic (P2 Prop. 3) | `kernelcal.terrain.channels.triple_spectral_diagnostic` |
| Bandwidth-constrained protocol (P2 Alg. 1) | `kernelcal.terrain.diagnostics.bandwidth_optimal_modes` |
| Observability ratio $R/\dot{I}_{\rm self}$ (P2 Table 2) | `kernelcal.terrain.diagnostics.observability_ratio` |
| OCN as MaxCal fixed point (P3 Thm. 7.3) | `kernelcal.terrain.channels.drainage_network_graph` |
| Max-Flow Min-Cut phase transition (P3 Prop. 7.5) | `kernelcal.terrain.diagnostics.phase_transition_sweep` |
| Topological biosignature $\Delta\beta_1$ (P4 Def. 1) | `kernelcal.terrain.biosig.topological_biosignature` |
| Detection threshold $R_{\min}$ (P4 Prop. 1) | `kernelcal.terrain.biosig.detection_threshold` |
| Cross-kernel factorization test (P4 Prop. 2) | `kernelcal.terrain.biosig.factorization_test` |
| Plume spectral entropy biosignature (P4 Prop. 3) | `kernelcal.terrain.biosig.plume_spectral_entropy` |

*† from the companion spectral paper (in preparation)*
*‡ from "Decentralised GP Bandits with Dynamic Kernels under MaxCal" (in preparation)*
*§ from fluid learning manuscript (in preparation)*

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

## External datasets used in demonstrations

The scripts in this repository use the following publicly available datasets.
Please cite them if you use the corresponding analyses.

| Dataset | Source | Script(s) | Notes |
|---|---|---|---|
| **Robbins (2018/2019) Global Lunar Crater Database** — 1.3 M craters, D ≥ 1 km, LRO WAC / LOLA / SELENE TC | [USGS Astrogeology](https://astrogeology.usgs.gov/search/map/moon_crater_database_v1_robbins) | `robbins_kernelcal.py`, `robbins_paper_figs.py` | k-NN proximity graph; used as methodological null to expose graph-construction invariance |
| **USGS 3DEP 1 m LiDAR DEM** — Coconino / Oak Creek Canyon, AZ Plateau *(planned future experiment)* | [USGS National Map](https://www.usgs.gov/the-national-map-data-delivery) | *(script not yet written — see ED0 in P4 §9.1)* | Rook-adjacency D8 channel graph for abiotic null calibration; DEM channel extraction methodology must be validated before scientific use |
| **MADNet HiRISE DTM mosaic** — Jezero Crater, Mars *(planned future experiment)* | [Tao et al. 2023, *Earth and Space Science*](https://doi.org/10.1029/2022EA002597); [FU Berlin data repository](https://refubium.fu-berlin.de/handle/fub188/41095) | *(script not yet written — see ED2 in P4 §9.1)* | Seam-free 1 m/pixel DTM; required for the Mars delta topology experiment; the HRSC DEM has persistent swath-boundary step edges that cannot be fully removed by post-hoc filtering |
| **OpenStreetMap street networks** — 5 world cities (Barcelona, Phoenix, Venice, Marrakech, Houston) | [© OpenStreetMap contributors](https://www.openstreetmap.org/copyright), via [OSMnx](https://github.com/gboeing/osmnx) | `osm_street_kernelcal.py` | Physically motivated edges (road segments); intersection nodes |
| **OpenStreetMap building footprints** — 5 world cities | [© OpenStreetMap contributors](https://www.openstreetmap.org/copyright), via [OSMnx](https://github.com/gboeing/osmnx) | `osm_urban_kernelcal.py` | k-NN proximity graphs on centroids (superseded by street-network analysis) |
| **DREAMS-lab LROC NAC MaskRCNN predictions** — highland crater patch | [DREAMS-lab/LROC_NAC_MaskRCNN_Prediction_Pipeline](https://github.com/DREAMS-lab/LROC_NAC_MaskRCNN_Prediction_Pipeline) | `lunar_kernelcal.py` | Results not used in paper; segmentation biased to one side of image |

### Graph construction provenance

Every edge in a calibration graph must have a physical referent.
The scripts implement three edge-construction methods with different validity status:

| Method | Physical referent | Status |
|---|---|---|
| **OSM road segment** (`osm_street_kernelcal.py`) | Built road = an act of construction by a planning controller | ✓ Physically motivated — confirmed |
| **D8 rook adjacency** (planned, see ED0/ED2 in P4) | Shared pixel boundary = water flows between neighbouring channel cells | ✓ Physically motivated — not yet implemented |
| **k-NN proximity** (`robbins_kernelcal.py`) | None — analyst-imposed distance threshold | ✗ Graph-construction artifact — retained as methodological null |

k-NN graphs on 2,000 points in a bounded domain produce nearly identical spectral signatures regardless of the generative process.
The Robbins crater analysis (`robbins_kernelcal.py`) explicitly demonstrates this invariance and is retained as a methodological transparency exhibit.

---

## Citation

Cite the primary paper for the framework, the dataset paper(s) for any
analysis that uses external data:

```bibtex
@article{das2026kerneldynamics,
  title   = {Kernel Dynamics under Path Entropy Maximization},
  author  = {Das, Jnaneshwar},
  journal = {arXiv preprint arXiv:2603.27880},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.27880}
}

@article{das2026maxcal,
  title   = {Spectral Kernel Dynamics via Maximum Caliber:
             Fixed Points, Geodesics, and Phase Transitions},
  author  = {Das, Jnaneshwar},
  journal = {arXiv preprint arXiv:2604.09745},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.09745}
}

@article{das2026biosignature,
  title   = {Spectral Kernel Dynamics as a Biosignature Framework:
             Topological Detection of Optimal Controllers on Planetary Surfaces},
  author  = {Das, Jnaneshwar},
  note    = {Manuscript, Arizona State University},
  year    = {2026}
}

%% ── External datasets ────────────────────────────────────────────────────

@article{robbins2019,
  title   = {A new global database of lunar impact craters $>$1--2~km:
             1.~Crater locations and sizes, comparisons with published
             databases, and global analysis},
  author  = {Robbins, Stuart J.},
  journal = {Journal of Geophysical Research: Planets},
  volume  = {124},
  number  = {4},
  pages   = {871--892},
  year    = {2019},
  doi     = {10.1029/2018JE005592}
}

@misc{robbins2018db,
  title        = {Moon Crater Database v1 Robbins},
  author       = {Robbins, Stuart J.},
  howpublished = {USGS Astrogeology Science Center},
  year         = {2018},
  note         = {Published 2018-08-15},
  url          = {https://astrogeology.usgs.gov/search/map/moon_crater_database_v1_robbins}
}

@software{boeing2017osmnx,
  title   = {{OSMnx}: New methods for acquiring, constructing, analysing,
             and visualising complex street networks},
  author  = {Boeing, Geoff},
  journal = {Computers, Environment and Urban Systems},
  volume  = {65},
  pages   = {126--139},
  year    = {2017},
  doi     = {10.1016/j.compenvurbsys.2017.05.004}
}

@article{tao2023madnet,
  title   = {A high-resolution digital terrain model mosaic of the Mars~2020
             {Perseverance} rover landing site, {Jezero} Crater, Mars, from
             {MADNet} deep-learning-based multi-view stereo surface modelling},
  author  = {Tao, Yu and Muller, Jan-Peter and Conway, Susan~J. and
             Putri, Alfiah~R.~D.},
  journal = {Earth and Space Science},
  volume  = {10},
  pages   = {e2022EA002597},
  year    = {2023},
  doi     = {10.1029/2022EA002597}
}
```

---

## Changelog

### v0.7.0 (April 2026)
- **Scope reduced to confirmed systems only** — empirical calibration now contains
  exactly two script families: OSM city street networks (active controller, confirmed)
  and Robbins lunar crater k-NN (methodological null, confirmed)
- **Removed DEM-based terrain scripts** — `terrain_channel_graph.py`, `badlands_kernelcal.py`,
  `artifact_filter.py`, and `tests/test_artifact_filter.py` deleted; D8 rook-adjacency
  channel extraction requires careful DEM selection and methodology validation before
  results are scientifically meaningful; deferred to ED0 in P4 §9.1
- **Paper (P4) further tightened** — AZ Plateau removed from empirical calibration;
  the one confirmed tier is the active-controller tier (5 cities); the abiotic and
  fossil-controller tiers are theoretically predicted; §9.1 now has five numbered
  experimental designs (ED0–ED4)

### v0.6.0 (April 2026)
- **Removed Jezero analysis** — `jezero_rook_kernelcal.py` and `jezero_kernelcal.py` deleted;
  the HRSC DEM contains persistent swath-seam step edges that survive all post-hoc filtering;
  Jezero analysis deferred to ED2 in P4 §9.1 pending a seam-free MADNet HiRISE DTM
- **Paper (P4) tightened** — Jezero removed from empirical calibration; §9.1
  "Future Experimental Designs" added

### v0.5.1 (April 2026)
- **Empirical calibration pipeline** — two-system controller hierarchy on real data
  - `terrain_channel_graph.py`: rook/queen adjacency terrain graphs on AZ Plateau USGS 3DEP DEM; visual verification of physically motivated edges vs. k-NN artifacts; AZ abiotic null: ΔH = −0.027, β₁ = 3
  - `osm_street_kernelcal.py`: OSM road-network graphs for 5 cities; physically motivated edges (road segments); spatial-patch bootstrap (N=300, 100 iterations); ΔH ∈ [−0.34, −0.24], Δβ₁/N ∈ [0.19, 0.61]
  - `robbins_kernelcal.py`, `robbins_paper_figs.py`: Robbins global lunar crater k-NN analysis retained as **methodological null** demonstrating graph-construction invariance; 5 regional sub-samples are spectrally identical despite geological diversity
- **Graph construction methodology** — documented physical vs. artifact edges; k-NN proximity graphs diagnosed as graph-construction artifacts for all point-cloud inputs
- **New data acknowledgements**: MADNet HiRISE DTM (Tao et al. 2023) cited as planned future dataset; OSM street networks added; graph provenance table added to README
- **Field notes 38–41** document the full methodology pivot and results

### v0.5.0 (April 2026)
- **New: `kernelcal.terrain`** — planetary terrain analysis and topological biosignature detection
  - `dem.py`: DEM → grid graph, D8 flow routing, flow accumulation, slope/curvature, synthetic test fixtures
  - `craters.py`: Hough-transform crater detection, rim graph, Betti numbers, abiotic null model
  - `channels.py`: D8 drainage network graphs, Strahler ordering, Hodge decomposition on edge signals, triple spectral diagnostic (P2 Prop. 3), topology budget (kmin)
  - `biosig.py`: topological biosignature Δβ₁ (P4 Def. 1), detection threshold, cross-kernel factorization test, plume spectral entropy biosignature
  - `diagnostics.py`: fixed-point kernel, spectral entropy, Fiedler-mode gap, stability–conservation tradeoff (Route 3 / P2 Prop. 1b), phase-transition sweep, observability ratio, bandwidth-optimal mode selection
  - 66 tests, stdlib-only (numpy + scipy)
- **Route 3 result** numerically verified and documented: conservation identity D_m = H_mm = −Δ′ for Gaussian MI source on P8 (`kernelcal/route3_conservation_test.py`)
- Paper series expanded to P1–P4; README and citation block updated
