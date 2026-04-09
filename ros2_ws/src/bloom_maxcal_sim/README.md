# bloom_maxcal_sim

ROS2 package implementing the adaptive algal-bloom sampling scenario from
**Section VI.C** of *"Kernel Dynamics under Path Entropy Maximization"*
([arXiv:2603.27880](https://arxiv.org/abs/2603.27880)).

A simulated 2D rover follows a spatiotemporally evolving algae bloom using a
**Maximum Caliber (MaxCal)** velocity controller backed by the
[`kernelcal`](../../) library.

---

## Mathematical model

### Bloom field

The bloom concentration field **b** : Ω × ℝ≥0 → [0,∞) is a superposition of
_N_ anisotropic Gaussian patches advecting under a **double-gyre stream function**
(Shadden et al., 2005):

```
b(x, y, t) = Σᵢ Aᵢ(t) · exp(-Qᵢ(x − μxᵢ(t), y − μyᵢ(t)))
```

where `Qᵢ` is a rotated quadratic form with semi-axes `σxᵢ, σyᵢ` and
orientation `θᵢ`.

**Advection velocity** (double-gyre, canonical benchmark for chaotic transport):

```
ψ(x̃, ỹ, t) = A · sin(π g(x̃, t)) · sin(π ỹ)
g(x̃, t)    = ε sin(ω t) x̃² + (1 − 2ε sin(ω t)) x̃
u = −(πA/Ly) sin(πg) cos(πỹ)
v = +(πA/Lx) cos(πg) sin(πỹ) · g'(x̃,t)
```

**Patch dynamics** (Euler-Maruyama):
- Centre advects with `(u,v)` plus Gaussian white-noise diffusion
- Amplitude follows logistic growth: `Ȧᵢ = rᵢ Aᵢ(1 − Aᵢ/Kᵢ)`
- Spatial spread: `σ² → σ² + 2Dᵢ dt` (Fickian broadening)

### Rover kinematics

2D differential-drive unicycle model:

```
ẋ = v cos θ,   ẏ = v sin θ,   θ̇ = ω
```

with acceleration limits and noisy bloom concentration observations.

### MaxCal velocity controller

At each control step the rover generates `N` candidate next-positions in a
lookahead ring. The **MaxCal distribution** over candidates is computed by
minimising the correct Caliber dual:

```
D(λ) = log Z(λ) + λ · F         (correct dual for entropy maximisation)
Z(λ) = Σᵢ qᵢ exp(−λ · fᵢ)
∂D/∂λ = F − ⟨f⟩_{p(λ)}          (gradient)
```

where:
- `fᵢ = bloom(xᵢ) + 0.4·|∇b(xᵢ)|` — bloom utility feature
- `F = ⟨f⟩_q + 2.3·std(f)` — targets above-average utility candidates
- `qᵢ = exp(−dᵢ²/2σ_q²)` — Gaussian proximity prior (energy model)

The optimal `λ* < 0` concentrates probability on **high-bloom, high-gradient**
candidates. The HS distance between successive kernel matrices `Kₜ`
(RBF kernel weighted by `pₜ`) tracks kernel trajectory evolution per the paper.

---

## Installation

### Prerequisites

- ROS 2 (Humble / Jazzy / Rolling)
- Python 3.9+
- `kernelcal` installed from the manuscript root:

```bash
cd /path/to/deepgis-maxcal-integration
pip install -e .
```

### Build the package

```bash
cd ros2_ws
colcon build --packages-select bloom_maxcal_sim
source install/setup.bash
```

---

## Running

### Full simulation (ROS2)

```bash
ros2 launch bloom_maxcal_sim bloom_sim.launch.py
```

Key arguments:
```bash
ros2 launch bloom_maxcal_sim bloom_sim.launch.py \
    gyre_A:=0.15 gyre_eps:=0.30 \
    n_candidates:=48 v_max:=1.5 \
    visualize:=true
```

### Standalone demo (no ROS required)

```bash
cd ros2_ws/src/bloom_maxcal_sim
python demo_bloom_maxcal.py --steps 300 --dt 1.0
python demo_bloom_maxcal.py --steps 500 --save   # saves bloom_maxcal_demo.gif
```

### Individual nodes

```bash
ros2 run bloom_maxcal_sim bloom_field_node
ros2 run bloom_maxcal_sim rover_sim_node
ros2 run bloom_maxcal_sim maxcal_controller_node
ros2 run bloom_maxcal_sim visualizer_node
```

---

## Topics

| Topic | Type | Description |
|---|---|---|
| `/bloom/field` | `Float32MultiArray` | Concentration grid (ny×nx) |
| `/bloom/velocity_field` | `Float32MultiArray` | Double-gyre (U,V) field |
| `/bloom/center` | `PointStamped` | Bloom centre of mass |
| `/bloom/params` | `Float32MultiArray` | `[Lx,Ly,nx,ny,t]` |
| `/cmd_vel` | `Twist` | Velocity command |
| `/rover/odom` | `Odometry` | Rover pose + velocity |
| `/rover/bloom_obs` | `Float32` | Noisy bloom observation |
| `/maxcal/waypoint` | `PointStamped` | Selected MaxCal target |
| `/maxcal/diagnostics` | `Float32MultiArray` | `[H, HS_dist, λ, b_obs, g_mag, L_nJ]` |

---

## Package structure

```
bloom_maxcal_sim/
├── bloom_maxcal_sim/
│   ├── bloom_field.py          # Double-gyre advecting Gaussian bloom
│   ├── rover_model.py          # 2D unicycle kinematics + sensing
│   ├── maxcal_bloom_follower.py# MaxCal controller (correct dual)
│   └── nodes/
│       ├── bloom_field_node.py
│       ├── rover_sim_node.py
│       ├── maxcal_controller_node.py
│       └── visualizer_node.py
├── launch/bloom_sim.launch.py
├── config/default.yaml
├── demo_bloom_maxcal.py        # Standalone demo
├── package.xml
└── setup.py
```

---

## Connection to the paper

| Paper concept | Code location |
|---|---|
| MaxCal path entropy (Eq. 1–2) | `_correct_maxcal()` in `maxcal_bloom_follower.py` |
| Kernel trajectory in HS metric (Eq. 5) | `_update_kernel_tracking()` |
| Landauer thermodynamic bound (Eq. 13) | `_landauer_step()` |
| Adaptive bloom sampling (Sec. VI.C) | Full package |
| Double-gyre advection | `bloom_field.py` → `advection_field_grid()` |
| RBF kernel over candidates | `kernelcal.kernel.space.rbf_kernel` |
