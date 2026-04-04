"""
2-D Toy Navigation Scenario — kernelcal evaluation
====================================================

Scenario
--------
A 10×10 grid of waypoints represents a field site.  Two clusters of "interesting"
terrain (high feature complexity) sit at opposite corners.  A third obstacle
cluster sits near the centre.

The script runs four phases and records wall-clock time, CPU time, and memory
allocation (and GPU utilisation when available) for every kernelcal call:

  1. SLAM warm-up   — SemanticSLAMKernelTracker ingests synthetic descriptor
                      frames as the rover sweeps across the grid.
  2. Autonomous nav — InformativePathPlanner selects the next waypoint at each
                      step, updating with battery level and SLAM novelty scores.
  3. Pilot demo     — A simulated human pilot drives two "expert" routes that
                      prefer the interesting corners and avoid the obstacle.
  4. Transfer       — HumanPilotDemonstrationLearner fits λ from the demos and
                      builds a transferred planner for a new (shifted) grid.

Outputs (saved in tests/figures/)
----------------------------------
  fig1_environment.png      — grid layout, terrain complexity, obstacle
  fig2_slam_evolution.png   — novelty score, loop-closure conf, HS trajectory
  fig3_maxcal_planner.png   — probability distribution snapshots (5 frames)
  fig4_coverage.png         — cumulative visit map vs. random baseline
  fig5_pilot_transfer.png   — learned λ, transferred vs. autonomous distribution
  fig6_compute.png          — wall-clock time per kernelcal call (all phases)
  fig7_kernel_stability.png — stability score & fixed-point flag over time
  fig8_cpu_gpu.png          — CPU time, memory allocation, and GPU telemetry

Run
---
    cd deepgis-maxcal-integration
    python tests/toy_navigation_2d.py
"""

import sys
import time
import tracemalloc
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))

from kernelcal.navigation.slam import SemanticSLAMKernelTracker
from kernelcal.navigation.planner import InformativePathPlanner
from kernelcal.navigation.pilot import HumanPilotDemonstrationLearner

# ──────────────────────────────────────────────────────────────────────────────
# GPU probe (optional — graceful fallback when no CUDA device present)
# ──────────────────────────────────────────────────────────────────────────────

_GPU_AVAILABLE = False
_nvml_handle = None

try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    _GPU_AVAILABLE = True
    print(f"GPU: {pynvml.nvmlDeviceGetName(_nvml_handle).decode()}")
except Exception:
    pass

if not _GPU_AVAILABLE:
    try:
        import torch
        if torch.cuda.is_available():
            _GPU_AVAILABLE = True
            print(f"GPU (torch): {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

def _gpu_snapshot() -> dict:
    """Return current GPU utilisation (%), memory used (MiB), and power (W)."""
    if _nvml_handle is not None:
        try:
            util  = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
            mem   = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
            power = pynvml.nvmlDeviceGetPowerUsage(_nvml_handle) / 1000.0
            return {
                "util_pct":  util.gpu,
                "mem_mib":   mem.used / 1024**2,
                "power_w":   power,
            }
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "util_pct": 0.0,
                "mem_mib":  torch.cuda.memory_allocated(0) / 1024**2,
                "power_w":  0.0,
            }
    except ImportError:
        pass
    return {"util_pct": 0.0, "mem_mib": 0.0, "power_w": 0.0}


# ──────────────────────────────────────────────────────────────────────────────
# Profiling context manager
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProfileResult:
    wall_s:   float = 0.0   # wall-clock seconds
    cpu_s:    float = 0.0   # process CPU seconds
    mem_kb:   float = 0.0   # peak memory delta (KB) during the call
    gpu_util: float = 0.0   # GPU utilisation % (snapshot before/after average)
    gpu_mem:  float = 0.0   # GPU memory used MiB (after)
    gpu_pow:  float = 0.0   # GPU power W (snapshot after)


class _Profiler:
    """Context manager: measures wall, CPU, memory, and GPU for one call."""

    result: ProfileResult

    def __enter__(self):
        self.result = ProfileResult()
        tracemalloc.start()
        self._gpu0 = _gpu_snapshot()
        self._t_wall = time.perf_counter()
        self._t_cpu  = time.process_time()
        return self

    def __exit__(self, *_):
        self.result.wall_s = time.perf_counter() - self._t_wall
        self.result.cpu_s  = time.process_time()  - self._t_cpu
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.result.mem_kb = peak / 1024.0
        gpu1 = _gpu_snapshot()
        self.result.gpu_util = (self._gpu0["util_pct"] + gpu1["util_pct"]) / 2
        self.result.gpu_mem  = gpu1["mem_mib"]
        self.result.gpu_pow  = gpu1["power_w"]


def _profile() -> _Profiler:
    return _Profiler()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

CMAP_PROB = "YlOrRd"
CMAP_NOVEL = "viridis"
CMAP_COVER = "Blues"

RNG = np.random.default_rng(42)


def _timer():
    """Thin wrapper kept for backward compat — delegates to _Profiler."""
    class _T:
        elapsed = 0.0
        def __enter__(self): self._t0 = time.perf_counter(); return self
        def __exit__(self, *_): self.elapsed = time.perf_counter() - self._t0
    return _T()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Build synthetic environment
# ──────────────────────────────────────────────────────────────────────────────

GRID_N = 10
xs = np.linspace(0, 1, GRID_N)
ys = np.linspace(0, 1, GRID_N)
XX, YY = np.meshgrid(xs, ys)
WAYPOINTS = np.column_stack([XX.ravel(), YY.ravel()])   # (100, 2)
N_WP = len(WAYPOINTS)

# Ground-truth terrain complexity: two Gaussian "hotspots" + one obstacle region
def terrain_complexity(wps):
    d1 = np.exp(-20 * np.sum((wps - [0.15, 0.85]) ** 2, axis=1))
    d2 = np.exp(-20 * np.sum((wps - [0.85, 0.15]) ** 2, axis=1))
    return (d1 + d2).clip(0, 1)

def obstacle_mask(wps, radius=0.12):
    centre = np.array([0.5, 0.5])
    return np.linalg.norm(wps - centre, axis=1) < radius

GT_COMPLEXITY = terrain_complexity(WAYPOINTS)
OBSTACLES = obstacle_mask(WAYPOINTS)
SAFE = ~OBSTACLES   # boolean mask of navigable waypoints

# Synthetic descriptor factory: high-complexity sites → high-norm descriptors
DESC_DIM = 16

def make_descriptors(wp_idx: int, n_desc: int = 12, t: float = 0.0) -> np.ndarray:
    """Generate synthetic SLAM descriptors for a given waypoint."""
    base_amp = 1.0 + 2.0 * GT_COMPLEXITY[wp_idx]
    noise = RNG.standard_normal((n_desc, DESC_DIM))
    # Gradually converge as t grows (simulates mapping a known area)
    convergence = np.exp(-0.05 * t)
    return base_amp * noise * convergence


# ──────────────────────────────────────────────────────────────────────────────
# 2. Phase 1 — SLAM warm-up
# ──────────────────────────────────────────────────────────────────────────────

print("Phase 1: SLAM warm-up …")

tracker = SemanticSLAMKernelTracker(
    descriptor_dim=DESC_DIM,
    descriptor_mode="cosine",
    fixed_point_tol=0.05,
    fixed_point_window=4,
)

slam_times = []
slam_cpu   = []
slam_mem   = []
slam_gpu_util = []
slam_gpu_mem  = []
novelty_history = []
stability_history = []
loop_closure_history = []
complexity_history = []

# Sweep row by row (raster scan)
sweep_order = np.arange(N_WP)
for step, idx in enumerate(sweep_order):
    descs = make_descriptors(idx, t=float(step))
    with _profile() as P:
        novelty = tracker.update(descs, keyframe_id=idx)
    slam_times.append(P.result.wall_s)
    slam_cpu.append(P.result.cpu_s)
    slam_mem.append(P.result.mem_kb)
    slam_gpu_util.append(P.result.gpu_util)
    slam_gpu_mem.append(P.result.gpu_mem)

    novelty_history.append(novelty)
    stability_history.append(tracker.map_stability_score())
    complexity_history.append(tracker.current_complexity())

    # Loop closure against the first stored keyframe
    if step > 5:
        conf, _ = tracker.loop_closure_confidence(
            make_descriptors(0, t=float(step))
        )
        loop_closure_history.append(conf)
    else:
        loop_closure_history.append(0.0)

print(f"  SLAM summary: {tracker.summary()}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Phase 2 — Autonomous informative path planning
# ──────────────────────────────────────────────────────────────────────────────

print("Phase 2: Informative path planner …")

BATTERY_START = 80_000.0   # J
BATTERY_DRAIN = 600.0      # J per step (≈ 300 W motor at ~2 steps/s)

planner = InformativePathPlanner(
    candidate_waypoints=WAYPOINTS,
    energy_budget_joules=BATTERY_START,
    joules_per_metre=300.0,
    novelty_weight=0.6,
    coverage_weight=0.4,
    fixed_point_tol=0.02,
    fixed_point_window=6,
)

plan_times    = []
plan_cpu      = []
plan_mem      = []
plan_gpu_util = []
plan_gpu_mem  = []
dist_snapshots = []    # (step, distribution)
battery_history = []
visit_path = []        # indices of visited waypoints
stability_plan_history = []

novelty_map = np.array(novelty_history)  # per-waypoint novelty from SLAM

battery = BATTERY_START
N_PLAN_STEPS = 60

for step in range(N_PLAN_STEPS):
    battery -= BATTERY_DRAIN
    semantic_scores = GT_COMPLEXITY.copy()
    semantic_scores[OBSTACLES] = 0.0   # obstacles have zero appeal

    with _profile() as P:
        planner.update(
            battery_joules_remaining=battery,
            semantic_scores=semantic_scores,
        )
        wp = planner.next_waypoint()

    plan_times.append(P.result.wall_s)
    plan_cpu.append(P.result.cpu_s)
    plan_mem.append(P.result.mem_kb)
    plan_gpu_util.append(P.result.gpu_util)
    plan_gpu_mem.append(P.result.gpu_mem)
    battery_history.append(battery)
    stability_plan_history.append(planner.patrol_stability_score())

    wp_idx = int(np.argmin(np.linalg.norm(WAYPOINTS - wp, axis=1)))
    visit_path.append(wp_idx)

    if step in (0, 10, 20, 35, 59):
        dist_snapshots.append((step, planner.distribution().copy()))

print(f"  Planner stats: {planner.statistics()}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Phase 3 — Simulated human-pilot demonstrations
# ──────────────────────────────────────────────────────────────────────────────

print("Phase 3: Human-pilot demonstrations …")

def pilot_route_1():
    """Pilot prefers top-left hotspot, avoids centre."""
    route = []
    for x in np.linspace(0.05, 0.45, 6):
        idx = int(np.argmin(np.linalg.norm(WAYPOINTS - [x, 0.85], axis=1)))
        if SAFE[idx]:
            route.append(idx)
    for y in np.linspace(0.85, 0.55, 4):
        idx = int(np.argmin(np.linalg.norm(WAYPOINTS - [0.15, y], axis=1)))
        if SAFE[idx]:
            route.append(idx)
    return np.array(route)

def pilot_route_2():
    """Pilot prefers bottom-right hotspot."""
    route = []
    for x in np.linspace(0.55, 0.95, 6):
        idx = int(np.argmin(np.linalg.norm(WAYPOINTS - [x, 0.15], axis=1)))
        if SAFE[idx]:
            route.append(idx)
    for y in np.linspace(0.15, 0.45, 4):
        idx = int(np.argmin(np.linalg.norm(WAYPOINTS - [0.85, y], axis=1)))
        if SAFE[idx]:
            route.append(idx)
    return np.array(route)

def energy_feature(wps):
    """Energy cost: distance from grid centre (proxy for drive distance)."""
    return np.linalg.norm(wps - 0.5, axis=1)

def novelty_feature(wps):
    """Semantic novelty proxy: terrain complexity."""
    return terrain_complexity(wps)

def obstacle_feature(wps):
    """Obstacle proximity: inverse distance from obstacle centre."""
    dist = np.linalg.norm(wps - [0.5, 0.5], axis=1)
    return 1.0 / (dist + 0.01)

learner = HumanPilotDemonstrationLearner(
    waypoints=WAYPOINTS,
    feature_fns=[energy_feature, novelty_feature, obstacle_feature],
)

pilot_times = []
pilot_mem   = []
for route in [pilot_route_1(), pilot_route_2(),
              pilot_route_1()[::-1], pilot_route_2()[::-1]]:
    with _profile() as P:
        learner.add_demonstration(route)
    pilot_times.append(P.result.wall_s)
    pilot_mem.append(P.result.mem_kb)

with _profile() as P:
    lambdas = learner.fit()
pilot_times.append(P.result.wall_s)
pilot_mem.append(P.result.mem_kb)

print(f"  Learned λ: {learner.learned_preferences()}")
print(f"  Log-likelihood: {learner.log_likelihood():.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Phase 4 — Transfer to shifted grid
# ──────────────────────────────────────────────────────────────────────────────

print("Phase 4: Transferring pilot preferences …")

# New grid shifted slightly (simulates adjacent field)
new_wps = WAYPOINTS + RNG.uniform(-0.05, 0.05, WAYPOINTS.shape)
new_wps = new_wps.clip(0, 1)

with _profile() as P:
    transferred_planner = learner.make_planner(new_wps)
transfer_time    = P.result.wall_s
transfer_mem_kb  = P.result.mem_kb

pilot_dist = learner.distribution()
transferred_dist = transferred_planner.distribution()
autonomous_dist = planner.distribution()


# ──────────────────────────────────────────────────────────────────────────────
# 6. Figure 1 — Environment overview
# ──────────────────────────────────────────────────────────────────────────────

print("Plotting …")

fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle("2-D Toy Navigation Environment", fontsize=14, fontweight="bold")

# Terrain complexity
ax = axes[0]
C = GT_COMPLEXITY.reshape(GRID_N, GRID_N)
im = ax.imshow(C, origin="lower", extent=[0, 1, 0, 1], cmap="YlGn",
               vmin=0, vmax=1, alpha=0.9)
ax.contourf(XX, YY, C, levels=5, cmap="YlGn", alpha=0.4)
# Obstacle
theta = np.linspace(0, 2 * np.pi, 60)
ax.fill(0.5 + 0.12 * np.cos(theta), 0.5 + 0.12 * np.sin(theta),
        color="red", alpha=0.35, label="Obstacle")
ax.scatter(*WAYPOINTS.T, s=12, c="k", alpha=0.4, zorder=3)
ax.set_title("Terrain complexity & obstacle")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.colorbar(im, ax=ax, label="complexity")
ax.legend(fontsize=8)

# Ground-truth complexity heat (full scatter)
ax = axes[1]
sc = ax.scatter(WAYPOINTS[:, 0], WAYPOINTS[:, 1], c=GT_COMPLEXITY,
                cmap="plasma", s=80, vmin=0, vmax=1, zorder=3)
ax.scatter(WAYPOINTS[OBSTACLES, 0], WAYPOINTS[OBSTACLES, 1],
           s=120, marker="X", c="red", label="Obstacle", zorder=4)
plt.colorbar(sc, ax=ax, label="GT complexity")
ax.set_title("Waypoint complexity map")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(fontsize=8)

# Pilot demonstration routes
ax = axes[2]
r1 = pilot_route_1(); r2 = pilot_route_2()
ax.scatter(*WAYPOINTS.T, s=18, c="lightgrey", zorder=2)
ax.scatter(WAYPOINTS[OBSTACLES, 0], WAYPOINTS[OBSTACLES, 1],
           s=120, marker="X", c="red", zorder=4, label="Obstacle")
ax.plot(WAYPOINTS[r1, 0], WAYPOINTS[r1, 1], "o-", c="royalblue",
        lw=2, ms=6, label="Pilot route 1")
ax.plot(WAYPOINTS[r2, 0], WAYPOINTS[r2, 1], "s-", c="darkorange",
        lw=2, ms=6, label="Pilot route 2")
ax.set_title("Human-pilot demonstrations")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(fontsize=8)

fig1.tight_layout()
fig1.savefig(FIGURES / "fig1_environment.png", dpi=150)
plt.close(fig1)
print("  Saved fig1_environment.png")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Figure 2 — SLAM evolution
# ──────────────────────────────────────────────────────────────────────────────

steps = np.arange(N_WP)

fig2, axes = plt.subplots(2, 2, figsize=(13, 8))
fig2.suptitle("SLAM Kernel Tracker Evolution", fontsize=14, fontweight="bold")

ax = axes[0, 0]
ax.plot(steps, novelty_history, color="steelblue", lw=1.5)
ax.set_title("Novelty score (HS distance per frame)")
ax.set_xlabel("Keyframe"); ax.set_ylabel("‖K(t)−K(t−1)‖_HS")
ax.axhline(0.05, ls="--", c="red", lw=1, label="fixed-point tol")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(steps, stability_history, color="seagreen", lw=1.5)
ax.set_title("Map stability score")
ax.set_xlabel("Keyframe"); ax.set_ylabel("Stability [0–1]")
ax.axhline(1.0, ls="--", c="gray", lw=1, alpha=0.5)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(steps, loop_closure_history, color="darkorange", lw=1.5)
ax.set_title("Loop-closure confidence (vs frame 0)")
ax.set_xlabel("Keyframe"); ax.set_ylabel("Confidence")
ax.axhline(0.5, ls="--", c="red", lw=1, label="0.5 threshold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(steps, complexity_history, color="mediumpurple", lw=1.5)
ax.set_title("Kernel complexity (spectral entropy)")
ax.set_xlabel("Keyframe"); ax.set_ylabel("Complexity")
ax.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(FIGURES / "fig2_slam_evolution.png", dpi=150)
plt.close(fig2)
print("  Saved fig2_slam_evolution.png")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Figure 3 — MaxCal distribution snapshots
# ──────────────────────────────────────────────────────────────────────────────

fig3, axes = plt.subplots(1, len(dist_snapshots), figsize=(4 * len(dist_snapshots), 4))
fig3.suptitle("InformativePathPlanner — MaxCal distribution snapshots",
              fontsize=13, fontweight="bold")

vmax = max(d.max() for _, d in dist_snapshots)

for ax, (step, dist) in zip(axes, dist_snapshots):
    D = dist.reshape(GRID_N, GRID_N)
    im = ax.imshow(D, origin="lower", extent=[0, 1, 0, 1],
                   cmap=CMAP_PROB, vmin=0, vmax=vmax)
    ax.scatter(WAYPOINTS[OBSTACLES, 0], WAYPOINTS[OBSTACLES, 1],
               marker="X", s=60, c="red", zorder=3)
    if step < len(visit_path):
        ax.scatter(*WAYPOINTS[visit_path[step]], s=100, marker="*",
                   c="white", edgecolors="k", zorder=4, linewidths=0.8)
    ax.set_title(f"step {step}", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="p")

fig3.tight_layout()
fig3.savefig(FIGURES / "fig3_maxcal_planner.png", dpi=150)
plt.close(fig3)
print("  Saved fig3_maxcal_planner.png")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Figure 4 — Coverage map vs random baseline
# ──────────────────────────────────────────────────────────────────────────────

# Random baseline: uniform random waypoint selection
random_visits = np.zeros(N_WP)
rng_idx = RNG.integers(0, N_WP, size=N_PLAN_STEPS)
for idx in rng_idx:
    random_visits[idx] += 1

maxcal_visits = np.zeros(N_WP)
for idx in visit_path:
    maxcal_visits[idx] += 1

fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
fig4.suptitle("Coverage: MaxCal Planner vs Random Baseline", fontsize=14, fontweight="bold")

vmax_v = max(maxcal_visits.max(), random_visits.max())

for ax, visits, title in zip(axes[:2],
                              [maxcal_visits, random_visits],
                              ["MaxCal planner", "Random baseline"]):
    V = visits.reshape(GRID_N, GRID_N)
    im = ax.imshow(V, origin="lower", extent=[0, 1, 0, 1],
                   cmap=CMAP_COVER, vmin=0, vmax=vmax_v)
    ax.scatter(WAYPOINTS[OBSTACLES, 0], WAYPOINTS[OBSTACLES, 1],
               marker="X", s=80, c="red", zorder=3)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="visits")

# Weighted coverage: visits × complexity
ax = axes[2]
maxcal_info = (maxcal_visits * GT_COMPLEXITY).reshape(GRID_N, GRID_N)
random_info  = (random_visits  * GT_COMPLEXITY).reshape(GRID_N, GRID_N)
diff = maxcal_info - random_info
vabs = np.abs(diff).max()
im2 = ax.imshow(diff, origin="lower", extent=[0, 1, 0, 1],
                cmap="RdYlGn", vmin=-vabs, vmax=vabs)
ax.scatter(WAYPOINTS[OBSTACLES, 0], WAYPOINTS[OBSTACLES, 1],
           marker="X", s=80, c="black", zorder=3)
ax.set_title("Info gain: MaxCal − Random\n(green = MaxCal better)")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.colorbar(im2, ax=ax, label="Δ(visits × complexity)")

fig4.tight_layout()
fig4.savefig(FIGURES / "fig4_coverage.png", dpi=150)
plt.close(fig4)
print("  Saved fig4_coverage.png")


# ──────────────────────────────────────────────────────────────────────────────
# 10. Figure 5 — Pilot λ and transfer
# ──────────────────────────────────────────────────────────────────────────────

fig5, axes = plt.subplots(1, 3, figsize=(15, 5))
fig5.suptitle("Human-Pilot Transfer — Inverse MaxCal", fontsize=14, fontweight="bold")

# Learned Lagrange multipliers
ax = axes[0]
prefs = learner.learned_preferences()
bars = ax.bar(list(prefs.keys()), list(prefs.values()),
              color=["steelblue", "seagreen", "tomato"])
ax.axhline(0, c="k", lw=0.8)
ax.set_title("Learned Lagrange multipliers (λ)")
ax.set_ylabel("λ value")
ax.set_xlabel("Feature")
for bar, val in zip(bars, prefs.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, val + (0.01 if val >= 0 else -0.04),
            f"{val:.2f}", ha="center", va="bottom", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# Pilot-inferred distribution
ax = axes[1]
if pilot_dist is not None:
    D_pilot = pilot_dist.reshape(GRID_N, GRID_N)
    im = ax.imshow(D_pilot, origin="lower", extent=[0, 1, 0, 1],
                   cmap=CMAP_PROB)
    ax.scatter(WAYPOINTS[OBSTACLES, 0], WAYPOINTS[OBSTACLES, 1],
               marker="X", s=80, c="white", zorder=3)
    # Overlay pilot routes
    ax.plot(WAYPOINTS[r1, 0], WAYPOINTS[r1, 1], "o-", c="cyan", lw=1.5, ms=4,
            label="Route 1")
    ax.plot(WAYPOINTS[r2, 0], WAYPOINTS[r2, 1], "s-", c="lime", lw=1.5, ms=4,
            label="Route 2")
    ax.set_title("Recovered pilot distribution p_λ")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="p")
    ax.legend(fontsize=7)

# Transferred vs autonomous
ax = axes[2]
if transferred_dist is not None and autonomous_dist is not None:
    td = transferred_dist / transferred_dist.sum()
    ad = autonomous_dist / autonomous_dist.sum()
    diff = (td - ad).reshape(GRID_N, GRID_N)
    vabs = np.abs(diff).max()
    im2 = ax.imshow(diff, origin="lower", extent=[0, 1, 0, 1],
                    cmap="RdYlGn", vmin=-vabs, vmax=vabs)
    ax.scatter(WAYPOINTS[OBSTACLES, 0], WAYPOINTS[OBSTACLES, 1],
               marker="X", s=80, c="black", zorder=3)
    ax.set_title("Transferred − Autonomous\n(green = pilot prefers here)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im2, ax=ax, label="Δp")

fig5.tight_layout()
fig5.savefig(FIGURES / "fig5_pilot_transfer.png", dpi=150)
plt.close(fig5)
print("  Saved fig5_pilot_transfer.png")


# ──────────────────────────────────────────────────────────────────────────────
# 11. Figure 6 — Compute profiling
# ──────────────────────────────────────────────────────────────────────────────

fig6, axes = plt.subplots(2, 2, figsize=(13, 8))
fig6.suptitle("kernelcal Compute Profile", fontsize=14, fontweight="bold")

# SLAM per-call timing
ax = axes[0, 0]
ax.plot(np.array(slam_times) * 1e3, color="steelblue", lw=1.2, alpha=0.8)
ax.axhline(np.mean(slam_times) * 1e3, ls="--", c="red", lw=1.5,
           label=f"mean {np.mean(slam_times)*1e3:.2f} ms")
ax.set_title("SemanticSLAMKernelTracker.update() latency")
ax.set_xlabel("Keyframe"); ax.set_ylabel("Latency (ms)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Planner per-call timing
ax = axes[0, 1]
ax.plot(np.array(plan_times) * 1e3, color="seagreen", lw=1.2, alpha=0.8)
ax.axhline(np.mean(plan_times) * 1e3, ls="--", c="red", lw=1.5,
           label=f"mean {np.mean(plan_times)*1e3:.2f} ms")
ax.set_title("InformativePathPlanner.update() latency")
ax.set_xlabel("Planning step"); ax.set_ylabel("Latency (ms)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Phase summary bar chart
ax = axes[1, 0]
phases = ["SLAM\ntotal", "Planner\ntotal", "Pilot demo\nadd×4", "Pilot\nfit", "Transfer"]
totals_ms = [
    sum(slam_times) * 1e3,
    sum(plan_times) * 1e3,
    sum(pilot_times[:4]) * 1e3,
    pilot_times[4] * 1e3,
    transfer_time * 1e3,
]
colors = ["steelblue", "seagreen", "gold", "darkorange", "mediumpurple"]
bars = ax.bar(phases, totals_ms, color=colors, edgecolor="k", linewidth=0.7)
for bar, val in zip(bars, totals_ms):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8)
ax.set_title("Phase total wall-clock time")
ax.set_ylabel("Time (ms)")
ax.grid(True, alpha=0.3, axis="y")

# CDF of latencies
ax = axes[1, 1]
for times, label, c in [(slam_times, "SLAM update", "steelblue"),
                          (plan_times, "Planner update", "seagreen")]:
    sorted_t = np.sort(np.array(times) * 1e3)
    cdf = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
    ax.plot(sorted_t, cdf, lw=2, label=label, color=c)
ax.axvline(1.0, ls="--", c="red", lw=1, label="1 ms")
ax.set_title("CDF of per-call latency")
ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Cumulative fraction")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

fig6.tight_layout()
fig6.savefig(FIGURES / "fig6_compute.png", dpi=150)
plt.close(fig6)
print("  Saved fig6_compute.png")


# ──────────────────────────────────────────────────────────────────────────────
# 12. Figure 7 — Kernel stability and patrol convergence
# ──────────────────────────────────────────────────────────────────────────────

fig7, axes = plt.subplots(2, 2, figsize=(13, 8))
fig7.suptitle("Kernel Stability & Patrol Convergence", fontsize=14, fontweight="bold")

plan_steps = np.arange(N_PLAN_STEPS)

ax = axes[0, 0]
ax.plot(plan_steps, stability_plan_history, color="darkorange", lw=1.8)
ax.set_title("Planner stability score over steps")
ax.set_xlabel("Planning step"); ax.set_ylabel("Stability [0–1]")
ax.axhline(1.0, ls="--", c="gray", lw=1, alpha=0.6, label="Fixed point")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
batt_frac = np.array(battery_history) / BATTERY_START
ax.plot(plan_steps, batt_frac * 100, color="tomato", lw=1.8)
ax.set_title("Battery remaining")
ax.set_xlabel("Planning step"); ax.set_ylabel("Battery (%)")
ax.axhline(20, ls="--", c="red", lw=1, label="20 % warning")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Path visualisation
ax = axes[1, 0]
ax.scatter(*WAYPOINTS.T, s=14, c=GT_COMPLEXITY, cmap="YlGn",
           vmin=0, vmax=1, alpha=0.6, zorder=2)
ax.scatter(WAYPOINTS[OBSTACLES, 0], WAYPOINTS[OBSTACLES, 1],
           s=120, marker="X", c="red", zorder=4, label="Obstacle")
path_xy = WAYPOINTS[visit_path]
ax.plot(path_xy[:, 0], path_xy[:, 1], "b-", lw=0.8, alpha=0.5, zorder=3)
ax.scatter(path_xy[0, 0], path_xy[0, 1], s=120, marker="^",
           c="lime", edgecolors="k", zorder=5, label="Start")
ax.scatter(path_xy[-1, 0], path_xy[-1, 1], s=120, marker="v",
           c="red", edgecolors="k", zorder=5, label="End")
ax.set_title("Autonomous path (MaxCal planner)")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(fontsize=8)

# Entropy of the planner distribution over time
ax = axes[1, 1]
# Recompute entropy from visit counts (proxy)
cumulative_entropy = []
for k in range(1, N_PLAN_STEPS + 1):
    counts = np.zeros(N_WP)
    for idx in visit_path[:k]:
        counts[idx] += 1
    p = counts / counts.sum()
    ent = -np.sum(p[p > 0] * np.log(p[p > 0]))
    cumulative_entropy.append(ent)

ax.plot(plan_steps, cumulative_entropy, color="mediumpurple", lw=1.8)
ax.set_title("Empirical path entropy (coverage diversity)")
ax.set_xlabel("Planning step"); ax.set_ylabel("Entropy (nats)")
ax.axhline(np.log(N_WP), ls="--", c="gray", lw=1, alpha=0.6,
           label=f"Max = log({N_WP})")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig7.tight_layout()
fig7.savefig(FIGURES / "fig7_kernel_stability.png", dpi=150)
plt.close(fig7)
print("  Saved fig7_kernel_stability.png")


# ──────────────────────────────────────────────────────────────────────────────
# 13. Figure 8 — CPU / GPU / Memory deep profile
# ──────────────────────────────────────────────────────────────────────────────

steps_slam = np.arange(N_WP)
steps_plan = np.arange(N_PLAN_STEPS)

fig8 = plt.figure(figsize=(16, 14))
fig8.suptitle(
    "kernelcal CPU / GPU / Memory Profile"
    + (" (GPU present)" if _GPU_AVAILABLE else " (CPU only — no GPU detected)"),
    fontsize=14, fontweight="bold",
)
gs = gridspec.GridSpec(4, 3, figure=fig8, hspace=0.55, wspace=0.38)

# ── Row 0: wall-clock breakdown ──────────────────────────────────────────────
ax = fig8.add_subplot(gs[0, :2])
ax.plot(steps_slam, np.array(slam_times) * 1e3,
        color="steelblue", lw=1.2, alpha=0.8, label="SLAM update")
ax.axhline(np.mean(slam_times) * 1e3, ls="--", c="steelblue", lw=1.2,
           label=f"SLAM mean {np.mean(slam_times)*1e3:.2f} ms")
ax2 = ax.twinx()
ax2.plot(steps_plan, np.array(plan_times) * 1e3,
         color="seagreen", lw=1.2, alpha=0.8, label="Planner update")
ax2.axhline(np.mean(plan_times) * 1e3, ls="--", c="seagreen", lw=1.2,
            label=f"Planner mean {np.mean(plan_times)*1e3:.2f} ms")
ax.set_xlabel("Step"); ax.set_ylabel("SLAM latency (ms)", color="steelblue")
ax2.set_ylabel("Planner latency (ms)", color="seagreen")
ax.set_title("Wall-clock latency per call")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)

# ── Row 0 col 2: CPU vs wall-clock scatter ──────────────────────────────────
ax = fig8.add_subplot(gs[0, 2])
ax.scatter(np.array(slam_times) * 1e3, np.array(slam_cpu) * 1e3,
           s=18, c="steelblue", alpha=0.6, label="SLAM")
ax.scatter(np.array(plan_times) * 1e3, np.array(plan_cpu) * 1e3,
           s=18, c="seagreen", alpha=0.6, label="Planner")
_lim = max(np.array(slam_times + plan_times).max() * 1e3,
           np.array(slam_cpu + plan_cpu).max() * 1e3) * 1.1
ax.plot([0, _lim], [0, _lim], "k--", lw=0.8, label="CPU=Wall (ideal)")
ax.set_xlabel("Wall-clock (ms)"); ax.set_ylabel("CPU time (ms)")
ax.set_title("CPU time vs wall-clock\n(below line = waiting / memory)")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
ax.set_xlim(0); ax.set_ylim(0)

# ── Row 1: CPU time series ───────────────────────────────────────────────────
ax = fig8.add_subplot(gs[1, :2])
ax.plot(steps_slam, np.array(slam_cpu) * 1e3,
        color="steelblue", lw=1.1, alpha=0.8, label="SLAM CPU")
ax2 = ax.twinx()
ax2.plot(steps_plan, np.array(plan_cpu) * 1e3,
         color="seagreen", lw=1.1, alpha=0.8, label="Planner CPU")
ax.set_xlabel("Step"); ax.set_ylabel("SLAM CPU (ms)", color="steelblue")
ax2.set_ylabel("Planner CPU (ms)", color="seagreen")
ax.set_title("CPU time per call (process_time)")
ax.grid(True, alpha=0.3)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

# ── Row 1 col 2: CPU efficiency ratio ───────────────────────────────────────
ax = fig8.add_subplot(gs[1, 2])
slam_eff  = np.array(slam_cpu)  / (np.array(slam_times)  + 1e-12)
plan_eff  = np.array(plan_cpu)  / (np.array(plan_times)  + 1e-12)
ax.hist(slam_eff.clip(0, 1.5), bins=20, color="steelblue", alpha=0.6,
        label="SLAM", density=True)
ax.hist(plan_eff.clip(0, 1.5), bins=20, color="seagreen",  alpha=0.6,
        label="Planner", density=True)
ax.axvline(1.0, ls="--", c="red", lw=1, label="100% CPU utilisation")
ax.set_xlabel("CPU / wall ratio"); ax.set_ylabel("Density")
ax.set_title("CPU utilisation distribution")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# ── Row 2: Memory ────────────────────────────────────────────────────────────
ax = fig8.add_subplot(gs[2, :2])
ax.fill_between(steps_slam, np.array(slam_mem),
                color="steelblue", alpha=0.35, label="SLAM peak mem")
ax.plot(steps_slam, np.array(slam_mem), color="steelblue", lw=0.8)
ax2 = ax.twinx()
ax2.fill_between(steps_plan, np.array(plan_mem),
                 color="seagreen", alpha=0.35, label="Planner peak mem")
ax2.plot(steps_plan, np.array(plan_mem), color="seagreen", lw=0.8)
ax.set_xlabel("Step")
ax.set_ylabel("SLAM peak mem (KB)", color="steelblue")
ax2.set_ylabel("Planner peak mem (KB)", color="seagreen")
ax.set_title("tracemalloc peak memory allocation per call")
ax.grid(True, alpha=0.3)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

# ── Row 2 col 2: per-phase memory bar ───────────────────────────────────────
ax = fig8.add_subplot(gs[2, 2])
phase_names  = ["SLAM\n(mean/call)", "Planner\n(mean/call)",
                "Pilot\nadd×4", "Pilot\nfit", "Transfer"]
phase_mem_kb = [
    float(np.mean(slam_mem)),
    float(np.mean(plan_mem)),
    float(np.mean(pilot_mem[:4])),
    float(pilot_mem[4]),
    float(transfer_mem_kb),
]
colors_bar = ["steelblue", "seagreen", "gold", "darkorange", "mediumpurple"]
bars = ax.bar(phase_names, phase_mem_kb, color=colors_bar, edgecolor="k", lw=0.7)
for bar, val in zip(bars, phase_mem_kb):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + max(phase_mem_kb) * 0.01,
            f"{val:.0f}", ha="center", va="bottom", fontsize=7)
ax.set_ylabel("Peak mem (KB)")
ax.set_title("Memory per phase")
ax.grid(True, alpha=0.3, axis="y")

# ── Row 3: GPU (or placeholder) ─────────────────────────────────────────────
if _GPU_AVAILABLE and any(v > 0 for v in slam_gpu_util + plan_gpu_util):
    ax = fig8.add_subplot(gs[3, :2])
    ax.plot(steps_slam, slam_gpu_util, color="tomato", lw=1.2, label="SLAM GPU util %")
    ax.plot(steps_plan, plan_gpu_util, color="darkorange", lw=1.2, label="Planner GPU util %")
    ax.set_xlabel("Step"); ax.set_ylabel("GPU utilisation (%)")
    ax.set_title("GPU utilisation during kernelcal calls")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig8.add_subplot(gs[3, 2])
    ax.plot(steps_slam, slam_gpu_mem, color="mediumpurple", lw=1.2, label="SLAM")
    ax.plot(steps_plan, plan_gpu_mem, color="royalblue",    lw=1.2, label="Planner")
    ax.set_xlabel("Step"); ax.set_ylabel("GPU memory (MiB)")
    ax.set_title("GPU memory footprint")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
else:
    ax = fig8.add_subplot(gs[3, :])
    ax.set_axis_off()

    # Summary table showing all metrics numerically
    col_labels = ["Metric", "SLAM update", "Planner update", "Pilot fit", "Transfer"]
    rows = [
        ["Wall mean (ms)",
         f"{np.mean(slam_times)*1e3:.3f}",
         f"{np.mean(plan_times)*1e3:.3f}",
         f"{pilot_times[-1]*1e3:.3f}",
         f"{transfer_time*1e3:.3f}"],
        ["CPU mean (ms)",
         f"{np.mean(slam_cpu)*1e3:.3f}",
         f"{np.mean(plan_cpu)*1e3:.3f}",
         "—", "—"],
        ["CPU/wall ratio",
         f"{np.mean(slam_eff):.2f}",
         f"{np.mean(plan_eff):.2f}",
         "—", "—"],
        ["Peak mem (KB)",
         f"{np.mean(slam_mem):.0f}",
         f"{np.mean(plan_mem):.0f}",
         f"{pilot_mem[-1]:.0f}",
         f"{transfer_mem_kb:.0f}"],
        ["GPU util (%)",
         "n/a", "n/a", "n/a", "n/a"],
    ]
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    ax.set_title("Numeric summary — CPU only (no GPU detected)", pad=12)

fig8.savefig(FIGURES / "fig8_cpu_gpu.png", dpi=150, bbox_inches="tight")
plt.close(fig8)
print("  Saved fig8_cpu_gpu.png")


# ──────────────────────────────────────────────────────────────────────────────
# 15. Summary
# ──────────────────────────────────────────────────────────────────────────────

slam_eff_mean  = float(np.mean(np.array(slam_cpu) / (np.array(slam_times) + 1e-12)))
plan_eff_mean  = float(np.mean(np.array(plan_cpu) / (np.array(plan_times) + 1e-12)))

print()
print("=" * 65)
print("Toy scenario summary")
print("=" * 65)
print(f"  SLAM frames processed  :  {N_WP}")
print(f"  Mean SLAM wall          :  {np.mean(slam_times)*1e3:.3f} ms")
print(f"  Mean SLAM CPU           :  {np.mean(slam_cpu)*1e3:.3f} ms  (CPU/wall {slam_eff_mean:.2f})")
print(f"  Mean SLAM peak mem      :  {np.mean(slam_mem):.0f} KB")
print(f"  Planner steps           :  {N_PLAN_STEPS}")
print(f"  Mean planner wall       :  {np.mean(plan_times)*1e3:.3f} ms")
print(f"  Mean planner CPU        :  {np.mean(plan_cpu)*1e3:.3f} ms  (CPU/wall {plan_eff_mean:.2f})")
print(f"  Mean planner peak mem   :  {np.mean(plan_mem):.0f} KB")
print(f"  Pilot demos             :  {len(learner._demonstrations)}")
print(f"  Fit wall / peak mem     :  {pilot_times[-1]*1e3:.2f} ms  /  {pilot_mem[-1]:.0f} KB")
print(f"  Transfer wall / mem     :  {transfer_time*1e3:.2f} ms  /  {transfer_mem_kb:.0f} KB")
print(f"  GPU available           :  {_GPU_AVAILABLE}")
print(f"  Learned λ               :  {dict(zip(prefs.keys(), lambdas.round(3)))}")
print(f"  Final stability score   :  {stability_plan_history[-1]:.3f}")
print(f"  Patrol classification   :  {planner.classify()}")
print()
print(f"  Figures saved to: {FIGURES}/")
print("=" * 65)
