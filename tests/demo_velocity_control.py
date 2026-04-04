"""
Terrain-Aware Velocity Control Demo — kernelcal
=================================================

Simulates the Earth Rover traversing a 12×12 grid with:
  - Two high-complexity terrain patches (ORB-SLAM3-like sparse feature clouds)
  - A central obstacle zone (zero navigable features)
  - A narrow "canyon" corridor that forces the rover to slow and look ahead
  - A "tracking lost" event mid-traverse (dropped SLAM, rover stops)
  - Human-pilot demonstrations biasing the path toward safe corridors

The demo wires together all four navigation modules:

  SemanticSLAMKernelTracker   →  novelty, stability, complexity
       ↓
  TerrainKernelVelocityController  →  v_cmd(t)
       ↓                                        ↓
  InformativePathPlanner       (waypoint)   velocity clamping
       ↓
  HumanPilotDemonstrationLearner (λ transfer)

Outputs (tests/figures/)
------------------------
  vel_fig1_overview.png      — environment with velocity heat-map overlay
  vel_fig2_velocity_profile.png — v(t) and all contributing factors over time
  vel_fig3_factor_breakdown.png — per-factor contribution waterfall
  vel_fig4_slam_kernel.png   — SLAM novelty / stability / complexity driving v
  vel_fig5_path_comparison.png  — MaxCal path vs pilot-transferred path vs random
  vel_fig6_speed_map.png     — spatial map of commanded speeds at each waypoint
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).parent.parent))

from kernelcal.navigation.slam import SemanticSLAMKernelTracker, descriptors_to_kernel
from kernelcal.navigation.planner import InformativePathPlanner
from kernelcal.navigation.pilot import HumanPilotDemonstrationLearner
from kernelcal.navigation.velocity import (
    TerrainKernelVelocityController, VelocityBand,
    map_points_to_kernel, TRACKING_OK, TRACKING_LOST, TRACKING_NOT_INITIALISED,
)

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)
RNG = np.random.default_rng(7)

# Speed-coded colormap: green (fast) → yellow → red (slow/stop)
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "speed", ["#d73027", "#fdae61", "#fee090", "#a6d96a", "#1a9641"]
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Environment
# ──────────────────────────────────────────────────────────────────────────────

GRID_N = 12
xs = np.linspace(0, 1, GRID_N)
ys = np.linspace(0, 1, GRID_N)
XX, YY = np.meshgrid(xs, ys)
WPS = np.column_stack([XX.ravel(), YY.ravel()])   # (144, 2)
N = len(WPS)

DESC_DIM = 24

def terrain_complexity(wps):
    """Two hotspots, one narrow corridor ridge."""
    h1 = np.exp(-18 * np.sum((wps - [0.12, 0.88]) ** 2, axis=1))   # top-left
    h2 = np.exp(-18 * np.sum((wps - [0.88, 0.12]) ** 2, axis=1))   # bottom-right
    ridge = np.exp(-50 * (wps[:, 0] - 0.5) ** 2) * (wps[:, 1] > 0.35) * (wps[:, 1] < 0.65)
    return (h1 + h2 + 0.5 * ridge).clip(0, 1)

def make_obstacle(wps):
    """Central blob + narrow passage forces slow traversal."""
    blob  = np.linalg.norm(wps - [0.5, 0.5], axis=1) < 0.15
    wall1 = (np.abs(wps[:, 0] - 0.5) < 0.07) & (wps[:, 1] < 0.38)
    wall2 = (np.abs(wps[:, 0] - 0.5) < 0.07) & (wps[:, 1] > 0.62)
    return blob | wall1 | wall2

GT_COMPLEXITY = terrain_complexity(WPS)
OBSTACLES     = make_obstacle(WPS)
SAFE          = ~OBSTACLES

# Fake "tracking lost" zone: bottom-left quadrant (dark / featureless)
TRACKING_ZONE = (WPS[:, 0] < 0.25) & (WPS[:, 1] < 0.30)


def make_descriptors(wp_idx: int, t: float = 0.0) -> np.ndarray:
    n_desc = max(4, int(4 + 20 * GT_COMPLEXITY[wp_idx]))
    amp    = 0.5 + 1.5 * GT_COMPLEXITY[wp_idx]
    decay  = np.exp(-0.03 * t)
    return amp * RNG.standard_normal((n_desc, DESC_DIM)) * decay


def make_map_points(wp_idx: int) -> np.ndarray:
    """Fake 3-D ORB-SLAM3 map points near a waypoint."""
    n_pts = max(3, int(5 + 40 * GT_COMPLEXITY[wp_idx]))
    centre = np.array([WPS[wp_idx, 0], WPS[wp_idx, 1], 0.0])
    spread = 0.1 + 0.4 * GT_COMPLEXITY[wp_idx]
    return centre + RNG.standard_normal((n_pts, 3)) * spread


# ──────────────────────────────────────────────────────────────────────────────
# 2. Build informative path (MaxCal planner)
# ──────────────────────────────────────────────────────────────────────────────

planner = InformativePathPlanner(
    WPS,
    energy_budget_joules=200_000.0,
    joules_per_metre=350.0,
    novelty_weight=0.65,
    coverage_weight=0.35,
    fixed_point_tol=0.02,
)
semantic_scores = GT_COMPLEXITY.copy()
semantic_scores[OBSTACLES] = 0.0
planner.update(
    current_position=WPS[0],
    battery_joules_remaining=200_000.0,
    semantic_scores=semantic_scores,
)

# Generate an 80-step autonomous path
AUTO_STEPS = 80
auto_path  = []
battery    = 200_000.0
for _ in range(AUTO_STEPS):
    battery -= 400.0
    planner.update(battery_joules_remaining=battery, semantic_scores=semantic_scores)
    wp  = planner.next_waypoint()
    idx = int(np.argmin(np.linalg.norm(WPS - wp, axis=1)))
    auto_path.append(idx)


# ──────────────────────────────────────────────────────────────────────────────
# 3. SLAM tracker + velocity controller along the auto path
# ──────────────────────────────────────────────────────────────────────────────

tracker = SemanticSLAMKernelTracker(
    descriptor_dim=DESC_DIM, descriptor_mode="cosine",
    fixed_point_tol=0.06, fixed_point_window=5,
)
ctrl = TerrainKernelVelocityController(
    band=VelocityBand(v_min=0.0, v_max=3.0, v_nominal=1.5, v_crawl=0.25),
    novelty_safe=0.08, novelty_danger=1.2,
    min_novelty_factor=0.05,
    min_stability_factor=0.20,
    complexity_ref=2.2,
    smoothing_alpha=0.35,
)

novelty_hist    = []
stability_hist  = []
complexity_hist = []
v_cmd_hist      = []
tracking_hist   = []
waypoint_speeds = {}   # idx → last commanded speed (for spatial map)

for step, idx in enumerate(auto_path):
    # SLAM tracking state
    if TRACKING_ZONE[idx]:
        t_state = TRACKING_LOST if step % 8 < 3 else TRACKING_NOT_INITIALISED
    else:
        t_state = TRACKING_OK

    # Feed SLAM tracker
    descs   = make_descriptors(idx, t=float(step))
    novelty = tracker.update(descs, keyframe_id=step)
    stab    = tracker.map_stability_score()
    cplx    = tracker.current_complexity()

    # Map points → look-ahead kernel (next waypoint)
    lookahead_k = None
    if step + 1 < len(auto_path):
        next_idx = auto_path[step + 1]
        pts = make_map_points(next_idx)
        lookahead_k = map_points_to_kernel(pts, fov_radius=2.0)

    # Velocity controller
    # Pass current kernel from tracker's trajectory
    cur_K = tracker._prev_kernel
    v_cmd = ctrl.update(
        novelty=novelty,
        stability=stab,
        current_kernel=cur_K,
        complexity=cplx,
        tracking_state=t_state,
        next_waypoint_kernel=lookahead_k,
    )

    novelty_hist.append(novelty)
    stability_hist.append(stab)
    complexity_hist.append(cplx)
    v_cmd_hist.append(v_cmd)
    tracking_hist.append(t_state)
    waypoint_speeds[idx] = v_cmd

steps = np.arange(AUTO_STEPS)
v_arr = np.array(v_cmd_hist)
factors = ctrl.factor_histories()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Human-pilot demonstrations → transfer planner
# ──────────────────────────────────────────────────────────────────────────────

def energy_fn(wps): return np.linalg.norm(wps - 0.5, axis=1)
def novelty_fn(wps): return terrain_complexity(wps)
def obstacle_fn(wps): return 1.0 / (np.linalg.norm(wps - [0.5, 0.5], axis=1) + 0.05)

learner = HumanPilotDemonstrationLearner(
    WPS, feature_fns=[energy_fn, novelty_fn, obstacle_fn]
)

# Pilot hugs the top-left hotspot and right side (avoids centre obstacle)
for start_x, path_xs in [
    (0.08, np.linspace(0.08, 0.40, 7)),
    (0.92, np.linspace(0.92, 0.60, 7)),
]:
    route = []
    for x in path_xs:
        y = 0.88 - x * 0.3
        idx = int(np.argmin(np.linalg.norm(WPS - [x, y], axis=1)))
        if SAFE[idx]: route.append(idx)
    if route: learner.add_demonstration(np.array(route))

# Bottom-right hotspot routes
for y_start in [0.12, 0.18]:
    route = []
    for x in np.linspace(0.88, 0.55, 6):
        idx = int(np.argmin(np.linalg.norm(WPS - [x, y_start], axis=1)))
        if SAFE[idx]: route.append(idx)
    if route: learner.add_demonstration(np.array(route))

learner.fit()
transfer_planner = learner.make_planner(WPS)

# 80-step path from transfer planner + velocity controller (same SLAM history)
transfer_path = []
transfer_speeds = {}
tracker2 = SemanticSLAMKernelTracker(
    descriptor_dim=DESC_DIM, descriptor_mode="cosine",
    fixed_point_tol=0.06, fixed_point_window=5,
)
ctrl2 = TerrainKernelVelocityController(
    band=VelocityBand(v_min=0.0, v_max=3.0, v_crawl=0.25),
    novelty_safe=0.08, novelty_danger=1.2,
    min_novelty_factor=0.05, smoothing_alpha=0.35,
)

for step in range(AUTO_STEPS):
    transfer_planner.update(semantic_scores=semantic_scores)
    wp  = transfer_planner.next_waypoint()
    idx = int(np.argmin(np.linalg.norm(WPS - wp, axis=1)))
    transfer_path.append(idx)
    t_state = TRACKING_LOST if TRACKING_ZONE[idx] and step % 8 < 3 else TRACKING_OK
    nov  = tracker2.update(make_descriptors(idx, float(step)), keyframe_id=step)
    v_t  = ctrl2.update(novelty=nov, stability=tracker2.map_stability_score(),
                        current_kernel=tracker2._prev_kernel,
                        complexity=tracker2.current_complexity(),
                        tracking_state=t_state)
    transfer_speeds[idx] = v_t

# Random path
random_path = [int(RNG.integers(0, N)) for _ in range(AUTO_STEPS)]


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

print("Plotting velocity control demo …")

# ── Fig 1: Environment overview with velocity overlay ────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig1.suptitle("Terrain-Aware Velocity Control — Environment Overview",
              fontsize=13, fontweight="bold")

ax = axes[0]
C = GT_COMPLEXITY.reshape(GRID_N, GRID_N)
im = ax.imshow(C, origin="lower", extent=[0, 1, 0, 1],
               cmap="YlGn", vmin=0, vmax=1, alpha=0.85)
ax.contourf(XX, YY, C, levels=5, cmap="YlGn", alpha=0.3)
obs_pts = WPS[OBSTACLES]
ax.scatter(obs_pts[:, 0], obs_pts[:, 1], marker="s", s=30, c="dimgrey", zorder=3,
           label="Obstacle")
tlost = WPS[TRACKING_ZONE & ~OBSTACLES]
ax.scatter(tlost[:, 0], tlost[:, 1], marker="x", s=22, c="firebrick", zorder=3,
           label="Tracking lost zone", alpha=0.7)
plt.colorbar(im, ax=ax, label="Terrain complexity")
ax.set_title("Terrain complexity & zones"); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(fontsize=7)

ax = axes[1]
# Speed heatmap on grid
speed_grid = np.full(N, np.nan)
for idx, v in waypoint_speeds.items():
    speed_grid[idx] = v
speed_img = speed_grid.reshape(GRID_N, GRID_N)
im2 = ax.imshow(speed_img, origin="lower", extent=[0, 1, 0, 1],
                cmap=SPEED_CMAP, vmin=0, vmax=3.0)
ax.scatter(obs_pts[:, 0], obs_pts[:, 1], marker="s", s=30, c="k", zorder=3)
plt.colorbar(im2, ax=ax, label="v_cmd (m/s)")
ax.set_title("Commanded speed at each waypoint"); ax.set_xlabel("x"); ax.set_ylabel("y")

ax = axes[2]
path_xy = WPS[auto_path]
segs = np.stack([path_xy[:-1], path_xy[1:]], axis=1)
v_segs = v_arr[:-1]
lc = LineCollection(segs, cmap=SPEED_CMAP, norm=Normalize(0, 3.0), lw=2.5, zorder=3)
lc.set_array(v_segs)
ax.add_collection(lc)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.scatter(obs_pts[:, 0], obs_pts[:, 1], marker="s", s=30, c="dimgrey", zorder=4)
ax.scatter(WPS[TRACKING_ZONE & ~OBSTACLES, 0], WPS[TRACKING_ZONE & ~OBSTACLES, 1],
           marker="x", s=22, c="firebrick", zorder=4, alpha=0.6)
ax.scatter(*path_xy[0], s=120, marker="^", c="lime", edgecolors="k", zorder=5, label="Start")
ax.scatter(*path_xy[-1], s=120, marker="v", c="red",  edgecolors="k", zorder=5, label="End")
plt.colorbar(
    ScalarMappable(norm=Normalize(0, 3.0), cmap=SPEED_CMAP), ax=ax, label="v_cmd (m/s)"
)
ax.set_title("Autonomous path — speed-coloured"); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(fontsize=7)

fig1.tight_layout()
fig1.savefig(FIGURES / "vel_fig1_overview.png", dpi=150)
plt.close(fig1)
print("  Saved vel_fig1_overview.png")


# ── Fig 2: Velocity profile over time ────────────────────────────────────────
fig2, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig2.suptitle("Velocity Profile and Driving Signals", fontsize=13, fontweight="bold")

ax = axes[0]
ax.fill_between(steps, v_arr, alpha=0.3, color="steelblue")
ax.plot(steps, v_arr, color="steelblue", lw=2, label="v_cmd (m/s)")
ax.axhline(3.0, ls="--", c="green",  lw=1, alpha=0.6, label="v_max 3 m/s")
ax.axhline(0.25, ls="--", c="orange", lw=1, alpha=0.6, label="v_crawl 0.25 m/s")
# Mark tracking-lost steps
lost_mask = np.array(tracking_hist) == TRACKING_LOST
if lost_mask.any():
    ax.fill_between(steps, 0, 3.0, where=lost_mask,
                    color="red", alpha=0.15, label="Tracking LOST")
ax.set_ylabel("v_cmd (m/s)"); ax.set_ylim(0, 3.3)
ax.set_title("Forward velocity command")
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(steps, novelty_hist,   color="tomato",     lw=1.4, label="HS novelty")
ax.plot(steps, complexity_hist, color="mediumpurple", lw=1.4, label="Complexity")
ax2 = ax.twinx()
ax2.plot(steps, stability_hist, color="seagreen", lw=1.4, ls="--", label="Stability")
ax.axhline(0.08, ls=":", c="tomato", lw=1, alpha=0.6, label="novelty_safe")
ax.axhline(1.20, ls=":", c="red",   lw=1, alpha=0.6, label="novelty_danger")
ax.set_ylabel("Novelty / Complexity", color="tomato")
ax2.set_ylabel("Stability [0–1]", color="seagreen")
ax.set_title("SLAM kernel signals driving velocity")
l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3)

ax = axes[2]
f = factors
ax.stackplot(
    steps,
    [1 - f["novelty_factor"],
     1 - f["stability_factor"],
     1 - f["complexity_factor"],
     1 - f["tracking_factor"]],
    labels=["Novelty penalty", "Stability penalty", "Complexity penalty", "Tracking penalty"],
    colors=["tomato", "gold", "mediumpurple", "firebrick"],
    alpha=0.65,
)
ax.set_xlabel("Step"); ax.set_ylabel("Speed reduction (stacked)")
ax.set_ylim(0); ax.set_title("Cumulative speed-reduction factors")
ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(FIGURES / "vel_fig2_velocity_profile.png", dpi=150)
plt.close(fig2)
print("  Saved vel_fig2_velocity_profile.png")


# ── Fig 3: Factor breakdown waterfall ────────────────────────────────────────
fig3, axes = plt.subplots(2, 2, figsize=(13, 8))
fig3.suptitle("Speed Factor Decomposition", fontsize=13, fontweight="bold")

factor_names = ["novelty_factor", "stability_factor", "complexity_factor", "tracking_factor"]
colors_f = ["tomato", "gold", "mediumpurple", "firebrick"]
titles_f = ["Novelty factor σ_nov(t)", "Stability factor σ_stab(t)",
            "Complexity factor σ_cplx(t)", "Tracking factor σ_track(t)"]

for ax, fn, c, title in zip(axes.ravel(), factor_names, colors_f, titles_f):
    data = factors[fn]
    ax.plot(steps, data, color=c, lw=1.5)
    ax.fill_between(steps, data, 1.0, color=c, alpha=0.2, label="speed reduction")
    ax.fill_between(steps, 0, data, color=c, alpha=0.35, label="speed contribution")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, ls="--", c="gray", lw=0.8)
    ax.set_title(title); ax.set_xlabel("Step"); ax.set_ylabel("Factor [0–1]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

fig3.tight_layout()
fig3.savefig(FIGURES / "vel_fig3_factor_breakdown.png", dpi=150)
plt.close(fig3)
print("  Saved vel_fig3_factor_breakdown.png")


# ── Fig 4: SLAM kernel state ──────────────────────────────────────────────────
fig4, axes = plt.subplots(2, 2, figsize=(13, 8))
fig4.suptitle("SLAM Kernel State Driving Velocity Controller",
              fontsize=13, fontweight="bold")

ax = axes[0, 0]
ax.plot(steps, novelty_hist, color="tomato", lw=1.4)
ax.axhline(0.08, ls="--", c="green", lw=1,  label="safe threshold")
ax.axhline(1.20, ls="--", c="red",   lw=1,  label="danger threshold")
ax.set_title("HS novelty ‖K(t)−K(t−1)‖"); ax.set_ylabel("Novelty")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(steps, stability_hist, color="seagreen", lw=1.4)
ax.set_title("Map stability score"); ax.set_ylabel("Stability [0–1]")
ax.axhline(1.0, ls="--", c="gray", lw=0.8); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(steps, complexity_hist, color="mediumpurple", lw=1.4)
ax.axhline(2.2, ls="--", c="orange", lw=1, label="complexity_ref")
ax.set_title("Spectral complexity"); ax.set_ylabel("Complexity (nats)")
ax.set_xlabel("Step"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
# Phase portrait: novelty vs velocity
sc = ax.scatter(novelty_hist, v_cmd_hist, c=steps, cmap="viridis", s=15, alpha=0.7)
ax.set_xlabel("Novelty ‖ΔK‖"); ax.set_ylabel("v_cmd (m/s)")
ax.set_title("Phase portrait: novelty → velocity")
plt.colorbar(sc, ax=ax, label="Step")
ax.grid(True, alpha=0.3)

fig4.tight_layout()
fig4.savefig(FIGURES / "vel_fig4_slam_kernel.png", dpi=150)
plt.close(fig4)
print("  Saved vel_fig4_slam_kernel.png")


# ── Fig 5: Path comparison ────────────────────────────────────────────────────
fig5, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig5.suptitle("Path Comparison: MaxCal / Pilot-Transfer / Random",
              fontsize=13, fontweight="bold")

def _draw_path(ax, path, speeds, title, cmap=SPEED_CMAP, vmax=3.0):
    wxy = WPS[path]
    segs = np.stack([wxy[:-1], wxy[1:]], axis=1)
    spd  = np.array([speeds.get(i, 0.0) for i in path[:-1]])
    lc   = LineCollection(segs, cmap=cmap, norm=Normalize(0, vmax), lw=2.0)
    lc.set_array(spd)
    ax.add_collection(lc)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.scatter(obs_pts[:, 0], obs_pts[:, 1], marker="s", s=28, c="dimgrey", zorder=4)
    ax.scatter(WPS[TRACKING_ZONE & ~OBSTACLES, 0],
               WPS[TRACKING_ZONE & ~OBSTACLES, 1],
               marker="x", s=18, c="firebrick", zorder=4, alpha=0.5)
    ax.scatter(*wxy[0], s=100, marker="^", c="lime",  edgecolors="k", zorder=5)
    ax.scatter(*wxy[-1], s=100, marker="v", c="red",  edgecolors="k", zorder=5)
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(
        ScalarMappable(norm=Normalize(0, vmax), cmap=cmap), ax=ax, label="v_cmd (m/s)"
    )

_draw_path(axes[0], auto_path,     waypoint_speeds,  "MaxCal planner")
_draw_path(axes[1], transfer_path, transfer_speeds,  "Pilot-transfer")
random_speeds = {i: RNG.uniform(0, 3) for i in random_path}
_draw_path(axes[2], random_path,   random_speeds,    "Random baseline")

fig5.tight_layout()
fig5.savefig(FIGURES / "vel_fig5_path_comparison.png", dpi=150)
plt.close(fig5)
print("  Saved vel_fig5_path_comparison.png")


# ── Fig 6: Spatial speed map ──────────────────────────────────────────────────
fig6, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig6.suptitle("Spatial Speed Maps — Averaged Over Full Traverse",
              fontsize=13, fontweight="bold")

def _spatial_map(path, speeds, ax, title):
    v_grid = np.zeros(N)
    cnt    = np.zeros(N)
    for idx in path:
        v_grid[idx] += speeds.get(idx, 0.0)
        cnt[idx] += 1
    with np.errstate(invalid="ignore"):
        v_grid = np.where(cnt > 0, v_grid / cnt, np.nan)
    img = v_grid.reshape(GRID_N, GRID_N)
    im  = ax.imshow(img, origin="lower", extent=[0, 1, 0, 1],
                    cmap=SPEED_CMAP, vmin=0, vmax=3.0)
    ax.scatter(obs_pts[:, 0], obs_pts[:, 1], marker="s", s=28, c="k", zorder=3)
    ax.scatter(WPS[TRACKING_ZONE & ~OBSTACLES, 0],
               WPS[TRACKING_ZONE & ~OBSTACLES, 1],
               marker="x", s=18, c="white", zorder=4)
    plt.colorbar(im, ax=ax, label="Mean v_cmd (m/s)")
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")

_spatial_map(auto_path,     waypoint_speeds,  axes[0], "MaxCal planner")
_spatial_map(transfer_path, transfer_speeds,  axes[1], "Pilot-transfer")
_spatial_map(random_path,   random_speeds,    axes[2], "Random baseline")

fig6.tight_layout()
fig6.savefig(FIGURES / "vel_fig6_speed_map.png", dpi=150)
plt.close(fig6)
print("  Saved vel_fig6_speed_map.png")


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("Velocity control demo summary")
print("=" * 65)
s = ctrl.summary()
print(f"  Steps                 :  {s['n_steps']}")
print(f"  Mean v_cmd            :  {s['mean_v']:.2f} m/s")
print(f"  Min  v_cmd            :  {s['min_v']:.2f} m/s")
print(f"  Max  v_cmd            :  {s['max_v']:.2f} m/s")
print(f"  Full-speed steps (%)  :  {s['full_speed_pct']:.1f}%")
print(f"  Crawl steps           :  {s['crawl_steps']}")
print(f"  Stops (v < 0.05)      :  {s['stops']}")
print(f"  Tracking-lost steps   :  {int(np.sum(np.array(tracking_hist)==TRACKING_LOST))}")
print(f"  SLAM fixed-point      :  {s['fixed_point_reached']}")
print(f"  Learned pilot λ       :  {learner.learned_preferences()}")
print()
print(f"  Figures saved to: {FIGURES}/")
print("=" * 65)
