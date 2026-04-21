#!/usr/bin/env python3
"""Graph-based bishop-rocks explorer with live visualization.

Analogue of ``drone_dem_betti_adaptive_experiment.py`` in the
``software-kernelcal-deepgis-integration`` package, but the underlying graph is
built over **rock centroids** (point data) instead of DEM pixels.

Inputs (expected under ``unprocessed/bishop-root/`` next to this script):
  - ``rocks-coord-list.csv`` — no header, (lon, lat)      ~82k centroids
  - ``rock_traits_full.csv`` — lon, lat, area_m2, major_axis_m,
    minor_axis_m, eccentricity, orientation_deg, elevation_rel   ~14k rocks

The explorer sweeps a circular window of radius ``--window-m`` over the traits
bounding box. At every step it:

1. Projects all lon/lat to a local equirectangular frame in **metres**.
2. Collects the trait rocks inside the window.
3. Builds a local **k-NN graph** where edges are added only between rocks that
   have trait rows in ``rock_traits_full.csv``. Rocks without traits remain
   isolated (no incident edges).
4. Runs scipy's connected-components and the smallest Laplacian eigenvalue
   (Fiedler) on the k-NN graph.
5. Scores candidate motion directions from local k-NN topology + exploration
   bonus (unseen rocks) and moves toward the best candidate.
6. Updates a live 2x3 matplotlib figure:
     (0,0) full map, rocks colored by area + window + path
     (0,1) local graph — nodes colored by component, cyan = k-NN edges
     (0,2) area histogram over rocks inside the window
     (1,0) rolling topology history (n_nodes, n_components, Fiedler)
     (1,1) rolling trait stats (median area, median eccentricity)
     (1,2) cumulative explored: eccentricity vs area scatter, colored by step

A summary PNG plus (optional) MP4 / GIF are written to ``--out``.

Usage
-----
    python3 bishop_rocks_graph_explorer.py                 # default scan
    python3 bishop_rocks_graph_explorer.py --show          # interactive window
    python3 bishop_rocks_graph_explorer.py --steps 120 --knn 8
    python3 bishop_rocks_graph_explorer.py --save-mp4 run.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation as mpl_animation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

# Ensure direct script execution can import the local ``kernelcal`` package.
# This mirrors sibling example scripts moved under ``examples/<category>/``.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Shared lon/lat -> local metres projection lives in the kernelcal package
# so this script, the drone-DEM explorer, and any future examples share the
# same implementation instead of redefining it. Imported names are kept at
# module scope for back-compat with callers such as ``tests/test_bishop_rocks_explorer.py``
# that reach for ``bishop_rocks_graph_explorer.LocalFrame``.
from kernelcal.geo3d import METERS_PER_DEG_LAT, LocalFrame  # noqa: F401

HERE = Path(__file__).resolve().parent
# Matches sibling scripts (``bishop_kernelcal.py``, ``bishop_trait_analysis.py``):
# rock centroids + per-rock traits CSVs are expected under
# ``<repo>/datasets/bishop_scarp/`` by default.  Env vars allow local override
# without editing the script.
DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "KERNELCAL_BISHOP_DATA_DIR",
        str(REPO_ROOT / "datasets" / "bishop_scarp"),
    )
)
DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "KERNELCAL_BISHOP_FIG_DIR",
        str(REPO_ROOT / "bishop_figures" / "rocks_explorer"),
    )
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, LocalFrame]:
    coords = pd.read_csv(data_dir / "rocks-coord-list.csv", header=None,
                         names=["lon", "lat"])
    traits = pd.read_csv(data_dir / "rock_traits_full.csv")

    lon0 = float(traits["lon"].mean())
    lat0 = float(traits["lat"].mean())
    frame = LocalFrame(lon0=lon0, lat0=lat0)

    traits["x_m"], traits["y_m"] = frame.to_xy(traits["lon"].to_numpy(),
                                               traits["lat"].to_numpy())
    coords["x_m"], coords["y_m"] = frame.to_xy(coords["lon"].to_numpy(),
                                               coords["lat"].to_numpy())
    return coords, traits, frame


# ---------------------------------------------------------------------------
# Graph construction over a point set
# ---------------------------------------------------------------------------


def _edge_set_from_neighbors(neighbors: list[np.ndarray], *,
                             skip_self: bool = True) -> set[tuple[int, int]]:
    es: set[tuple[int, int]] = set()
    for i, idxs in enumerate(neighbors):
        for j in idxs:
            if skip_self and int(j) == i:
                continue
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            es.add((a, b))
    return es


def knn_edges(xy: np.ndarray, k: int) -> set[tuple[int, int]]:
    if len(xy) < 2:
        return set()
    tree = cKDTree(xy)
    kq = min(k + 1, len(xy))   # +1 because query returns self
    _, idx = tree.query(xy, k=kq)
    if idx.ndim == 1:
        idx = idx[:, None]
    return _edge_set_from_neighbors([row for row in idx])


def trait_mask_for_coords(
    coords: pd.DataFrame,
    traits: pd.DataFrame,
    *,
    decimals: int = 8,
    tol_m: float = 1.0,
) -> np.ndarray:
    """Boolean mask over `coords`: True when that rock has trait data.

    Prefers nearest-neighbour matching in projected metres (`x_m`,`y_m`) with
    tolerance `tol_m`, which is robust to CSV precision drift between the two
    source files. Falls back to rounded lon/lat key matching when projected
    columns are unavailable.
    """
    if len(coords) == 0:
        return np.empty((0,), dtype=bool)
    if len(traits) == 0:
        return np.zeros((len(coords),), dtype=bool)

    if {"x_m", "y_m"}.issubset(coords.columns) and {"x_m", "y_m"}.issubset(traits.columns):
        cxy = np.c_[coords["x_m"].to_numpy(dtype=float), coords["y_m"].to_numpy(dtype=float)]
        txy = np.c_[traits["x_m"].to_numpy(dtype=float), traits["y_m"].to_numpy(dtype=float)]
        tree = cKDTree(cxy)
        d, idx = tree.query(txy, k=1)
        mask = np.zeros((len(coords),), dtype=bool)
        good = np.isfinite(d) & (d <= float(tol_m))
        if np.any(good):
            mask[np.asarray(idx[good], dtype=int)] = True
        return mask

    tr_lon = np.round(traits["lon"].to_numpy(dtype=float), decimals=decimals)
    tr_lat = np.round(traits["lat"].to_numpy(dtype=float), decimals=decimals)
    trait_keys = {(float(lo), float(la)) for lo, la in zip(tr_lon.tolist(), tr_lat.tolist())}

    co_lon = np.round(coords["lon"].to_numpy(dtype=float), decimals=decimals)
    co_lat = np.round(coords["lat"].to_numpy(dtype=float), decimals=decimals)
    mask = np.array(
        [(float(lo), float(la)) in trait_keys for lo, la in zip(co_lon.tolist(), co_lat.tolist())],
        dtype=bool,
    )
    return mask


def knn_edges_trait_only(xy: np.ndarray, k: int, has_trait: np.ndarray) -> set[tuple[int, int]]:
    """k-NN edges between nodes with traits only; others stay isolated."""
    n = int(len(xy))
    if n < 2:
        return set()
    m = np.asarray(has_trait, dtype=bool)
    if m.size != n:
        raise ValueError("has_trait mask length must match xy length")
    active = np.where(m)[0]
    if active.size < 2:
        return set()

    sub_edges = knn_edges(xy[active], k)
    out: set[tuple[int, int]] = set()
    for i_sub, j_sub in sub_edges:
        i = int(active[int(i_sub)])
        j = int(active[int(j_sub)])
        a, b = (i, j) if i < j else (j, i)
        out.add((a, b))
    return out


def adjacency_from_edges(n: int, edges: Iterable[tuple[int, int]]) -> csr_matrix:
    e = list(edges)
    if not e:
        return csr_matrix((n, n))
    arr = np.asarray(e, dtype=int)
    rows = np.concatenate([arr[:, 0], arr[:, 1]])
    cols = np.concatenate([arr[:, 1], arr[:, 0]])
    data = np.ones(len(rows), dtype=float)
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def fiedler_value(A: csr_matrix) -> float:
    n = A.shape[0]
    if n < 3 or A.nnz == 0:
        return 0.0
    L = laplacian(A, normed=False).astype(float)
    k = min(2, n - 1)
    try:
        vals = eigsh(L, k=k, sigma=0.0, which="LM", return_eigenvectors=False)
    except Exception:
        try:
            vals = eigsh(L, k=k, which="SM", return_eigenvectors=False)
        except Exception:
            return float("nan")
    vals = np.sort(np.real(vals))
    # Fiedler = 2nd smallest eigenvalue of L
    return float(vals[1]) if len(vals) > 1 else float(vals[0])


def fiedler_value_and_ipr(A: csr_matrix) -> tuple[float, float]:
    """Return (lambda2, IPR of Fiedler mode) for an undirected graph."""
    n = A.shape[0]
    if n < 3 or A.nnz == 0:
        return 0.0, 0.0
    L = laplacian(A, normed=False).astype(float)
    k = min(3, n - 1)
    try:
        vals, vecs = eigsh(L, k=k, which="SM", return_eigenvectors=True)
    except Exception:
        # Fall back to value-only path if eigenvectors are numerically unstable.
        return fiedler_value(A), 0.0
    vals = np.real(vals)
    vecs = np.real(vecs)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    lam2 = float(vals[1]) if len(vals) > 1 else float(vals[0])
    if vecs.shape[1] <= 1:
        return lam2, 0.0
    v2 = vecs[:, 1]
    denom = float(np.sum(v2 * v2))
    if denom <= 1e-12:
        return lam2, 0.0
    ipr = float(np.sum(v2**4) / (denom * denom))
    return lam2, ipr


# ---------------------------------------------------------------------------
# Scan path (spiral)
# ---------------------------------------------------------------------------


def spiral_path(xmin: float, xmax: float, ymin: float, ymax: float,
                step: float, n_steps: int) -> np.ndarray:
    """Rectangular spiral starting from the centre, spacing ~ ``step`` metres."""
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    xs = [cx]; ys = [cy]
    dx, dy = step, 0.0
    leg_len = 1
    direction = 0
    while len(xs) < n_steps:
        for _ in range(2):
            for _ in range(leg_len):
                if len(xs) >= n_steps:
                    break
                xs.append(xs[-1] + dx); ys.append(ys[-1] + dy)
            # rotate 90° CCW
            dx, dy = -dy, dx
            direction = (direction + 1) % 4
        leg_len += 1
    pts = np.asarray([xs, ys], dtype=float).T
    # Clamp to bbox so we don't spiral off the scarp
    pts[:, 0] = np.clip(pts[:, 0], xmin, xmax)
    pts[:, 1] = np.clip(pts[:, 1], ymin, ymax)
    return pts[:n_steps]


# ---------------------------------------------------------------------------
# Directional candidate planner (non-quadrant)
# ---------------------------------------------------------------------------


MOVE_DIRS: tuple[tuple[str, tuple[float, float]], ...] = (
    ("E",  ( 1.0,  0.0)),
    ("NE", ( 1.0,  1.0)),
    ("N",  ( 0.0,  1.0)),
    ("NW", (-1.0,  1.0)),
    ("W",  (-1.0,  0.0)),
    ("SW", (-1.0, -1.0)),
    ("S",  ( 0.0, -1.0)),
    ("SE", ( 1.0, -1.0)),
    ("STAY", (0.0, 0.0)),
)

# Anti "race-around" defaults (kept internal to keep CLI simple).
_BACKTRACK_PENALTY = 0.75
_BACKTRACK_COS_THRESHOLD = -0.8
_LOOP_RETURN_PENALTY = 1.0
_LOOP_RETURN_RADIUS_FRACTION = 0.4  # of step_m


def _normalize(vx: float, vy: float) -> tuple[float, float]:
    n = float(np.hypot(vx, vy))
    if n <= 1e-9:
        return 0.0, 0.0
    return vx / n, vy / n

# ---------------------------------------------------------------------------
# Explorer
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    cx: float
    cy: float
    n_nodes: int
    n_edges_knn: int
    n_components: int     # β₀
    beta1: int            # cycles in local k-NN graph
    fiedler: float        # λ₂
    median_area: float
    median_ecc: float
    chosen_direction: str
    chosen_score: float


def run(args: argparse.Namespace) -> None:
    coords, traits, _frame = load(args.data_dir)
    print(f"coords CSV : {len(coords):>7,} rocks")
    print(f"traits CSV : {len(traits):>7,} rocks, columns={list(traits.columns)}")

    tx = traits["x_m"].to_numpy()
    ty = traits["y_m"].to_numpy()
    area = traits["area_m2"].to_numpy()
    ecc = traits["eccentricity"].to_numpy()
    elev = traits["elevation_rel"].to_numpy() if "elevation_rel" in traits.columns else None
    cx_bg = coords["x_m"].to_numpy()
    cy_bg = coords["y_m"].to_numpy()

    xmin, xmax = float(tx.min()), float(tx.max())
    ymin, ymax = float(ty.min()), float(ty.max())
    print(f"traits bbox : x {xmin:.1f} .. {xmax:.1f} m   "
          f"y {ymin:.1f} .. {ymax:.1f} m   "
          f"(Δx={xmax-xmin:.0f}m, Δy={ymax-ymin:.0f}m)")

    step_m = args.step_m if args.step_m > 0 else 0.6 * args.window_m
    rng = np.random.default_rng(args.decision_noise_seed)
    loop_return_radius_m = _LOOP_RETURN_RADIUS_FRACTION * float(step_m)
    if args.motion_policy == "spiral":
        spiral_pts = spiral_path(xmin, xmax, ymin, ymax, step=step_m, n_steps=args.steps)
    else:
        spiral_pts = None

    trait_tree = cKDTree(np.c_[tx, ty])
    # All coord rocks are candidate nodes; only nodes with trait rows can
    # receive k-NN edges.
    coord_tree = cKDTree(np.c_[cx_bg, cy_bg])
    coord_has_trait = trait_mask_for_coords(coords, traits)
    print(
        f"trait-linked coord rocks: {int(np.count_nonzero(coord_has_trait)):,} / {len(coords):,} "
        "(only these are edge-eligible)"
    )

    # Path is built online (the adaptive planner chooses next center from
    # candidate move directions scored on local k-NN topology).
    path_xy: list[tuple[float, float]] = []
    seed_xy = (
        float(spiral_pts[0, 0]) if spiral_pts is not None else
        (float(args.seed_x) if args.seed_x is not None else 0.5 * (xmin + xmax)),
        float(spiral_pts[0, 1]) if spiral_pts is not None else
        (float(args.seed_y) if args.seed_y is not None else 0.5 * (ymin + ymax)),
    )

    # ------------------------------------------------------------------
    # Figure layout — mirrors the 2x3 live panel from the DEM explorer.
    # ------------------------------------------------------------------
    if args.show:
        plt.ion()
    fig, ax = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    ax00, ax01, ax02 = ax[0, 0], ax[0, 1], ax[0, 2]
    ax10, ax11, ax12 = ax[1, 0], ax[1, 1], ax[1, 2]

    # (0,0) Full map
    ax00.scatter(cx_bg, cy_bg, s=0.8, color="lightgray", alpha=0.35, linewidths=0)
    if elev is not None:
        elev_arr = np.asarray(elev, dtype=float)
        finite_elev = elev_arr[np.isfinite(elev_arr)]
        if finite_elev.size:
            vmin = float(np.nanpercentile(finite_elev, 2.0))
            vmax = float(np.nanpercentile(finite_elev, 98.0))
            if vmax <= vmin:
                vmin = float(np.nanmin(finite_elev))
                vmax = float(np.nanmax(finite_elev) + 1e-6)
        else:
            vmin, vmax = 0.0, 1.0
        color_values = elev_arr
        color_norm = Normalize(vmin=vmin, vmax=vmax)
        color_label = "relative elevation"
        title_suffix = "elevation"
    else:
        # Backward-compatible fallback for older trait exports.
        area_clip = np.clip(area, max(area.min(), 0.05), area.max())
        color_values = area_clip
        color_norm = Normalize(vmin=float(np.nanmin(area_clip)),
                               vmax=float(np.nanmax(area_clip)))
        color_label = "area (m²)"
        title_suffix = "area"
    sc_all = ax00.scatter(
        tx, ty,
        c=color_values, cmap="viridis",
        norm=color_norm,
        s=np.clip(area * 1.2, 1.5, 40),
        linewidths=0, alpha=0.85,
    )
    fig.colorbar(sc_all, ax=ax00, shrink=0.8, pad=0.02, label=color_label)
    window = Circle(seed_xy, args.window_m,
                    ec="crimson", fc="none", lw=1.6, alpha=0.9)
    ax00.add_patch(window)
    path_line, = ax00.plot([], [], "-", color="crimson", lw=0.8, alpha=0.8)
    # Yellow arrow showing chosen next direction.
    dir_arrow, = ax00.plot([], [], "-", color="gold", lw=2.0, alpha=0.9)
    dir_head = ax00.scatter([], [], marker=(3, 0, 0), s=60, color="gold")
    ax00.set_aspect("equal")
    ax00.set_xlabel("x (m, local)")
    ax00.set_ylabel("y (m, local)")
    ax00.set_title(f"Bishop scarp — rocks colored by {title_suffix}\n"
                   "(red: scan window, gold: next move)")
    ax00.grid(alpha=0.3)

    # (0,1) Local graph
    ax01.set_aspect("equal")
    ax01.set_xlabel("x (m, local)")
    ax01.set_ylabel("y (m, local)")
    ax01.set_title(f"Local k-NN graph — all rocks (r={args.window_m} m)")
    ax01.grid(alpha=0.3)
    lc_knn = LineCollection([], colors="cyan", linewidths=1.0, alpha=0.9)
    ax01.add_collection(lc_knn)
    local_nodes = ax01.scatter([], [], s=[], c=[], cmap="tab20", linewidths=0)
    ax01.legend(
        handles=[
            Line2D([0], [0], color="cyan", lw=1.2,
                   label=f"k-NN edge  (k={args.knn})"),
            Line2D([0], [0], marker="o", color="k", lw=0, markersize=6,
                   label="node = any rock (color = component)"),
        ],
        loc="lower left", fontsize=8, framealpha=0.85,
    )
    # Panel background so white edges are visible.
    ax01.set_facecolor("#1f1f1f")

    # (0,2) local area histogram
    ax02.set_title("Area histogram (rocks inside window)")
    ax02.set_xlabel("area (m²)")
    ax02.set_ylabel("count")
    ax02.set_yscale("log")
    ax02.grid(alpha=0.3)
    hist_bins = np.geomspace(max(area.min(), 0.01), area.max(), 40)
    (hist_bar,) = ax02.plot([], [], drawstyle="steps-mid", color="steelblue", lw=1.2)

    # (1,0) topology history
    ax10.set_title("Topology per step (k-NN graph)")
    ax10.set_xlabel("step")
    ax10.grid(alpha=0.3)
    nodes_line, = ax10.plot([], [], label="n_nodes", color="tab:blue")
    comp_line, = ax10.plot([], [], label="n_components", color="tab:red")
    fied_line, = ax10.plot([], [], label="Fiedler × 10", color="tab:brown")
    ax10.legend(loc="upper left", fontsize=8)

    # (1,1) trait stats
    ax11.set_title("Trait medians per step (inside window)")
    ax11.set_xlabel("step")
    ax11.grid(alpha=0.3)
    med_area_line, = ax11.plot([], [], label="median area_m²", color="tab:green")
    med_ecc_line, = ax11.plot([], [], label="median eccentricity × 50",
                              color="tab:purple")
    ax11.legend(loc="upper left", fontsize=8)

    # (1,2) cumulative explored: ecc vs area colored by step
    ax12.set_title("Cumulative explored rocks (ecc vs area)")
    ax12.set_xlabel("area (m²)")
    ax12.set_ylabel("eccentricity")
    ax12.set_xscale("log")
    ax12.grid(alpha=0.3)
    cum_scat = ax12.scatter([], [], c=[], s=8, cmap="plasma", linewidths=0)
    cbar_cum = fig.colorbar(cum_scat, ax=ax12, shrink=0.8, pad=0.02, label="step")

    # ------------------------------------------------------------------
    # Step loop  (online: plan next center from directional candidates)
    # ------------------------------------------------------------------
    records: list[StepRecord] = []
    seen_nodes: set[int] = set()         # coord indices — all rocks seen so far
    seen_trait_nodes: set[int] = set()   # trait indices — for cumulative scatter
    metric_history: list[np.ndarray] = []   # spectral/topological vectors for novelty
    next_center: list[tuple[float, float]] = [seed_xy]  # mutable across frames

    # Update ax10 (topology history) to include β₁.
    beta1_line, = ax10.plot([], [], label="β₁ (cycles)", color="tab:orange")
    ax10.legend(loc="upper left", fontsize=8)

    def update_step(step: int) -> Iterable:
        # Spiral planner overrides online choice.
        if spiral_pts is not None:
            cx, cy = float(spiral_pts[step, 0]), float(spiral_pts[step, 1])
        else:
            cx, cy = next_center[0]
        path_xy_snapshot = path_xy + [(cx, cy)]

        # Graph built over ALL coord rocks inside the window.
        coord_idx = np.asarray(coord_tree.query_ball_point((cx, cy), r=args.window_m), dtype=int)
        n_local = len(coord_idx)
        local_xy = np.c_[cx_bg[coord_idx], cy_bg[coord_idx]] if n_local else np.empty((0, 2))

        # Trait rocks inside the window — for stats / histogram only.
        trait_idx = np.asarray(trait_tree.query_ball_point((cx, cy), r=args.window_m), dtype=int)
        n_traits_local = len(trait_idx)

        local_has_trait = coord_has_trait[coord_idx] if n_local else np.empty((0,), dtype=bool)
        if n_local >= 2:
            e_knn = knn_edges_trait_only(local_xy, args.knn, local_has_trait)
        else:
            e_knn = set()

        A_knn = adjacency_from_edges(max(n_local, 1), e_knn)
        if n_local > 0:
            # Always run connected-components, even for empty-edge graphs.
            # (If there are N isolated nodes, beta0 must be N, not 1.)
            n_comp, labels = connected_components(A_knn, directed=False)
        else:
            n_comp, labels = (0, np.empty((0,), dtype=int))
        fied_full = fiedler_value(A_knn) if n_local >= 3 else 0.0
        n_edges_full = int(A_knn.nnz // 2)
        beta1_full = max(0, n_edges_full - n_local + int(n_comp))

        # ---- non-quadrant move selection from directional candidates -------
        prev_dir_vec = (0.0, 0.0)
        if len(records) >= 1:
            prev_rec = records[-1]
            vx = cx - prev_rec.cx
            vy = cy - prev_rec.cy
            prev_dir_vec = _normalize(vx, vy)

        best_name = "STAY"
        best_score = -np.inf
        best_dir = (0.0, 0.0)
        seen_arr = np.fromiter(seen_nodes, dtype=int) if seen_nodes else np.empty((0,), dtype=int)
        best_metric_vec = np.zeros(4, dtype=float)
        scored_candidates: list[tuple[float, str, tuple[float, float], np.ndarray]] = []
        for name, (dx, dy) in MOVE_DIRS:
            cand_x = float(np.clip(cx + step_m * dx, xmin, xmax))
            cand_y = float(np.clip(cy + step_m * dy, ymin, ymax))
            cand_idx = np.asarray(coord_tree.query_ball_point((cand_x, cand_y), r=args.window_m), dtype=int)
            n_cand = int(len(cand_idx))
            if n_cand == 0:
                continue
            else:
                cand_xy = np.c_[cx_bg[cand_idx], cy_bg[cand_idx]]
                cand_has_trait = coord_has_trait[cand_idx] if n_cand else np.empty((0,), dtype=bool)
                cand_edges = (
                    knn_edges_trait_only(cand_xy, args.knn, cand_has_trait)
                    if n_cand >= 2 else set()
                )
                cand_A = adjacency_from_edges(max(n_cand, 1), cand_edges)
                if n_cand > 0 and cand_A.nnz > 0:
                    cand_beta0, _ = connected_components(cand_A, directed=False)
                else:
                    cand_beta0 = n_cand
                cand_fied, cand_ipr = fiedler_value_and_ipr(cand_A) if n_cand >= 3 else (0.0, 0.0)
                cand_beta1 = max(0, int(cand_A.nnz // 2) - n_cand + int(cand_beta0))
                n_new = int(np.count_nonzero(~np.isin(cand_idx, seen_arr))) if seen_arr.size else n_cand
                if args.motion_policy == "spectral-leaf":
                    beta0_norm = float(cand_beta0) / max(1.0, float(n_cand))
                    beta1_norm = float(cand_beta1) / max(1.0, float(n_cand))
                    leafness = 1.0 - float(np.clip(beta1_norm, 0.0, 1.0))
                    lambda_inv = min(50.0, 1.0 / (1e-6 + max(0.0, float(cand_fied))))
                    unseen_frac = float(n_new) / max(1.0, float(n_cand))
                    metric_vec = np.array([beta0_norm, beta1_norm, float(cand_fied), float(cand_ipr)], dtype=float)
                    novelty = 0.0
                    if metric_history:
                        hist = np.vstack(metric_history)
                        novelty = float(np.min(np.linalg.norm(hist - metric_vec[None, :], axis=1)))
                    score = (
                        args.w_lambda_inv * lambda_inv
                        + args.w_beta1_tree * leafness
                        + args.w_beta0_spectral * beta0_norm
                        + args.w_ipr * float(cand_ipr)
                        + args.w_novelty * novelty
                        + args.w_unseen * unseen_frac
                    )
                else:
                    beta0_norm = float(cand_beta0) / max(1.0, float(n_cand))
                    beta1_norm = float(cand_beta1) / max(1.0, float(n_cand))
                    unseen_frac = float(n_new) / max(1.0, float(n_cand))
                    metric_vec = np.array([float(cand_beta0), float(cand_beta1),
                                           float(cand_fied), float(n_new)], dtype=float)
                    score = (
                        args.w_beta1 * beta1_norm
                        + args.w_fiedler * float(cand_fied)
                        - args.w_beta0 * beta0_norm
                        + args.w_unseen * unseen_frac
                    )
                if name == "STAY":
                    score -= 0.5 * max(1.0, args.w_unseen)
                mv = _normalize(dx, dy)
                if mv != (0.0, 0.0):
                    score += args.w_momentum * (mv[0] * prev_dir_vec[0] + mv[1] * prev_dir_vec[1])

                # Anti "race-around" guards:
                # 1) penalize immediate backtracking (direction reversal),
                # 2) penalize tight two-step loops (A->B->A-style returns).
                if len(records) >= 1 and mv != (0.0, 0.0):
                    prev_mv = _normalize(cx - records[-1].cx, cy - records[-1].cy)
                    if prev_mv != (0.0, 0.0):
                        cosang = mv[0] * prev_mv[0] + mv[1] * prev_mv[1]
                        if cosang <= _BACKTRACK_COS_THRESHOLD:
                            score -= _BACKTRACK_PENALTY
                if len(records) >= 2:
                    dist_2back = float(np.hypot(cand_x - records[-2].cx, cand_y - records[-2].cy))
                    if dist_2back <= loop_return_radius_m:
                        score -= _LOOP_RETURN_PENALTY

                # Optional stochastic exploration: small Gaussian perturbation
                # to break deterministic lock-in under near-tied scores.
                if float(args.decision_noise_sigma) > 0.0:
                    score += float(rng.normal(0.0, float(args.decision_noise_sigma)))
                scored_candidates.append((float(score), name, (dx, dy), metric_vec))

        if scored_candidates:
            best_score = max(s for s, *_ in scored_candidates)
            eps = 1e-6 * max(1.0, abs(best_score))
            tied = [cand for cand in scored_candidates if abs(cand[0] - best_score) <= eps]
            chosen = tied[len(records) % len(tied)]
            _, best_name, best_dir, best_metric_vec = chosen

        # if all candidates are poor (empty neighborhoods), drift toward bbox center
        if not np.isfinite(best_score):
            tgt = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)])
            vec = tgt - np.array([cx, cy])
            nrm = float(np.linalg.norm(vec))
            dirv = (vec / nrm) if nrm > 1e-6 else np.array([1.0, 0.0])
        else:
            dirv = np.array(_normalize(best_dir[0], best_dir[1]))
        new_cx = float(np.clip(cx + step_m * dirv[0], xmin, xmax))
        new_cy = float(np.clip(cy + step_m * dirv[1], ymin, ymax))
        next_center[0] = (new_cx, new_cy)

        med_area = float(np.median(area[trait_idx])) if n_traits_local else float("nan")
        med_ecc = float(np.median(ecc[trait_idx])) if n_traits_local else float("nan")

        rec = StepRecord(
            step=step, cx=cx, cy=cy,
            n_nodes=n_local,
            n_edges_knn=len(e_knn),
            n_components=int(n_comp),
            beta1=int(beta1_full),
            fiedler=float(fied_full),
            median_area=med_area,
            median_ecc=med_ecc,
            chosen_direction=best_name,
            chosen_score=float(best_score if np.isfinite(best_score) else 0.0),
        )
        if records and records[-1].step == step:
            records[-1] = rec          # FuncAnimation can call frame 0 twice
        else:
            records.append(rec)
            # Track chosen-state spectral/topological fingerprint for novelty.
            if np.isfinite(best_score):
                metric_history.append(best_metric_vec.copy())
            if spiral_pts is None and len(path_xy) < step + 1:
                path_xy.append((cx, cy))
        seen_nodes.update(int(i) for i in coord_idx)
        seen_trait_nodes.update(int(i) for i in trait_idx)

        # ------------------------- update artists --------------------------
        window.center = (cx, cy)
        # Path history trail.
        if spiral_pts is not None:
            path_line.set_data(spiral_pts[: step + 1, 0],
                               spiral_pts[: step + 1, 1])
        else:
            pxy = np.asarray(path_xy_snapshot, dtype=float)
            path_line.set_data(pxy[:, 0], pxy[:, 1])

        # Chosen direction arrow.
        dir_arrow.set_data([cx, cx + step_m * dirv[0]],
                           [cy, cy + step_m * dirv[1]])
        dir_head.set_offsets(np.array([[cx + step_m * dirv[0],
                                        cy + step_m * dirv[1]]]))

        # Local graph panel.
        knn_segs = [np.array([local_xy[a], local_xy[b]]) for a, b in e_knn]
        lc_knn.set_segments(knn_segs)

        node_sizes = np.full(n_local, 8.0) if n_local else np.array([])
        local_nodes.set_offsets(local_xy if n_local else np.empty((0, 2)))
        local_nodes.set_sizes(node_sizes)
        local_nodes.set_array(labels.astype(float) if n_local else np.array([]))

        pad = 0.15 * args.window_m
        lim = args.window_m + pad
        ax01.set_xlim(cx - lim, cx + lim)
        ax01.set_ylim(cy - lim, cy + lim)

        # Area histogram panel — trait rocks in FoV only.
        if n_traits_local:
            hist, edges = np.histogram(area[trait_idx], bins=hist_bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            hist_bar.set_data(centers, np.maximum(hist, 1e-1))
            ax02.set_xlim(hist_bins[0], hist_bins[-1])
            ax02.set_xscale("log")
            ax02.set_ylim(0.8, max(10, hist.max() * 1.5))

        # Topology history.
        xs = [r.step for r in records]
        nodes_line.set_data(xs, [r.n_nodes for r in records])
        comp_line.set_data(xs, [r.n_components for r in records])
        fied_line.set_data(xs, [10.0 * r.fiedler for r in records])
        beta1_line.set_data(xs, [r.beta1 for r in records])
        ax10.relim(); ax10.autoscale_view()

        # Trait stats.
        med_area_line.set_data(xs, [r.median_area for r in records])
        med_ecc_line.set_data(xs, [50.0 * r.median_ecc for r in records])
        ax11.relim(); ax11.autoscale_view()

        # Cumulative explored scatter — trait rocks only (have area + ecc).
        if seen_trait_nodes:
            sn = np.fromiter(seen_trait_nodes, dtype=int)
            a_seen = area[sn]
            e_seen = ecc[sn]
            cum_scat.set_offsets(np.c_[a_seen, e_seen])
            cum_scat.set_array(np.arange(len(sn)))
            cbar_cum.mappable.set_clim(0, max(1, len(sn)))
            ax12.set_xlim(max(a_seen.min(), 0.01) * 0.8, a_seen.max() * 1.2)
            ax12.set_ylim(-0.02, max(1.02, float(np.nanmax(e_seen)) + 0.02))

        fig.suptitle(
            f"Bishop rocks explorer — step {step+1}/{args.steps}   "
            f"policy={args.motion_policy}   "
            f"window=({cx:.1f}, {cy:.1f}) m   "
            f"graph_nodes={n_local}  traits={n_traits_local}   "
            f"β₀={n_comp}   β₁={beta1_full}   "
            f"λ₂={fied_full:.3g}   →{best_name}  (score={best_score:.2f})",
            fontsize=11,
        )

        return (
            path_line, window, dir_arrow, dir_head,
            lc_knn, local_nodes,
            hist_bar,
            nodes_line, comp_line, fied_line, beta1_line,
            med_area_line, med_ecc_line, cum_scat,
        )

    if args.show:
        for s in range(args.steps):
            update_step(s)
            fig.canvas.draw_idle()
            plt.pause(args.pause_s)
        plt.ioff()
    elif args.save_mp4 or args.save_gif:
        ani = mpl_animation.FuncAnimation(
            fig, update_step, frames=args.steps,
            interval=max(20, int(args.pause_s * 1000)),
            blit=False, repeat=False,
        )
        if args.save_mp4:
            out = Path(args.save_mp4)
            out.parent.mkdir(parents=True, exist_ok=True)
            ani.save(str(out), writer=mpl_animation.FFMpegWriter(fps=args.fps))
            print(f"wrote MP4  : {out}")
        else:
            out = Path(args.save_gif)
            out.parent.mkdir(parents=True, exist_ok=True)
            ani.save(str(out), writer=mpl_animation.PillowWriter(fps=args.fps))
            print(f"wrote GIF  : {out}")
    else:
        for s in range(args.steps):
            update_step(s)

    args.out.mkdir(parents=True, exist_ok=True)
    final_png = args.out / "bishop_rocks_explorer_final.png"
    fig.savefig(final_png, dpi=180)
    print(f"wrote final: {final_png}")

    # Summary CSV of per-step records
    summary = pd.DataFrame(
        dict(
            step=[r.step for r in records],
            cx_m=[r.cx for r in records],
            cy_m=[r.cy for r in records],
            n_nodes=[r.n_nodes for r in records],
            n_edges_knn=[r.n_edges_knn for r in records],
            beta0_components=[r.n_components for r in records],
            beta1_cycles=[r.beta1 for r in records],
            fiedler=[r.fiedler for r in records],
            median_area_m2=[r.median_area for r in records],
            median_eccentricity=[r.median_ecc for r in records],
            chosen_direction=[r.chosen_direction for r in records],
            chosen_score=[r.chosen_score for r in records],
        )
    )
    summary_csv = args.out / "bishop_rocks_explorer_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"wrote summary: {summary_csv}  ({len(summary)} steps, "
          f"{len(seen_nodes):,} rocks explored)")

    if args.show:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help=f"Folder with the two CSVs (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Output folder (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--steps", type=int, default=80,
                   help="Number of explorer steps (default: 80)")
    p.add_argument("--window-m", type=float, default=40.0,
                   help="Scan window radius in metres (default: 40)")
    p.add_argument("--step-m", type=float, default=0.0,
                   help="Move step size in metres (default: 0.6 * window-m)")
    p.add_argument("--knn", type=int, default=6,
                   help="k for k-NN graph (default: 6)")
    p.add_argument("--motion-policy", choices=["greedy", "spectral-leaf", "spiral"], default="greedy",
                   help="Direction policy. 'greedy' (default) scores candidate move "
                        "directions by local k-NN topology + unseen rocks; "
                        "'spectral-leaf' uses only spectral/topological metrics "
                        "(β0, β1, λ2, IPR(v2), novelty, momentum); "
                        "'spiral' walks a fixed outward spiral.")
    p.add_argument("--seed-x", type=float, default=None,
                   help="Starting x in metres (default: bbox centre)")
    p.add_argument("--seed-y", type=float, default=None,
                   help="Starting y in metres (default: bbox centre)")
    p.add_argument("--w-beta1", type=float, default=1.0,
                   help="Weight on normalized β₁/n (cycle density) in greedy direction score")
    p.add_argument("--w-fiedler", type=float, default=20.0,
                   help="Weight on Fiedler λ₂ in greedy direction score")
    p.add_argument("--w-beta0", type=float, default=0.5,
                   help="Penalty weight on normalized β₀/n (fragmentation)")
    p.add_argument("--w-unseen", type=float, default=5.0,
                   help="Weight on unseen-rock fraction in candidate next FoV")
    p.add_argument("--w-momentum", type=float, default=0.45,
                   help="Momentum bonus term for directional continuity")
    p.add_argument("--w-lambda-inv", type=float, default=3.0,
                   help="Spectral-leaf: weight on inverse Fiedler term 1/(eps+λ₂)")
    p.add_argument("--w-beta1-tree", type=float, default=2.0,
                   help="Spectral-leaf: weight on tree-likeness proxy (1 - β₁/n)")
    p.add_argument("--w-beta0-spectral", type=float, default=0.5,
                   help="Spectral-leaf: weight on component density β₀/n")
    p.add_argument("--w-ipr", type=float, default=1.5,
                   help="Spectral-leaf: weight on IPR of Fiedler eigenvector")
    p.add_argument("--w-novelty", type=float, default=1.0,
                   help="Spectral-leaf: weight on metric novelty vs previous steps")
    p.add_argument(
        "--decision-noise-sigma",
        type=float,
        default=0.0,
        help="Std-dev of Gaussian noise added to candidate scores (0 disables).",
    )
    p.add_argument(
        "--decision-noise-seed",
        type=int,
        default=None,
        help="Random seed for decision noise (for reproducible stochastic runs).",
    )
    p.add_argument("--pause-s", type=float, default=0.08,
                   help="Seconds to pause between frames when --show")
    p.add_argument("--show", action="store_true",
                   help="Open a live interactive window (uses plt.ion)")
    p.add_argument("--save-mp4", type=str, default="",
                   help="Also save animation as MP4 to this path")
    p.add_argument("--save-gif", type=str, default="",
                   help="Also save animation as GIF to this path")
    p.add_argument("--fps", type=int, default=12, help="FPS for saved animation")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
