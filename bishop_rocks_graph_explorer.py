#!/usr/bin/env python3
"""Graph-based bishop-rocks explorer with live visualization.

Analogue of ``drone_dem_betti_adaptive_experiment.py`` in the
``software-kernelcal-deepgis-integration`` package, but the underlying graph is
built over **rock centroids** (point data) instead of DEM pixels.

Inputs (expected under ``unprocessed/bishop-root/`` next to this script):
  - ``rocks-coord-list.csv`` — no header, (lon, lat)      ~82k centroids
  - ``rock_traits_full.csv`` — lon, lat, area_m2, major_axis_m,
    minor_axis_m, eccentricity, orientation_deg, elevation_rel   ~14k rocks

The explorer sweeps a circular window of radius ``--window-m`` along a spiral
path covering the traits bounding box. At every step it:

1. Projects all lon/lat to a local equirectangular frame in **metres** so that
   the ``--radius-m`` (default 10 m) neighbour rule is geometrically correct.
2. Collects the trait rocks inside the window.
3. Builds two graphs over those local rocks:
     * ``k-NN``  — each rock connected to its ``--knn`` nearest neighbours.
     * ``radius`` — each rock connected to all rocks within ``--radius-m`` m.
4. Runs scipy's connected-components and the smallest Laplacian eigenvalue
   (Fiedler) on the k-NN graph.
5. Updates a live 2x3 matplotlib figure:
     (0,0) full map, rocks colored by area + window + path
     (0,1) local graph — nodes colored by component, cyan = k-NN edges,
           white = radius-only edges
     (0,2) area histogram over rocks inside the window
     (1,0) rolling topology history (n_nodes, n_components, Fiedler)
     (1,1) rolling trait stats (median area, median eccentricity)
     (1,2) cumulative explored: eccentricity vs area scatter, colored by step

A summary PNG plus (optional) MP4 / GIF are written to ``--out``.

Usage
-----
    python3 bishop_rocks_graph_explorer.py                 # default scan
    python3 bishop_rocks_graph_explorer.py --show          # interactive window
    python3 bishop_rocks_graph_explorer.py --steps 120 --knn 8 --radius-m 10
    python3 bishop_rocks_graph_explorer.py --save-mp4 run.mp4
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation as mpl_animation
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
# Matches sibling scripts (``bishop_kernelcal.py``, ``bishop_trait_analysis.py``):
# rock centroids + per-rock traits CSVs live under ``datasets/bishop_scarp/``.
DEFAULT_DATA_DIR = HERE / "datasets" / "bishop_scarp"
DEFAULT_OUT_DIR = HERE / "bishop_figures" / "rocks_explorer"


# ---------------------------------------------------------------------------
# Geo → local metres projection
# ---------------------------------------------------------------------------

METERS_PER_DEG_LAT = 111_320.0


@dataclass(frozen=True)
class LocalFrame:
    """Equirectangular projection around a central lat (good for <50 km regions)."""

    lon0: float
    lat0: float

    def to_xy(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lat_rad = np.deg2rad(self.lat0)
        x = (lon - self.lon0) * METERS_PER_DEG_LAT * np.cos(lat_rad)
        y = (lat - self.lat0) * METERS_PER_DEG_LAT
        return x.astype(float), y.astype(float)


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


def radius_edges(xy: np.ndarray, r: float) -> set[tuple[int, int]]:
    if len(xy) < 2:
        return set()
    tree = cKDTree(xy)
    neighbors = tree.query_ball_point(xy, r=r)
    return _edge_set_from_neighbors([np.asarray(n, dtype=int) for n in neighbors])


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


# ---------------------------------------------------------------------------
# Per-quadrant Betti + Fiedler scoring  (NE / NW / SW / SE)
# ---------------------------------------------------------------------------


QUADRANTS: tuple[tuple[str, tuple[float, float]], ...] = (
    ("NE", ( 1.0,  1.0)),
    ("NW", (-1.0,  1.0)),
    ("SW", (-1.0, -1.0)),
    ("SE", ( 1.0, -1.0)),
)


@dataclass
class QuadrantScore:
    name: str
    direction: tuple[float, float]      # unit vector
    n_nodes: int
    beta0: int                          # components  (β₀)
    beta1: int                          # cycles  = E - V + β₀
    fiedler: float                      # λ₂ of induced k-NN Laplacian
    score: float


def quadrant_metrics(
    local_xy: np.ndarray,
    knn_A: csr_matrix,
    center: tuple[float, float],
    w_beta1: float,
    w_fiedler: float,
    w_beta0: float = 0.0,
    w_unseen: float = 0.0,
    unseen_mask: np.ndarray | None = None,
    prev_dir: tuple[float, float] | None = None,
    w_momentum: float = 0.0,
) -> list[QuadrantScore]:
    """Split the FoV rocks into NE/NW/SW/SE and score each quadrant's subgraph.

    Score per quadrant::

        info  = w_beta1 * β₁ + w_fiedler * λ₂ * n + w_beta0 * β₀ + w_unseen * n_new
        score = info * (1 + w_momentum * cos(prev_dir, quad_dir))

    The multiplicative momentum factor is in ``[1 - w_momentum, 1 + w_momentum]``
    so re-visiting the previous direction's opposite quadrant is strongly
    discounted (prevents step-by-step oscillation).
    """
    dx = local_xy[:, 0] - center[0]
    dy = local_xy[:, 1] - center[1]
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    out: list[QuadrantScore] = []

    for name, (sx, sy) in QUADRANTS:
        mask = (np.sign(dx) == sx) & (np.sign(dy) == sy)
        n = int(mask.sum())
        unit = (sx * inv_sqrt2, sy * inv_sqrt2)
        if n == 0:
            out.append(QuadrantScore(name, unit, 0, 0, 0, 0.0, 0.0))
            continue

        sub_idx = np.where(mask)[0]
        A_sub = knn_A[sub_idx][:, sub_idx]
        if A_sub.nnz == 0:
            beta0 = n
            beta1 = 0
            fied = 0.0
        else:
            beta0, _ = connected_components(A_sub, directed=False)
            n_edges = int(A_sub.nnz // 2)          # symmetric CSR
            beta1 = max(0, n_edges - n + int(beta0))
            fied = fiedler_value(A_sub) if n >= 3 else 0.0

        n_new = 0
        if unseen_mask is not None and n > 0:
            n_new = int(np.count_nonzero(unseen_mask[sub_idx]))

        info = (
            w_beta1   * float(beta1)
            + w_fiedler * float(fied) * max(n, 1)   # λ₂ scaled by size
            + w_beta0   * float(beta0)
            + w_unseen  * float(n_new)              # raw new-rock count
        )
        if prev_dir is not None and w_momentum != 0.0:
            cos = float(unit[0] * prev_dir[0] + unit[1] * prev_dir[1])
            # Bound momentum factor to [1 - w_momentum, 1 + w_momentum] but never
            # allow the score to flip sign (keeps strong β₁ positive).
            factor = max(1.0 - abs(w_momentum), 1.0 + w_momentum * cos)
            info *= factor

        out.append(QuadrantScore(name, unit, n, int(beta0), int(beta1), float(fied),
                                 float(info)))
    return out


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
# Explorer
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    cx: float
    cy: float
    n_nodes: int
    n_edges_knn: int
    n_edges_rad: int
    n_components: int     # β₀
    beta1: int            # cycles in local k-NN graph
    fiedler: float        # λ₂
    median_area: float
    median_ecc: float
    chosen_quadrant: str
    chosen_score: float


def run(args: argparse.Namespace) -> None:
    coords, traits, _frame = load(args.data_dir)
    print(f"coords CSV : {len(coords):>7,} rocks")
    print(f"traits CSV : {len(traits):>7,} rocks, columns={list(traits.columns)}")

    tx = traits["x_m"].to_numpy()
    ty = traits["y_m"].to_numpy()
    area = traits["area_m2"].to_numpy()
    ecc = traits["eccentricity"].to_numpy()
    cx_bg = coords["x_m"].to_numpy()
    cy_bg = coords["y_m"].to_numpy()

    xmin, xmax = float(tx.min()), float(tx.max())
    ymin, ymax = float(ty.min()), float(ty.max())
    print(f"traits bbox : x {xmin:.1f} .. {xmax:.1f} m   "
          f"y {ymin:.1f} .. {ymax:.1f} m   "
          f"(Δx={xmax-xmin:.0f}m, Δy={ymax-ymin:.0f}m)")

    step_m = args.step_m if args.step_m > 0 else 0.6 * args.window_m
    if args.planner == "spiral":
        spiral_pts = spiral_path(xmin, xmax, ymin, ymax, step=step_m, n_steps=args.steps)
    else:
        spiral_pts = None

    trait_tree = cKDTree(np.c_[tx, ty])

    # Path is built online (the adaptive planner chooses next center from the
    # scored quadrants of the current FoV).
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
    area_clip = np.clip(area, max(area.min(), 0.05), area.max())
    sc_all = ax00.scatter(
        tx, ty,
        c=area_clip, cmap="viridis",
        norm=LogNorm(vmin=max(area.min(), 0.05), vmax=area.max()),
        s=np.clip(area * 1.2, 1.5, 40),
        linewidths=0, alpha=0.85,
    )
    fig.colorbar(sc_all, ax=ax00, shrink=0.8, pad=0.02, label="area (m², log)")
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
    ax00.set_title("Bishop scarp — rocks colored by area\n"
                   "(red: scan window, gold: next move)")
    ax00.grid(alpha=0.3)

    # (0,1) Local graph
    ax01.set_aspect("equal")
    ax01.set_xlabel("x (m, local)")
    ax01.set_ylabel("y (m, local)")
    ax01.set_title(f"Local graph inside window (r={args.window_m} m)")
    ax01.grid(alpha=0.3)
    lc_rad = LineCollection([], colors="white", linewidths=0.8, alpha=0.55)
    lc_knn = LineCollection([], colors="cyan", linewidths=1.0, alpha=0.9)
    ax01.add_collection(lc_rad)
    ax01.add_collection(lc_knn)
    local_nodes = ax01.scatter([], [], s=[], c=[], cmap="tab20", linewidths=0)
    # Quadrant dividers (the two diagonals through window center).
    quad_divider1, = ax01.plot([], [], "w--", lw=0.7, alpha=0.6)
    quad_divider2, = ax01.plot([], [], "w--", lw=0.7, alpha=0.6)
    # Per-quadrant text labels (updated every frame).
    quad_labels: dict[str, plt.Text] = {
        name: ax01.text(0, 0, "", color="white", fontsize=8, ha="center",
                        va="center", alpha=0.9,
                        bbox=dict(facecolor="#000000", alpha=0.35,
                                  edgecolor="none", pad=1.2))
        for name, _ in QUADRANTS
    }
    ax01.legend(
        handles=[
            Line2D([0], [0], color="cyan", lw=1.2,
                   label=f"k-NN edge  (k={args.knn})"),
            Line2D([0], [0], color="white", lw=1.0,
                   label=f"radius edge  (r={args.radius_m:.0f} m)"),
            Line2D([0], [0], marker="o", color="k", lw=0, markersize=6,
                   label="node (color = component)"),
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
    # Step loop  (online: plan next center from current-FoV quadrants)
    # ------------------------------------------------------------------
    records: list[StepRecord] = []
    seen_nodes: set[int] = set()
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

        idx = trait_tree.query_ball_point((cx, cy), r=args.window_m)
        idx = np.asarray(idx, dtype=int)

        local_xy = np.c_[tx[idx], ty[idx]] if len(idx) else np.empty((0, 2))
        n_local = len(idx)

        if n_local >= 2:
            e_knn = knn_edges(local_xy, args.knn)
            e_rad = radius_edges(local_xy, args.radius_m)
        else:
            e_knn = set(); e_rad = set()

        rad_only = e_rad - e_knn
        A_knn = adjacency_from_edges(max(n_local, 1), e_knn)
        n_comp, labels = (1, np.zeros(max(n_local, 1), dtype=int))
        if n_local > 0 and A_knn.nnz > 0:
            n_comp, labels = connected_components(A_knn, directed=False)
        fied_full = fiedler_value(A_knn) if n_local >= 3 else 0.0
        n_edges_full = int(A_knn.nnz // 2)
        beta1_full = max(0, n_edges_full - n_local + int(n_comp))

        # ---- per-quadrant scoring -----------------------------------------
        unseen_mask = None
        if args.w_unseen > 0.0 and n_local:
            unseen_mask = np.fromiter(
                (int(i) not in seen_nodes for i in idx), dtype=bool, count=n_local,
            )

        prev_dir_vec: tuple[float, float] | None = None
        if len(records) >= 1:
            prev_rec = records[-1]
            vx = cx - prev_rec.cx
            vy = cy - prev_rec.cy
            norm = float(np.hypot(vx, vy))
            if norm > 1e-6:
                prev_dir_vec = (vx / norm, vy / norm)

        if n_local >= 4:
            quads = quadrant_metrics(
                local_xy, A_knn, (cx, cy),
                w_beta1=args.w_beta1, w_fiedler=args.w_fiedler,
                w_beta0=args.w_beta0, w_unseen=args.w_unseen,
                unseen_mask=unseen_mask,
                prev_dir=prev_dir_vec, w_momentum=args.w_momentum,
            )
        else:
            quads = [QuadrantScore(name, (sx/np.sqrt(2), sy/np.sqrt(2)),
                                   0, 0, 0, 0.0, 0.0)
                     for name, (sx, sy) in QUADRANTS]

        # Pick best (random tie-break to avoid deterministic stalls).
        best_score = max(q.score for q in quads)
        candidates = [q for q in quads if q.score == best_score]
        best = candidates[0] if len(candidates) == 1 else \
               candidates[step % len(candidates)]

        # --- figure out next center (for step + 1) -------------------------
        # Direction bias: if all scores are ~0 (empty FoV), bail toward bbox
        # centre so the explorer doesn't sit on an empty spot forever.
        if best_score <= 0.0:
            tgt = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)])
            vec = tgt - np.array([cx, cy])
            nrm = float(np.linalg.norm(vec))
            dirv = (vec / nrm) if nrm > 1e-6 else np.array([1.0, 0.0])
        else:
            dirv = np.array(best.direction)

        new_cx = float(np.clip(cx + step_m * dirv[0], xmin, xmax))
        new_cy = float(np.clip(cy + step_m * dirv[1], ymin, ymax))
        next_center[0] = (new_cx, new_cy)

        med_area = float(np.median(area[idx])) if n_local else float("nan")
        med_ecc = float(np.median(ecc[idx])) if n_local else float("nan")

        rec = StepRecord(
            step=step, cx=cx, cy=cy,
            n_nodes=n_local,
            n_edges_knn=len(e_knn),
            n_edges_rad=len(e_rad),
            n_components=int(n_comp),
            beta1=int(beta1_full),
            fiedler=float(fied_full),
            median_area=med_area,
            median_ecc=med_ecc,
            chosen_quadrant=best.name,
            chosen_score=float(best.score),
        )
        if records and records[-1].step == step:
            records[-1] = rec          # FuncAnimation can call frame 0 twice
        else:
            records.append(rec)
            if spiral_pts is None and len(path_xy) < step + 1:
                path_xy.append((cx, cy))
        seen_nodes.update(int(i) for i in idx)

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
        rad_segs = [np.array([local_xy[a], local_xy[b]]) for a, b in rad_only]
        lc_knn.set_segments(knn_segs)
        lc_rad.set_segments(rad_segs)

        node_sizes = np.clip(area[idx] * 2.5, 5, 80) if n_local else np.array([])
        local_nodes.set_offsets(local_xy if n_local else np.empty((0, 2)))
        local_nodes.set_sizes(node_sizes)
        local_nodes.set_array(labels if n_local else np.array([]))

        pad = 0.15 * args.window_m
        lim = args.window_m + pad
        ax01.set_xlim(cx - lim, cx + lim)
        ax01.set_ylim(cy - lim, cy + lim)

        # Quadrant dividers (two diagonals through window center).
        d = args.window_m
        quad_divider1.set_data([cx - d, cx + d], [cy - d, cy + d])
        quad_divider2.set_data([cx - d, cx + d], [cy + d, cy - d])

        # Per-quadrant annotations.
        off = 0.55 * args.window_m
        for q in quads:
            lx = cx + q.direction[0] * off
            ly = cy + q.direction[1] * off
            is_best = (q.name == best.name)
            color = "gold" if is_best else "white"
            txt = (f"{q.name}  n={q.n_nodes}\n"
                   f"β₀={q.beta0}  β₁={q.beta1}\n"
                   f"λ₂={q.fiedler:.3g}\n"
                   f"score={q.score:.2f}")
            quad_labels[q.name].set_position((lx, ly))
            quad_labels[q.name].set_text(txt)
            quad_labels[q.name].set_color(color)
            quad_labels[q.name].set_fontweight("bold" if is_best else "normal")

        # Area histogram panel.
        if n_local:
            hist, edges = np.histogram(area[idx], bins=hist_bins)
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

        # Cumulative explored scatter.
        if seen_nodes:
            sn = np.fromiter(seen_nodes, dtype=int)
            a_seen = area[sn]
            e_seen = ecc[sn]
            cum_scat.set_offsets(np.c_[a_seen, e_seen])
            cum_scat.set_array(np.arange(len(sn)))
            cbar_cum.mappable.set_clim(0, max(1, len(sn)))
            ax12.set_xlim(max(a_seen.min(), 0.01) * 0.8, a_seen.max() * 1.2)
            ax12.set_ylim(-0.02, max(1.02, float(np.nanmax(e_seen)) + 0.02))

        fig.suptitle(
            f"Bishop rocks explorer — step {step+1}/{args.steps}   "
            f"window=({cx:.1f}, {cy:.1f}) m   "
            f"n_nodes={n_local}   β₀={n_comp}   β₁={beta1_full}   "
            f"λ₂={fied_full:.3g}   →{best.name}  (score={best.score:.2f})",
            fontsize=11,
        )

        return (
            path_line, window, dir_arrow, dir_head,
            lc_knn, lc_rad, local_nodes,
            quad_divider1, quad_divider2,
            *quad_labels.values(),
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
            n_edges_rad=[r.n_edges_rad for r in records],
            beta0_components=[r.n_components for r in records],
            beta1_cycles=[r.beta1 for r in records],
            fiedler=[r.fiedler for r in records],
            median_area_m2=[r.median_area for r in records],
            median_eccentricity=[r.median_ecc for r in records],
            chosen_quadrant=[r.chosen_quadrant for r in records],
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
                   help="Spiral step size in metres (default: 0.6 * window-m)")
    p.add_argument("--knn", type=int, default=6,
                   help="k for k-NN graph (default: 6)")
    p.add_argument("--radius-m", type=float, default=10.0,
                   help="Radius in metres for radius graph (default: 10)")
    p.add_argument("--planner", choices=["quadrant", "spiral"], default="quadrant",
                   help="Next-step planner. 'quadrant' (default) scores the 4 FoV "
                        "quadrants by β₀/β₁/λ₂ and moves toward the best one; "
                        "'spiral' walks a fixed outward spiral.")
    p.add_argument("--seed-x", type=float, default=None,
                   help="Starting x in metres (default: bbox centre)")
    p.add_argument("--seed-y", type=float, default=None,
                   help="Starting y in metres (default: bbox centre)")
    p.add_argument("--w-beta1", type=float, default=1.0,
                   help="Weight on β₁ (cycles) in quadrant score")
    p.add_argument("--w-fiedler", type=float, default=20.0,
                   help="Weight on Fiedler λ₂ (scaled by quadrant size)")
    p.add_argument("--w-beta0", type=float, default=0.0,
                   help="Weight on β₀ (components)")
    p.add_argument("--w-unseen", type=float, default=5.0,
                   help="Weight on raw unseen-rock count per quadrant "
                        "(exploration bonus; dominant driver after the first pass)")
    p.add_argument("--w-momentum", type=float, default=0.45,
                   help="Momentum bonus as fraction of info score "
                        "(cos(prev_dir, quad_dir) * w_momentum multiplies the "
                        "quadrant score; keep < 1; prevents oscillation)")
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
