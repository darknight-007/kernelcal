#!/usr/bin/env python3
"""
Synthetic planetary-style terrain for topology / spectral compression
experiments (replaces dependence on a specific external OBJ).

Base mesh: triangulated regular-grid height field (DEM-style): samples z_ij
on a rectangular planar grid; each cell is two triangles. This is simple,
predictable, and standard in terrain modeling.

Procedural features (applied in order on the height grid):
  1. Large-scale macro relief — fractional Brownian motion (fBm) built from
     octave-wise upsampled Gaussian noise (Hurst exponent H, multi-scale).
  2. Crater — subtract a smooth bowl and add a raised annulus (loop-like
     topographic rim); cratered synthetic terrain is common in planetary tests.
  3. Incised channels — dendritic / braided drainage carved along polylines
     (Gaussian trench profile).
  4. Optional loop channel — closed or rejoining polyline so an *extracted*
     channel centreline graph can carry a nontrivial cycle (β₁ of that graph);
     the surface mesh remains a topological disk (β₁(mesh)=0).
  5. Triangulation — two triangles per quad, alternating diagonal.

Ground-truth Betti numbers for the disk chart are fixed by construction; a
coarse 12×12 grid instance is checked with kernelcal.geo3d.hodge.betti_numbers.

Usage
-----
  cd software-kernelcal-deepgis-integration
  python3 synthetic_planetary_mesh_experiment.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scipy import sparse as sp
from scipy.sparse.linalg import eigsh

from kernelcal.geo3d.bounds import estimate_compression_bounds
from kernelcal.geo3d.hodge import betti_numbers
from kernelcal.geo3d.mesh import mesh_combinatorial_laplacian


def fbm_height_field(
    ny: int,
    nx: int,
    *,
    hurst: float = 0.72,
    octaves: int = 7,
    amplitude: float = 0.11,
    seed: int = 0,
) -> np.ndarray:
    """2D fractional-Brownian-style relief: summed octave noises, amplitude ~ 2^{-o H}."""
    rng = np.random.default_rng(seed)
    z = np.zeros((ny, nx), dtype=float)
    for o in range(octaves):
        h = max(2, ny // (2 ** (o + 2)) + 1)
        w = max(2, nx // (2 ** (o + 2)) + 1)
        layer = rng.standard_normal((h, w))
        zh = (ny - 1) / max(h - 1, 1)
        zw = (nx - 1) / max(w - 1, 1)
        up = zoom(layer, (zh, zw), order=1)
        up = up[:ny, :nx]
        if up.shape[0] < ny or up.shape[1] < nx:
            tmp = np.zeros((ny, nx))
            tmp[: up.shape[0], : up.shape[1]] = up
            up = tmp
        gain = 2.0 ** (-o * hurst)
        z += amplitude * gain * up
    z -= float(np.mean(z))
    return z


def add_crater_bowl_and_annular_rim(
    X: np.ndarray,
    Y: np.ndarray,
    z: np.ndarray,
    *,
    xc: float = 0.08,
    yc: float = 0.06,
    bowl_scale: float = 0.38,
    bowl_depth: float = 0.26,
    rim_radius: float = 0.36,
    rim_width: float = 0.07,
    rim_height: float = 0.11,
) -> np.ndarray:
    """Subtract a Gaussian bowl; add a raised annulus (rim) at mid radius."""
    R2 = (X - xc) ** 2 + (Y - yc) ** 2
    z = z - bowl_depth * np.exp(-R2 / bowl_scale)
    r = np.sqrt(np.maximum(R2, 1e-14))
    rim = rim_height * np.exp(-((r - rim_radius) ** 2) / (2.0 * rim_width**2))
    return z + rim


def _min_dist_sq_to_segments(
    X: np.ndarray, Y: np.ndarray, px: np.ndarray, py: np.ndarray
) -> np.ndarray:
    """Minimum squared distance from each point to polyline segments (px,py)."""
    dmin = np.full_like(X, np.inf, dtype=float)
    for i in range(len(px) - 1):
        x0, y0 = px[i], py[i]
        x1, y1 = px[i + 1], py[i + 1]
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy + 1e-18
        t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0.0, 1.0)
        px_ = x0 + t * dx
        py_ = y0 + t * dy
        d2 = (X - px_) ** 2 + (Y - py_) ** 2
        dmin = np.minimum(dmin, d2)
    return dmin


def carve_trench_from_distance(
    z: np.ndarray, dist_sq: np.ndarray, *, depth: float, width_sq: float
) -> np.ndarray:
    """Subtract Gaussian trench profile vs distance^2 to centreline."""
    return z - depth * np.exp(-dist_sq / width_sq)


def polyline_dendritic_drainage() -> tuple[np.ndarray, np.ndarray]:
    """Main stem + bifurcations (planar coords in [-1,1]^2)."""
    # Main: upland to basin
    t = np.linspace(0, 1, 80)
    mx = 0.15 + 0.55 * t
    my = 0.55 - 0.95 * t
    # Branch A (left)
    u = np.linspace(0, 1, 40)
    ax = 0.35 + 0.25 * u
    ay = 0.15 - 0.45 * u
    # Branch B (right)
    bx = 0.42 + 0.3 * u
    by = 0.08 - 0.5 * u
    px = np.concatenate([mx, np.full(1, np.nan), ax, np.full(1, np.nan), bx])
    py = np.concatenate([my, np.full(1, np.nan), ay, np.full(1, np.nan), by])
    return px, py


def polyline_braided_threads() -> list[tuple[np.ndarray, np.ndarray]]:
    """Three parallel-ish threads (braided) for secondary incision."""
    threads: list[tuple[np.ndarray, np.ndarray]] = []
    for off in (-0.12, 0.0, 0.12):
        s = np.linspace(-0.85, 0.4, 50)
        threads.append(
            (
                0.2 * np.sin(2.8 * s + off) + 0.15 * s,
                s + 0.15 * off,
            )
        )
    return threads


def polyline_closed_loop_channel(
    *,
    xc: float = -0.42,
    yc: float = -0.35,
    rx: float = 0.22,
    ry: float = 0.16,
    n: int = 72,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed ellipse — centreline graph has one cycle (β₁=1 for that abstract graph)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return xc + rx * np.cos(th), yc + ry * np.sin(th)


def carve_polylines_sequential(
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    *,
    depth: float,
    width: float,
) -> np.ndarray:
    """Skip NaN breaks between polylines."""
    starts = np.where(np.isnan(px))[0]
    idx0 = 0
    for st in list(starts) + [len(px)]:
        segx = px[idx0:st]
        segy = py[idx0:st]
        idx0 = st + 1
        if len(segx) < 2:
            continue
        d2 = _min_dist_sq_to_segments(X, Y, segx, segy)
        z = carve_trench_from_distance(z, d2, depth=depth, width_sq=width**2)
    return z


def build_heightfield_surface(
    nx: int,
    ny: int,
    *,
    seed: int = 0,
    include_loop_channel: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices (V,3), faces (F,3)): DEM-style grid, two triangles per cell."""
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    # (1) Macro relief: fBm-style
    z = fbm_height_field(ny, nx, hurst=0.72, octaves=7, amplitude=0.11, seed=seed)

    # (2) Crater: bowl + annular rim
    z = add_crater_bowl_and_annular_rim(X, Y, z)

    # Mild regional tilt (regional drainage bias)
    z += 0.06 * X - 0.04 * Y

    # (3) Dendritic / braided incised channels
    dx, dy = polyline_dendritic_drainage()
    z = carve_polylines_sequential(z, X, Y, dx, dy, depth=0.055, width=0.045)
    for sx, sy in polyline_braided_threads():
        d2 = _min_dist_sq_to_segments(X, Y, np.asarray(sx), np.asarray(sy))
        z = carve_trench_from_distance(z, d2, depth=0.028, width_sq=0.022**2)

    # (4) Optional closed loop channel (abstract channel graph has a cycle)
    if include_loop_channel:
        lx, ly = polyline_closed_loop_channel()
        d2 = _min_dist_sq_to_segments(X, Y, lx, ly)
        z = carve_trench_from_distance(z, d2, depth=0.04, width_sq=0.028**2)

    verts: list[list[float]] = []
    for j in range(ny):
        for i in range(nx):
            verts.append([float(X[j, i]), float(Y[j, i]), float(z[j, i])])
    V = np.asarray(verts, dtype=float)
    vid = lambda i, j: j * nx + i
    faces: list[list[int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    F = np.asarray(faces, dtype=int)
    return V, F


def build_two_component_patch(
    n_patch: int,
    *,
    gap: float = 2.5,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint height-field patches (islands) in one vertex list — β₀ = 2 by construction."""
    V1, F1 = build_heightfield_surface(n_patch, n_patch, seed=seed)
    V2, F2 = build_heightfield_surface(n_patch, n_patch, seed=seed + 17)
    n1 = V1.shape[0]
    V2 = V2.copy()
    V2[:, 0] += gap
    V = np.vstack([V1, V2])
    F2o = F2 + n1
    F = np.vstack([F1, F2o])
    return V, F


def _smallest_eigenpairs(L_dense: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ascending (λ[:k], Φ[:, :k]) using sparse Lanczos when large."""
    n = L_dense.shape[0]
    k = min(k, n - 1)
    if k < 1:
        raise ValueError("k must be positive")
    Ls = sp.csr_matrix(L_dense)
    # Smallest magnitude: graph Laplacian PSD, first mode ~0
    w, v = eigsh(Ls, k=k, which="SM")
    idx = np.argsort(w)
    return w[idx], v[:, idx]


def run_experiment() -> dict:
    # 65×65 grid → 4,225 vertices — vertex Laplacian only (fast);
    # full Hodge Betti via dense L₁ is omitted at this resolution (see small check below).
    nx = ny = 65
    max_k = 256
    t0 = time.perf_counter()
    V, F = build_heightfield_surface(nx, ny, seed=42)
    t_mesh = time.perf_counter() - t0

    t1 = time.perf_counter()
    # Disk-shaped chart: β₀=1, β₁=β₂=0 (triangulated topological disk)
    b0, b1, b2 = 1, 0, 0
    betti_note = "by construction (single connected height-field chart)"
    t_betti = time.perf_counter() - t1

    t2 = time.perf_counter()
    L = mesh_combinatorial_laplacian(V.shape[0], F)
    lam, Phi = _smallest_eigenpairs(L, k=max_k + 8)
    t_eig = time.perf_counter() - t2

    k_list = [64, 128, 256]
    rows = []
    t_rows = []
    for k in k_list:
        tk = time.perf_counter()
        if k > Phi.shape[1]:
            continue
        b = estimate_compression_bounds(
            V,
            F,
            Phi,
            lam,
            n_modes=k,
            coeff_only=True,
            betti=(b0, b1, b2),
        )
        t_rows.append(time.perf_counter() - tk)
        rows.append(
            {
                "k": k,
                "relative_distortion": float(b.relative_distortion),
                "topology_preserved": b.topology_preserved,
                "compression_ratio": float(b.compression_ratio),
            }
        )

    # Two-component ablation: k=1 should fail topology preservation if β₀=2
    t3 = time.perf_counter()
    V2, F2 = build_two_component_patch(32, gap=2.8, seed=3)
    b0t, b1t, b2t = 2, 0, 0
    L2 = mesh_combinatorial_laplacian(V2.shape[0], F2)
    lam2, Phi2 = _smallest_eigenpairs(L2, k=32)
    b_k1 = estimate_compression_bounds(
        V2, F2, Phi2, lam2, n_modes=1, coeff_only=True, betti=(b0t, b1t, b2t)
    )
    b_k2 = estimate_compression_bounds(
        V2, F2, Phi2, lam2, n_modes=2, coeff_only=True, betti=(b0t, b1t, b2t)
    )
    t_two = time.perf_counter() - t3

    # Cross-check Betti on a tiny mesh (fast dense Hodge)
    Vc, Fc = build_heightfield_surface(12, 12, seed=99)
    b_chk = betti_numbers(Vc.shape[0], Fc)

    return {
        "single_chart": {
            "grid": [nx, ny],
            "n_vertices": int(V.shape[0]),
            "n_faces": int(F.shape[0]),
            "procedural_features": {
                "macro_relief": "fbm (octave fBm via upsampled Gaussian noise, H=0.72)",
                "crater": "Gaussian bowl subtract + annular rim add",
                "channels_dendritic_braided": "polylines: main stem + branches; three braided threads",
                "loop_channel": "closed ellipse trench (centreline graph has one cycle; mesh remains a disk)",
            },
            "betti": [b0, b1, b2],
            "betti_provenance": betti_note,
            "betti_sanity_check_12x12": list(b_chk),
            "timing_s": {
                "mesh": t_mesh,
                "betti": t_betti,
                "eigenpairs": t_eig,
                "per_k": t_rows,
            },
            "compression": rows,
        },
        "two_component": {
            "n_vertices": int(V2.shape[0]),
            "n_faces": int(F2.shape[0]),
            "betti": [b0t, b1t, b2t],
            "k1_topology_preserved": b_k1.topology_preserved,
            "k1_rel_distortion": float(b_k1.relative_distortion),
            "k2_topology_preserved": b_k2.topology_preserved,
            "k2_rel_distortion": float(b_k2.relative_distortion),
            "timing_s": t_two,
        },
    }


def main() -> None:
    out = run_experiment()
    print(json.dumps(out, indent=2))
    out_path = ROOT / "datasets" / "synthetic_mesh_compression_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
