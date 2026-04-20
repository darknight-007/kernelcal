#!/usr/bin/env python3
"""Q14-style leakage-injection simulation on RivGraph-derived networks.

Purpose
-------
Create a controlled temporal simulation (no external time series required) on
RivGraph-derived graph topology (e.g., Brahmaputra or mouse-brain vessel mask),
then measure theorem-informed diagnostics:
  - D_t  : aggregate per-mode conservation residual
  - H[h*]: spectral entropy of fixed-point weights
  - Delta': per-frame stability margin (min -H_diag)

The script writes:
  1) experiment_matrix.json (predeclared sweep design),
  2) metrics.csv / metrics.json for each scenario,
  3) temporal frame PNGs and animated GIFs for selected scenarios.

Usage
-----
  cd software-kernelcal-deepgis-integration
  python3 q14_brahmaputra_temporal_sim.py --rivgraph-repo ../rivgraph --dataset brahmaputra
  python3 q14_brahmaputra_temporal_sim.py --rivgraph-repo ../rivgraph --dataset mouse_brain --mouse-mask /path/to/brain_scan_handcleaned.png
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

import rivgraph_kernelcal_bridge as bridge


@dataclass(frozen=True)
class Scenario:
    family: str
    amplitude: float
    frames: int
    seed: int


def _guard_optional_numpy1_extensions() -> None:
    """Prevent noisy optional imports (numexpr/bottleneck) under NumPy 2.

    RivGraph pulls pandas through geopandas. Pandas treats numexpr/bottleneck
    as optional accelerators, but in mixed environments those modules can be
    NumPy-1.x-compiled wheels, causing repeated ABI traceback noise.
    For this simulation we block those optional imports up front so pandas
    falls back to pure-NumPy code paths.
    """

    blocked = {"bottleneck", "numexpr"}
    real_import_module = importlib.import_module

    def _patched_import_module(name: str, package: str | None = None):
        root = name.split(".", 1)[0]
        if root in blocked:
            raise ImportError(
                f"{root} intentionally blocked for NumPy-2 compatibility "
                "in q14_brahmaputra_temporal_sim."
            )
        return real_import_module(name, package)

    importlib.import_module = _patched_import_module


def _normalized_xy(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = positions[:, 0].astype(float)
    c = positions[:, 1].astype(float)
    c0 = (c - c.mean()) / (c.std() + 1e-12)
    r0 = (r - r.mean()) / (r.std() + 1e-12)
    return c0, r0


def _base_field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Smooth analytic scene field defined on node coordinates."""
    g1 = np.sin(1.6 * x) + 0.6 * np.cos(1.2 * y)
    g2 = np.exp(-((x - 0.8) ** 2 + (y + 0.5) ** 2) / 1.7)
    return g1 + 0.8 * g2


def _inject_field(
    family: str,
    amplitude: float,
    t_idx: int,
    t_total: int,
    x: np.ndarray,
    y: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    tau = 0.0 if t_total <= 1 else t_idx / float(t_total - 1)
    phase = 2.0 * np.pi * tau

    if family == "sensor_drift":
        drift = amplitude * tau * (0.9 * x - 0.4 * y)
        return base + drift

    if family == "registration_error":
        # Coordinate-space shift of the analytic scene model.
        # Interpreted as a registration mismatch against a fixed graph.
        dx = amplitude * np.cos(phase)
        dy = amplitude * np.sin(phase)
        return _base_field(x + dx, y + dy)

    if family == "calibration_shift":
        gain = 1.0 + amplitude * np.sin(phase)
        offset = 0.5 * amplitude * np.cos(phase)
        return gain * base + offset

    raise ValueError(f"Unknown family: {family}")


def _laplacian_sparse_from_terrain_graph(tg) -> coo_matrix:
    n = int(tg.positions.shape[0])
    edges = np.asarray(tg.edges, dtype=np.int64)
    w = np.asarray(tg.weights, dtype=float)
    if edges.size == 0:
        return coo_matrix((n, n), dtype=float)

    i = edges[:, 0]
    j = edges[:, 1]
    data = np.concatenate([w, w])
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    D = coo_matrix((deg, (np.arange(n), np.arange(n))), shape=(n, n)).tocsr()
    return (D - A).tocoo()


def _fixed_point_from_weights(
    lambdas: np.ndarray,
    w: np.ndarray,
    *,
    sigma2: float = 1.0,
    mu2: float = 2.0,
    tau_prior: float = 1.0,
    n_iter: int = 120,
    eps: float = 1e-12,
) -> np.ndarray:
    h0 = np.exp(-tau_prior * np.maximum(lambdas, 0.0))
    h = np.maximum(h0.copy(), eps)
    for _ in range(n_iter):
        T = mu2 * w / (2.0 * (sigma2 + h))
        h = h0 * np.exp(-1.0 - T)
        h = np.maximum(h, eps)
    return h


def _diagnostics_from_coeffs(lambdas: np.ndarray, coeffs: np.ndarray) -> dict[str, float]:
    w = np.maximum(coeffs**2, 1e-10)
    h = _fixed_point_from_weights(lambdas, w)

    # Mode-wise conservation residual (Algorithm-1 style, absolute sum).
    d_m = -1.0 / h + (2.0 * w) / (2.0 * (1.0 + h) ** 2)  # mu2=2,sigma2=1
    d_t = float(np.sum(np.abs(d_m)))

    # H[h*]
    h_bar = h / (h.sum() + 1e-12)
    entropy = float(-np.sum(h_bar * np.log(h_bar + 1e-12)))

    # Delta' = min_l(1/h - mu2*w/(2(1+h)^2))
    delta_prime = float(np.min(1.0 / h - (2.0 * w) / (2.0 * (1.0 + h) ** 2)))
    return {"D_t": d_t, "H": entropy, "Delta_prime": delta_prime}


def _save_animation(frames: list[Path], gif_path: Path, fps: int) -> bool:
    try:
        import imageio.v2 as imageio
    except Exception:
        return False
    imgs = [imageio.imread(p) for p in frames]
    duration = 1.0 / max(1, fps)
    imageio.mimsave(gif_path, imgs, duration=duration, loop=0)
    return True


def _plot_sequence(
    out_dir: Path,
    scenario: Scenario,
    network_label: str,
    x: np.ndarray,
    y: np.ndarray,
    series_signal: np.ndarray,
    series_metrics: list[dict[str, float]],
    *,
    fps: int,
) -> Path | None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    dts = np.array([m["D_t"] for m in series_metrics], dtype=float)
    hs = np.array([m["H"] for m in series_metrics], dtype=float)
    t = np.arange(scenario.frames)
    frames: list[Path] = []

    vmin = float(np.min(series_signal))
    vmax = float(np.max(series_signal))
    for i in range(scenario.frames):
        sig = series_signal[i]
        fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

        sc0 = ax[0, 0].scatter(x, y, c=sig, s=4, cmap="viridis", vmin=vmin, vmax=vmax)
        ax[0, 0].set_title(f"Signal on {network_label} graph (t={i})")
        ax[0, 0].set_axis_off()
        fig.colorbar(sc0, ax=ax[0, 0], shrink=0.8)

        delta = sig - series_signal[0]
        sc1 = ax[0, 1].scatter(x, y, c=delta, s=4, cmap="coolwarm")
        ax[0, 1].set_title("Injected perturbation vs t=0")
        ax[0, 1].set_axis_off()
        fig.colorbar(sc1, ax=ax[0, 1], shrink=0.8)

        ax[1, 0].plot(t, dts, color="tab:red", lw=1.8)
        ax[1, 0].scatter([i], [dts[i]], color="black", s=25, zorder=3)
        for thr, lab in ((1.0, "low"), (5.0, "med"), (20.0, "patch")):
            ax[1, 0].axhline(thr, color="gray", ls="--", lw=0.8)
            ax[1, 0].text(0.2, thr + 0.05, lab, fontsize=8, color="gray")
        ax[1, 0].set_title("D_t trajectory")
        ax[1, 0].set_xlabel("Frame")
        ax[1, 0].set_ylabel("D_t")

        ax[1, 1].plot(t, hs, color="tab:blue", lw=1.8)
        ax[1, 1].scatter([i], [hs[i]], color="black", s=25, zorder=3)
        ax[1, 1].set_title("H[h*] trajectory")
        ax[1, 1].set_xlabel("Frame")
        ax[1, 1].set_ylabel("H[h*]")

        fig.suptitle(
            f"Q14 simulation | family={scenario.family} amplitude={scenario.amplitude:g}",
            fontsize=12,
        )
        fp = frame_dir / f"frame_{i:03d}.png"
        fig.savefig(fp, dpi=130)
        plt.close(fig)
        frames.append(fp)

    gif_path = out_dir / "temporal_sequence.gif"
    if _save_animation(frames, gif_path, fps=fps):
        return gif_path
    return None


def _default_matrix(frames: int, seed: int) -> list[Scenario]:
    return [
        Scenario("sensor_drift", 0.04, frames, seed),
        Scenario("sensor_drift", 0.08, frames, seed),
        Scenario("sensor_drift", 0.12, frames, seed),
        Scenario("registration_error", 0.08, frames, seed),   # normalized units
        Scenario("registration_error", 0.16, frames, seed),
        Scenario("registration_error", 0.24, frames, seed),
        Scenario("calibration_shift", 0.05, frames, seed),
        Scenario("calibration_shift", 0.10, frames, seed),
        Scenario("calibration_shift", 0.15, frames, seed),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rivgraph-repo", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("datasets/q14_brahmaputra_sim"))
    p.add_argument(
        "--dataset",
        choices=("brahmaputra", "mouse_brain"),
        default="brahmaputra",
        help="Base graph source for the simulation.",
    )
    p.add_argument(
        "--mouse-mask",
        type=Path,
        default=None,
        help="Path to mouse brain mask image (recommended: brain_scan_handcleaned.png).",
    )
    p.add_argument(
        "--mouse-threshold",
        type=float,
        default=0.5,
        help="Threshold for converting grayscale mouse mask to binary.",
    )
    p.add_argument(
        "--mouse-dark-vessels",
        action="store_true",
        help="Treat darker pixels as vessel foreground (mask = img < threshold).",
    )
    p.add_argument(
        "--mouse-prune-dangling",
        action="store_true",
        help="Delete 1-connected dangling links for mouse network before graphiphy.",
    )
    p.add_argument("--k-modes", type=int, default=64)
    p.add_argument("--frames", type=int, default=28)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument(
        "--visualize-medium-only",
        action="store_true",
        help="Render visuals only for middle amplitude per family (faster).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    _guard_optional_numpy1_extensions()

    # Build selected graph once, then run synthetic temporal injections on it.
    repo = bridge.ensure_rivgraph_import(args.rivgraph_repo)
    if args.dataset == "brahmaputra":
        from rivgraph.classes import river

        mask = repo / "examples" / "data" / "Brahmaputra_Braided_River" / "Brahmaputra_mask.tif"
        results = out_root / "rivgraph_results"
        results.mkdir(parents=True, exist_ok=True)

        rnet = river("Q14_brahma", str(mask), str(results), exit_sides="NS", verbose=False)
        rnet.skeletonize()
        rnet.compute_network()
        rnet.prune_network()
        rnet.compute_link_width_and_length()

        tg, lap_label = bridge.resolve_laplacian_terrain_graph(
            mode="rivgraph",
            iskel=rnet.Iskel,
            links=rnet.links,
            nodes=rnet.nodes,
            imshape=rnet.imshape,
            weight_by_length=False,
            max_skel_nodes=12000,
            skeleton_connectivity=8,
            rivgraph_weight="len_adj",
        )
        network_label = "Brahmaputra"
        print(f"[info] Dataset      : {args.dataset}")
        print(f"[info] Laplacian mode: {lap_label}")
    else:
        # Non-river mask route following RivGraph mouse_brain example:
        # mask -> skeleton -> links/nodes -> graphiphy -> TerrainGraph.
        from rivgraph import im_utils as iu
        from rivgraph import ln_utils as lnu
        from rivgraph import mask_to_graph as m2g
        from scipy.ndimage import distance_transform_edt
        from skimage import io

        default_mouse_mask = (
            repo / "examples" / "data" / "Mouse_Brain" / "brain_scan_handcleaned.png"
        )
        mask_path = (args.mouse_mask or default_mouse_mask).resolve()
        if not mask_path.is_file():
            raise SystemExit(
                f"Mouse mask not found: {mask_path}\n"
                "Pass --mouse-mask /absolute/path/to/brain_scan_handcleaned.png"
            )

        img = io.imread(mask_path, as_gray=True)
        if args.mouse_dark_vessels:
            Imask = np.asarray(img < args.mouse_threshold, dtype=bool)
        else:
            Imask = np.asarray(img > args.mouse_threshold, dtype=bool)
        Imask = iu.largest_blobs(Imask, action="keep")

        Iskel = m2g.skeletonize_mask(Imask)
        links, nodes = m2g.skel_to_graph(Iskel)
        Idt = distance_transform_edt(Imask)
        links = lnu.link_widths_and_lengths(links, Idt)

        if args.mouse_prune_dangling:
            dangling_link_ids = []
            for nconn in nodes["conn"]:
                if len(nconn) == 1:
                    dangling_link_ids.append(nconn[0])
            for dli in dangling_link_ids:
                links, nodes = lnu.delete_link(links, nodes, dli)

        tg = bridge.terrain_graph_from_graphiphy(
            links,
            nodes,
            Imask.shape,
            weight_key="len_adj",
        )
        lap_label = f"rivgraph graphiphy (len_adj, N={tg.elevations.shape[0]})"
        network_label = "Mouse_Brain"
        print(f"[info] Dataset      : {args.dataset}")
        print(f"[info] Mouse mask   : {mask_path}")
        print(f"[info] Laplacian mode: {lap_label}")

    Ls = _laplacian_sparse_from_terrain_graph(tg).tocsr()
    n = Ls.shape[0]
    k = min(max(6, int(args.k_modes)), max(6, n - 2))
    vals, vecs = eigsh(Ls, k=k, which="SM")
    idx = np.argsort(vals)
    lambdas = np.maximum(vals[idx], 0.0)
    Phi = vecs[:, idx]

    x, y = _normalized_xy(tg.positions)
    base = _base_field(x, y)

    scenarios = _default_matrix(args.frames, args.seed)
    matrix_path = out_root / "experiment_matrix.json"
    matrix_payload = [s.__dict__ for s in scenarios]
    matrix_path.write_text(json.dumps(matrix_payload, indent=2), encoding="utf-8")

    rows: list[dict[str, float | str | int]] = []
    for family in ("sensor_drift", "registration_error", "calibration_shift"):
        fam = [s for s in scenarios if s.family == family]
        # Visualize only the middle amplitude unless explicitly disabled.
        mid_amp = fam[len(fam) // 2].amplitude
        for sc in fam:
            seq = []
            metrics = []
            for t in range(sc.frames):
                sig = _inject_field(sc.family, sc.amplitude, t, sc.frames, x, y, base)
                coeffs = Phi.T @ sig
                m = _diagnostics_from_coeffs(lambdas, coeffs)
                seq.append(sig)
                metrics.append(m)

            dts = np.array([m["D_t"] for m in metrics], dtype=float)
            hs = np.array([m["H"] for m in metrics], dtype=float)
            dp = np.array([m["Delta_prime"] for m in metrics], dtype=float)
            row = {
                "family": sc.family,
                "amplitude": sc.amplitude,
                "frames": sc.frames,
                "D_t_mean": float(dts.mean()),
                "D_t_max": float(dts.max()),
                "D_t_patch_rate": float(np.mean(dts > 20.0)),
                "H_mean": float(hs.mean()),
                "Delta_prime_mean": float(dp.mean()),
            }
            rows.append(row)

            do_vis = (not args.visualize_medium_only) or np.isclose(sc.amplitude, mid_amp)
            if do_vis:
                vis_dir = out_root / "visuals" / f"{sc.family}_amp_{sc.amplitude:.3f}"
                gif = _plot_sequence(
                    vis_dir,
                    sc,
                    network_label,
                    x,
                    y,
                    np.asarray(seq),
                    metrics,
                    fps=args.fps,
                )
                if gif is not None:
                    print(f"[visual] {gif}")

    json_path = out_root / "metrics.json"
    csv_path = out_root / "metrics.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] matrix  : {matrix_path}")
    print(f"[done] metrics : {json_path}")
    print(f"[done] csv     : {csv_path}")
    print(f"[done] visuals : {out_root / 'visuals'}")


if __name__ == "__main__":
    main()

