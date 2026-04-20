#!/usr/bin/env python3
"""Encoder → bytes → decoder demo for graph spectral telemetry + kernelcal diagnostics.

Lossless: npz payload restores ``TerrainGraph`` and Laplacian exactly.
Spectral-only reconstruction ``L_hat = U_k Λ_k U_k^T`` is shown as lossy
with Frobenius error vs true ``L``.

Run from this directory::

  python3 run_graph_codec_demo.py
  python3 run_graph_codec_demo.py --nrows 64 --ncols 64 --k-modes 24
  python3 run_graph_codec_demo.py --demo brahmaputra --rivgraph-repo /path/to/rivgraph
  python3 run_graph_codec_demo.py --demo meandering --rivgraph-repo /path/to/rivgraph
  python3 run_graph_codec_demo.py --demo colville --rivgraph-repo /path/to/rivgraph --plot-dir ./plots

  ``Mouse_Brain`` in RivGraph is a PNG notebook workflow, not a bundled GeoTIFF river mask; run
  ``examples/mouse_brain_example.py`` to build a mask, export a binary GeoTIFF, then you could
  point ``rivgraph_kernelcal_bridge.py --mask`` at that file (same codec path as Brahmaputra).

With ``--plot-dir``, writes Fiedler comparison PNGs and a **sender vs reconstructed**
geometry figure (nodes + edges on the mask). The reconstructed panel annotates
**full telemetry** vs a **compressed sparse-only npz** (edges + weights + positions
+ elevations, no ``λ,U``), **sparse npz vs naive dense ``L``** (``n²×8``), and
**full/sparse** spectral overhead — honest ratios, same ``np.savez_compressed``
family as the codec.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal, cast

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from kernelcal.terrain.dem import TerrainGraph, dem_to_graph, synthetic_channel_dem, terrain_graph_laplacian
from kernelcal.terrain.graph_codec import (
    decode_graph_packet,
    encode_graph_packet,
    kernelcal_graph_diagnostics,
    lossless_sparse_graph_npz_bytes,
    low_rank_laplacian_from_packet,
    packet_from_npz_bytes,
    packet_to_npz_bytes,
    packet_from_stream_bytes,
    packet_to_stream_bytes,
    write_packet_stream_file,
    reconstruction_frobenius,
    topology_guard_k,
)


def _fiedler_vector(L: np.ndarray) -> np.ndarray:
    """First nontrivial Laplacian eigenvector (algebraic connectivity / Fiedler mode)."""
    L = np.asarray(L, dtype=float)
    w, U = np.linalg.eigh(L)
    if U.shape[1] < 2:
        return np.zeros(L.shape[0], dtype=float)
    return U[:, 1].copy()


def _align_sign(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return ``b`` with global sign chosen so ``dot(a, b) >= 0``."""
    b = np.asarray(b, dtype=float).copy()
    if float(np.dot(a, b)) < 0.0:
        b *= -1.0
    return b


def write_codec_decode_plots(
    plot_dir: Path,
    tg: TerrainGraph,
    L_true: np.ndarray,
    L_rx: np.ndarray,
    *,
    backdrop: np.ndarray | None,
    suptitle: str,
    filename_stem: str = "codec_fiedler",
) -> None:
    """Save PNGs comparing sender Laplacian vs lossless decoded Laplacian on node layout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = tg.positions[:, 0]
    cols = tg.positions[:, 1]
    u0 = _fiedler_vector(L_true)
    u1 = _fiedler_vector(L_rx)
    u1a = _align_sign(u0, u1)
    diff = np.abs(u0 - u1a)

    def _scatter(ax, z: np.ndarray, title: str, cbar_label: str) -> None:
        if backdrop is not None:
            ax.imshow(np.asarray(backdrop), cmap="Greys", origin="upper", alpha=0.35)
        sc = ax.scatter(
            cols,
            rows,
            c=z,
            cmap="coolwarm",
            s=8.0,
            linewidths=0,
            alpha=0.9,
        )
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, shrink=0.72, fraction=0.046, label=cbar_label)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    _scatter(axes[0], u0, "Sender $u_1$ (Fiedler)", r"$u_1$")
    _scatter(axes[1], u1a, "Decoded $u_1$ (lossless)", r"$u_1$")
    _scatter(axes[2], diff, r"$\left|u_1^{\mathrm{send}} - u_1^{\mathrm{dec}}\right|$", "|Δ|")
    fig.suptitle(suptitle, fontsize=11)
    out = plot_dir / f"{filename_stem}_sender_decoded_diff.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  Codec plot written: {out.resolve()}")


def _terrain_graph_edge_segments(tg: TerrainGraph) -> list[np.ndarray]:
    """Line segments in matplotlib (x=col, y=row) space, ``origin='upper'``."""
    pos = np.asarray(tg.positions, dtype=float)
    edges = np.asarray(tg.edges, dtype=np.int64).reshape(-1, 2)
    segs: list[np.ndarray] = []
    for i, j in edges:
        ia, jb = int(i), int(j)
        r0, c0 = pos[ia, 0], pos[ia, 1]
        r1, c1 = pos[jb, 0], pos[jb, 1]
        segs.append(np.array([[c0, r0], [c1, r1]], dtype=float))
    return segs


def _edge_weight_map(tg: TerrainGraph) -> dict[tuple[int, int], float]:
    """Undirected edge -> weight map for fast sender/decoder diffs."""
    e = np.asarray(tg.edges, dtype=np.int64).reshape(-1, 2)
    w = np.asarray(tg.weights, dtype=float).reshape(-1)
    out: dict[tuple[int, int], float] = {}
    for (i, j), ww in zip(e, w):
        a, b = (int(i), int(j)) if int(i) <= int(j) else (int(j), int(i))
        out[(a, b)] = float(ww)
    return out


def _segments_from_edge_keys(tg: TerrainGraph, keys: set[tuple[int, int]]) -> list[np.ndarray]:
    """Line segments for a selected set of undirected edge keys."""
    pos = np.asarray(tg.positions, dtype=float)
    segs: list[np.ndarray] = []
    for i, j in keys:
        r0, c0 = pos[int(i), 0], pos[int(i), 1]
        r1, c1 = pos[int(j), 0], pos[int(j), 1]
        segs.append(np.array([[c0, r0], [c1, r1]], dtype=float))
    return segs


def write_reconstructed_graph_geometry_plot(
    plot_dir: Path,
    tg_sender: TerrainGraph,
    tg_decoded: TerrainGraph,
    *,
    backdrop: np.ndarray | None,
    suptitle: str,
    stem: str = "codec_graph",
    payload_bytes: int | None = None,
    n_nodes: int | None = None,
) -> None:
    """Save a 1×2 figure: sender vs losslessly decoded graph (nodes + chord edges)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    def _draw(ax, tg: TerrainGraph, title: str) -> None:
        if backdrop is not None:
            ax.imshow(np.asarray(backdrop), cmap="Greys", origin="upper", alpha=0.42)
        segs = _terrain_graph_edge_segments(tg)
        if segs:
            ax.add_collection(
                LineCollection(
                    segs,
                    colors="steelblue",
                    linewidths=0.65,
                    alpha=0.85,
                )
            )
        rows = tg.positions[:, 0]
        cols = tg.positions[:, 1]
        ax.scatter(
            cols,
            rows,
            s=6.0,
            c="crimson",
            alpha=0.75,
            linewidths=0,
            zorder=3,
        )
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title)

    em_s = _edge_weight_map(tg_sender)
    em_d = _edge_weight_map(tg_decoded)
    ks = set(em_s.keys())
    kd = set(em_d.keys())
    k_common = ks & kd
    k_sender_only = ks - kd
    k_decoded_only = kd - ks

    n_s = int(np.asarray(tg_sender.positions).shape[0])
    n_d = int(np.asarray(tg_decoded.positions).shape[0])
    pos_delta = np.nan
    if n_s == n_d and n_s > 0:
        ds = np.asarray(tg_sender.positions, dtype=float) - np.asarray(tg_decoded.positions, dtype=float)
        pos_delta = float(np.max(np.linalg.norm(ds, axis=1)))

    w_delta = 0.0
    if len(k_common) > 0:
        w_delta = max(abs(em_s[k] - em_d[k]) for k in k_common)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), constrained_layout=True)
    _draw(axes[0], tg_sender, "Sender graph (nodes + edges)")
    _draw(axes[1], tg_decoded, "Reconstructed graph (lossless decode)")
    if backdrop is not None:
        axes[2].imshow(np.asarray(backdrop), cmap="Greys", origin="upper", alpha=0.42)
    seg_common = _segments_from_edge_keys(tg_sender, k_common)
    seg_sender_only = _segments_from_edge_keys(tg_sender, k_sender_only)
    seg_decoded_only = _segments_from_edge_keys(tg_decoded, k_decoded_only)
    if seg_common:
        axes[2].add_collection(
            LineCollection(seg_common, colors="0.55", linewidths=0.65, alpha=0.55)
        )
    if seg_sender_only:
        axes[2].add_collection(
            LineCollection(seg_sender_only, colors="crimson", linewidths=1.3, alpha=0.95)
        )
    if seg_decoded_only:
        axes[2].add_collection(
            LineCollection(seg_decoded_only, colors="limegreen", linewidths=1.3, alpha=0.95)
        )
    axes[2].set_aspect("equal")
    axes[2].set_axis_off()
    axes[2].set_title("Edge audit overlay\ngray=shared, red=sender-only, green=decoded-only")

    if payload_bytes is not None and n_nodes is not None and n_nodes > 0 and payload_bytes > 0:
        dense_L_bytes = int(n_nodes) * int(n_nodes) * 8
        b_sparse = lossless_sparse_graph_npz_bytes(tg_sender, compress=True, store_cell_index=False)
        kb_full = payload_bytes / 1024.0
        kb_sparse = b_sparse / 1024.0
        mb_dense = dense_L_bytes / (1024.0 * 1024.0)
        sparse_vs_dense = dense_L_bytes / float(b_sparse) if b_sparse > 0 else float("nan")
        spectral_over = payload_bytes / float(b_sparse) if b_sparse > 0 else float("nan")
        lines = (
            f"Full telemetry: {kb_full:.1f} KiB (sparse+λ,U)",
            f"Sparse-only npz: {kb_sparse:.1f} KiB (honest baseline)",
            f"Sparse npz vs dense L: {sparse_vs_dense:.1f}× smaller",
            f"Full / sparse-only: {spectral_over:.2f}× (spectral overhead)",
            f"(dense L ref: {mb_dense:.2f} MiB = n²×8)",
        )
        axes[1].text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=axes[1].transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.4", alpha=0.92),
            zorder=10,
        )

    audit_lines = (
        f"nodes sender/decoded: {n_s}/{n_d}",
        f"edges sender/decoded: {len(ks)}/{len(kd)}",
        f"edge diff sender-only: {len(k_sender_only)}",
        f"edge diff decoded-only: {len(k_decoded_only)}",
        f"max |Δweight| (shared): {w_delta:.3e}",
        f"max ||Δpos||: {pos_delta:.3e}" if np.isfinite(pos_delta) else "max ||Δpos||: n/a",
    )
    axes[2].text(
        0.02,
        0.02,
        "\n".join(audit_lines),
        transform=axes[2].transAxes,
        va="bottom",
        ha="left",
        fontsize=8.0,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.4", alpha=0.92),
        zorder=10,
    )

    fig.suptitle(suptitle, fontsize=11)
    out = plot_dir / f"{stem}_sender_vs_reconstructed.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  Codec plot written: {out.resolve()}")


def _print_diag(label: str, d: dict[str, float]) -> None:
    print(
        f"  {label}: spectral_entropy={d['spectral_entropy']:.6f}  "
        f"fixed_point_residual={d['fixed_point_residual']:.3e}"
    )


def chain_terrain_graph(n: int = 24) -> TerrainGraph:
    """Path graph on n nodes (β₁=0): illustrates topology guard with small k."""
    n = int(n)
    edges = np.column_stack((np.arange(0, n - 1, dtype=np.int32), np.arange(1, n, dtype=np.int32)))
    weights = np.ones(n - 1, dtype=float)
    positions = np.column_stack((np.arange(n, dtype=float), np.zeros(n, dtype=float)))
    elevations = np.zeros(n, dtype=float)
    h, w = 1, n
    cell_index = np.full((h, w), -1, dtype=np.int32)
    cell_index[0, :] = np.arange(n, dtype=np.int32)
    return TerrainGraph(
        positions=positions,
        elevations=elevations,
        edges=edges,
        weights=weights,
        shape=(h, w),
        cell_index=cell_index,
    )


def _mask_backdrop_from_tg(tg: TerrainGraph) -> np.ndarray:
    h, w = int(tg.shape[0]), int(tg.shape[1])
    m = np.zeros((h, w), dtype=float)
    ci = np.asarray(tg.cell_index, dtype=np.int32)
    if ci.shape == (h, w):
        m[ci >= 0] = 1.0
    return m


def run_once(
    *,
    nrows: int,
    ncols: int,
    k_modes: int,
    seed: int,
    plot_dir: Path | None,
) -> None:
    rng = np.random.default_rng(seed)
    dem = synthetic_channel_dem(nrows, ncols, n_tributaries=3)
    dem = dem + 0.05 * rng.standard_normal(dem.shape)
    tg = dem_to_graph(dem, connectivity=8, weight="uniform")
    L_true = terrain_graph_laplacian(tg)
    n = L_true.shape[0]

    pkt = encode_graph_packet(tg, k_modes=min(k_modes, n))
    ok, msg = topology_guard_k(pkt.k_transmitted(), pkt.beta0, pkt.beta1)
    print(f"Graph: n_nodes={n}  |E|={pkt.edges.shape[0]}  beta0={pkt.beta0}  beta1={pkt.beta1}")
    print(f"Topology guard: {msg}  ->  {'OK' if ok else 'WARN'}")

    raw = packet_to_npz_bytes(pkt, compress=True)
    print(f"Telemetry payload (npz compressed): {len(raw):,} bytes")
    raw_kcg = packet_to_stream_bytes(pkt)
    print(f"Telemetry payload (KCG stream):    {len(raw_kcg):,} bytes")

    pkt_rx = packet_from_npz_bytes(raw)
    tg_rx = decode_graph_packet(pkt_rx)
    L_rx = terrain_graph_laplacian(tg_rx)
    fro_lossless = reconstruction_frobenius(L_true, L_rx)
    print(f"Lossless Laplacian ||L - L_decoded||_F = {fro_lossless:.3e} (expect 0)")
    tg_rx_kcg = decode_graph_packet(packet_from_stream_bytes(raw_kcg))
    L_rx_kcg = terrain_graph_laplacian(tg_rx_kcg)
    print(f"Lossless Laplacian ||L - L_decoded_kcg||_F = {reconstruction_frobenius(L_true, L_rx_kcg):.3e}")

    d0 = kernelcal_graph_diagnostics(L_true)
    d1 = kernelcal_graph_diagnostics(L_rx)
    print("kernelcal.terrain diagnostics (sender vs receiver, lossless):")
    _print_diag("sender  ", d0)
    _print_diag("receiver", d1)

    L_low = low_rank_laplacian_from_packet(pkt_rx)
    fro_low = reconstruction_frobenius(L_true, L_low)
    print(f"Low-rank spectral L_hat (k={pkt_rx.k_transmitted()}): ||L - L_hat||_F = {fro_low:.6f}")
    d_low = kernelcal_graph_diagnostics(L_low)
    print("Diagnostics on L_hat (lossy):")
    _print_diag("L_hat   ", d_low)

    k_full = n
    pkt_full = encode_graph_packet(tg, k_modes=k_full)
    L_full_hat = low_rank_laplacian_from_packet(pkt_full)
    print(
        f"Sanity: full-rank k=n ({k_full}): ||L - L_hat||_F = "
        f"{reconstruction_frobenius(L_true, L_full_hat):.3e}"
    )

    if plot_dir is not None:
        write_packet_stream_file(pkt, str(Path(plot_dir) / "codec_synthetic.kcg"))
        write_codec_decode_plots(
            plot_dir,
            tg,
            L_true,
            L_rx,
            backdrop=_mask_backdrop_from_tg(tg),
            suptitle="Graph codec (synthetic channel DEM) — Fiedler: sender vs lossless decode",
            filename_stem="codec_synthetic",
        )
        write_reconstructed_graph_geometry_plot(
            plot_dir,
            tg,
            tg_rx,
            backdrop=_mask_backdrop_from_tg(tg),
            suptitle="Graph codec (synthetic) — geometry after decode",
            stem="codec_synthetic",
            payload_bytes=len(raw),
            n_nodes=n,
        )


RIVGRAPH_DATASETS: dict[str, dict[str, str]] = {
    "brahmaputra": {
        "kind": "river",
        "title": "Brahmaputra braided river",
        "stem": "codec_brahmaputra",
        "subdir": "Brahmaputra_Braided_River",
        "mask": "Brahmaputra_mask.tif",
        "run_name": "Brahma_graphcodec",
    },
    "meandering": {
        "kind": "river",
        "title": "Meandering river",
        "stem": "codec_meandering",
        "subdir": "Meandering_River",
        "mask": "meander_mask.tif",
        "run_name": "Meander_graphcodec",
    },
    "colville": {
        "kind": "delta",
        "title": "Colville delta",
        "stem": "codec_colville",
        "subdir": "Colville_Delta",
        "mask": "Colville_mask.tif",
        "shoreline": "Colville_shoreline.shp",
        "inlets": "Colville_inlet_nodes.shp",
        "run_name": "Colville_graphcodec",
    },
}


def run_rivgraph_dataset(
    dataset: str,
    *,
    rivgraph_repo: Path | None,
    k_modes: int,
    laplacian_mode: str,
    exit_sides: str,
    laplacian_max_nodes: int,
    skeleton_connectivity: int,
    verbose: bool,
    plot_dir: Path | None,
    prune_less: bool,
) -> None:
    """RivGraph example mask → ``TerrainGraph`` → graph codec (see ``RIVGRAPH_DATASETS``)."""
    import rivgraph_kernelcal_bridge as bridge

    cfg = RIVGRAPH_DATASETS.get(dataset)
    if cfg is None:
        raise SystemExit(f"Unknown RivGraph dataset: {dataset}")

    repo = bridge.ensure_rivgraph_import(rivgraph_repo)
    from rivgraph.classes import delta as delta_cls
    from rivgraph.classes import river

    base = (repo / "examples" / "data" / cfg["subdir"]).resolve()
    mask_path = (base / cfg["mask"]).resolve()
    results_folder = (base / "Results_graph_codec").resolve()
    if not mask_path.is_file():
        raise SystemExit(f"Mask not found for {dataset}: {mask_path}")

    results_folder.mkdir(parents=True, exist_ok=True)
    run_name = cfg["run_name"]
    for stale in (f"{run_name}_fixlinks.csv", "fixlinks.csv"):
        sp = results_folder / stale
        if sp.exists():
            sp.unlink()

    print(f"--- {cfg['title']} (RivGraph → kernelcal graph codec) ---")
    print(f"  dataset        : {dataset}")
    print(f"  mask           : {mask_path}")
    print(f"  results_folder : {results_folder}")

    if cfg["kind"] == "river":
        print(f"  exit_sides     : {exit_sides}")
        rnet = river(
            run_name,
            str(mask_path),
            str(results_folder),
            exit_sides=exit_sides,
            verbose=verbose,
        )
        rnet.skeletonize()
        rnet.compute_network()
    elif cfg["kind"] == "delta":
        shoreline = (base / cfg["shoreline"]).resolve()
        inlet_nodes = (base / cfg["inlets"]).resolve()
        print(f"  shoreline      : {shoreline}")
        print(f"  inlet_nodes    : {inlet_nodes}")
        if not shoreline.is_file():
            raise SystemExit(f"Shoreline not found: {shoreline}")
        if not inlet_nodes.is_file():
            raise SystemExit(f"Inlet nodes not found: {inlet_nodes}")
        rnet = delta_cls(
            run_name,
            str(mask_path),
            str(results_folder),
            verbose=verbose,
        )
        rnet.skeletonize()
        rnet.compute_network()
        rnet.prune_network(
            path_shoreline=str(shoreline),
            path_inletnodes=str(inlet_nodes),
            prune_less=prune_less,
        )
    else:
        raise SystemExit(f"Unknown kind {cfg['kind']!r}")

    riv_w = None
    sk_conn = cast(Literal[4, 8], int(skeleton_connectivity))
    tg, laplacian_label = bridge.resolve_laplacian_terrain_graph(
        mode=laplacian_mode,
        iskel=rnet.Iskel,
        links=rnet.links,
        nodes=rnet.nodes,
        imshape=rnet.imshape,
        weight_by_length=False,
        max_skel_nodes=laplacian_max_nodes,
        skeleton_connectivity=sk_conn,
        rivgraph_weight=riv_w,
    )
    L_true = terrain_graph_laplacian(tg)
    n = L_true.shape[0]
    print(f"  Laplacian      : {laplacian_label}")
    print(f"  RivGraph links : {len(rnet.links['id'])}")

    pkt = encode_graph_packet(tg, k_modes=min(k_modes, n))
    ok, msg = topology_guard_k(pkt.k_transmitted(), pkt.beta0, pkt.beta1)
    print(f"Graph: n_nodes={n}  |E|={pkt.edges.shape[0]}  beta0={pkt.beta0}  beta1={pkt.beta1}")
    print(f"Topology guard: {msg}  ->  {'OK' if ok else 'WARN'}")

    raw = packet_to_npz_bytes(pkt, compress=True)
    print(f"Telemetry payload (npz compressed): {len(raw):,} bytes")
    raw_kcg = packet_to_stream_bytes(pkt)
    print(f"Telemetry payload (KCG stream):    {len(raw_kcg):,} bytes")

    pkt_rx = packet_from_npz_bytes(raw)
    tg_rx = decode_graph_packet(pkt_rx)
    L_rx = terrain_graph_laplacian(tg_rx)
    print(f"Lossless Laplacian ||L - L_decoded||_F = {reconstruction_frobenius(L_true, L_rx):.3e}")
    tg_rx_kcg = decode_graph_packet(packet_from_stream_bytes(raw_kcg))
    L_rx_kcg = terrain_graph_laplacian(tg_rx_kcg)
    print(f"Lossless Laplacian ||L - L_decoded_kcg||_F = {reconstruction_frobenius(L_true, L_rx_kcg):.3e}")

    d0 = kernelcal_graph_diagnostics(L_true)
    d1 = kernelcal_graph_diagnostics(L_rx)
    print("kernelcal.terrain diagnostics (sender vs receiver, lossless):")
    _print_diag("sender  ", d0)
    _print_diag("receiver", d1)

    L_low = low_rank_laplacian_from_packet(pkt_rx)
    print(
        f"Low-rank spectral L_hat (k={pkt_rx.k_transmitted()}): "
        f"||L - L_hat||_F = {reconstruction_frobenius(L_true, L_low):.6f}"
    )
    _print_diag("L_hat (lossy)", kernelcal_graph_diagnostics(L_low))

    stem = cfg["stem"]
    title = cfg["title"]
    if plot_dir is not None:
        write_packet_stream_file(pkt, str(Path(plot_dir) / f"{stem}.kcg"))
        write_codec_decode_plots(
            plot_dir,
            tg,
            L_true,
            L_rx,
            backdrop=np.asarray(rnet.Imask, dtype=float),
            suptitle=f"{title} — Fiedler: sender vs lossless decode",
            filename_stem=stem,
        )
        write_reconstructed_graph_geometry_plot(
            plot_dir,
            tg,
            tg_rx,
            backdrop=np.asarray(rnet.Imask, dtype=float),
            suptitle=f"{title} — reconstructed graph (lossless decode vs sender)",
            stem=stem,
            payload_bytes=len(raw),
            n_nodes=n,
        )


def run_chain(*, n_nodes: int, k_modes: int, plot_dir: Path | None) -> None:
    print("--- Path graph (acyclic, beta1=0) ---")
    tg = chain_terrain_graph(n_nodes)
    L_true = terrain_graph_laplacian(tg)
    n = L_true.shape[0]
    pkt = encode_graph_packet(tg, k_modes=min(k_modes, n))
    ok, msg = topology_guard_k(pkt.k_transmitted(), pkt.beta0, pkt.beta1)
    print(f"Graph: n_nodes={n}  |E|={pkt.edges.shape[0]}  beta0={pkt.beta0}  beta1={pkt.beta1}")
    print(f"Topology guard: {msg}  ->  {'OK' if ok else 'WARN'}")
    raw = packet_to_npz_bytes(pkt, compress=True)
    print(f"Telemetry payload (npz compressed): {len(raw):,} bytes")
    raw_kcg = packet_to_stream_bytes(pkt)
    print(f"Telemetry payload (KCG stream):    {len(raw_kcg):,} bytes")
    tg_rx = decode_graph_packet(packet_from_npz_bytes(raw))
    L_rx = terrain_graph_laplacian(tg_rx)
    fro_lossless = reconstruction_frobenius(L_true, L_rx)
    print(f"Lossless ||L - L_decoded||_F = {fro_lossless:.3e}")
    tg_rx_kcg = decode_graph_packet(packet_from_stream_bytes(raw_kcg))
    L_rx_kcg = terrain_graph_laplacian(tg_rx_kcg)
    print(f"Lossless ||L - L_decoded_kcg||_F = {reconstruction_frobenius(L_true, L_rx_kcg):.3e}")
    d0 = kernelcal_graph_diagnostics(L_true)
    _print_diag("sender diagnostics", d0)

    if plot_dir is not None:
        write_packet_stream_file(pkt, str(Path(plot_dir) / "codec_chain.kcg"))
        write_codec_decode_plots(
            plot_dir,
            tg,
            L_true,
            L_rx,
            backdrop=_mask_backdrop_from_tg(tg),
            suptitle="Path graph codec — Fiedler: sender vs lossless decode",
            filename_stem="codec_chain",
        )
        write_reconstructed_graph_geometry_plot(
            plot_dir,
            tg,
            tg_rx,
            backdrop=_mask_backdrop_from_tg(tg),
            suptitle="Path graph — reconstructed graph (lossless decode)",
            stem="codec_chain",
            payload_bytes=len(raw),
            n_nodes=n,
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--demo",
        choices=("synthetic", "chain", "both", "brahmaputra", "meandering", "colville"),
        default="both",
        help="synthetic | chain | both | brahmaputra | meandering | colville (last three: RivGraph + codec)",
    )
    p.add_argument(
        "--rivgraph-repo",
        type=Path,
        default=None,
        help="RivGraph clone root (adds repo and _deps to sys.path). Used for RivGraph demos.",
    )
    p.add_argument(
        "--laplacian-mode",
        choices=("auto", "skeleton", "junction", "rivgraph"),
        default="rivgraph",
        help="TerrainGraph / Laplacian topology (default: rivgraph graphiphy).",
    )
    p.add_argument(
        "--exit-sides",
        default="NS",
        help="River exit sides (N,E,S,W) for --demo brahmaputra or meandering.",
    )
    p.add_argument(
        "--laplacian-max-nodes",
        type=int,
        default=12000,
        help="Max skeleton pixels before auto mode falls back to rivgraph.",
    )
    p.add_argument(
        "--skeleton-connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="Skeleton 4/8-neighbourhood if skeleton Laplacian is used.",
    )
    p.add_argument("--verbose", action="store_true", help="RivGraph verbose logging.")
    p.add_argument(
        "--prune-less",
        action="store_true",
        help="Colville only: pass prune_less=True to prune_network.",
    )
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="If set, save codec comparison PNG(s) under this directory (Fiedler sender/decoded/diff).",
    )
    p.add_argument("--nrows", type=int, default=48)
    p.add_argument("--ncols", type=int, default=48)
    p.add_argument("--k-modes", type=int, default=16, help="Smallest Laplacian eigenpairs stored")
    p.add_argument("--chain-nodes", type=int, default=24, help="Path length for --demo chain")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.plot_dir is not None:
        os.environ.setdefault("MPLBACKEND", "Agg")
    if args.demo in ("brahmaputra", "meandering", "colville"):
        run_rivgraph_dataset(
            args.demo,
            rivgraph_repo=args.rivgraph_repo,
            k_modes=args.k_modes,
            laplacian_mode=args.laplacian_mode,
            exit_sides=args.exit_sides,
            laplacian_max_nodes=args.laplacian_max_nodes,
            skeleton_connectivity=args.skeleton_connectivity,
            verbose=args.verbose,
            plot_dir=args.plot_dir,
            prune_less=args.prune_less,
        )
        return
    if args.demo in ("synthetic", "both"):
        run_once(
            nrows=args.nrows,
            ncols=args.ncols,
            k_modes=args.k_modes,
            seed=args.seed,
            plot_dir=args.plot_dir,
        )
    if args.demo in ("chain", "both"):
        if args.demo == "both":
            print()
        run_chain(n_nodes=args.chain_nodes, k_modes=args.k_modes, plot_dir=args.plot_dir)


if __name__ == "__main__":
    main()
