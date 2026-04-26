"""experiments/planned_vs_organic_dH.py

PR-C of CR-2026-04-26 -- planned-vs-organic ΔH receipt.

Two viewports (or two synthetic layouts), each built in two graph
modes:

    * ``road_knn`` -- adjacency from k-NN on road-network distance
      between building centroids.
    * ``knn``      -- adjacency from k-NN on Euclidean distance.

For each (viewport × mode) pair we compute the Laplacian spectrum and
report:

    * normalised spectral entropy ``H = -Σ p_i log p_i / log k``,
      where ``p_i = λ_i / Σ λ_j`` over the non-zero spectrum;
    * ``β₀``: the connected-component count;
    * ``H_grid - H_fringe`` per mode;
    * the difference of differences ``ΔΔH = ΔH_road - ΔH_knn``,
      which is the load-bearing claim from Field Notes 38/39.

Two run modes
-------------
``--mode synthetic`` (default; no network)
    Builds purely synthetic grid / fringe layouts via
    :mod:`kernelcal.urban.synthetic`.  This is the one shipped in
    the receipt: it tests the *structural* prediction (orthogonal
    grid spectrum is more concentrated than branching spectrum).

``--mode live-osm``
    Fetches buildings + road graph for the bboxes in the YAML
    configs, then runs through ``buildings_to_graph_via_roads_from_bbox``
    and ``buildings_to_graph_from_bbox``.  Requires Overpass /
    osmnx access.  The numbers from this run are committed as a
    follow-up addendum to the result report; the synthetic numbers
    stand on their own as the structural receipt.

Usage
-----
    python experiments/planned_vs_organic_dH.py \\
        --grid-config experiments/configs/synthetic_grid.yaml \\
        --fringe-config experiments/configs/synthetic_fringe.yaml \\
        --output experiments/output/planned_vs_organic_dH.json

    python experiments/planned_vs_organic_dH.py --mode live-osm \\
        --grid-config experiments/configs/tempe_grid.yaml \\
        --fringe-config experiments/configs/sonoran_fringe.yaml \\
        --output experiments/output/planned_vs_organic_dH_live.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np


# ---------------------------------------------------------------------------\
# YAML loading without requiring PyYAML for the simple flat configs.
# ---------------------------------------------------------------------------\

def _load_yaml(path: Path) -> dict:
    """Load a YAML config -- prefers PyYAML, falls back to a tiny parser
    for the flat key:value subset our configs use.

    Restricted to:
        * top-level ``key: value`` pairs (no nested mappings, no lists)
        * scalars: int, float, bool, str (unquoted), null
        * comments starting with ``#``
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]

        return yaml.safe_load(text) or {}
    except ImportError:
        pass

    out: dict = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{path}: cannot parse line {raw!r}")
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not value or value.lower() == "null":
            out[key] = None
            continue
        if value.lower() in ("true", "false"):
            out[key] = (value.lower() == "true")
            continue
        try:
            if "." in value or "e" in value.lower():
                out[key] = float(value)
            else:
                out[key] = int(value)
        except ValueError:
            out[key] = value.strip("'\"")
    return out


# ---------------------------------------------------------------------------\
# Synthetic-mode runner
# ---------------------------------------------------------------------------\

def _run_synthetic_layout(name: str, cfg: Mapping) -> dict:
    """Build a synthetic CityGraph in both road_knn and knn modes."""
    from kernelcal.urban import (
        make_fringe_layout,
        make_fringe_road_segments,
        make_grid_layout,
        make_grid_road_segments,
        spectral_diagnostics,
        synthetic_city_graph,
    )

    layout = str(cfg.get("layout", name))

    if layout == "grid":
        positions = make_grid_layout(
            n_blocks_x=int(cfg.get("n_blocks_x", 8)),
            n_blocks_y=int(cfg.get("n_blocks_y", 8)),
            block_size_m=float(cfg.get("block_size_m", 80.0)),
            jitter_m=float(cfg.get("jitter_m", 5.0)),
            seed=int(cfg.get("seed", 42)),
        )
        road_nodes, road_edges = make_grid_road_segments(
            n_blocks_x=int(cfg.get("n_blocks_x", 8)),
            n_blocks_y=int(cfg.get("n_blocks_y", 8)),
            block_size_m=float(cfg.get("block_size_m", 80.0)),
        )
    elif layout == "fringe":
        positions = make_fringe_layout(
            n_buildings=int(cfg.get("n_buildings", 64)),
            n_seeds=int(cfg.get("n_seeds", 6)),
            scale_m=float(cfg.get("scale_m", 800.0)),
            cluster_sigma_m=float(cfg.get("cluster_sigma_m", 25.0)),
            seed=int(cfg.get("seed", 42)),
        )
        road_nodes, road_edges = make_fringe_road_segments(
            n_seeds=int(cfg.get("n_seeds", 6)),
            scale_m=float(cfg.get("scale_m", 800.0)),
            branch_length_m=float(cfg.get("branch_length_m", 120.0)),
            n_branches_per_seed=int(cfg.get("n_branches_per_seed", 3)),
            seed=int(cfg.get("seed", 42)),
        )
    else:
        raise ValueError(
            f"Unknown synthetic layout {layout!r}; expected 'grid' or 'fringe'"
        )

    k = int(cfg.get("k", 8))
    sigma_frac = float(cfg.get("sigma_frac", 0.05))

    cg_road = synthetic_city_graph(
        name=f"{name}_road_knn",
        place=f"synthetic:{layout}",
        positions=positions,
        road_nodes=road_nodes,
        road_edges=road_edges,
        k=k,
        sigma_frac=sigma_frac,
    )
    cg_knn = synthetic_city_graph(
        name=f"{name}_knn",
        place=f"synthetic:{layout}",
        positions=positions,
        road_nodes=None,
        road_edges=None,
        k=k,
        sigma_frac=sigma_frac,
    )

    return {
        "name": name,
        "layout": layout,
        "n_nodes": int(positions.shape[0]),
        "modes": {
            "road_knn": spectral_diagnostics(cg_road.eigvals)
            | {"road_meta": cg_road.road_meta},
            "knn": spectral_diagnostics(cg_knn.eigvals),
        },
    }


# ---------------------------------------------------------------------------\
# Live-OSM runner
# ---------------------------------------------------------------------------\

def _run_live_osm(name: str, cfg: Mapping, cache_dir: Path) -> dict:
    """Run the full bbox -> CityGraph pipeline in both modes against
    live OSM.  Requires osmnx + Overpass connectivity.
    """
    from kernelcal.urban import (
        buildings_to_graph_from_bbox,
        buildings_to_graph_via_roads_from_bbox,
        spectral_diagnostics,
    )

    bbox = (
        float(cfg["south"]),
        float(cfg["west"]),
        float(cfg["north"]),
        float(cfg["east"]),
    )
    k = int(cfg.get("k", 8))
    n_max = int(cfg.get("n_max", 1500))
    sigma_frac = float(cfg.get("sigma_frac", 0.05))

    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[live-osm:{name}] fetching road_knn for bbox={bbox} ...", flush=True)
    cg_road = buildings_to_graph_via_roads_from_bbox(
        south=bbox[0], west=bbox[1], north=bbox[2], east=bbox[3],
        name=f"{name}_road_knn",
        k=k, n_max=n_max, sigma_frac=sigma_frac,
        cache_dir=cache_dir,
    )

    print(f"[live-osm:{name}] fetching knn for bbox={bbox} ...", flush=True)
    cg_knn = buildings_to_graph_from_bbox(
        south=bbox[0], west=bbox[1], north=bbox[2], east=bbox[3],
        name=f"{name}_knn",
        k=k, n_max=n_max, sigma_frac=sigma_frac,
        cache_dir=cache_dir,
    )

    return {
        "name": name,
        "bbox": list(bbox),
        "n_nodes_road_knn": int(cg_road.positions.shape[0]),
        "n_nodes_knn": int(cg_knn.positions.shape[0]),
        "modes": {
            "road_knn": spectral_diagnostics(cg_road.eigvals)
            | {"road_meta": cg_road.road_meta},
            "knn": spectral_diagnostics(cg_knn.eigvals),
        },
    }


# ---------------------------------------------------------------------------\
# Result composition
# ---------------------------------------------------------------------------\

def _compose_receipt(grid_result: dict, fringe_result: dict) -> dict:
    """Compose the ΔH receipt summary from the two per-viewport results."""
    H = lambda r, mode: r["modes"][mode]["spectral_entropy_normalised"]
    B0 = lambda r, mode: r["modes"][mode]["beta_0"]

    receipt = {
        "viewports": {"grid": grid_result, "fringe": fringe_result},
        "delta_H": {
            "road_knn": H(grid_result, "road_knn") - H(fringe_result, "road_knn"),
            "knn":      H(grid_result, "knn")      - H(fringe_result, "knn"),
        },
        "beta_0": {
            "grid": {
                "road_knn": B0(grid_result, "road_knn"),
                "knn":      B0(grid_result, "knn"),
            },
            "fringe": {
                "road_knn": B0(fringe_result, "road_knn"),
                "knn":      B0(fringe_result, "knn"),
            },
        },
    }
    receipt["delta_delta_H"] = (
        receipt["delta_H"]["road_knn"] - receipt["delta_H"]["knn"]
    )

    # Field Notes 38/39 prediction discriminator.  Positive ΔH on
    # road_knn = grids are more ordered than fringes once you account
    # for the road structure.  ΔΔH > 0 = the road structure is the
    # load-bearing piece of the difference.
    p = receipt["delta_H"]["road_knn"]
    receipt["interpretation"] = {
        "fn_38_39_grid_more_ordered_than_fringe_in_road_knn": bool(p < 0.0),
        "fn_38_39_road_structure_amplifies_difference": bool(receipt["delta_delta_H"] < 0.0),
    }
    # NOTE: H = -Σ p log p with p = λ/Σλ; a *more concentrated* spectrum
    # (a "controller" pulling mass to a few low modes) has *smaller* H.
    # So "grid more ordered" predicts H_grid < H_fringe i.e. ΔH < 0.
    return receipt


def _print_summary(receipt: dict, mode: str) -> None:
    print()
    print(f"=== planned-vs-organic ΔH receipt ({mode}) ===")
    grid = receipt["viewports"]["grid"]
    fringe = receipt["viewports"]["fringe"]
    H_grid_road = grid["modes"]["road_knn"]["spectral_entropy_normalised"]
    H_grid_knn  = grid["modes"]["knn"]["spectral_entropy_normalised"]
    H_fr_road   = fringe["modes"]["road_knn"]["spectral_entropy_normalised"]
    H_fr_knn    = fringe["modes"]["knn"]["spectral_entropy_normalised"]

    print(f"  grid    ({grid['n_nodes']:>4d} nodes):  H_road = {H_grid_road:.4f}   H_knn = {H_grid_knn:.4f}")
    print(f"  fringe  ({fringe['n_nodes']:>4d} nodes):  H_road = {H_fr_road:.4f}   H_knn = {H_fr_knn:.4f}")
    print(f"  ΔH (road_knn): {receipt['delta_H']['road_knn']:+.4f}")
    print(f"  ΔH (knn):      {receipt['delta_H']['knn']:+.4f}")
    print(f"  ΔΔH:           {receipt['delta_delta_H']:+.4f}")
    print(f"  β₀ grid    road_knn={receipt['beta_0']['grid']['road_knn']}  knn={receipt['beta_0']['grid']['knn']}")
    print(f"  β₀ fringe  road_knn={receipt['beta_0']['fringe']['road_knn']}  knn={receipt['beta_0']['fringe']['knn']}")
    interp = receipt["interpretation"]
    print(f"  FN 38/39: grid more ordered than fringe (road_knn): "
          f"{interp['fn_38_39_grid_more_ordered_than_fringe_in_road_knn']}")
    print(f"  FN 38/39: road structure amplifies the difference: "
          f"{interp['fn_38_39_road_structure_amplifies_difference']}")
    print()


# ---------------------------------------------------------------------------\
# CLI
# ---------------------------------------------------------------------------\

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("synthetic", "live-osm"),
        default="synthetic",
        help="synthetic (default) or live-osm.",
    )
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=Path("experiments/configs/synthetic_grid.yaml"),
    )
    parser.add_argument(
        "--fringe-config",
        type=Path,
        default=Path("experiments/configs/synthetic_fringe.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/output/planned_vs_organic_dH.json"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./cache/planned_vs_organic"),
        help="Cache dir for live-OSM mode.",
    )
    args = parser.parse_args(argv)

    grid_cfg = _load_yaml(args.grid_config)
    fringe_cfg = _load_yaml(args.fringe_config)

    t0 = time.time()
    if args.mode == "synthetic":
        grid_result = _run_synthetic_layout("grid", grid_cfg)
        fringe_result = _run_synthetic_layout("fringe", fringe_cfg)
    else:
        grid_result = _run_live_osm("grid", grid_cfg, args.cache_dir)
        fringe_result = _run_live_osm("fringe", fringe_cfg, args.cache_dir)
    t1 = time.time()

    receipt = _compose_receipt(grid_result, fringe_result)
    receipt["mode"] = args.mode
    receipt["runtime_seconds"] = float(t1 - t0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2))

    _print_summary(receipt, args.mode)
    print(f"  wrote {args.output} ({t1 - t0:.2f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
