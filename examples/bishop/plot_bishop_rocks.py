#!/usr/bin/env python3
"""Plot bishop scarp rocks: a map colored by rock area, plus trait histograms.

Inputs (expected in `unprocessed/bishop-root/` next to this script):
  - rocks-coord-list.csv  (no header, columns: lon, lat)        ~82k rows
  - rock_traits_full.csv  (header: lon,lat,area_m2,major_axis_m,
                           minor_axis_m,eccentricity,
                           orientation_deg,elevation_rel)       ~14k rows

Usage:
    python3 plot_bishop_rocks.py
    python3 plot_bishop_rocks.py --show
    python3 plot_bishop_rocks.py --out ~/tmp
    python3 plot_bishop_rocks.py --data-dir /path/with/csvs

Requires: pandas, numpy, matplotlib.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
# Matches sibling scripts (``bishop_kernelcal.py``, ``bishop_trait_analysis.py``):
# rock centroids + per-rock traits CSVs live under ``datasets/bishop_scarp/``.
DEFAULT_DATA_DIR = HERE / "datasets" / "bishop_scarp"
DEFAULT_OUT_DIR = HERE / "bishop_figures"

COORD_CSV = "rocks-coord-list.csv"
TRAITS_CSV = "rock_traits_full.csv"


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    coord_path = data_dir / COORD_CSV
    traits_path = data_dir / TRAITS_CSV
    coords = pd.read_csv(coord_path, header=None, names=["lon", "lat"])
    traits = pd.read_csv(traits_path)
    return coords, traits


def plot_map(coords: pd.DataFrame, traits: pd.DataFrame, out: Path) -> Path:
    """Map view of all rocks; the traits subset is colored by area (log scale)."""
    fig, ax = plt.subplots(figsize=(11, 11))

    ax.scatter(
        coords["lon"], coords["lat"],
        s=1.5, color="lightgray", alpha=0.4, linewidths=0,
        label=f"all rocks  (n={len(coords):,})",
    )

    area = traits["area_m2"].to_numpy()
    sc = ax.scatter(
        traits["lon"], traits["lat"],
        c=area,
        s=np.clip(area * 4.0, 3, 140),
        cmap="viridis",
        norm=LogNorm(vmin=max(area.min(), 0.05), vmax=area.max()),
        alpha=0.85, linewidths=0,
        label=f"traits rocks  (n={len(traits):,})",
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("area (m², log scale)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.set_title("Bishop scarp rocks — colored and sized by area_m²")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    return out


def plot_histograms(traits: pd.DataFrame, out: Path) -> Path:
    """2×3 grid of histograms for the numeric trait columns."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    def _hist(ax, data, bins, color, title, xlabel, logy=False):
        ax.hist(data, bins=bins, color=color, edgecolor="white", linewidth=0.3)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    _hist(axes[0, 0], traits["area_m2"], 80, "steelblue",
          "Rock area", "area (m²)", logy=True)

    axes[0, 1].hist(traits["major_axis_m"], bins=80, alpha=0.75,
                    label="major", color="firebrick", edgecolor="white", linewidth=0.3)
    axes[0, 1].hist(traits["minor_axis_m"], bins=80, alpha=0.75,
                    label="minor", color="seagreen", edgecolor="white", linewidth=0.3)
    axes[0, 1].set_xlabel("axis length (m)")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].set_title("Ellipse axes")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    _hist(axes[0, 2], traits["eccentricity"], 60, "darkorange",
          "Eccentricity", "eccentricity")

    _hist(axes[1, 0], np.mod(traits["orientation_deg"], 180.0), 36, "mediumpurple",
          "Orientation (mod 180°)", "orientation (°)")

    _hist(axes[1, 1], traits["elevation_rel"], 80, "teal",
          "Relative elevation", "elevation_rel (m)")

    ax = axes[1, 2]
    sc = ax.scatter(
        traits["major_axis_m"], traits["minor_axis_m"],
        c=traits["area_m2"],
        s=6, alpha=0.5, cmap="viridis", linewidths=0,
        norm=LogNorm(vmin=max(traits["area_m2"].min(), 0.05),
                     vmax=traits["area_m2"].max()),
    )
    lim = max(traits["major_axis_m"].max(), traits["minor_axis_m"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("major_axis (m)"); ax.set_ylabel("minor_axis (m)")
    ax.set_title("Major × minor (color = area, log)")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02, label="area (m²)")

    fig.suptitle(f"Bishop scarp — rock traits  (n={len(traits):,})",
                 y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    return out


def summary(traits: pd.DataFrame) -> str:
    cols = ["area_m2", "major_axis_m", "minor_axis_m",
            "eccentricity", "orientation_deg", "elevation_rel"]
    return traits[cols].describe().round(3).to_string()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"Folder with the two CSVs (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Output folder for PNGs (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--show", action="store_true",
                        help="Open an interactive window after saving")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    coords, traits = load(args.data_dir)

    print(f"{COORD_CSV:>24}: {len(coords):>7,} rows")
    print(f"{TRAITS_CSV:>24}: {len(traits):>7,} rows, columns={list(traits.columns)}")
    print("\nTrait summary:")
    print(summary(traits))

    map_png = plot_map(coords, traits, args.out / "bishop_rocks_map.png")
    hist_png = plot_histograms(traits, args.out / "bishop_rocks_traits_hist.png")
    print(f"\nWrote: {map_png}")
    print(f"Wrote: {hist_png}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
