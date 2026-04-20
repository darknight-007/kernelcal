"""
kernelcal.spectral.experiments_plots
====================================
Shared plotting scaffolding for :mod:`kernelcal.spectral.experiments`.

This module owns the matplotlib helpers used across the six/seven
verification experiments (mode-color palettes, mode labels, figure save
helper, and the topology schematics).

Per-experiment figure generation remains inside ``experiments.py`` for
historical reasons — each ``experiment_N_*`` function interleaves numerics
and plotting, and a clean split requires API changes to every experiment.
Those helpers re-import ``_mode_colors`` / ``_mode_labels`` / ``_save``
from here, so a future full split can migrate experiment-by-experiment
without any shared-helper churn.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


__all__ = [
    "mode_colors",
    "mode_labels",
    "save_figure",
    "draw_topology",
    "save_topology_schematics",
    # Back-compat aliases — these were the historical names used inside
    # experiments.py before the plotting split.
    "_mode_colors",
    "_mode_labels",
    "_save",
    "_draw_topology",
    "_save_topology_schematics",
]


def mode_colors(N: int):
    """Return N distinct colors from the tab10/tab20 colormap."""
    cmap = plt.cm.get_cmap("tab10" if N <= 10 else "tab20", N)
    return [cmap(i) for i in range(N)]


def mode_labels(N: int):
    return [f"l={l}" for l in range(N)]


def save_figure(fig, path: Path, name: str) -> None:
    """Save ``fig`` to ``path / name`` at 150 dpi with tight bbox, then close it."""
    path.mkdir(parents=True, exist_ok=True)
    fpath = path / name
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {fpath}")


def draw_topology(ax, coords, edges, choke_edge, title: str):
    """Draw a simple node-edge topology schematic on ``ax``."""
    for (i, j) in edges:
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        is_choke = (i, j) == choke_edge or (j, i) == choke_edge
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#d62728" if is_choke else "#1f77b4",
            lw=2.6 if is_choke else 1.8,
            alpha=0.95,
            zorder=1,
        )
    xs = [coords[k][0] for k in sorted(coords)]
    ys = [coords[k][1] for k in sorted(coords)]
    ax.scatter(xs, ys, s=55, color="#4c78a8", edgecolor="white", linewidth=0.8, zorder=2)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_topology_schematics(output_dir: Path) -> None:
    """Save the Exp 7 river + trunk-roots topology schematic figure."""
    river_coords = {i: (i, 0.0) for i in range(10)}
    river_coords.update({
        10: (2.2, 1.2),
        11: (4.0, 1.3),
        12: (4.4, -1.2),
        13: (6.2, 1.2),
        14: (7.2, 1.1),
        15: (8.2, -1.0),
    })
    river_edges = [(i, i + 1) for i in range(9)] + [
        (2, 10), (4, 11), (4, 12), (6, 13), (7, 14), (8, 15),
    ]

    trunk_coords = {i: (0.0, i) for i in range(8)}
    trunk_coords.update({
        8: (-1.0, -0.2),  9: (-1.3, 0.5), 10: (-0.6, 0.8),
        11: (1.0, 0.2), 12: (1.3, 0.9),
        13: (-1.0, 6.3), 14: (-0.6, 7.0), 15: (1.0, 6.4), 16: (0.7, 7.1),
    })
    trunk_edges = [(i, i + 1) for i in range(7)] + [
        (0, 8), (0, 9), (0, 10), (1, 11), (1, 12),
        (6, 13), (6, 14), (7, 15), (7, 16),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    draw_topology(
        axes[0], river_coords, river_edges,
        choke_edge=(4, 5),
        title="River-channel topology\n(choke edge 4-5 in red)",
    )
    draw_topology(
        axes[1], trunk_coords, trunk_edges,
        choke_edge=(3, 4),
        title="Trunk+roots topology\n(constriction edge 3-4 in red)",
    )
    fig.suptitle("Exp 7 topology schematics", fontsize=11)
    fig.tight_layout()
    save_figure(fig, output_dir, "exp7_topologies.pdf")


# Back-compat: historically these helpers were prefixed with ``_`` inside
# experiments.py and referred to by that name.  Keep aliases so any
# external caller that reached in continues to work.
_mode_colors = mode_colors
_mode_labels = mode_labels
_save = save_figure
_draw_topology = draw_topology
_save_topology_schematics = save_topology_schematics
