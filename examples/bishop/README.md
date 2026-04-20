# examples/bishop

Bishop Tuff rock-trait spectral analyses and adaptive graph exploration.

| Script                           | What it does                                                                                              |
| -------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `bishop_kernelcal.py`            | End-to-end kernelcal pipeline: load traits, build graph, compute kernel, write `bishop_figures/`.         |
| `bishop_mode_decomposition.py`   | Per-trait spectral mode decomposition + bandwise weight plots.                                            |
| `bishop_rocks_graph_explorer.py` | Quadrant-adaptive graph explorer over rock centroids.  Also exercised by `tests/test_bishop_rocks_explorer.py`. |
| `bishop_trait_analysis.py`       | Scarp vs. off-scarp trait contrast and cross-kernel figures.                                              |
| `plot_bishop_rocks.py`           | Static scatter/histogram figures for the Bishop rock dataset.                                             |

Dataset: `datasets/bishop_scarp/` (repo-root; gitignored).
Figures: `bishop_figures/` (repo-root; gitignored, except for the live
explorer frame referenced from the top-level README).
