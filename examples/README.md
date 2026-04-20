# kernelcal examples

Runnable experiment drivers and demo scripts that exercise the `kernelcal`
package against real-world datasets (terrain meshes, ROS2 point clouds,
satellite rasters, transformer activations, graph networks).

These scripts are **not library code**.  They are the sources of truth for
the figures and tables published alongside the paper, and they double as
executable documentation for each integration thread.

## Layout

| Directory            | Focus area                                                            |
| -------------------- | --------------------------------------------------------------------- |
| `attention/`         | Transformer attention kernels: training dynamics, EEG, wall-power     |
| `bishop/`            | Bishop Tuff rock-trait analysis and adaptive graph exploration        |
| `compression/`       | Spectral-kernel compression: terrain OBJ, random meshes, graph codec  |
| `controller/`        | MaxCal-driven control experiments: phase-space, conservation sweeps   |
| `fires/`             | Bobcat Fire vegetation segmentation and kernelcal bridge              |
| `lunar/`             | Lunar DEM + CERES / small-body kernelcal drivers                      |
| `navigation/`        | 2-D toy navigation and velocity-control demos                         |
| `rivgraph/`          | RivGraph integration batches and temporal river-network sims          |
| `robbins/`           | Robbins lunar-crater phase-space + paper-figure drivers               |
| `urban/`             | OSM street-network spectral analysis                                  |

Each subdirectory has its own `README.md` with invocation examples.

## Common conventions

1. **Path anchoring.** Every script computes `KCAL_ROOT` (or `ROOT`) as
   `Path(__file__).resolve().parent.parent.parent` and uses that anchor
   for kernelcal imports, datasets, and figure outputs.  This means you
   can run them from *any* working directory:

       python examples/bishop/bishop_kernelcal.py

2. **Figure output.**  Figure directories (e.g. `bishop_figures/`,
   `terrain_figures/`) live at the **repo root** — not inside
   `examples/<category>/`.  The directories are gitignored; only the
   handful of figures referenced inline from the top-level `README.md`
   are committed as explicit exceptions.

3. **Dataset inputs.**  Scripts look for inputs under the repo-root
   `datasets/` tree (also gitignored).  See each script's docstring for
   the exact path and how to regenerate or obtain the data.

4. **Console-script entry points.**  A subset of frequently-run CLIs are
   also registered as console scripts in `pyproject.toml`:

       kernelcal-rivgraph-bridge     # kernelcal.integrations.rivgraph
       kernelcal-landauer            # kernelcal.attention.landauer
       kernelcal-attention-training  # kernelcal.attention.training
       kernelcal-sigma-m-p8          # kernelcal.thermodynamics.sigma_m_p8

   Everything else is invoked directly with `python examples/…`.
