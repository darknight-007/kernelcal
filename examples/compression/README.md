# examples/compression

Spectral-kernel compression drivers for large meshes and graphs.

| Script                                | What it does                                                                                                          |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `run_graph_codec_demo.py`             | Round-trip demo of `kernelcal.geo3d` point-cloud → graph-kernel codec on synthetic inputs.                            |
| `run_terrain_compression.py`          | Nyström compression sweep on a real terrain OBJ (artburysol175).  Pass the OBJ path as argv[1] or set `KERNELCAL_TERRAIN_OBJ`. |
| `synthetic_planetary_mesh_experiment.py` | Synthetic planetary-scale meshes, compression bounds vs. topology.                                                |

All three write payloads (`.kcmesh`) and decoded OBJs alongside the input,
and figures under `terrain_figures/` at the repo root.
