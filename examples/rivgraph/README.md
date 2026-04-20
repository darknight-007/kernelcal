# examples/rivgraph

RivGraph integration — convert river-network skeleton graphs from the
[`rivgraph`](https://github.com/jonschwenk/RivGraph) package into
kernelcal spectral kernels.

| Script                                          | What it does                                                                     |
| ----------------------------------------------- | -------------------------------------------------------------------------------- |
| `rivgraph_graph_analysis.py`                    | Single-timestep kernelcal analysis on a RivGraph network.                        |
| `run_rivgraph_kernelcal_integration_batch.py`   | Batch driver over a directory of RivGraph outputs; writes `integration_batch_outputs/`. |
| `q14_brahmaputra_temporal_sim.py`               | Q14: temporal simulation of the Brahmaputra river network.                      |

The canonical library-side bridge lives at
`kernelcal.integrations.rivgraph` and is also exposed as the
`kernelcal-rivgraph-bridge` console script.  A root-level
`rivgraph_kernelcal_bridge.py` shim forwards to it for backward
compatibility.
