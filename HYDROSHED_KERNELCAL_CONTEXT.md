# HydroSHEDS DEM in the KernelCal Graph-Spectral Framework

## Purpose

This note defines how to treat a HydroSHEDS DEM as a physically grounded testbed for KernelCal spectral-kernel dynamics, graph Hodge analysis, Betti-number topology, and optimal stream-network inference.

The immediate dataset anchor in this workspace is:

- `datasets/hydroshed-dem/na_con_3s/na_con_3s.tif`

Auxiliary metadata indicates elevation values roughly spanning `-39 m` to `3895 m`, with mean around `670 m` over valid pixels. This is a plausible continental-scale relief field for drainage-graph experiments.

## Why HydroSHEDS is a good fit

HydroSHEDS provides exactly the structure needed for physically motivated edges:

- DEM cell adjacency has a hydrologic interpretation (local flow transfer).
- D8 flow routing yields directed drainage pathways from elevation gradients.
- Thresholded flow accumulation gives channel masks and stream skeletons.
- Resulting channel graphs carry explicit topological signatures (`beta0`, `beta1`) and support spectral diagnostics on graph Laplacians.

This avoids the k-NN proximity artifact problem discussed in this repository, where edges can be analyst-imposed rather than process-grounded.

## Mathematical embedding in KernelCal

Given a DEM patch \( Z \in \mathbb{R}^{n_r \times n_c} \):

1. Build a terrain graph \( G=(V,E) \) from grid cells (`dem_to_graph`).
2. Build a drainage graph \( G_d \) from D8 flow (`drainage_network_graph`).
3. Form Laplacian \( L = D - W \) (`terrain_graph_laplacian` or `drainage_graph_laplacian`).
4. Solve fixed-point spectral kernel \( h^* \) under Gaussian-MI source:
   \[
   T_\ell[h] = \frac{\mu_2 w_\ell}{2(\sigma^2 + h_\ell)},\quad
   h^{(n+1)}_\ell = h_{0,\ell}\exp\{-1 - T_\ell[h^{(n)}]\}.
   \]
5. Compute diagnostics:
   - Spectral entropy \( H[h^*] \)
   - Stability margin / Fiedler-mode gap \( \Delta' \)
   - Stability-conservation residual \( D_m \)
   - Topology budget \( k_{\min} = \beta_0 + \beta_1 \)

In this framing, HydroSHEDS is not just an elevation raster; it is a generator of physically constrained graph trajectories in kernel space.

## Hodge-Laplacian interpretation for streams

For stream-network analysis, use edge signals on the drainage graph (for example, accumulation differences along edges):

- Incidence map \( B_1 \) defines edge gradients.
- The graph Hodge decomposition splits edge flow into:
  - gradient-like component (potential flow),
  - harmonic/residual component (cycle-associated structure in graph setting),
  - curl proxy energy (implemented as residual-from-gradient energy in `kernelcal.terrain.channels`).

Operationally, this is encoded by:

- `hodge_edge_decompose(...)`
- `curl_energy(...)`

In braided/anabranched systems, nontrivial cycle structure tends to increase residual/harmonic energy, complementing \( \beta_1 \) and spectral-entropy signals.

## Betti numbers and hydro-topological meaning

For a channel graph:

- \( \beta_0 \): number of connected drainage components.
- \( \beta_1 \): number of independent loops/cycles (braiding, anastomosis, avulsion remnants).

The repository implements:

- direct graph-level \( \beta_1 \) from \( E - V + \beta_0 \),
- `topology_budget(dg)` returning \( k_{\min} = \beta_0 + \beta_1 \).

Interpretation:

- low \( \beta_1 \): mostly tree-like drainage organization,
- elevated \( \beta_1 \): loop-rich transport structure and potential multichannel complexity.

## Triple spectral diagnostic for channelization

HydroSHEDS-derived drainage graphs can be classified with the P2 triple criterion:

\[
D_{\text{channel}} =
[H[h^*] < H_{\text{flat}}]
\land
[E_{\text{curl}} > E_{\text{flat}}]
\land
[\beta_1 \ge \beta_1^*].
\]

Mapped to code:

- `triple_spectral_diagnostic(dg, dg_flat=None)`

This gives a physically interpretable channel/no-channel discriminator using spectral concentration, non-potential flow structure, and topology together.

## Optimal stream network in the KernelCal sense

In this framework, an "optimal" stream network is one that approaches a MaxCal-consistent fixed point under physical constraints and topology conservation:

1. **Hydrologic feasibility:** edges come from D8 flow, not arbitrary proximity.
2. **Topological sufficiency:** preserve at least \( k_{\min}=\beta_0+\beta_1 \) spectral modes.
3. **Stability:** positive stability margin \( \Delta' \), low fixed-point residual.
4. **Information efficiency:** under bandwidth limits, allocate modes via
   `bandwidth_optimal_modes(...)` beyond topologically obligate modes.
5. **Fragility-aware optimization:** identify critical nodes and fragmentation trajectories:
   - `identify_critical_nodes(...)`
   - `critical_fragmentation_curve(...)`

This makes "optimal stream network" a joint criterion over flow physics, topology, and information-constrained spectral representation.

## Recommended HydroSHEDS workflow (manuscript-ready)

1. Tile HydroSHEDS into hydrologically coherent patches (watershed-aligned if possible).
2. For each tile:
   - preprocess sinks/depressions (if needed),
   - compute D8 flow and accumulation,
   - extract channel mask at threshold sweep,
   - build `DrainageGraph`.
3. Compute per-tile metrics:
   - \( \beta_0, \beta_1, k_{\min} \),
   - \( H[h^*] \), \( \Delta' \), conservation deficit \( D_m \),
   - curl energy and triple diagnostic flag,
   - critical-node fragmentation slopes.
4. Define an objective for "optimality" (example):
   \[
   J = \alpha\,\text{stability}(\Delta')
     - \beta\,\text{deficit}(|D_m|)
     + \gamma\,\text{channel\_confidence}
     - \eta\,\text{fragility}.
   \]
5. Compare objective across thresholds/tiles to identify robust stream-network regimes.

## Suggested framing text for the manuscript

HydroSHEDS DEMs are treated as physically constrained generators of drainage graphs, where adjacency and flow paths arise from terrain-controlled transport rather than analyst-imposed proximity. Within KernelCal, each drainage graph induces a Laplacian spectrum and a MaxCal fixed-point kernel \( h^* \), enabling a unified readout of entropy concentration, stability margin, and conservation deficit. Hodge-style edge decomposition and Betti-number topology provide complementary structure diagnostics: \( \beta_1 \) captures loop-rich braiding, while residual flow energy quantifies departures from pure potential flow. An optimal stream network is then defined as a topology-preserving, stability-favoring, information-efficient fixed-point regime under bandwidth and observation constraints.

## Practical caveats

- Continental HydroSHEDS rasters are large; dense eigensolves require tiling/coarsening.
- D8 single-flow routing can under-represent divergent flow on flats; consider sensitivity checks.
- Channel-threshold choice strongly affects \( \beta_1 \); report threshold sweeps, not a single cutoff.
- Use physically interpretable edge definitions consistently across study regions.
