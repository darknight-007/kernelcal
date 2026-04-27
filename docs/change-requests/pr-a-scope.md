# PR-A scope: integration spine + vectorized multi-component fluid solver

| Field | Value |
|---|---|
| CR | [CR-2026-04-26-integration-spine-and-bookkeeping](./2026-04-26-integration-spine-and-bookkeeping.md) |
| PR | PR-A |
| Status | A.0, A.1 complete (committed); A.2–A.5 pending |
| Estimated effort | 1.5–2 weeks (revised on review of CR), ~1800 LOC including tests |
| Reviewer | Owner of `kernelcal.fluid` (TBD) |

This scope doc decomposes PR-A into sub-tasks with explicit
dependencies, surface-area estimates, and acceptance criteria.  It
exists because the original CR proposed PR-A at "~1 week, ~1500 LOC"
without acknowledging that the existing single-component fluid solver
in `kernelcal/fluid/dynamics.py` cannot be lifted to multi-component
without first being vectorized.

## 0. Why this is bigger than the CR initially said

Two structural problems with the current solver:

1. **Time complexity.**  `simulate_kernel_fluid` has Python `for`
   loops over edges and per-node list comprehensions.  Reading the
   actual code:

   * `_edge_laplacian_term` calls `np.mean([u[i, k] for k in nbr_i])`
     per edge per step — a Python list-comp inside a Python loop.
   * The continuity update is
     `drho[i] = -np.sum([F[i, j] for j in graph.adjacency[i]])` —
     same shape.

   At the 20-node reference graph this is fine; at a Tempe viewport
   (`n ~ 1000`, `E ~ 8000`) this is roughly `O(steps × E × deg) ~
   100 × 8000 × 8 ~ 6.4M Python-level operations`, blowing past the
   "~1 second per timestep" budget the CR's §2 commits to.  Lifting
   this to multi-component (multiplying by `n_categories ~ 10`)
   makes the CR's "<60 s for 100 steps" target permanently
   unreachable.

2. **Mass-conservation contract is violated.**  The current solver
   applies `rho = np.maximum(rho, config.rho_floor)` followed by
   `rho /= float(np.sum(rho))` after the continuity step.  Both
   of those silently absorb mass:

   * `np.maximum(rho, rho_floor)` adds mass at any node where the
     unsmoothed continuity update produced `rho < rho_floor`.
   * `rho /= sum(rho)` then renormalises away both that added
     mass *and* any genuine drift the discretisation produced.

   PR-B's ledger is supposed to record every transformation; lifted
   as-is to multi-component, the renormalisation step would silently
   zero exactly the off-diagonal mass the ledger is meant to expose,
   and PR-B's `B3` closure test would pass vacuously.

The fix for both is the same: a sparse-Laplacian, conservative
discretisation that lives in `kernelcal/fluid/sparse.py` and is the
default solver from PR-A.5 onwards.

## 1. Sub-task breakdown

### A.0 Sparse-Laplacian solver (NEW, blocks A.2) — **DONE**

Files: `kernelcal/fluid/sparse.py` (~390 LOC, shipped),
`tests/test_fluid_sparse_vectorization.py` (~370 LOC, 22 tests, all pass).

Effort: shipped in ~1 day on top of the scoped 3-day budget.

Status: see commit `kernelcal: …` (PR-A.0 of CR-2026-04-26).  Two
design refinements were made on review of the implementation against
the spec; both are recorded here so the spec and shipped code agree.

Replace per-step Python loops with sparse linear algebra.  Concretely:

1. **Edge-indexed state.**  Replace `u` of shape `(n, n)` and `F` of
   shape `(n, n)` (each ≈99% zero on a real graph) with `u_e` and
   `F_e` of shape `(E,)`.  Sign convention: for canonical edge
   `e = (i, j)` with `i < j`, `u_e > 0` means "flow from `i` to
   `j`".

2. **Sparse incidence matrix.**  Build once at `FluidGraph`
   construction time: `B` of shape `(E, n)`, `B[e, j] = +1`,
   `B[e, i] = -1` for edge `e = (i, j)`, `i < j`.  Stored as a
   `scipy.sparse.csr_matrix`.

3. **Vectorised gradients.**  ``grad_p_per_edge = (B @ p) /
   edge_lengths`` (length-`E`).  Same for `grad_phi`.

4. **Vectorised edge-Laplacian smoothing.**  The current
   `_edge_laplacian_term` averages neighbour-edge flux at each
   endpoint.  Implement as
   ```
   incident_avg = (|B|.T @ u_e) / node_degree   # shape (n,)
   u_lap = incident_avg[B.T_dst] + incident_avg[B.T_src] - 2.0 * u_e
   ```
   where `|B|` is the absolute value (incidence pattern).  Closed
   form, no Python loops.

5. **Vectorised continuity.**  `drho = -B.T @ F_e` is the discrete
   divergence of flux.  By construction
   `sum(drho) = sum(B.T @ F_e) = sum(B @ 1)·F_e = 0` -- conservation
   is exact, not approximate.

6. **Log, do not drop, the floor + renormalise hack** (revised
   on implementation review).  The original spec said "drop
   renormalisation".  In practice removing both the floor and the
   renormalise step makes the basic explicit-Euler discretisation
   numerically unstable on the 20-node reference at the existing
   `dt = 0.01` — `rho` goes mildly negative, edge fluxes amplify, and
   the simulation diverges within ~50 steps.  The legacy solver was
   being silently stabilised by the floor + renormalise pair.  The
   shipped sparse solver therefore **keeps** both operations but
   **logs** them: every step records
   `floor_mass_inserted[t] = sum(max(rho_floor - rho, 0))` and
   `renormalize_correction[t] = sum(rho_after_floor) - 1.0`.  These
   are the ledger signals PR-B's runtime ledger consumes; PR-B's
   §B3 closure test will balance against them rather than against a
   silently zero residual.  Two new fields on
   `FluidSimulationResult` (`floor_mass_inserted`,
   `renormalize_correction`, both `(steps,)` arrays) carry the
   signal; the legacy solver was updated to populate them too.

7. **Update sweep semantics: Jacobi, not Gauss-Seidel** (added on
   implementation review).  The legacy solver's per-edge Python
   loop was accidentally Gauss-Seidel: each edge's update saw the
   just-updated state of edges processed earlier in the sweep.
   That semantics does not vectorise.  The sparse solver is
   explicitly Jacobi (one pre-step snapshot of `u`, all derived
   quantities computed from the snapshot, atomic write at end of
   step).  Jacobi and Gauss-Seidel agree at stationary state and
   to `O(dt)` on transients; tests verify the attractor matches
   legacy within 1% L1 after long-time integration.

8. **Diagnostics.**  Re-implement `flux_to_node_*`, `dissipation`,
   `entropy`, `concentration_m2`, `mass_error` against the new
   edge-indexed arrays.  `mass_error` is computed *after* the
   renormalise step (same convention as legacy) so it stays below
   `1e-12`.

Acceptance criteria as shipped:

* **A0a (operator parity).**  `edge_gradient`,
  `node_signed_inflow`, `edge_laplacian_smoothing`, `edge_flux`,
  and `continuity_drho` each match a brute-force reference within
  `1e-12` on randomly-sampled inputs over the 20-node reference
  graph.  ✓
* **A0b (mass conservation post-renorm).**  `mass_error` stays
  below `1e-12` over 500 steps on the 20-node reference, and
  `continuity_drho(F_edge)` sums to zero within `1e-12` for any
  flux profile.  ✓
* **A0c (ledger signals are exposed).**
  `floor_mass_inserted` and `renormalize_correction` are both
  `(steps,)` arrays of finite values; on a well-conditioned
  problem (small `dt`) the floor never triggers and the renorm
  correction is below `1e-9`.  ✓
* **A0d (attractor parity vs legacy).**  After 4000 steps at
  `dt=0.005` on the 20-node reference, sparse and legacy reach
  the same stationary distribution within `1e-2` L1.  ✓
* **A0e (performance).**  Sparse solver is ≥ 5× faster than
  legacy on the 20-node reference for 200 steps.  Measured 6.7×
  (legacy ≈170 ms, sparse ≈25 ms).  At `n=200, E≈400`, sparse runs
  200 steps in ≈36 ms — well under the CR §2 "~1 second per
  timestep" budget extrapolated to a Tempe viewport.  ✓

The original A0 ("`1e-9` pointwise parity vs legacy") was found to
be mathematically impossible: Jacobi and Gauss-Seidel sweeps
necessarily disagree at `O(dt)` per step, so over 2000 steps at
`dt=0.01` the disagreement is `O(20)` -- 1e-9 parity would require
either replicating the legacy Gauss-Seidel sweep (which doesn't
vectorise) or running at `dt < 1e-13` (which doesn't terminate).
A0d above is the honest replacement: same fixed point, not same
trajectory.

### A.1 `kernelcal.urban.adapter.to_fluid_graph` — **DONE**

Files: `kernelcal/urban/adapter.py` (~155 LOC, shipped),
`tests/test_urban_to_fluid_adapter.py` (~330 LOC, 18 tests, all
pass).

Effort: shipped in ~1 hour; the actual time went to one
non-trivial design call described below, not the mechanical adapter
code.

Edge-length convention as scoped: ``edge_lengths = 1 / max(W_ij,
weight_floor)`` with ``weight_floor=1e-6`` default; high-weight
(close in substrate) edges produce short fluid-domain edges.

Acceptance criterion **A1** as shipped: every CityGraph round-trips
through ``to_fluid_graph`` into a ``FluidGraph`` with the same
**topological** β₀ (count of connected components of the ``W > 0``
adjacency).  The original spec said "same connected-component
count" without specifying topological vs spectral; in
``road_knn`` mode these can legitimately differ when two clusters
are bridged by a numerically tiny weight that is below the
spectral ``betti_zero`` tolerance but still represents a valid
graph edge.  The adapter, by construction, creates exactly one
edge per positive ``W_ij``, so topological β₀ is preserved
exactly.  The companion test
``test_topological_and_spectral_can_diverge_in_road_knn`` documents
the divergence and pins the inequality (spectral β₀ ≥ topological
β₀) so the choice is auditable.

Helper exported alongside the adapter:
:func:`fluid_graph_connected_components` builds the topological β₀
of any FluidGraph from its ``adjacency_mask`` via
``scipy.sparse.csgraph.connected_components``.  PR-B's ledger
closure test and PR-A.5's pipeline smoke can reuse it without
re-deriving the adjacency.

End-to-end smoke included in the test file: a grid CityGraph and
a fringe CityGraph each round-trip into a (FluidGraph,
SparseFluidGraph) pair and run ``simulate_kernel_fluid_sparse`` for
20–50 steps with mass conservation at machine epsilon.

### A.2 Multi-component lift (UPDATED to depend on A.0)

Files: `kernelcal/fluid/multicomponent.py` (~450 LOC),
`tests/test_multicomponent_fluid.py` (~250 LOC).

Effort: ~3 days.  Depends on A.0 (sparse solver) being merged.

State shapes:

* `rho` of shape `(C, n)` with `C = len(taxonomy.categories)`.
* `rho_unknown` of shape `(n,)`.
* `u` of shape `(C, E)`.
* `phi` (per-category) of shape `(C, n)`.

Per-step inner loop is *exactly* the A.0 loop, broadcast over
category axis.  No Python `for c in range(C)` — broadcast in numpy.

Simplex projection: at each step, project `(rho, rho_unknown)` onto
the simplex `sum_c rho_c + rho_unknown = 1` per node via
Bregman/KL projection.  Drift between pre-projection and
post-projection is the genuine `simplex_projection` event PR-B logs;
this is *not* the silent renormalisation hack of the legacy solver.

Acceptance criteria:

* **A2** simplex residual `< 1e-9` per step on 20-node ring with 3 categories.
* **A3** mass error per category `< 1e-7 × initial_mass_c` over 1000 steps.
* **A2-extra (new)** with `V_c = 0` and `rho_unknown(0) = 0.5`, the
  fixed point is `rho_c = 0.5 / n_categories` everywhere.

### A.3 `heat_map_from_scene_graph` (UPDATED for missing footprints)

Files: `kernelcal/distinction_game/heat_map.py` (~180 LOC),
`tests/test_heat_map_ic_builder.py` (~100 LOC).

Effort: ~1.5 days.

The current `CityGraph` does *not* persist per-node polygon
footprints — only centroids.  Three modes:

* `centroid_nn` (default, no extra data): each scene-graph region
  contributes its full posterior to the CityGraph node nearest its
  centroid (kd-tree query).  This is what works against the
  `CityGraph` schema today.
* `containment` (opt-in): requires `CityGraph.footprints` populated
  (added under A.3 as an optional attribute).  Falls back to
  `centroid_nn` with a `RuntimeWarning` when missing.
* `iou_weighted` (opt-in): same.

A SceneGraph region whose centroid is more than `r_max` from any
CityGraph node lands in the `rho_unknown` channel — formalises the
"Goedel-slot" channel of FN 102.

Acceptance criterion **A4**: round-trip a synthetic SceneGraph with
known posteriors per region onto a synthetic 5-node CityGraph;
per-category totals match within `1e-6`.

### A.4 `kernelcal.pipeline.run_viewport_pipeline` (unchanged from CR)

Files: `kernelcal/pipeline.py` (~120 LOC).

Effort: ~1 day.

Wires together: `fetch_buildings_bbox` → `fetch_road_graph_bbox` →
`buildings_to_graph_via_roads_from_bbox` → `to_fluid_graph` →
`heat_map_from_scene_graph` → `simulate_multicomponent_fluid`.

### A.5 Tempe smoke test (unchanged from CR)

Files: `tests/test_pipeline_tempe_viewport.py` (~200 LOC).

Effort: ~1 day.

Cached OSM fixtures committed under `tests/fixtures/tempe_pipeline/`
to avoid live Overpass calls in CI.  Live-OSM run lives in
`experiments/` not `tests/`.

Acceptance criteria **A5**, **A6** as in CR.

## 2. Total effort and timeline

| Sub-task | Effort | Depends on |
|---|---|---|
| A.0 sparse solver | 3 days | -- |
| A.1 adapter | 0.5 day | -- |
| A.2 multi-component lift | 3 days | A.0 |
| A.3 heat-map IC | 1.5 days | A.2 (for `MultiComponentDensity`) |
| A.4 pipeline driver | 1 day | A.1, A.2, A.3 |
| A.5 Tempe smoke test | 1 day | A.4 |
| Buffer + review-loop | 1 day | -- |
| **Total** | **~11 working days** | -- |

Two-week PR-A is realistic; the original "~1 week" estimate was not.

## 3. Out-of-scope for PR-A (deferred)

* **GPU sparse mat-vec.**  The sparse solver targets CPU.  GPU-level
  speedups via cupy/jax are deferred to a separate PR after PR-E.
* **Implicit time integration.**  The current explicit Euler scheme
  is preserved; Crank–Nicolson / SDIRK lifts are deferred.
* **DEM-flow drainage Laplacian** (already deferred in CR §3).
* **Time-varying / multi-modal road graphs** (already deferred in CR §3).

## 4. Open questions to resolve before implementation

1. **Legacy solver disposition.**  Once A.0 ships, do we keep
   `simulate_kernel_fluid` as a deprecated alias for one cycle, or
   drop it immediately?  Recommendation: keep with a
   `DeprecationWarning` for one minor version, then remove in
   the version after PR-B.  This preserves any external scripts in
   `experiments/` that imported it.

2. **Edge-length convention in `to_fluid_graph`.**  The CR proposes
   `1 / max(W_ij, eps)`.  An alternative is to keep the raw network
   distance from `road_meta` (when available), which preserves
   physical units.  Recommendation: use the network distance when
   `graph_mode='road_knn'`, fall back to `1 / W_ij` when
   `graph_mode='knn'`.  Document the choice in `to_fluid_graph`'s
   docstring.

3. **Where does `rho_unknown` live in the public API?**  Inside
   `MultiComponentDensity` as a separate field (CR's choice), or as
   a special category `'unknown'` added to the taxonomy?
   Recommendation: separate field — keeps `taxonomy.categories` as
   the named taxonomy and `rho_unknown` as the explicit Goedel-slot
   channel that PR-E's M1/M2/M3 classifier can target without
   string-matching.

## 5. Acceptance criteria summary (PR-A as a whole)

| # | Criterion | Verified by |
|---|---|---|
| **A0** | Sparse solver matches legacy within `1e-9` on the 20-node ring | `tests/test_fluid_sparse_vectorization.py` |
| A1 | `to_fluid_graph` round-trips connected components | `tests/test_urban_to_fluid_adapter.py` |
| A2 | Multi-component fluid preserves simplex to `1e-9` | `tests/test_multicomponent_fluid.py` |
| A3 | Multi-component fluid preserves total mass per category to `1e-7` over 1000 steps | `tests/test_multicomponent_fluid.py` |
| A4 | Heat-map IC builder round-trips synthetic posteriors | `tests/test_heat_map_ic_builder.py` |
| A5 | Tempe smoke test runs end-to-end in `< 60` s | `tests/test_pipeline_tempe_viewport.py` |
| A6 | `rho_unknown` is non-trivial at end of Tempe smoke test | same |

A0 is the key new gate — without it, A5's runtime budget is
unattainable.

## 6. Cross-references

* CR-2026-04-26 §A — original PR-A scope.
* CR-2026-04-26 Revisions §1, §2 — vectorization and conservative discretisation.
* `kernelcal/fluid/dynamics.py` — current single-component solver.
* `kernelcal/urban/city_graph.py` — `CityGraph` schema (no footprints today).
