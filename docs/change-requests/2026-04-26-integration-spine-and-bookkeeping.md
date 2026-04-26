# CR-2026-04-26: Integration Spine and Bookkeeping

| Field | Value |
|---|---|
| CR ID | CR-2026-04-26-integration-spine-and-bookkeeping |
| Date | 2026-04-26 |
| Author | J. Das (ASU SESE) |
| Status | **Accepted (with revisions)** |
| Target package | kernelcal |
| Estimated effort | 5–6 weeks of focused work, 6 PRs |
| Reviewers | (TBD; PR-A reviewer should own `kernelcal.fluid`) |

## Revisions accepted on review (2026-04-26)

Six revisions were accepted relative to the originally proposed text;
the body of the CR below is the original.  In order of priority:

1. **PR-A scope expansion: vectorize the fluid solver before lifting to
   multi-component.**  The current `simulate_kernel_fluid`
   (`kernelcal/fluid/dynamics.py`) is implemented with Python loops
   over edges and per-node list comprehensions.  Multiplying that by
   `n_categories` puts the Tempe-scale "<60 s for 100 steps" target
   permanently out of reach.  PR-A therefore now includes a
   sparse-Laplacian vectorization sub-task (~3 days, ~300 LOC) before
   the multi-component lift in §A.2.  Effort estimate revised from
   "~1 week, ~1500 LOC" to "~1.5–2 weeks, ~1800 LOC".

2. **PR-A scope expansion: replace the renormalize-each-step
   conservation hack with a conservative discretization.**  The
   current single-component code enforces `sum(rho) == 1` by
   `rho /= sum(rho)` after every step.  Lifted as-is to
   multi-component, this would silently zero exactly the off-diagonal
   mass that PR-B's ledger is supposed to expose, and `B3`'s closure
   test would pass vacuously.  PR-A's §A.2 must replace this with an
   explicit conservative continuity discretization (upwind or
   gateway-flux); PR-B's `simplex_projection` ledger entry is reserved
   for genuine Bregman/KL projection drift only.

3. **Promote PR-C to Week 1 (parallel with PR-A).**  As §C
   acknowledges, PR-C depends on PR-A only nominally; the empirical
   apparatus already exists in `kernelcal.urban` (`road_knn` mode +
   bbox fetchers shipped in commits f539b2c, 27045ca, ce177c2).  The
   ΔH receipt has been overdue since April 13; gating it on the
   integration spine is a cosmetic dependency and forfeits a free
   piece of empirical evidence for a week.  Sequencing diagram
   updated: PR-C runs in Week 1 alongside PR-A.

4. **Reconcile PR-D with PR-7's already-shipped `CityPriorStack`.**
   PR-7 (commit landing 2026-04-26) shipped a per-class plug-in
   factor registry in `kernelcal.distinction_game.city_priors`, plus
   per-class compatibility tables (`PHX_PARENT_CHILD_COMPATIBLE` and
   friends) that overlap conceptually with PR-D's proposed
   `CategoryMeta.factor_set`.  Resolution: PR-D keeps
   `CategoryMeta.support_graph`, `cadence`,
   `relaxation_time_seconds`; **drops `CategoryMeta.factor_set`** and
   instead extends `CityPriorStack` so each `CityPriorSource` can
   self-filter to entities whose `support_graph` matches the source.
   This avoids a parallel "factor" registry growing beside the typed
   one PR-7 introduced.

5. **PR-B hook target correction.**  The original §B.3 says "Inside
   `fit_kernel_mix`, after computing posteriors..."; in fact
   `fit_kernel_mix` is the MaxCal λ-weight fitter, not the per-claim
   Q_s applier.  The correct hook points are
   `factor_graph.UnaryPerceptualFactor.__init__` (where Q_s rows are
   summed) and `collapse.collapse_scene_graphs` (where per-claim mass
   actually flows into per-category posteriors).  Spec is corrected
   in implementation; the body below intentionally preserves the
   original wording for archival fidelity.

6. **PR-F.2 acceptance criterion F3 reclassified as a result, not a
   gate.**  The original "at least one category exhibits non-monotone
   TE in CR" is a falsifier of P13 H1 dressed as an acceptance
   criterion.  The mitigation in §7 already treats PR-F.1 this way;
   PR-F.2 should be treated identically.  Renamed to
   "F3-result: report TE-vs-CR per category and whether
   non-monotonicity is present"; if the answer is no, that is a
   real result and the PR still merges.

### Additional out-of-scope deferral

* **GTSAM/Hydra continuous-SLAM coupling.**  The `factor_graph.py`
  docstring already mentions this as a future adapter.  Explicitly
  outside the scope of CR-2026-04-26 and any of its constituent PRs;
  belongs in a separate CR after PR-A and PR-B have stabilised the
  categorical / fluid layer.

### Sequencing summary (revised)

| Week | Active PR(s) | Deliverable at end |
|---|---|---|
| 1 | PR-A **and** PR-C in parallel | End-to-end pipeline running on Tempe; planned-vs-organic ΔH receipt |
| 2 | PR-A continued; PR-B starts | Vectorized + conservative multi-component fluid; runtime ledger |
| 3 | PR-D | Canopy + curb substrates; per-category metadata |
| 4 | PR-E | NESS / anomaly / counterfactual endpoints |
| 5 | PR-F | Bishop→Phoenix splay receipt; P13 multi-component diagnostics |

PR-A's effort estimate now spans Weeks 1–2.

### Considered, not adopted

* **Inverting PR-A and PR-B.**  Implementing the journal first
  against the existing single-component fluid would expose the
  renormalize bug as a closure-test failure and motivate the
  discretization redesign with concrete numbers.  Not adopted because
  it doubles the journal-hooking work (single-component first, then
  multi-component) without changing the eventual deliverable.  Kept
  on file for future reference if PR-A's redesign turns out to be
  riskier than estimated.

---

## 0. TL;DR

The four kernelcal modules that should be talking — distinction_game, urban, fluid, and a not-yet-existing ledger — sit next to each other in the same package but share no types. Note 108's heat-map cube has no executable form despite every primitive being in the repo. This CR proposes six PRs that close the gap. The single non-negotiable is PR-A (integration spine); without it every other PR is premature. The single highest-conceptual-leverage move is PR-B (runtime ledger), which makes EIC-2026 enforced by the system rather than by author discipline. The two "receipts" PRs (PR-C, PR-F) close empirical loops that have been on the program since April 13 and April 25 respectively.

## 1. Background

The kernel-dynamics corpus has, since at least April 13 2026, been describing a four-level stack: kernel field theory (P0) → kernel-fluids (P13) → civic continuity equation (Field Note 102) → per-tile categorical heat-map cube (Field Note 108). Field Note 109 documents that lineage explicitly. Every layer is internally consistent on paper.

In code, however:

- `kernelcal.distinction_game` (PR-1, commit 4781102, Field Note 106) ships per-region categorical inference and produces SceneGraph objects.
- `kernelcal.urban` (commits f539b2c, 27045ca, ce177c2, Field Note 107) ships bbox-driven OSM fetchers and the road_knn graph mode.
- `kernelcal.fluid` (companion to P13) ships single-component continuity + momentum on a synthetic 20-node ring-with-chords graph (E0 sanity only).
- `kernelcal.ledger` does not exist; EIC-2026 is enforced by author discipline.
- The substrate bundle from Field Note 108 has only its car slice (via road_knn); canopy and curb substrates are not built.

The three operational modules share no types and there is no end-to-end pipeline from a viewport bbox to a heat-map cube. This CR proposes the work to close that gap and to deliver the empirical receipts the framework has been promising.

## 2. Goal

After this CR is implemented, kernelcal will:

- Run an end-to-end pipeline from a WGS84 bbox to a multi-channel heat-map cube `{H_c(r, t)}_{c, r, t}` on real urban CityGraphs, in ~1 second per timestep on a Tempe-scale viewport.
- Enforce EIC-2026 at runtime via a `kernelcal.ledger` module that records every mass transformation in named accounts.
- Cover the three operationally-named categories from the design corpus (parked cars, trash bins, unhealthy trees) on their natural substrates.
- Emit NESS distributions, anomaly differentials, and counterfactual differential heat-maps as first-class endpoints.
- Have shipped the planned-vs-organic ΔH receipt (Field Notes 38/39, overdue since April 13) and the Bishop→Phoenix splayed-prior receipt (Field Note 96).

## 3. Out-of-scope (deliberately deferred)

- Time-varying / multi-modal road graphs (Field Note 107 X2): rush-hour vs night, walk vs bike vs drive. Additive after PR-E. Defer to a separate CR.
- Drainage / stormwater Laplacian (Field Note 107 X4, Field Note 105): needs DEM flow-accumulation integration. Larger; defer to its own CR after PR-D.
- Trait-conditioned edge weights (Field Note 107 X1): nice-to-have; doesn't unblock anything.
- PR-3 empirical Q_s refit (Field Note 106 forward): R3-rung work owed after PR-F has shown the splay-vs-no-splay effect numerically.
- `k_cross(X, Y | Z)` formal implementation (Field Note 100): requires PR-A and PR-B to be in place; defer.
- (Added on review) **GTSAM/Hydra continuous-SLAM coupling.**  Belongs to a separate CR after the categorical / fluid layer stabilises.

## 4. PR sequence and dependency graph

```
                           PR-A (integration spine)        [BLOCKING]
                          /     |        \      |
                         /      |         \     |
                       PR-B   PR-C       PR-D   |
                       (ledger)(ΔH receipt)   (substrates)
                         |              \      /
                         |               \    /
                         +---------------> PR-E
                                       (NESS / anomaly / counterfactual)
                                            |
                                            v
                                          PR-F
                                       (Bishop→Phoenix + P13 diagnostics)
```

PR-A blocks everything **except PR-C** (see Revisions §3).  PR-B and
PR-C can run in parallel once their respective dependencies are in
place.  PR-D needs PR-A only.  PR-E needs PR-A and PR-D.  PR-F needs
PR-A, PR-B, and PR-D.

---

## PR-A — Integration spine

Scope. Make `distinction_game`, `urban`, and `fluid` one pipeline. Lift the fluid module from single-component to multi-component. Build the heat-map IC from a SceneGraph joined onto a CityGraph.

Estimated effort. ~1.5–2 weeks, ~1800 lines including tests (revised on review; was "~1 week, ~1500 lines").

Files

```
kernelcal/fluid/sparse.py                     NEW   ~300 LOC  (vectorization)
kernelcal/urban/adapter.py                    NEW   ~80 LOC
kernelcal/fluid/multicomponent.py             NEW   ~450 LOC
kernelcal/fluid/__init__.py                   MOD   +30 LOC
kernelcal/distinction_game/heat_map.py        NEW   ~180 LOC
kernelcal/distinction_game/__init__.py        MOD   +10 LOC
kernelcal/pipeline.py                         NEW   ~120 LOC
tests/test_fluid_sparse_vectorization.py      NEW   ~120 LOC  (parity vs. dense)
tests/test_urban_to_fluid_adapter.py          NEW   ~120 LOC
tests/test_multicomponent_fluid.py            NEW   ~250 LOC
tests/test_heat_map_ic_builder.py             NEW   ~100 LOC
tests/test_pipeline_tempe_viewport.py         NEW   ~200 LOC
```

### A.1 — `kernelcal.urban.adapter.to_fluid_graph`

```python
def to_fluid_graph(
    city_graph: CityGraph,
    *,
    use_weighted_lengths: bool = True,
) -> FluidGraph:
    """Adapter from CityGraph (Note 107) to FluidGraph (P13 / kernelcal.fluid).

    Preserves nodes, edges, and (optionally) edge weights from the
    Gaussian-weighted Laplacian as inverse edge_lengths so high-weight
    edges become short paths in the fluid layer.
    """
```

Acceptance. Every CityGraph from `tests/test_urban_road_knn.py` round-trips through `to_fluid_graph` and back into a graph with the same connected-component count.

### A.2 — `kernelcal.fluid.multicomponent`

Multi-component lift of P13 / `simulate_kernel_fluid`. Lift scalar `rho(k, t)` to vector `{rho_c(k, t)}_{c in c*}` plus a `rho_unknown(k, t)` channel.

```python
@dataclass(frozen=True)
class MultiComponentDensity:
    fluid_graph: FluidGraph
    taxonomy: Taxonomy
    rho: np.ndarray               # shape (n_categories, n_nodes)
    rho_unknown: np.ndarray       # shape (n_nodes,)
    t: float

    def simplex_residual(self) -> np.ndarray: ...
    def total_mass_per_category(self) -> np.ndarray: ...


@dataclass(frozen=True)
class MultiComponentSimulationConfig:
    dt: float = 0.05
    viscosity: float = 0.1
    project_simplex_every_step: bool = True
    pressure_per_category: bool = True
    ledger: Optional["LedgerJournal"] = None  # Set in PR-B


def simulate_multicomponent_fluid(
    initial: MultiComponentDensity,
    potentials: Mapping[str, PotentialLandscape],
    config: MultiComponentSimulationConfig,
    n_steps: int,
) -> List[MultiComponentDensity]:
    """Multi-component continuity + momentum on a FluidGraph."""
```

Mathematical content. P13 §2.1, lifted per-species, with the simplex constraint enforced as a Bregman / KL projection at each step.  The continuity discretization MUST be conservative; see Revisions §2.

Acceptance.

- `simplex_residual` is bounded by `1e-9` at every step on the 20-node ring with three categories.
- Mass error per category is bounded by `1e-7 * total_mass_initial` over 1000 steps.
- When all `V_c` are zero and `rho_unknown(0) = 0.5`, the system reaches a uniform fixed point with `rho_c = (1 - 0.5) / n_categories` per node within numerical tolerance.

### A.3 — `kernelcal.distinction_game.heat_map.heat_map_from_scene_graph`

```python
def heat_map_from_scene_graph(
    scene_graph: SceneGraph,
    city_graph: CityGraph,
    taxonomy: Taxonomy,
    *,
    aggregation: Literal["containment", "centroid_nn", "iou_weighted"] = "centroid_nn",
) -> MultiComponentDensity:
    """Build the Note 108 heat-map cube IC at t=0."""
```

Acceptance. Round-trip test: a synthetic SceneGraph with known posteriors per region, joined onto a synthetic 5-node CityGraph, produces a `MultiComponentDensity` whose per-category totals match the synthetic ground truth within `1e-6`.

> **Implementation note (revisions §3 of the schema kind):** the
> `containment` and `iou_weighted` modes require per-node polygon
> footprints, which `CityGraph` does not currently persist in cache.
> Default is `centroid_nn`; the polygon-based modes opt-in and require
> populating `CityGraph.footprints` (added in PR-A).

### A.4 — `kernelcal.pipeline.run_viewport_pipeline`

```python
def run_viewport_pipeline(
    south: float, west: float, north: float, east: float,
    *,
    taxonomy: Taxonomy = PHX_URBAN_V0,
    kernel_sources: Sequence[str] = ("osm", "mr_rocks", "mr_house"),
    n_fluid_steps: int = 100,
    config: MultiComponentSimulationConfig = MultiComponentSimulationConfig(),
    cache_dir: Path = Path("./cache/pipeline"),
) -> ViewportPipelineResult:
    """End-to-end driver."""
```

### A.5 — Smoke test on Tempe

`tests/test_pipeline_tempe_viewport.py`:

- `bbox = (33.4140, -111.9520, 33.4400, -111.9100)` (≈ ASU Tempe campus + Mill Avenue).
- Fetch buildings + roads via existing bbox APIs.
- Build SceneGraph with hand-set MR-rocks / MR-house / OSM claims (synthetic for now).
- Run 100 fluid steps with default config.
- Assert: per-step simplex residual < 1e-6 at all nodes; mass error per category < 1e-5 * initial_mass_c at end; `rho_unknown(t=100)` is non-trivial; pipeline runtime < 60 s on a single-thread CPU (after vectorization sub-task).

### A.6 — Acceptance criteria (PR-A as a whole)

| # | Criterion | How verified |
|---|---|---|
| A0 | Sparse-Laplacian solver matches dense reference within `1e-9` on the 20-node ring | `tests/test_fluid_sparse_vectorization.py` |
| A1 | `to_fluid_graph` round-trips connected components | `tests/test_urban_to_fluid_adapter.py` |
| A2 | Multi-component fluid preserves simplex to `1e-9` | `tests/test_multicomponent_fluid.py` |
| A3 | Multi-component fluid preserves total mass per category to `1e-7` over 1000 steps | `tests/test_multicomponent_fluid.py` |
| A4 | Heat-map IC builder round-trips synthetic posteriors | `tests/test_heat_map_ic_builder.py` |
| A5 | Tempe smoke test runs end-to-end in `< 60` s | `tests/test_pipeline_tempe_viewport.py` |
| A6 | `rho_unknown` is non-trivial at end of Tempe smoke test | same |

---

## PR-B — Runtime ledger (the bookkeeping spine)

Scope. Make EIC-2026 enforced by the system rather than by author discipline. Implement `kernelcal.ledger` and `kernelcal.fluid.afterglow.AfterglowLedger`. Hook into the three places mass moves: Q_s applications, multi-component fluid steps, and gateway flux integration.

Depends on. PR-A.

Estimated effort. ~1 week, ~900 lines including tests.

Files

```
kernelcal/ledger/__init__.py                  NEW   ~30 LOC
kernelcal/ledger/journal.py                   NEW   ~250 LOC
kernelcal/ledger/accounts.py                  NEW   ~100 LOC
kernelcal/ledger/hooks.py                     NEW   ~120 LOC
kernelcal/fluid/afterglow.py                  NEW   ~220 LOC
kernelcal/distinction_game/factor_graph.py    MOD   +30 LOC (UnaryPerceptualFactor hook)
kernelcal/distinction_game/collapse.py        MOD   +30 LOC (collapse hook)
kernelcal/fluid/multicomponent.py             MOD   +40 LOC (hook in)
tests/test_ledger_journal.py                  NEW   ~180 LOC
tests/test_afterglow_ledger.py                NEW   ~150 LOC
tests/test_bookkeeping_closure.py             NEW   ~200 LOC  (synthetic-graph closure)
```

> **Hook target correction (Revisions §5):** the original §B.3 said
> "Inside `fit_kernel_mix`, after computing posteriors..."; the actual
> per-claim Q_s application happens in
> `factor_graph.UnaryPerceptualFactor.__init__` and
> `collapse.collapse_scene_graphs`.  The hooks live there.

### B.1 — `kernelcal.ledger.journal`

```python
@dataclass(frozen=True)
class LedgerEntry:
    t: float
    source_account: str
    dest_account: str
    mass: float
    transform_type: str   # 'q_s_leak' | 'fluid_step' | 'gateway_flux'
                          # | 'simplex_projection' | 'afterglow_decay'
    tier: int             # EIC-2026 tier disposition (1, 2, or 3)
    provenance: dict


@dataclass
class LedgerJournal:
    entries: List[LedgerEntry] = field(default_factory=list)
    def record(self, entry: LedgerEntry) -> None: ...
    def closure_residual(self) -> dict[str, float]: ...
    def to_parquet(self, path: Path) -> None: ...
    @classmethod
    def from_parquet(cls, path: Path) -> "LedgerJournal": ...
```

Account naming convention. Hierarchical, dot-separated:

- `Q_s.<source_id>.<native_label>` for per-source label-semantics accounts.
- `H_<category>` for per-category live mass.
- `H_unknown` for the Goedel-slot channel.
- `nu_t.<category>` for the afterglow ledger per category.
- `gateway.{in,out}.<region_id>` for gateway flux.

### B.2 — `kernelcal.fluid.afterglow.AfterglowLedger`

```python
@dataclass
class AfterglowLedger:
    """Exponentially-decaying mass buffer parallel to the live fabric."""
    fluid_graph: FluidGraph
    taxonomy: Taxonomy
    nu: np.ndarray
    gamma: float
    journal: Optional[LedgerJournal] = None

    def step(self, dt: float, departures: np.ndarray) -> None: ...
    def total_afterglow(self) -> np.ndarray: ...
```

### B.3 — Hooks

`UnaryPerceptualFactor` records per-claim per-source Q_s off-diagonal mass leakage; `simulate_multicomponent_fluid` records per-step source/sink mass and simplex-projection drift.

### B.4 — Synthetic-graph bookkeeping closure test

> **Revised on review:** the originally proposed property test ran the
> full Tempe pipeline under Hypothesis, hitting Overpass and 100-step
> fluid runs per sample.  Replaced with a synthetic 12-node, 4-category
> closure test (~50 steps) under Hypothesis; the Tempe end-to-end stays
> as a single non-property smoke that verifies the journal is non-empty
> and closure-passing.

### B.5 — Acceptance criteria

| # | Criterion | How verified |
|---|---|---|
| B1 | `LedgerJournal` records every Q_s leakage | `tests/test_ledger_journal.py` |
| B2 | `AfterglowLedger` preserves total mass = live + decayed under closure | `tests/test_afterglow_ledger.py` |
| B3 | Synthetic-graph closure: every transformation has a ledger entry | `tests/test_bookkeeping_closure.py` |
| B4 | Parquet round-trip of journal preserves all entries bit-exactly | `tests/test_ledger_journal.py` |
| B5 | Tempe smoke from PR-A produces a non-empty, closure-passing journal when ledger is configured | extends `test_pipeline_tempe_viewport.py` |

---

## PR-C — Planned-vs-organic ΔH receipt (Field Notes 38/39, overdue)

Scope. Run the planned-vs-organic spectral comparison that has been on the program since April 13. Pre-PR-A this was conceivable; pre-Field-Note-107's `road_knn` it was infeasible because Euclidean Laplacians smear across freeways. Now it is one orchestrator call away.

> **Revised on review (Revisions §3):** PR-C is **promoted to Week 1
> alongside PR-A**.  Its dependency on PR-A is purely cosmetic; the
> machinery already exists in `kernelcal.urban`.

Depends on. None operationally; uses `kernelcal.urban` as it ships today.

Estimated effort. ~2 days, ~250 lines + a result report.

Files

```
experiments/planned_vs_organic_dH.py          NEW   ~250 LOC
experiments/configs/tempe_grid.yaml           NEW   ~30 LOC
experiments/configs/sonoran_fringe.yaml       NEW   ~30 LOC
docs/results/2026-04-26-planned-vs-organic-dH.md   NEW   ~200 LOC report
```

### C.1 — Experiment

Two viewports, both `road_knn` mode with σ matched per Field Note 107:

- **Tempe grid**: bbox `(33.420, -111.945, 33.460, -111.905)`. Phoenix-grid orthogonal block structure.
- **Sonoran fringe**: bbox `(33.55, -112.10, 33.59, -112.06)` (Cave Creek / Phoenix Mountain Preserve fringe). Organic, branching street structure.

For each:

- Fetch buildings + road graph.
- Build CityGraph (`graph_mode='road_knn'`).
- Compute Laplacian spectrum: top-50 eigenvalues, β₀, spectral entropy `H = -sum p_i log p_i` where `p_i = lambda_i / sum lambda_j`.
- Repeat with `graph_mode='knn'` (Euclidean baseline) to confirm σ-matching makes the spectra comparable across modes.

Report: `H_grid - H_fringe` in `road_knn` mode (predicted positive: grids are more regular); same in Euclidean mode (the smearing baseline); the difference of differences.

### C.2 — Acceptance criteria

| # | Criterion | How verified |
|---|---|---|
| C1 | Spectra computed for both viewports in both modes | `experiments/planned_vs_organic_dH.py` produces 4 spectra |
| C2 | σ-matching verified across modes | test in `tests/test_urban_road_knn.py` extended |
| C3 | Result report committed with explicit ΔH numbers | `docs/results/2026-04-26-planned-vs-organic-dH.md` |
| C4 | β₀ in fringe `road_knn` ≥ β₀ in fringe Euclidean (disconnection-as-signal) | reported in C3 |

Falsifier. If `H_grid ≈ H_fringe` in `road_knn` mode, the Field Notes 38/39 prediction fails and a separate post-mortem CR is filed.

---

## PR-D — Substrate bundle build-out (canopy + curb)

Scope. Extend the substrate bundle from `{S_car}` to `{S_car, S_tree, S_trash}` so heat maps for the three operationally-named categories can run on their natural substrates. Add per-category metadata to `Taxonomy`.

> **Revised on review (Revisions §4):** drop `CategoryMeta.factor_set`;
> reconcile with `kernelcal.distinction_game.city_priors` from PR-7
> (already shipped).  `CityPriorStack` is extended so each
> `CityPriorSource` self-filters to entities whose `support_graph`
> matches the source.

Depends on. PR-A.

Estimated effort. ~1 week, ~700 lines + tests.

Files

```
kernelcal/urban/canopy.py                     NEW   ~320 LOC
kernelcal/urban/curb.py                       NEW   ~170 LOC
kernelcal/distinction_game/taxonomy.py        MOD   +60 LOC (CategoryMeta, no factor_set)
kernelcal/distinction_game/city_priors.py     MOD   +40 LOC (support_graph filter)
tests/test_canopy.py                          NEW   ~140 LOC
tests/test_curb.py                            NEW   ~100 LOC
tests/test_taxonomy_meta.py                   NEW   ~80 LOC
tests/test_city_priors_support_graph.py       NEW   ~80 LOC
```

### D.1 — `kernelcal.urban.canopy.fetch_canopy_partition_bbox`

```python
def fetch_canopy_partition_bbox(
    south: float, west: float, north: float, east: float,
    *,
    ndvi_source: Literal["sentinel2", "naip", "user_raster"] = "sentinel2",
    ndvi_threshold: float = 0.4,
    min_polygon_area_m2: float = 4.0,
    cache_dir: Path = Path("./cache/canopy"),
) -> CanopyGraph:
    """Build the canopy substrate (the SAM-vacuum slice for trees)."""
```

### D.2 — `kernelcal.urban.curb.fetch_curb_partition_bbox`

```python
def fetch_curb_partition_bbox(
    south: float, west: float, north: float, east: float,
    *,
    road_buffer_m: float = 5.0,
    cache_dir: Path = Path("./cache/curb"),
) -> CurbGraph:
    """Build the curb substrate (the SAM-vacuum slice for trash bins
    and street vendors)."""
```

### D.3 — `Taxonomy.CategoryMeta`

```python
@dataclass(frozen=True)
class CategoryMeta:
    name: str
    support_graph: Literal["road_knn", "canopy_knn", "curb_knn", "drainage", "any"]
    cadence: Literal["streaming", "hourly", "daily", "weekly", "annual"]
    relaxation_time_seconds: float
```

(`factor_set` dropped on review; see Revisions §4.)

`PHX_URBAN_V0` is updated to include `CategoryMeta` for each of its 10 categories.

### D.4 — Acceptance criteria

| # | Criterion | How verified |
|---|---|---|
| D1 | Canopy partition produces non-trivial polygons on a Tempe NDVI tile | `tests/test_canopy.py` with cached NDVI |
| D2 | Canopy footprint does not overlap building footprint | same |
| D3 | Curb partition produces non-empty intersections on Mill Ave | `tests/test_curb.py` |
| D4 | `PHX_URBAN_V0` carries `CategoryMeta` for all 10 categories | `tests/test_taxonomy_meta.py` |
| D5 | Heat-map IC builder routes mass correctly per category support | extends A.3 tests |
| D6 | `CityPriorSource.factors_for` self-filters by `support_graph` | `tests/test_city_priors_support_graph.py` |

---

## PR-E — NESS, anomaly, counterfactual endpoints

Scope. Deliver the operational endpoints a civic-twin deployment actually queries.

Depends on. PR-A, PR-D.

Estimated effort. ~1 week, ~750 lines.

Files

```
kernelcal/fluid/ness.py                       NEW   ~210 LOC
kernelcal/fluid/anomaly.py                    NEW   ~120 LOC
kernelcal/distinction_game/unknown.py         NEW   ~210 LOC
kernelcal/distinction_game/counterfactual.py  NEW   ~270 LOC
tests/test_ness_estimator.py                  NEW   ~140 LOC
tests/test_anomaly_differential.py            NEW   ~100 LOC
tests/test_unknown_classifier.py              NEW   ~150 LOC
tests/test_counterfactual.py                  NEW   ~180 LOC
```

### E.1 — `kernelcal.fluid.ness.estimate_ness`

```python
def estimate_ness(
    history: List[MultiComponentDensity],
    *,
    periods: Sequence[Literal["daily", "weekly", "seasonal"]] = ("daily", "weekly"),
) -> NESSDistribution:
    """Time-averaged H_c with periodic Fourier decomposition."""
```

### E.2 — `kernelcal.fluid.anomaly.differential`

```python
def anomaly_differential(
    current: MultiComponentDensity,
    ness: NESSDistribution,
    *,
    significance: Literal["zscore", "kl", "tv"] = "zscore",
) -> AnomalyMap:
    """H_c(r,t) - <H_c>_T(r, phase(t)) with significance scoring."""
```

### E.3 — `kernelcal.distinction_game.unknown.classify`

```python
class UnknownMode(Enum):
    M1_LOCALIZED_NOVEL = "tall_narrow"
    M2_DOMAIN_SHIFT = "tall_broad"
    M3_UNNAMED_RECURRENT = "periodic_structure"


def classify_unknown(
    rho_unknown_history: List[np.ndarray],
    fluid_graph: FluidGraph,
) -> List[UnknownEvent]:
    """Detect the three operational modes of H_unknown."""
```

### E.4 — `kernelcal.distinction_game.counterfactual.intervene`

```python
def intervene(
    scene_graph: SceneGraph,
    city_graph: CityGraph,
    intervention: Intervention,
    *,
    n_fluid_steps: int = 100,
) -> DifferentialHeatMap:
    """Perturb G_t by intervention; return the differential heat map
    delta_H_c(r) = H_c^after - H_c^before."""
```

### E.5 — Acceptance criteria

| # | Criterion | How verified |
|---|---|---|
| E1 | NESS estimate stable across two halves of a synthetic 10000-step history (run on the 20-node reference graph) | `tests/test_ness_estimator.py` |
| E2 | Injected anomaly is detected with significance > 3σ | `tests/test_anomaly_differential.py` |
| E3 | M1/M2/M3 classifier hits >80% accuracy on labelled synthetic events | `tests/test_unknown_classifier.py` |
| E4 | Counterfactual differential is non-trivial when adding a node, zero when intervention is a no-op | `tests/test_counterfactual.py` |
| E5 | Counterfactual respects the simplex closure pre-and-post intervention | same |

---

## PR-F — Bishop→Phoenix splay receipt + P13 multi-component diagnostics

Scope. Two empirical receipts that close loops the corpus has been promising.

Depends on. PR-A, PR-B, PR-D.

Estimated effort. ~3 days, ~400 lines + result reports.

Files

```
experiments/bishop_to_phoenix_splay.py        NEW   ~200 LOC
experiments/p13_multicomponent_diagnostics.py NEW   ~180 LOC
kernelcal/fluid/diagnostics.py                NEW   ~150 LOC (TE/CR/WI lifted to multi-component)
docs/results/2026-04-XX-bishop-phoenix-splay.md NEW
docs/results/2026-04-XX-p13-multicomponent-tempe.md NEW
```

### F.1 — Bishop→Phoenix splay validation

Two pipeline runs on the same Phoenix viewport:

| Run | `Q_s.mr_rocks` | Expected `H_unknown` |
|---|---|---|
| F.1a | Bishop-fit prior (no splay): `P(rock | building) ≈ 0.0` | M2 mode fires |
| F.1b | Splayed prior (Field Note 96): `P(rock | building) ≈ 0.45` | M2 mode suppressed |

Report the M2 anomaly mass under each condition.

### F.2 — P13 multi-component diagnostics on Tempe

Implement multi-component TE/CR/WI in `kernelcal.fluid.diagnostics`. Run on Tempe `road_knn` CityGraph with sweep over adhesion / noise (P13 E1 controls) per category.

### F.3 — Acceptance criteria

| # | Criterion | How verified |
|---|---|---|
| F1 | Bishop-prior run produces M2 anomaly > 3× splay-prior run on Phoenix | `experiments/bishop_to_phoenix_splay.py` + report |
| F2 | TE-vs-CR curve produced per category for Tempe | `experiments/p13_multicomponent_diagnostics.py` + report |
| **F3-result** | **Report TE-vs-CR per category and whether non-monotonicity (P13 H1) is present.  Either outcome is a real result.** | report numeric (revised on review; was acceptance criterion) |
| F4 | All result reports committed under `docs/results/` | git log |

---

## 5. Sequencing summary (revised)

| Week | Active PR(s) | Deliverable at end |
|---|---|---|
| 1 | PR-A starts; **PR-C in parallel** | Vectorized + conservative fluid solver landed; planned-vs-organic ΔH receipt shipped |
| 2 | PR-A continues; PR-B starts | Multi-component fluid + Tempe smoke test; runtime ledger |
| 3 | PR-D | Canopy + curb substrates; per-category metadata; `CityPriorStack` substrate filter |
| 4 | PR-E | NESS / anomaly / counterfactual endpoints |
| 5 | PR-F | Bishop→Phoenix splay receipt; P13 diagnostics on Tempe |

After Week 5, the "kernel-fluids lineage" of Field Note 109 is implemented end-to-end and the framework has shipped four empirical receipts (PR-A Tempe smoke, PR-C ΔH, PR-F.1 splay, PR-F.2 P13 diagnostics).

## 6. Cross-references

- Field Note 38 — urban graph baseline (April 13)
- Field Note 39 — graph-construction critique
- Field Note 96 — Bishop / Phoenix / Las Vegas rock-detector superclass
- Field Note 100 — source-conditioned PGMs and affordances
- Field Note 102 — civic continuity equation
- Field Note 106 — `distinction_game` PR-1 scaffold
- Field Note 107 — urban `road_knn` and OSM bbox APIs
- Field Note 108 — heat maps as state vector of civic hydrodynamics
- Field Note 109 — kernel-fluids lineage (the upstream context)
- P0-misc-infogeo-kernel-field-theory-tech-report/main.tex — kernel field equation, Landauer speed limit, representational afterglow
- P13-journal-Kernel Rivers, Lakes, and Granular-to-Fluid Dynamics/main.tex — abstract single-component fluid layer
- **PR-7 (this repo, 2026-04-26) — `CityPriorStack` + new factor types** (`UnaryClassPriorFactor`, `UnaryGroundElevationFactor`, `PairwiseParentChildFactor`).  PR-D consolidates with this rather than introducing a parallel registry.

## 7. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Multi-component simplex projection introduces numerical drift | Medium | Bregman projection with explicit per-step tolerance check; ledger captures the drift |
| `kernelcal.fluid` E0 ring-with-chords assumptions don't generalise to 10⁴-node CityGraphs | **High → Mitigated by Revisions §1** | PR-A vectorizes the solver via sparse Laplacians before the multi-component lift; sparse path is the default for PR-A.5 onwards |
| Sentinel-2 / NAIP NDVI fetcher requires API keys not yet in CI | High | Cache NDVI tiles in `cache/canopy/` with committed test fixtures; mock the fetch in CI |
| Property-based bookkeeping closure test is too strict / too slow | **Mitigated by Revisions B.4** | Closure test runs on a synthetic 12-node graph; Tempe smoke is a single non-property test |
| PR-F.1 splay receipt fails | Low–Medium | Treated as a real result; updates Field Note 96 |
| Renormalize-each-step conservation hack survives into multi-component | **Mitigated by Revisions §2** | PR-A explicitly replaces it with a conservative discretization; PR-B's `simplex_projection` ledger entry is reserved for genuine projection drift |

## 8. Decision asks (resolved)

| Ask | Resolution |
|---|---|
| Approve PR sequencing | **Approved with revisions** (vectorize, promote PR-C, reconcile PR-D with PR-7) |
| Confirm `docs/change-requests/` as canonical CR location | **Approved.** This file establishes the convention; see `INDEX.md` |
| Reviewer for PR-A | Owner of `kernelcal.fluid` (TBD) — they pay the bill on the discretization redesign |
| Confirm out-of-scope items | **Approved.** Plus GTSAM/Hydra coupling explicitly deferred |

End of CR-2026-04-26-integration-spine-and-bookkeeping.
