# Planned-vs-organic ΔH receipt (PR-C of CR-2026-04-26)

| Field | Value |
|---|---|
| Date | 2026-04-26 |
| CR | [CR-2026-04-26-integration-spine-and-bookkeeping](../change-requests/2026-04-26-integration-spine-and-bookkeeping.md) |
| PR | PR-C |
| Status | **Synthetic receipt: shipped.** Live-OSM follow-up: pending Overpass run. |
| Code | `experiments/planned_vs_organic_dH.py`, `kernelcal/urban/spectrum.py`, `kernelcal/urban/synthetic.py` |
| Receipt artifact | `experiments/output/planned_vs_organic_dH.json` |

## 0. TL;DR

The Field Notes 38/39 prediction has *two* components and they fare
*differently* under the synthetic receipt:

* **β₀ disconnection-as-signal: confirmed.**  In ``road_knn`` mode the
  organic fringe layout fragments into 6 connected components vs 1 for
  the regular grid; the same fringe in Euclidean ``knn`` shows 5,
  i.e. the road-aware mode genuinely amplifies disconnection on
  organic fabric.  This is the load-bearing piece of Field Note 107
  and it survives.

* **Spectral entropy direction: refuted, in the sign reported by FN
  38/39.**  Under the normalised spectral entropy diagnostic
  ``H = -Σ p_i log p_i / log k`` (with ``p_i = λ_i / Σ λ_j`` over the
  non-zero spectrum), the regular grid has *higher* normalised
  entropy than the organic fringe in both modes (ΔH_road = +0.059,
  ΔH_knn = +0.013).  Mechanism: clustered fringe layouts produce a
  few sharp dominant eigen-modes per cluster, which are *more*
  spectrally concentrated than the broad, gently-spread spectrum of a
  regular grid.

* **ΔΔH: positive (+0.046), not negative.**  Said differently, the
  road structure amplifies ΔH in the *opposite* direction to the FN
  38/39 prediction.  This is a real result; it updates Field Notes
  38/39 to attach the structural-controller signature to β₀ rather
  than to spectral entropy alone.

The receipt is therefore a **partial confirmation, partial
falsification** of the corpus prediction — exactly the kind of
honest discriminator the receipt was meant to deliver.  See §6 for
recommended downstream changes.

## 1. Setup

Two synthetic CityGraphs, each built in two graph modes:

| Viewport | Layout | n_buildings | k | σ_frac |
|---|---|---|---|---|
| `synthetic_grid.yaml` | Manhattan grid 8×8, 80 m blocks, 5 m jitter | 64 | 8 | 0.05 |
| `synthetic_fringe.yaml` | 6 clusters around branching road network, 800 m extent | 64 | 8 | 0.05 |

Both layouts produced by `kernelcal.urban.synthetic`; spectral
diagnostics computed by `kernelcal.urban.spectrum.spectral_diagnostics`
with σ-matching identical to live `buildings_to_graph`.

Why synthetic for the receipt (and not Tempe / Sonoran fringe live
OSM): the Field Notes 38/39 prediction is fundamentally about graph
structure, not specifically about the Tempe / Cave Creek bboxes.
Doing the receipt on synthetic layouts that explicitly instantiate the
two structural archetypes removes the confound that real Tempe and
real Cave Creek differ in many other ways (population density,
building area, OSM tag completeness).  The live-OSM run remains a
follow-up confirmation against a less-controlled empirical setting;
its result will be appended to this report under §7 once it has
been performed.

## 2. Numbers

Quoted directly from `experiments/output/planned_vs_organic_dH.json`
(committed alongside this report).  Reproducible from a fresh checkout
via:

```bash
python experiments/planned_vs_organic_dH.py \
    --grid-config experiments/configs/synthetic_grid.yaml \
    --fringe-config experiments/configs/synthetic_fringe.yaml \
    --output experiments/output/planned_vs_organic_dH.json
```

### Per-viewport diagnostics

| Viewport | Mode | n_eigvals | β₀ | H (normalised) | H (nats) |
|---|---|---|---|---|---|
| grid     | road_knn | 64 | **1** | 0.9779 | 4.052 |
| grid     | knn      | 64 | **1** | 0.9796 | 4.058 |
| fringe   | road_knn | 64 | **6** | 0.9189 | 3.731 |
| fringe   | knn      | 64 | **5** | 0.9669 | 3.943 |

### Differences

| Quantity | Value | Interpretation |
|---|---|---|
| ΔH (road_knn) | **+0.0590** | Grid has higher normalised entropy than fringe under road-aware mode |
| ΔH (knn)      | **+0.0126** | Same direction under Euclidean k-NN |
| ΔΔH = ΔH_road - ΔH_knn | **+0.0464** | Road-aware mode *amplifies* the difference (in the +direction) |

### β₀ panel (the disconnection-as-signal claim)

| Viewport | road_knn β₀ | knn β₀ | Δ |
|---|---|---|---|
| grid | 1 | 1 | 0 |
| fringe | **6** | 5 | +1 |

## 3. Interpretation

### 3.1 What FN 38/39 predicted

> "Grid layouts are more controller-shaped than organic fabric;
> their Laplacian spectra concentrate on a few low modes; spectral
> entropy is therefore lower for grids."

### 3.2 What the receipt shows

* **Spectral entropy is higher for grids, not lower.**  The mechanism
  visible in the top-50 spectrum block of the JSON: a regular grid
  has *all* eigenvalues clustered near a uniform value (the regular
  grid's spectrum is approximately a 2-D cosine product, so its
  eigenvalues are densely packed) → high normalised entropy.  The
  fringe has six sharp peaks (one per cluster) and very few off-peak
  modes → lower normalised entropy.  The "controller pulling mass to
  a few modes" intuition was correct, but the mechanism that produces
  it is *clustering*, not *regularity*.

* **β₀ flips up under road_knn for fringe but stays at 1 for grid.**
  This is the disconnection-as-signal claim from Field Note 107 in
  numeric form.  The road network in the synthetic fringe doesn't
  bridge all six clusters, and the road-aware k-NN respects that;
  Euclidean k-NN partially papers over the disconnection (β₀ = 5
  rather than 6, because the sparse Euclidean tail link still
  creates one bridge between two close clusters).  This piece of FN
  38/39 / FN 107 is robust.

### 3.3 What this updates

We propose two specific updates for the corpus:

1. **The "controller signature" should be attached to β₀, not to
   spectral entropy.**  β₀ in road_knn mode is monotone in
   "fragmentation" — the right scalar for the disconnection-as-signal
   story.  Spectral entropy mixes in a "concentration vs spread"
   axis that, on these synthetic layouts, points the *other* way
   from naïve expectation.

2. **Use spectral entropy as a *concentration* diagnostic, not as a
   *regularity* diagnostic.**  Lower H means a few modes carry most
   of the spectral mass; this happens equally for "controller-shaped
   regularity" and for "high-cluster-purity organic fabric".  The
   diagnostic is useful, but it is not by itself a grid-vs-organic
   discriminator.

## 4. β₀ as the receipt-grade scalar

A clean restatement of the FN 38/39 prediction that *does* hold in
this receipt:

> Under road-aware k-NN with σ matched to the layout, β₀ of the
> Laplacian is monotone in the organic-ness of the layout: regular
> grids saturate at β₀ = 1; clustered organic fabric returns β₀ ≥
> 2; informal-settlement fabric (not exercised here) is expected to
> climb further.

This is the falsifier worth tracking in PR-F (Bishop→Phoenix splay
receipt) and PR-E (anomaly differential): a viewport whose β₀
*decreases* over time is collapsing topologically; a viewport whose
β₀ *increases* is fragmenting.

## 5. σ-matching verification

`sigma_matched_spectrum_diff` between the two grid modes (road_knn vs
knn) is **0.030** total-variation; between the two fringe modes it is
**0.092** total-variation.  Both are well below the 0.5 threshold
above which the two modes would be sampling structurally incomparable
graphs.  σ-matching as implemented in `_adaptive_sigma` is therefore
adequate for the cross-mode comparison this receipt depends on.

## 6. Acceptance criteria status

| # | Criterion | Status |
|---|---|---|
| C1 | Spectra computed for both viewports in both modes | ✓ — 4 spectra in `experiments/output/planned_vs_organic_dH.json` |
| C2 | σ-matching verified across modes | ✓ — TV distance 0.030 (grid) and 0.092 (fringe), well below 0.5 |
| C3 | Result report committed with explicit ΔH numbers | ✓ — this file |
| C4 | β₀ in fringe road_knn ≥ β₀ in fringe Euclidean (disconnection-as-signal) | ✓ — 6 ≥ 5 |

The original CR (PR-C §C.2) listed C4 as "β₀ in fringe road_knn ≥ β₀
in fringe Euclidean" without specifying strictness; the receipt
shows strict (`6 > 5`) which is the strong form.

## 7. Live-OSM follow-up (pending)

The committed YAML configs `experiments/configs/tempe_grid.yaml` and
`experiments/configs/sonoran_fringe.yaml` are the original CR's
viewports.  The live-OSM run

```bash
python experiments/planned_vs_organic_dH.py --mode live-osm \
    --grid-config experiments/configs/tempe_grid.yaml \
    --fringe-config experiments/configs/sonoran_fringe.yaml \
    --output experiments/output/planned_vs_organic_dH_live.json
```

is **deferred** until executed on a workstation with Overpass
connectivity and an osmnx install.  When run, append the resulting
ΔH / ΔΔH / β₀ numbers as §7.1 below; the synthetic numbers above are
the structural-prediction receipt regardless of whether the live OSM
numbers point in the same direction (and we expect them to: the
synthetic grid was tuned to match Tempe's block-density, and the
synthetic fringe to match the Cave Creek cluster scale).

### 7.1 Live-OSM numbers (TBD)

> _to be filled in by whoever runs the live-OSM mode._

## 8. Reproducibility

* Code: commit hash of this CR's PR-C merge.
* Random seed: `42` (set in both YAML configs).
* Numpy / SciPy: as pinned by `kernelcal/pyproject.toml` (`numpy>=1.24`, `scipy>=1.11`).
* Runtime: 4.25 s on a single-thread CPU (Apr 26 2026 reference machine).

## 9. Cross-references

* Field Note 38 — urban graph baseline (April 13 2026).
* Field Note 39 — graph-construction critique.
* Field Note 107 — `road_knn` mode and OSM bbox APIs.
* CR-2026-04-26 — Integration Spine and Bookkeeping (this PR's CR).
* `tests/test_planned_vs_organic.py` — automated pin of the receipt's β₀ and structure.

