"""
Per-source confusion matrices ``Q_s(ŷ_s | c)``.

Per §3.0 / §4 of the design doc, each source's *label semantics* live
in a small confusion matrix that is **fitted, not asserted**:

.. math::

    Q_s[i, j]  =  P\\big(\\hat y_s = y_i \\mid c = c_j\\big)

with rows indexed by the source's native vocabulary ``Y_s`` (e.g.
``["rock"]`` for MR-rocks, ``["house_undamaged", "house_damage_0", …]``
for MR-house) and columns indexed by the shared taxonomy ``c*``
(:class:`~kernelcal.distinction_game.taxonomy.Taxonomy`). Each column
sums to 1.

PR-1 ships **hand-coded priors only** — written down from domain
knowledge so the rest of the pipeline has something concrete to fuse.
PR-3 adds a real :func:`fit_q_s` that consumes ``(claim, true_category)``
pairs (the §8.1 R1 rung — "relabel only, no GPU"). The hand-coded
priors are deliberately *honest*:

* MR-rocks gets the **splayed prior** of design-doc §3.4 + Figs.
  0.7a/b — ``Q_MR_rocks(rock | building) ≈ 0.45`` and
  ``Q_MR_rocks(rock | debris) ≈ 0.85``, encoding the empirical
  observation that the rocks kernel fires confidently on Phoenix
  rooftops.
* MR-house gets a *peaked* prior on building because the Eureka
  taxonomy decomposes a single super-class (built structure) into
  damage levels.
* OSM gets near-deterministic priors on its anchor categories
  (``building``, ``road``, ``tree``, ``water``) — these are the §4
  OSM-anchor constraints.
* Grounding-DINO is keyed by the text phrase and gets a phrase-
  appropriate peak.
* SAM2 (class-agnostic) gets a uniform prior — per design-doc §3.1,
  SAM contributes to the *region set*, not to category logits, so
  ``Q_SAM`` should be uninformative until a phrase is attached.

These priors are tuned to be *honest about ignorance*: every column
keeps a small mass on ``unknown`` so a low-confidence claim cannot
collapse the posterior. PR-3 will replace whatever's most wrong with
empirical fits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .taxonomy import PHX_URBAN_V0, Taxonomy


# ---------------------------------------------------------------------------
# ConfusionMatrix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfusionMatrix:
    """A per-source confusion matrix ``Q_s(ŷ_s | c)``.

    The matrix has shape ``(|Y_s|, |c*|)`` with **column-stochastic**
    columns (``Q[:, j].sum() == 1`` for every category ``j``). Row
    labels are stored separately so the source can keep a stable native
    vocabulary even when ``c*`` evolves.

    Attributes
    ----------
    source_id
        Canonical kernel identifier (e.g. ``"osm"``, ``"mr_rocks"``).
    taxonomy
        The :class:`Taxonomy` whose categories index the columns.
    native_labels
        Ordered tuple of the source's own output labels indexing rows.
    matrix
        ``(|Y_s|, |c*|)`` array of conditional probabilities; columns
        sum to 1. Validated on construction.
    description
        Human-readable provenance note (where these numbers came from,
        what evidence supports them).
    """

    source_id: str
    taxonomy: Taxonomy
    native_labels: tuple
    matrix: np.ndarray
    description: str = ""

    # ---- Validation ----------------------------------------------------

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.ndim != 2:
            raise ValueError(
                f"Q_{self.source_id} must be 2-D; got shape {m.shape}"
            )
        if m.shape[0] != len(self.native_labels):
            raise ValueError(
                f"Q_{self.source_id} row count {m.shape[0]} does not match "
                f"native_labels ({len(self.native_labels)})"
            )
        if m.shape[1] != self.taxonomy.n:
            raise ValueError(
                f"Q_{self.source_id} column count {m.shape[1]} does not "
                f"match taxonomy {self.taxonomy.name!r} ({self.taxonomy.n})"
            )
        if (m < 0).any():
            raise ValueError(f"Q_{self.source_id} has negative entries")
        col_sums = m.sum(axis=0)
        if not np.allclose(col_sums, 1.0, atol=1e-6):
            raise ValueError(
                f"Q_{self.source_id} columns do not sum to 1 "
                f"(deltas = {col_sums - 1.0})"
            )
        # Replace the input matrix with a clean float64 copy so users
        # can't mutate underlying data through their original handle.
        object.__setattr__(self, "matrix", m.copy())

    # ---- Lookups -------------------------------------------------------

    @property
    def n_native(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_categories(self) -> int:
        return self.matrix.shape[1]

    def native_index(self, label: str) -> int:
        """Row index of ``label`` in :attr:`native_labels`."""
        try:
            return self.native_labels.index(label)
        except ValueError as exc:
            raise KeyError(
                f"native label {label!r} not in Q_{self.source_id}; "
                f"valid: {list(self.native_labels)}"
            ) from exc

    def likelihood_row(self, native_label: str) -> np.ndarray:
        """Return ``Q_s(ŷ = native_label | c)`` as a 1-D array of length
        ``|c*|``.

        This is the row of the confusion matrix corresponding to a
        given native output, treated as a per-category likelihood
        vector. :func:`build_scene_graph` multiplies these (in log
        space) across kernels with weights ``λ_s`` to form the fused
        posterior.
        """
        return self.matrix[self.native_index(native_label)].copy()

    def log_likelihood_row(
        self,
        native_label: str,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """``log Q_s(ŷ | c)`` for a fixed ``ŷ``.

        ``eps`` floors the log to avoid ``-inf`` when a column has
        zero mass on the given native label; tune downward in PR-3
        once Q_s is empirically fitted and zeros become real.
        """
        return np.log(np.maximum(self.matrix[self.native_index(native_label)], eps))

    # ---- Serialization -------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_id": self.source_id,
            "taxonomy": self.taxonomy.name,
            "native_labels": list(self.native_labels),
            "matrix": self.matrix.tolist(),
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Helpers for hand-coded priors
# ---------------------------------------------------------------------------

def _peaked_column(
    taxonomy: Taxonomy,
    peaks: Mapping[str, float],
    *,
    floor: float = 0.0,
) -> np.ndarray:
    """Build a single category-conditional column ``Q_s[:, j]``.

    ``peaks`` maps native-label string → probability mass; remaining
    mass is uniformly spread over native labels not in ``peaks`` if
    ``floor > 0``. Caller asserts the keys correspond to the row order
    used by :class:`ConfusionMatrix`.
    """
    raise NotImplementedError("not used; kept for clarity of design")  # pragma: no cover


def _column_stochastic(
    matrix: np.ndarray,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """Renormalise so each column of ``matrix`` sums to 1 (in place-safe way).

    Adds ``eps`` to every entry first to keep zeros from forming hard
    barriers. The :class:`ConfusionMatrix` validator will then accept
    the result.
    """
    out = np.asarray(matrix, dtype=np.float64) + eps
    out = out / out.sum(axis=0, keepdims=True)
    return out


def _binary_fire_matrix(fire_per_category: np.ndarray) -> np.ndarray:
    """Build a column-stochastic ``(2, |C|)`` matrix from a per-category
    fire rate.

    Used by single-foreground heads (MR-rocks, SAM) where the kernel
    emits exactly one foreground label or stays silent. Row 0 is
    ``P(fire | c) = fire_per_category[c]``, row 1 is ``P(no_fire | c)
    = 1 - fire_per_category[c]``. The orchestrator only ever queries
    row 0; row 1 exists so that the column-stochastic constraint is
    non-trivial and Q's column structure carries real per-category
    information.

    Inputs are clipped to ``[1e-3, 1 - 1e-3]`` so that no column can
    collapse to a delta and the validator's ``allclose`` check passes
    cleanly.
    """
    fire = np.clip(np.asarray(fire_per_category, dtype=np.float64), 1e-3, 1.0 - 1e-3)
    return np.stack([fire, 1.0 - fire], axis=0)


# ---------------------------------------------------------------------------
# Hand-coded priors for the kernels currently deployed on tesseract
# ---------------------------------------------------------------------------
#
# All priors are written against PHX_URBAN_V0:
#   index 0  unknown
#         1  building
#         2  road
#         3  vehicle
#         4  tree
#         5  vegetation_other
#         6  pavement
#         7  bare_ground
#         8  water
#         9  debris
#
# Each `_q_*` function returns a column-stochastic numpy matrix whose
# rows are the source's native labels, then `default_q_s` wraps it in
# a ConfusionMatrix.

def _q_osm() -> ConfusionMatrix:
    """OSM membership kernel — near-deterministic on its anchor tags.

    Native labels are OSM tag *values* used by ``deepgis-xr``'s OSM
    fetcher (``"building"``, ``"highway"``, ``"natural=tree"`` etc.).
    These are the §4 OSM-anchor constraints; coverage is the dominant
    error mode (a real building absent from OSM), which doesn't show
    up here — it shows up as *no claim at all* on that region.
    """
    tx = PHX_URBAN_V0
    natives = (
        "building",          # any building tag
        "highway",           # any highway tag
        "natural=tree",      # tagged single tree
        "landuse=grass",     # grassed lots / lawns
        "natural=water",     # water bodies
        "amenity=parking",   # parking lots
        "barrier",           # walls, fences (treated as building-adjacent)
    )
    Y, C = len(natives), tx.n
    mat = np.zeros((Y, C))

    # Column-by-column: P(ŷ_OSM = native | c)
    # Default off-diagonal mass spread among unrelated natives is small.
    def _set(category: str, dist: Mapping[str, float]) -> None:
        col = tx.index_of(category)
        for label, p in dist.items():
            mat[natives.index(label), col] = p

    # unknown: OSM almost never tags pure noise; small mass on every native
    for i in range(Y):
        mat[i, 0] = 1.0 / Y

    _set("building",          {"building":         0.93, "barrier":          0.04, "amenity=parking": 0.02, "natural=water": 0.01})
    _set("road",              {"highway":          0.92, "amenity=parking": 0.05, "building":         0.02, "natural=tree":   0.01})
    _set("vehicle",           {"highway":          0.30, "amenity=parking": 0.30, "building":         0.05, "natural=tree":   0.10, "landuse=grass": 0.10, "natural=water": 0.05, "barrier": 0.10})
    _set("tree",              {"natural=tree":     0.85, "landuse=grass":   0.10, "building":         0.02, "natural=water": 0.03})
    _set("vegetation_other",  {"landuse=grass":    0.80, "natural=tree":    0.10, "natural=water":    0.05, "building":       0.05})
    _set("pavement",          {"amenity=parking":  0.65, "highway":         0.25, "building":         0.05, "natural=tree":   0.05})
    _set("bare_ground",       {"landuse=grass":    0.30, "natural=tree":    0.10, "amenity=parking": 0.10, "natural=water": 0.05, "building": 0.05, "highway": 0.10, "barrier": 0.30})
    _set("water",             {"natural=water":    0.92, "landuse=grass":   0.05, "highway":          0.02, "natural=tree": 0.01})
    _set("debris",            {"barrier":          0.40, "amenity=parking": 0.20, "landuse=grass":   0.20, "natural=tree":   0.10, "building": 0.05, "highway": 0.05})

    return ConfusionMatrix(
        source_id="osm",
        taxonomy=tx,
        native_labels=natives,
        matrix=_column_stochastic(mat),
        description=(
            "OSM membership kernel (§3.2). Near-deterministic on the four "
            "anchor categories (building, road, tree, water); the long "
            "tail of less-confident tags (parking, barrier, grass) gets "
            "diffuse mass. PR-3 will refit from OSM-anchored regions in "
            "downtown PHX."
        ),
    )


def _q_grounding_dino() -> ConfusionMatrix:
    """Grounding-DINO with a fixed phrase set ``P``.

    Native labels are the literal text phrases. We assume the analyst
    queries the canonical set ``P = {"building", "house", "road",
    "car", "tree", "vegetation", "pavement", "ground", "water", "rock"}``;
    each phrase peaks on the most natural ``c*`` slot but reserves
    floor mass on the others to absorb GD's known habit of firing on
    visually-similar but semantically-different objects.
    """
    tx = PHX_URBAN_V0
    natives = (
        "building", "house", "road", "car", "tree",
        "vegetation", "pavement", "ground", "water", "rock",
    )
    Y, C = len(natives), tx.n
    mat = np.zeros((Y, C))

    def _set(category: str, dist: Mapping[str, float]) -> None:
        col = tx.index_of(category)
        for phrase, p in dist.items():
            mat[natives.index(phrase), col] = p

    # unknown: phrases all weakly fire (uninformative)
    for i in range(Y):
        mat[i, 0] = 1.0 / Y

    _set("building",          {"building": 0.50, "house": 0.40, "road": 0.02, "tree": 0.02, "rock": 0.02, "ground": 0.02, "vegetation": 0.01, "pavement": 0.01})
    _set("road",              {"road": 0.60, "pavement": 0.25, "ground": 0.05, "building": 0.02, "car": 0.05, "vegetation": 0.01, "tree": 0.01, "rock": 0.01})
    _set("vehicle",           {"car": 0.85, "road": 0.08, "pavement": 0.03, "building": 0.02, "tree": 0.01, "rock": 0.01})
    _set("tree",              {"tree": 0.80, "vegetation": 0.15, "ground": 0.02, "building": 0.01, "rock": 0.01, "road": 0.01})
    _set("vegetation_other",  {"vegetation": 0.70, "tree": 0.10, "ground": 0.10, "building": 0.05, "rock": 0.02, "water": 0.03})
    _set("pavement",          {"pavement": 0.55, "road": 0.30, "ground": 0.08, "building": 0.04, "rock": 0.01, "vegetation": 0.01, "car": 0.01})
    _set("bare_ground",       {"ground": 0.55, "rock": 0.20, "vegetation": 0.10, "pavement": 0.08, "road": 0.04, "tree": 0.02, "water": 0.01})
    _set("water",             {"water": 0.85, "ground": 0.05, "vegetation": 0.05, "pavement": 0.03, "tree": 0.01, "rock": 0.01})
    _set("debris",            {"rock": 0.55, "ground": 0.20, "building": 0.10, "pavement": 0.05, "vegetation": 0.05, "tree": 0.02, "road": 0.03})

    return ConfusionMatrix(
        source_id="grounding_dino",
        taxonomy=tx,
        native_labels=natives,
        matrix=_column_stochastic(mat),
        description=(
            "Grounding-DINO with phrase set "
            "{building, house, road, car, tree, vegetation, pavement, "
            "ground, water, rock} (§3.3). Each phrase's column peaks on "
            "the obvious c* slot; off-diagonal mass absorbs GD's known "
            "visual-similarity errors (e.g. firing on 'rock' for "
            "exposed soil)."
        ),
    )


def _q_sam2() -> ConfusionMatrix:
    """SAM2 — class-agnostic, contributes regions, not categories.

    Per design-doc §3.1, SAM has no ``Q_SAM`` in the confusion-matrix
    sense; we still need a stub so the fusion machinery has a uniform
    way to look up rows. We use the binary ``[fired, no_fire]``
    vocabulary so the matrix is column-stochastic with non-trivial
    column structure, and set ``P(fire | c) = const`` so SAM's
    contribution to the per-category log-likelihood is **flat**
    (zero log-evidence). Its real value comes from defining the
    region set ``R_t`` and the SAM-adjacency edges (§5).
    """
    tx = PHX_URBAN_V0
    natives = ("<sam_segment>", "<no_fire>")
    fire_rate = np.full(tx.n, 0.5)
    return ConfusionMatrix(
        source_id="sam2",
        taxonomy=tx,
        native_labels=natives,
        matrix=_binary_fire_matrix(fire_rate),
        description=(
            "SAM2 segmentation kernel (§3.1). Class-agnostic — flat "
            "fire rate across c* means SAM contributes zero log-"
            "evidence to the category posterior. Its contribution to "
            "the scene graph is regions and adjacency, not categories."
        ),
    )


def _q_grounded_sam2() -> ConfusionMatrix:
    """Grounded-SAM2 — text-conditioned SAM. Same prior as Grounding-DINO
    but emitted with mask geometry rather than just box geometry."""
    qd = _q_grounding_dino()
    return ConfusionMatrix(
        source_id="grounded_sam2",
        taxonomy=qd.taxonomy,
        native_labels=qd.native_labels,
        matrix=qd.matrix.copy(),
        description=(
            "Grounded-SAM2 — SAM masks anchored to a text phrase. "
            "Inherits the Grounding-DINO confusion matrix; the only "
            "difference is geometry (mask vs box)."
        ),
    )


def _q_mr_rocks() -> ConfusionMatrix:
    """Mask R-CNN rocks (Bishop NTL RGB) with the **deliberately-splayed**
    prior from design-doc §3.4 + Figs. 0.7a/b.

    Native vocabulary is ``[rock, <no_fire>]`` (single foreground class
    + implicit no-fire). The fire-rate vector
    ``P(ŷ = rock | c)`` reflects the empirical observation documented
    in the figures: this kernel fires *both* on real rocks *and* on
    Phoenix rooftops. Both contributions are real and roughly
    comparable in magnitude — debris ≈ 0.85, building ≈ 0.45.

    This is the central ``Q_s`` whose refit is the §8.1 R1-rung
    experiment: keep the kernel fixed, replace this fire-rate vector
    with one fitted from supervised PHX (rock vs roof vs other) pairs.
    """
    tx = PHX_URBAN_V0
    natives = ("rock", "<no_fire>")
    fire = np.zeros(tx.n)
    fire[tx.index_of("debris")]            = 0.85
    fire[tx.index_of("building")]          = 0.45  # the §3.4 splay
    fire[tx.index_of("bare_ground")]       = 0.30
    fire[tx.index_of("pavement")]          = 0.10
    fire[tx.index_of("vegetation_other")]  = 0.05
    fire[tx.index_of("tree")]              = 0.03
    fire[tx.index_of("road")]              = 0.05
    fire[tx.index_of("vehicle")]           = 0.02
    fire[tx.index_of("water")]             = 0.01
    fire[tx.index_of("unknown")]           = 0.10
    return ConfusionMatrix(
        source_id="mr_rocks",
        taxonomy=tx,
        native_labels=natives,
        matrix=_binary_fire_matrix(fire),
        description=(
            "Mask R-CNN rocks (Bishop NTL RGB e0049). Single-foreground "
            "head with the splayed prior of design-doc §3.4 + Figs. "
            "0.7a/b: P(ŷ=rock | c=debris) ≈ 0.85, P(ŷ=rock | c=building) "
            "≈ 0.45 — encodes the empirical observation that this "
            "kernel fires on Phoenix rooftops. PR-3 will replace these "
            "numbers with an empirical R1-rung fit; until then this "
            "prior is what makes the §3.0 distinction-kernel reframe "
            "actionable in the fusion code."
        ),
    )


def _q_mr_house() -> ConfusionMatrix:
    """Mask R-CNN house (Eureka aug_mult e0039) — binary fire-rate
    kernel peaked on ``c = building``.

    The eureka taxonomy decomposes a damage axis *within* a single
    super-class (built structure). Putting damage states into ``c*``
    is a deferred decision (PR-3 candidate); for PR-1 we treat
    MR-house as a binary fire-rate kernel like MR-rocks, with the
    six native damage classes collapsed to ``"house"`` for the
    purposes of category fusion. Damage level lives in the
    :attr:`KernelClaim.attributes` of each fire (e.g.
    ``{"damage": "house_damage_2"}``) so it can be rendered in the
    SceneGraph without polluting the c* posterior.

    The fire-rate vector is peaked on ``c = building`` (≈ 0.85)
    with substantial mass on ``c = debris`` (≈ 0.20) to absorb
    the head's known false-fires on rubble piles, and small mass
    on ``c = pavement`` for the parking-lot misfires.
    """
    tx = PHX_URBAN_V0
    natives = ("house", "<no_fire>")
    fire = np.zeros(tx.n)
    fire[tx.index_of("building")]         = 0.85
    fire[tx.index_of("debris")]           = 0.20
    fire[tx.index_of("pavement")]         = 0.15
    fire[tx.index_of("unknown")]          = 0.10
    fire[tx.index_of("bare_ground")]      = 0.08
    fire[tx.index_of("road")]             = 0.05
    fire[tx.index_of("vehicle")]          = 0.03
    fire[tx.index_of("vegetation_other")] = 0.03
    fire[tx.index_of("tree")]             = 0.02
    fire[tx.index_of("water")]            = 0.01
    return ConfusionMatrix(
        source_id="mr_house",
        taxonomy=tx,
        native_labels=natives,
        matrix=_binary_fire_matrix(fire),
        description=(
            "Mask R-CNN house (Eureka aug_mult e0039). Binary fire-rate "
            "kernel peaked on c=building (≈ 0.85) with a long tail on "
            "debris / pavement / bare_ground that absorbs the head's "
            "known PHX false-fires. The five native damage classes "
            "(house_undamaged, house_damage_0..3) are collapsed to a "
            "single 'house' fire signal here; per-claim damage lives "
            "in KernelClaim.attributes['damage'] so the SceneGraph can "
            "render it without affecting the c* posterior."
        ),
    )


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

_BUILTIN_PRIORS: Dict[str, callable] = {
    "osm":              _q_osm,
    "grounding_dino":   _q_grounding_dino,
    "sam2":             _q_sam2,
    "grounded_sam2":    _q_grounded_sam2,
    "mr_rocks":         _q_mr_rocks,
    "mr_house":         _q_mr_house,
}


def available_sources() -> Sequence[str]:
    """Canonical IDs of every source for which a default ``Q_s`` exists."""
    return tuple(_BUILTIN_PRIORS)


def default_q_s(
    source_id: str,
    *,
    taxonomy: Optional[Taxonomy] = None,
) -> ConfusionMatrix:
    """Hand-coded prior ``Q_s`` for a known source.

    Parameters
    ----------
    source_id
        Canonical kernel id; see :func:`available_sources`.
    taxonomy
        Optional override for the taxonomy. Currently only
        :data:`~kernelcal.distinction_game.taxonomy.PHX_URBAN_V0`
        is supported; passing anything else raises ``ValueError``.
        PR-3 will add per-locale taxonomies.
    """
    if source_id not in _BUILTIN_PRIORS:
        raise KeyError(
            f"no default Q_s for source {source_id!r}; "
            f"known sources: {list(_BUILTIN_PRIORS)}"
        )
    if taxonomy is not None and taxonomy is not PHX_URBAN_V0:
        raise ValueError(
            f"PR-1 only ships priors against {PHX_URBAN_V0.name!r}; "
            f"got {taxonomy.name!r}. Hand-write a Q_s or wait for PR-3."
        )
    return _BUILTIN_PRIORS[source_id]()


def default_q_s_table(
    source_ids: Optional[Sequence[str]] = None,
    *,
    taxonomy: Optional[Taxonomy] = None,
) -> Dict[str, ConfusionMatrix]:
    """Bulk lookup. Useful when the orchestrator needs every prior at once."""
    ids = source_ids if source_ids is not None else available_sources()
    return {sid: default_q_s(sid, taxonomy=taxonomy) for sid in ids}
