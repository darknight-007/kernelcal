"""
The shared category vocabulary ``c*`` over which every source's
:class:`~kernelcal.distinction_game.q_s.ConfusionMatrix` is defined.

Per §3.0 of the design doc, this is the *target* set of distinctions the
multi-kernel system is trying to ground; each source's native labels are
mapped into ``c*`` through its ``Q_s``. Adding or removing a category
forces every ``Q_s`` table to be re-padded — categories are part of the
public schema, not free parameters of the run.

PR-1 ships a single hand-curated taxonomy, ``PHX_URBAN_V0``, sized for
overhead Phoenix RGB imagery and the five kernels we currently have on
tesseract. Future locales will add additional :class:`Taxonomy`
instances (e.g. ``MARS_JEZERO_V0`` with crater / regolith / bedrock
categories — §10 of the design doc).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Taxonomy:
    """An ordered, named category vocabulary with optional super-classes.

    Indexing is positional: ``categories[0]`` is the canonical
    background / unknown class so that confusion matrices have a stable
    "no claim" column. Super-classes group fine categories for the
    coverage-matching constraint of §4 (the empirical rate of source
    ``s`` firing within super-class ``g`` is what gets matched).

    Examples
    --------
    >>> tx = PHX_URBAN_V0
    >>> tx.index_of("building")
    1
    >>> tx.super_class_of("vehicle")
    'transport'
    >>> tx.indices_in_super_class("vegetation")
    [4, 5]
    """

    name: str
    categories: Tuple[str, ...]
    super_classes: Dict[str, str] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if len(set(self.categories)) != len(self.categories):
            duplicates = [c for c in self.categories if self.categories.count(c) > 1]
            raise ValueError(
                f"taxonomy {self.name!r} has duplicate categories: "
                f"{sorted(set(duplicates))}"
            )
        unknown_super = set(self.super_classes) - set(self.categories)
        if unknown_super:
            raise ValueError(
                f"taxonomy {self.name!r} has super-class entries for "
                f"unknown categories: {sorted(unknown_super)}"
            )

    @property
    def n(self) -> int:
        """Number of categories (including the unknown / background slot)."""
        return len(self.categories)

    def index_of(self, category: str) -> int:
        """Position of ``category`` in :attr:`categories` (raises ``KeyError`` if absent)."""
        try:
            return self.categories.index(category)
        except ValueError as exc:
            raise KeyError(
                f"category {category!r} is not in taxonomy {self.name!r}; "
                f"valid categories: {list(self.categories)}"
            ) from exc

    def name_of(self, index: int) -> str:
        if not 0 <= index < self.n:
            raise IndexError(
                f"category index {index} out of bounds for taxonomy "
                f"{self.name!r} (n = {self.n})"
            )
        return self.categories[index]

    def super_class_of(self, category: str) -> Optional[str]:
        """Return the super-class of ``category``, or ``None`` if it has none."""
        return self.super_classes.get(category)

    def indices_in_super_class(self, super_class: str) -> List[int]:
        """All category indices belonging to ``super_class``.

        Returns an empty list if ``super_class`` is not used. The
        canonical unknown slot (index 0) is never grouped under a
        super-class.
        """
        return [
            self.index_of(c)
            for c, g in self.super_classes.items()
            if g == super_class and c in self.categories
        ]

    def to_dict(self) -> Dict[str, object]:
        """JSON-friendly view, suitable for embedding in a SceneGraph."""
        return {
            "name": self.name,
            "categories": list(self.categories),
            "super_classes": dict(self.super_classes),
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Phoenix urban v0
# ---------------------------------------------------------------------------
#
# The first taxonomy we ship — built around the live kernel set on
# tesseract (Grounding-DINO, SAM2, MR-rocks, MR-house, OSM) and Phoenix
# overhead RGB imagery from Cesium. Ten slots is small enough that every
# Q_s can be hand-written in one sitting, big enough that the rocks /
# house / OSM disagreement structure of the design doc shows up.

_PHX_CATEGORIES: Tuple[str, ...] = (
    "unknown",            # 0  canonical "no claim" / background
    "building",           # 1  any roofed structure
    "road",               # 2  paved drivable surface (highway)
    "vehicle",            # 3  car, truck, bus
    "tree",               # 4  woody single-trunk vegetation (canopy ≳ 2 m)
    "vegetation_other",   # 5  grass, lawn, shrub, garden
    "pavement",           # 6  driveway, parking, sidewalk, plaza
    "bare_ground",        # 7  dirt, sand, exposed regolith
    "water",              # 8  pool, canal, basin
    "debris",             # 9  rocks, rubble, construction spoil
)

_PHX_SUPER_CLASSES: Dict[str, str] = {
    "building":         "structure",
    "road":             "transport",
    "vehicle":          "transport",
    "pavement":         "transport",
    "tree":             "vegetation",
    "vegetation_other": "vegetation",
    "bare_ground":      "natural",
    "water":            "natural",
    "debris":           "natural",
}

PHX_URBAN_V0 = Taxonomy(
    name="phx_urban_v0",
    categories=_PHX_CATEGORIES,
    super_classes=_PHX_SUPER_CLASSES,
    description=(
        "Phoenix urban v0 — 10 categories sized to the kernel set live "
        "on tesseract as of 2026-04: Grounding-DINO, SAM2, MR-rocks, "
        "MR-house, OSM. Slot 0 is the canonical 'unknown' category that "
        "every Q_s reserves for low-confidence regions; slot 9 ('debris') "
        "absorbs the MR-rocks misfire on PHX rooftops in a soft way "
        "(see design doc §3.4 + Figs. 0.7a/b)."
    ),
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def by_name(name: str) -> Taxonomy:
    """Look up a built-in taxonomy by name. PR-1 only ships ``phx_urban_v0``."""
    if name == PHX_URBAN_V0.name:
        return PHX_URBAN_V0
    raise KeyError(
        f"unknown taxonomy {name!r}; available: [{PHX_URBAN_V0.name!r}]"
    )


def builtin_names() -> Sequence[str]:
    return (PHX_URBAN_V0.name,)
