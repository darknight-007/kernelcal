"""
kernelcal.semantic.registry — live ontology for human-in-the-loop SLAM.

A ``ClassRegistry`` holds the user-specified class vocabulary together with
per-class metadata that the rest of the pipeline needs:

* prototype embedding (mean feature vector of labelled instances),
* motion kind (``"static" | "semi_static" | "dynamic"``) — determines whether
  the class contributes to the static scene kernel K_static or to the
  dynamic distinction field c_dyn (see P2, Section 3.2),
* display colour (for DeepGIS overlay),
* prior probability (used as the MaxCal reference q(class) in the segmenter
  ensemble),
* aliases (synonyms for Grounding-DINO / CLIP text prompts).

The registry is deliberately **not exhaustive**: it starts from a user seed
vocabulary and grows when DeepGIS returns a ``LabelResponse`` with
``is_new_class=True``.  New-class extension is atomic: call
``apply_response(...)`` and the registry is consistent afterwards.

This module is pure Python; no torch, no ROS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Canonical motion kinds
# ---------------------------------------------------------------------------

MOTION_STATIC = "static"          # buildings, roads, trees (G ≈ 0 in P2 eq 13)
MOTION_SEMI_STATIC = "semi_static"  # parked cars, construction sites
MOTION_DYNAMIC = "dynamic"        # moving cars, pedestrians (needs G ≠ 0)

_VALID_MOTION = {MOTION_STATIC, MOTION_SEMI_STATIC, MOTION_DYNAMIC}


@dataclass
class ClassSpec:
    """Metadata for a single class in the registry."""

    name: str
    motion: str = MOTION_STATIC
    color_rgb: Tuple[int, int, int] = (200, 200, 200)
    prior: float = 1.0
    aliases: List[str] = field(default_factory=list)
    prototype: Optional[np.ndarray] = None   # (D,) mean embedding
    n_labelled: int = 0
    is_user_seed: bool = True                 # False if added via DeepGIS

    def __post_init__(self) -> None:
        if self.motion not in _VALID_MOTION:
            raise ValueError(
                f"motion must be one of {_VALID_MOTION}, got {self.motion!r}"
            )
        if self.prior < 0:
            raise ValueError("prior must be non-negative")

    def update_prototype(self, embedding: np.ndarray) -> None:
        """Running mean update of the prototype embedding."""
        emb = np.asarray(embedding, dtype=float).ravel()
        if self.prototype is None or self.prototype.shape != emb.shape:
            self.prototype = emb.copy()
            self.n_labelled = 1
            return
        n = self.n_labelled
        self.prototype = (self.prototype * n + emb) / (n + 1)
        self.n_labelled = n + 1

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.prototype is not None:
            d["prototype"] = self.prototype.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ClassSpec":
        d = dict(d)
        d["color_rgb"] = tuple(d.get("color_rgb", (200, 200, 200)))
        proto = d.get("prototype")
        if proto is not None:
            d["prototype"] = np.asarray(proto, dtype=float)
        return cls(**d)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ClassRegistry:
    """Live class vocabulary with prototype embeddings and motion tags.

    Parameters
    ----------
    seed_classes : list[ClassSpec] or None
        Initial user-specified vocabulary.  Example::

            ClassRegistry(seed_classes=[
                ClassSpec("tree",  motion="static",       color_rgb=(30,160,60)),
                ClassSpec("house", motion="static",       color_rgb=(180,120,60)),
                ClassSpec("road",  motion="static",       color_rgb=(90,90,90)),
                ClassSpec("car",   motion="dynamic",      color_rgb=(200,30,30)),
            ])

    always_include_other : bool
        If True (default) a catch-all ``"other"`` class with low prior is
        always available so the segmenter never has to commit to a seed class
        for an uncertain pixel.
    """

    OTHER_CLASS = "other"

    def __init__(
        self,
        seed_classes: Optional[Iterable[ClassSpec]] = None,
        always_include_other: bool = True,
    ) -> None:
        self._classes: Dict[str, ClassSpec] = {}
        if seed_classes is not None:
            for spec in seed_classes:
                self._classes[spec.name] = spec
        if always_include_other and self.OTHER_CLASS not in self._classes:
            self._classes[self.OTHER_CLASS] = ClassSpec(
                name=self.OTHER_CLASS,
                motion=MOTION_STATIC,
                color_rgb=(128, 128, 128),
                prior=0.1,
                is_user_seed=False,
            )

    # ------------------------------------------------------------------
    # Ontology accessors
    # ------------------------------------------------------------------

    @property
    def names(self) -> List[str]:
        return list(self._classes.keys())

    def __len__(self) -> int:
        return len(self._classes)

    def __contains__(self, name: str) -> bool:
        return name in self._classes

    def __getitem__(self, name: str) -> ClassSpec:
        return self._classes[name]

    def get(self, name: str, default: Optional[ClassSpec] = None) -> Optional[ClassSpec]:
        return self._classes.get(name, default)

    def specs(self) -> List[ClassSpec]:
        return list(self._classes.values())

    # ------------------------------------------------------------------
    # MaxCal prior vector — reference q(class) for the ensemble selector
    # ------------------------------------------------------------------

    def prior_vector(self) -> Tuple[List[str], np.ndarray]:
        """Return ``(names, q)`` where q is a normalised prior over classes."""
        names = self.names
        p = np.array([self._classes[n].prior for n in names], dtype=float)
        total = p.sum()
        if total <= 0:
            p = np.ones_like(p) / len(p)
        else:
            p = p / total
        return names, p

    # ------------------------------------------------------------------
    # Prototype tooling
    # ------------------------------------------------------------------

    def prototype_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Stack of known prototypes as an (M, D) matrix.

        Returns (names_with_prototype, embedding_matrix).  Classes without a
        prototype yet are omitted.
        """
        names = [n for n, s in self._classes.items() if s.prototype is not None]
        if not names:
            return [], np.zeros((0, 0))
        mat = np.stack([self._classes[n].prototype for n in names], axis=0)
        return names, mat

    def classify_by_prototype(
        self,
        embedding: np.ndarray,
        min_cosine: float = 0.5,
    ) -> Tuple[Optional[str], float]:
        """Nearest-prototype cosine classifier.

        Returns ``(class_name_or_None, cosine_similarity)``.  The caller
        decides, via ``min_cosine``, whether the match is good enough;
        otherwise it should raise a ``LabelRequest``.
        """
        names, protos = self.prototype_matrix()
        if not names:
            return None, -1.0
        emb = np.asarray(embedding, dtype=float).ravel()
        if emb.shape[0] != protos.shape[1]:
            return None, -1.0
        n_emb = emb / (np.linalg.norm(emb) + 1e-12)
        n_protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
        sims = n_protos @ n_emb
        idx = int(np.argmax(sims))
        sim = float(sims[idx])
        if sim < min_cosine:
            return None, sim
        return names[idx], sim

    # ------------------------------------------------------------------
    # HITL mutation
    # ------------------------------------------------------------------

    def apply_response(
        self,
        assigned_class: str,
        *,
        embedding: Optional[np.ndarray] = None,
        is_new_class: bool = False,
        motion: str = MOTION_STATIC,
        color_rgb: Tuple[int, int, int] = (200, 200, 200),
        aliases: Optional[Iterable[str]] = None,
    ) -> ClassSpec:
        """Apply a DeepGIS ``LabelResponse``.

        If ``is_new_class`` and the class already exists, the response is
        treated as an update (new aliases merged in).  If the class does not
        exist, it is created.  If ``is_new_class`` is False and the class is
        unknown, ``KeyError`` is raised (a guard against typos in responses).
        """
        if assigned_class in self._classes:
            spec = self._classes[assigned_class]
            if aliases:
                merged = list(dict.fromkeys(list(spec.aliases) + list(aliases)))
                spec.aliases = merged
            if embedding is not None:
                spec.update_prototype(np.asarray(embedding, dtype=float))
            return spec

        if not is_new_class:
            raise KeyError(
                f"Unknown class {assigned_class!r} and is_new_class=False"
            )

        spec = ClassSpec(
            name=assigned_class,
            motion=motion,
            color_rgb=color_rgb,
            prior=0.1,
            aliases=list(aliases) if aliases else [],
            is_user_seed=False,
        )
        if embedding is not None:
            spec.update_prototype(np.asarray(embedding, dtype=float))
        self._classes[assigned_class] = spec
        return spec

    # ------------------------------------------------------------------
    # Grounding-DINO prompt construction
    # ------------------------------------------------------------------

    def grounding_prompt(self) -> str:
        """Return a Grounding-DINO-style dot-separated prompt.

        Each class contributes its canonical name plus its aliases.  The
        ``"other"`` catch-all is excluded from the prompt (it has no visual
        meaning) but retained in the registry for unknown-region flagging.
        """
        tokens: List[str] = []
        for name, spec in self._classes.items():
            if name == self.OTHER_CLASS:
                continue
            tokens.append(name)
            tokens.extend(spec.aliases)
        return " . ".join(tokens) + (" ." if tokens else "")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"classes": [s.to_dict() for s in self._classes.values()]}

    @classmethod
    def from_dict(cls, d: dict) -> "ClassRegistry":
        specs = [ClassSpec.from_dict(x) for x in d.get("classes", [])]
        reg = cls(seed_classes=specs, always_include_other=False)
        return reg

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ClassRegistry":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------
    # Convenience builder — the most common seed
    # ------------------------------------------------------------------

    @classmethod
    def urban_default(cls) -> "ClassRegistry":
        """Seed with the four classes from the user's request."""
        return cls(seed_classes=[
            ClassSpec("tree", motion=MOTION_STATIC,
                      color_rgb=(30, 160, 60),
                      aliases=["tree crown", "foliage"]),
            ClassSpec("house", motion=MOTION_STATIC,
                      color_rgb=(180, 120, 60),
                      aliases=["building", "residential building", "home"]),
            ClassSpec("road", motion=MOTION_STATIC,
                      color_rgb=(90, 90, 90),
                      aliases=["street", "asphalt", "roadway"]),
            ClassSpec("car", motion=MOTION_DYNAMIC,
                      color_rgb=(200, 30, 30),
                      aliases=["vehicle", "automobile", "sedan"]),
        ])
