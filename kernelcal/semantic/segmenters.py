"""
kernelcal.semantic.segmenters — thin adapters over segmentation backends.

The kernelcal pipeline is deliberately backend-agnostic.  A segmenter is any
object implementing::

    class Segmenter(Protocol):
        name: str
        kind: str      # "closed_set" | "panoptic" | "open_vocab"
        cost_seconds: float  # amortised cost per image
        def segment(self, image, prompt=None) -> list[InstanceMask]: ...

``InstanceMask`` is a lightweight dataclass — no torch tensors, no ROS
types — so the ensemble can be unit-tested with a trivial stub segmenter
and the real Mask R-CNN / SAM / Mask2Former / Grounding-DINO backends are
loaded lazily behind try/except guards.

Backends shipped here
---------------------
* ``StubSegmenter``          — deterministic, for tests and CI.
* ``MaskRCNNSegmenter``      — torchvision Mask R-CNN on a user vocabulary.
* ``SAMSegmenter``           — Segment Anything (class-agnostic panoptic).
* ``Mask2FormerSegmenter``   — HuggingFace panoptic head.
* ``GroundingDINOSegmenter`` — open-vocab text-prompted.

All real backends are optional; if the corresponding package is not
installed, constructing the class raises ``ImportError`` with a clear
install hint and the rest of the module still imports fine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

KIND_CLOSED = "closed_set"
KIND_PANOPTIC = "panoptic"
KIND_OPEN_VOCAB = "open_vocab"


@dataclass
class InstanceMask:
    """One proposed instance from any segmenter.

    Attributes
    ----------
    mask : (H, W) bool array — binary mask in image coordinates.
    bbox : (x1, y1, x2, y2) int tuple — tight bounding box.
    score : float — confidence in [0, 1] (segmenter-specific meaning).
    proposed_label : str | None — closed-set class name, or None if the
        segmenter is class-agnostic (SAM, Mask2Former stuff-id) or open-vocab.
    embedding : (D,) float array or None — visual feature descriptor
        (ORB-style bag, CLIP-ViT pooled vector, etc.).
    source : str — name of the originating segmenter.
    extras : dict — segmenter-specific payload (class_probs, prompt phrase, …).
    """

    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    score: float
    proposed_label: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    source: str = "unknown"
    extras: dict = field(default_factory=dict)

    @property
    def area_px(self) -> int:
        return int(self.mask.sum())

    @property
    def centroid_xy(self) -> Tuple[float, float]:
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return 0.0, 0.0
        return float(xs.mean()), float(ys.mean())


class Segmenter(Protocol):
    name: str
    kind: str
    cost_seconds: float

    def segment(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None,
    ) -> List[InstanceMask]:
        ...


# ---------------------------------------------------------------------------
# StubSegmenter — deterministic fake for tests
# ---------------------------------------------------------------------------


class StubSegmenter:
    """A deterministic segmenter that returns preset instances.

    Useful for unit tests, CI, and plumbing work that should not depend on
    torchvision / transformers being installed.
    """

    def __init__(
        self,
        name: str = "stub",
        kind: str = KIND_CLOSED,
        cost_seconds: float = 0.01,
        instances_fn: Optional[Callable[[np.ndarray, Optional[str]], List[InstanceMask]]] = None,
        embedding_dim: int = 16,
    ) -> None:
        self.name = name
        self.kind = kind
        self.cost_seconds = cost_seconds
        self._instances_fn = instances_fn
        self._embedding_dim = embedding_dim

    def segment(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None,
    ) -> List[InstanceMask]:
        if self._instances_fn is not None:
            return self._instances_fn(image, prompt)
        h, w = image.shape[:2]
        # Deterministic quadrant masks with mildly varied scores.
        out: List[InstanceMask] = []
        for i, (x0, y0, lbl, score) in enumerate([
            (0, 0, "tree", 0.90),
            (w // 2, 0, "house", 0.80),
            (0, h // 2, "road", 0.95),
            (w // 2, h // 2, "car", 0.60),
        ]):
            m = np.zeros((h, w), dtype=bool)
            m[y0:y0 + h // 2, x0:x0 + w // 2] = True
            rng = np.random.default_rng(hash((self.name, lbl)) & 0xFFFF)
            emb = rng.standard_normal(self._embedding_dim)
            out.append(InstanceMask(
                mask=m,
                bbox=(x0, y0, x0 + w // 2, y0 + h // 2),
                score=score,
                proposed_label=lbl,
                embedding=emb,
                source=self.name,
                extras={"stub_index": i},
            ))
        return out


# ---------------------------------------------------------------------------
# Real backends — lazy / guarded
# ---------------------------------------------------------------------------


def _require(module: str, hint: str) -> Any:
    try:
        import importlib
        return importlib.import_module(module)
    except Exception as e:
        raise ImportError(
            f"Optional segmenter backend {module!r} is unavailable: {e}. "
            f"Install with: {hint}"
        )


class MaskRCNNSegmenter:
    """Adapter over torchvision's Mask R-CNN.

    Parameters
    ----------
    class_names : list[str]
        User-specified closed-set vocabulary.  The adapter does not retrain
        the head; it calls the pretrained model and filters detections whose
        top COCO label is mapped (via ``coco_map``) to one of ``class_names``.
    coco_map : dict[str, str]
        Mapping from COCO-80 labels to user vocabulary labels (many-to-one
        is allowed).  Unknown detections are dropped.
    score_thresh : float
    """

    name = "maskrcnn"
    kind = KIND_CLOSED
    cost_seconds = 0.25

    def __init__(
        self,
        class_names: Sequence[str],
        coco_map: Optional[dict[str, str]] = None,
        score_thresh: float = 0.5,
        device: str = "cpu",
    ) -> None:
        torch = _require("torch", "pip install torch torchvision")
        tv = _require("torchvision", "pip install torchvision")
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights

        self._torch = torch
        self._class_names = list(class_names)
        self._coco_map = dict(coco_map or {})
        self._score_thresh = score_thresh
        self._device = torch.device(device)

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self._weights = weights
        self._coco_labels = weights.meta["categories"]  # list len 91
        self._model = maskrcnn_resnet50_fpn(weights=weights).to(self._device).eval()

    @staticmethod
    def default_coco_map() -> dict[str, str]:
        """Minimal COCO→urban mapping for the {tree, house, road, car} seed."""
        return {
            "car": "car",
            "truck": "car",
            "bus": "car",
            "motorcycle": "car",
            "bicycle": "car",
            # COCO has no 'tree' / 'road' / 'house' — those arrive via SAM
            # + Grounding-DINO or via a user label round-trip.
        }

    def segment(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None,  # unused; closed-set
    ) -> List[InstanceMask]:
        torch = self._torch
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("MaskRCNNSegmenter expects (H,W,3) uint8 RGB.")
        x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        x = x.to(self._device)
        with torch.no_grad():
            out = self._model([x])[0]
        masks = out["masks"].cpu().numpy()[:, 0]    # (N, H, W)
        boxes = out["boxes"].cpu().numpy().astype(int)
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()

        instances: List[InstanceMask] = []
        for m, b, s, li in zip(masks, boxes, scores, labels):
            if s < self._score_thresh:
                continue
            coco_name = self._coco_labels[int(li)]
            mapped = self._coco_map.get(coco_name)
            if mapped is None or mapped not in self._class_names:
                continue
            bin_mask = m > 0.5
            if not bin_mask.any():
                continue
            instances.append(InstanceMask(
                mask=bin_mask,
                bbox=tuple(b.tolist()),
                score=float(s),
                proposed_label=mapped,
                source=self.name,
                extras={"coco_label": coco_name},
            ))
        return instances


class SAMSegmenter:
    """Adapter over Segment Anything (class-agnostic panoptic).

    Requires the ``segment_anything`` package and a downloaded checkpoint.
    """

    name = "sam"
    kind = KIND_PANOPTIC
    cost_seconds = 0.8

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_b",
        device: str = "cpu",
        points_per_side: int = 16,
        min_mask_region_area: int = 256,
    ) -> None:
        sam_mod = _require(
            "segment_anything",
            "pip install segment-anything",
        )
        torch = _require("torch", "pip install torch")

        sam = sam_mod.sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device).eval()
        self._torch = torch
        self._generator = sam_mod.SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            min_mask_region_area=min_mask_region_area,
        )

    def segment(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None,
    ) -> List[InstanceMask]:
        raw = self._generator.generate(image)
        out: List[InstanceMask] = []
        for r in raw:
            m = r["segmentation"].astype(bool)
            if not m.any():
                continue
            x, y, w, h = r["bbox"]
            out.append(InstanceMask(
                mask=m,
                bbox=(int(x), int(y), int(x + w), int(y + h)),
                score=float(r.get("stability_score", r.get("predicted_iou", 1.0))),
                proposed_label=None,  # class-agnostic
                source=self.name,
                extras={k: r[k] for k in ("area", "point_coords") if k in r},
            ))
        return out


class Mask2FormerSegmenter:
    """Adapter over HuggingFace Mask2Former panoptic model."""

    name = "mask2former"
    kind = KIND_PANOPTIC
    cost_seconds = 0.5

    def __init__(
        self,
        model_id: str = "facebook/mask2former-swin-base-coco-panoptic",
        device: str = "cpu",
        score_thresh: float = 0.5,
    ) -> None:
        torch = _require("torch", "pip install torch")
        tf = _require(
            "transformers",
            "pip install transformers",
        )
        self._torch = torch
        self._processor = tf.AutoImageProcessor.from_pretrained(model_id)
        self._model = tf.Mask2FormerForUniversalSegmentation.from_pretrained(
            model_id
        ).to(device).eval()
        self._device = device
        self._score_thresh = score_thresh

    def segment(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None,
    ) -> List[InstanceMask]:
        torch = self._torch
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        proc = self._processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[image.shape[:2]]
        )[0]
        seg = proc["segmentation"].cpu().numpy()
        id2label = self._model.config.id2label
        out: List[InstanceMask] = []
        for info in proc["segments_info"]:
            if info.get("score", 1.0) < self._score_thresh:
                continue
            seg_id = info["id"]
            m = seg == seg_id
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            label = id2label.get(info["label_id"], None)
            out.append(InstanceMask(
                mask=m,
                bbox=bbox,
                score=float(info.get("score", 1.0)),
                proposed_label=label,
                source=self.name,
                extras={"label_id": info["label_id"]},
            ))
        return out


class GroundingDINOSegmenter:
    """Adapter over Grounding-DINO (text-prompted, open vocab)."""

    name = "grounding_dino"
    kind = KIND_OPEN_VOCAB
    cost_seconds = 0.4

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cpu",
        box_thresh: float = 0.35,
        text_thresh: float = 0.25,
    ) -> None:
        torch = _require("torch", "pip install torch")
        tf = _require(
            "transformers",
            "pip install transformers",
        )
        self._torch = torch
        self._processor = tf.AutoProcessor.from_pretrained(model_id)
        self._model = tf.AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(device).eval()
        self._device = device
        self._box_thresh = box_thresh
        self._text_thresh = text_thresh

    def segment(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None,
    ) -> List[InstanceMask]:
        if not prompt:
            return []
        torch = self._torch
        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(
            self._device
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self._box_thresh,
            text_threshold=self._text_thresh,
            target_sizes=[image.shape[:2]],
        )[0]
        out: List[InstanceMask] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            m = np.zeros(image.shape[:2], dtype=bool)
            m[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = True
            if not m.any():
                continue
            out.append(InstanceMask(
                mask=m,
                bbox=(x1, y1, x2, y2),
                score=float(score),
                proposed_label=str(label),
                source=self.name,
                extras={"text_phrase": str(label)},
            ))
        return out


__all__ = [
    "InstanceMask",
    "Segmenter",
    "StubSegmenter",
    "MaskRCNNSegmenter",
    "SAMSegmenter",
    "Mask2FormerSegmenter",
    "GroundingDINOSegmenter",
    "KIND_CLOSED",
    "KIND_PANOPTIC",
    "KIND_OPEN_VOCAB",
]
