"""
Self-consistent prompt iteration for Grounding DINO.

Text prompts in Grounding DINO (`"rock . boulder . crater"`) implicitly
define a kernel over the image space: the text embedding induces a
similarity structure over visual features.  Changing the prompt changes
the kernel; successive prompt refinements are a trajectory through K.

The MaxCal fixed-point condition identifies the self-consistent prompt:
the text description whose induced kernel is a fixed point of the MaxCal
dynamics given the scene statistics.

Operationally: given an image and a detector callback, this module iterates
prompts until the distribution of detected object categories stabilises
(the kernel fixed-point condition).

The detector callback signature matches the Grounding DINO DeepGIS API:
    detections = detector_fn(image, prompt)
where detections is a list of dicts with keys 'label', 'confidence', 'box'.

The embedder callback (optional) maps a prompt string to a float vector,
enabling Hilbert-Schmidt distance computation in text embedding space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..kernel.space import (
    hilbert_schmidt_distance,
    kernel_from_embeddings,
    normalize_kernel,
)
from ..kernel.fixed_points import FixedPointDetector
from ..kernel.trajectory import KernelTrajectory


# ---------------------------------------------------------------------------
# Prompt kernel distance (in embedding space)
# ---------------------------------------------------------------------------

def prompt_kernel_distance(
    prompt1: str,
    prompt2: str,
    embedder_fn: Callable[[str], np.ndarray],
) -> float:
    """HS distance between the kernel matrices induced by two prompts.

    Each prompt is split on ' . ' into a list of concept tokens.  The
    embedder maps each token to a vector; the kernel matrix is the cosine
    similarity matrix over those vectors.

    Parameters
    ----------
    prompt1, prompt2 : str  Dot-separated concept strings.
    embedder_fn : callable str → (D,) array.

    Returns
    -------
    float : HS distance between the two prompt kernels.
    """
    def _prompt_to_kernel(prompt: str) -> np.ndarray:
        tokens = [t.strip() for t in prompt.split(".") if t.strip()]
        embeddings = np.array([embedder_fn(t) for t in tokens])
        return kernel_from_embeddings(embeddings)

    K1 = _prompt_to_kernel(prompt1)
    K2 = _prompt_to_kernel(prompt2)
    if K1.shape != K2.shape:
        # Pad the smaller kernel with zeros to match shapes
        n = max(K1.shape[0], K2.shape[0])
        def _pad(K, n):
            out = np.zeros((n, n))
            out[:K.shape[0], :K.shape[0]] = K
            return out
        K1, K2 = _pad(K1, n), _pad(K2, n)
    return hilbert_schmidt_distance(K1, K2)


# ---------------------------------------------------------------------------
# Detection statistics as kernel
# ---------------------------------------------------------------------------

def detections_to_distribution(
    detections: List[Dict],
    label_vocab: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Convert a list of detections to a label frequency distribution.

    Parameters
    ----------
    detections : list of dicts with 'label' and 'confidence' keys.
    label_vocab : fixed vocabulary.  If None, built from detections.

    Returns
    -------
    distribution : (|vocab|,) confidence-weighted frequency vector.
    vocab : list of label strings.
    """
    if not detections:
        vocab = label_vocab or []
        return np.zeros(len(vocab)), vocab

    labels = [d.get("label", "unknown") for d in detections]
    confs = np.array([d.get("confidence", 1.0) for d in detections])

    if label_vocab is None:
        label_vocab = sorted(set(labels))

    dist = np.zeros(len(label_vocab))
    for lbl, conf in zip(labels, confs):
        if lbl in label_vocab:
            dist[label_vocab.index(lbl)] += conf

    total = dist.sum()
    if total > 0:
        dist /= total
    return dist, label_vocab


def distribution_kernel(
    distributions: List[np.ndarray],
) -> np.ndarray:
    """Build a Hellinger kernel matrix from label distributions.

    K[i,j] = Σ_c √(p_i(c) · p_j(c))   (Bhattacharyya coefficient)
    """
    D = np.array(distributions, dtype=float)
    sqrt_D = np.sqrt(np.maximum(D, 0))
    K = sqrt_D @ sqrt_D.T
    # Ensure PSD
    eigvals = np.linalg.eigvalsh(K)
    if np.any(eigvals < 0):
        K += np.eye(len(K)) * (-np.min(eigvals) + 1e-9)
    return K


# ---------------------------------------------------------------------------
# Prompt refinement from detected categories
# ---------------------------------------------------------------------------

def refine_prompt(
    current_prompt: str,
    detections: List[Dict],
    top_k: int = 5,
    min_confidence: float = 0.2,
) -> str:
    """Generate a refined prompt from the dominant detected categories.

    Parameters
    ----------
    current_prompt : str  Current dot-separated prompt.
    detections : list of detection dicts.
    top_k : int  Keep the top_k most-confident labels.
    min_confidence : float  Filter out low-confidence detections.

    Returns
    -------
    str : refined dot-separated prompt.
    """
    filtered = [d for d in detections
                if d.get("confidence", 0.0) >= min_confidence]
    if not filtered:
        return current_prompt

    # Aggregate confidence per label
    label_conf: Dict[str, float] = {}
    for d in filtered:
        lbl = d.get("label", "")
        label_conf[lbl] = label_conf.get(lbl, 0.0) + d.get("confidence", 1.0)

    sorted_labels = sorted(label_conf, key=label_conf.get, reverse=True)
    new_labels = sorted_labels[:top_k]
    return " . ".join(new_labels)


# ---------------------------------------------------------------------------
# Self-consistent prompt iterator
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    step: int
    prompt: str
    n_detections: int
    distribution: np.ndarray
    kernel_distance_from_prev: float


class PromptKernelIterator:
    """Finds the fixed-point prompt for a given image via MaxCal iteration.

    The iteration:
      1. Run detector with current prompt → get detections.
      2. Build detection distribution over label vocabulary.
      3. Build Hellinger kernel from distribution.
      4. Check HS distance from previous kernel: if < tol → converged.
      5. Refine prompt from dominant detected categories → repeat.

    Parameters
    ----------
    detector_fn : callable (image, prompt) → list[dict]
        Must return dicts with 'label', 'confidence', 'box' keys.
    embedder_fn : callable str → (D,) array, optional.
        Used for prompt-space kernel distance (Thread 7 in integration doc).
    max_steps : int  Maximum iteration steps.
    tol : float  HS-distance convergence threshold.
    top_k : int  Max labels in refined prompt.
    min_confidence : float  Detection confidence filter.
    """

    def __init__(
        self,
        detector_fn: Callable[[Any, str], List[Dict]],
        embedder_fn: Optional[Callable[[str], np.ndarray]] = None,
        max_steps: int = 20,
        tol: float = 1e-2,
        top_k: int = 5,
        min_confidence: float = 0.2,
    ):
        self.detector_fn = detector_fn
        self.embedder_fn = embedder_fn
        self.max_steps = max_steps
        self.tol = tol
        self.top_k = top_k
        self.min_confidence = min_confidence

        self._history: List[IterationRecord] = []
        self._trajectory = KernelTrajectory(name="prompt kernel")
        self._fp_detector = FixedPointDetector(tol=tol, window=3)

    def iterate(
        self,
        image: Any,
        initial_prompt: str,
        label_vocab: Optional[List[str]] = None,
    ) -> str:
        """Run the self-consistent prompt search.

        Parameters
        ----------
        image : image in whatever format detector_fn expects.
        initial_prompt : str  Starting prompt.
        label_vocab : list of str or None.

        Returns
        -------
        str : converged self-consistent prompt (or best found in max_steps).
        """
        self._history.clear()
        self._fp_detector = FixedPointDetector(tol=self.tol, window=3)
        self._trajectory = KernelTrajectory(name="prompt kernel")

        prompt = initial_prompt
        prev_K: Optional[np.ndarray] = None
        dist_prev: Optional[np.ndarray] = None
        vocab = label_vocab

        for step in range(self.max_steps):
            detections = self.detector_fn(image, prompt)
            dist, vocab = detections_to_distribution(detections, vocab)

            # Need at least 2 steps to build a kernel
            if dist_prev is not None:
                K = distribution_kernel([dist_prev, dist])
            else:
                K = np.array([[float(np.dot(dist + 1e-9, dist + 1e-9))]])

            d_from_prev = (
                hilbert_schmidt_distance(prev_K, K)
                if prev_K is not None and prev_K.shape == K.shape
                else float("inf")
            )

            self._history.append(IterationRecord(
                step=step,
                prompt=prompt,
                n_detections=len(detections),
                distribution=dist.copy(),
                kernel_distance_from_prev=d_from_prev,
            ))
            self._trajectory.add(t=float(step), K=K)
            self._fp_detector.update(K)

            if self._fp_detector.is_fixed_point():
                break

            prompt = refine_prompt(
                prompt, detections,
                top_k=self.top_k,
                min_confidence=self.min_confidence,
            )
            prev_K = K
            dist_prev = dist

        return prompt

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def converged(self) -> bool:
        return self._fp_detector.is_fixed_point()

    def n_steps(self) -> int:
        return len(self._history)

    def convergence_curve(self) -> np.ndarray:
        """HS distances between consecutive prompt kernels."""
        return np.array([r.kernel_distance_from_prev for r in self._history])

    def prompt_trajectory(self) -> List[str]:
        return [r.prompt for r in self._history]

    def final_distribution(self) -> Optional[np.ndarray]:
        if self._history:
            return self._history[-1].distribution
        return None

    def summary(self) -> dict:
        return {
            "n_steps": self.n_steps(),
            "converged": self.converged(),
            "final_prompt": self._history[-1].prompt if self._history else "",
            "initial_prompt": self._history[0].prompt if self._history else "",
            "convergence_curve": self.convergence_curve().tolist(),
            "stability_score": self._fp_detector.stability_score(),
            "total_detections": sum(r.n_detections for r in self._history),
        }

    def __repr__(self) -> str:
        return (f"PromptKernelIterator("
                f"steps={self.n_steps()}, "
                f"converged={self.converged()}, "
                f"stability={self._fp_detector.stability_score():.3f})")
