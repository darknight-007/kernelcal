"""kernelcal.urban.spectrum -- spectral diagnostics on CityGraph Laplacians.

Small, dependency-light helpers for the PR-C planned-vs-organic ΔH
receipt and for downstream city-scale diagnostics.

The Laplacian of a Gaussian-weighted proximity graph is positive
semi-definite with multiplicity-of-zero equal to the number of
connected components.  These helpers turn that spectrum into three
scalar diagnostics the receipt asks for:

* :func:`spectral_entropy` -- normalised Shannon entropy of the
  non-zero eigenvalue distribution.  Smaller H means a more
  controller-shaped spectrum (concentrated on a few low modes); larger
  H means a broader, less ordered spectrum.

* :func:`betti_zero` -- the count of near-zero eigenvalues of L,
  i.e. the connected-component count.  Lifts the road-graph
  ``disconnection-as-signal`` of Field Note 107 to a scalar.

* :func:`normalised_top_k_spectrum` -- the top-k normalised
  eigenvalues, used both for the report tables and for σ-matching
  verification across graph modes.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def betti_zero(eigvals: np.ndarray, *, tol: float = 1e-9) -> int:
    """Count near-zero eigenvalues of a graph Laplacian.

    For a Gaussian-weighted graph Laplacian this equals the number of
    connected components (β₀).
    """
    arr = np.asarray(eigvals, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"eigvals must be 1-D; got shape {arr.shape}")
    return int(np.count_nonzero(np.abs(arr) <= float(tol)))


def spectral_entropy(
    eigvals: np.ndarray,
    *,
    tol: float = 1e-9,
    normalise: bool = True,
) -> float:
    """Shannon entropy of the non-zero Laplacian spectrum.

    Parameters
    ----------
    eigvals
        1-D array of Laplacian eigenvalues.  Negative entries are
        clipped to zero (they only appear from numerical drift).
    tol
        Eigenvalues with magnitude below ``tol`` are treated as the
        connected-component zero modes and dropped before the
        probability normalisation.
    normalise
        If ``True`` (default), divide by ``log(k)`` where ``k`` is the
        number of non-zero eigenvalues, giving a value in ``[0, 1]``
        that is comparable across graphs of different sizes.  If
        ``False``, return the raw nat-entropy.

    Returns
    -------
    H
        Spectral entropy in nats (or in ``[0, 1]`` if ``normalise``).

    Notes
    -----
    For a regular grid the spectrum concentrates on a few well-spaced
    plateaux (a strong controller has shifted the modal mass downward),
    yielding a smaller H.  For an organic fabric the spectrum is more
    uniformly spread, yielding a larger H.  The Field Notes 38/39
    prediction is therefore ``H_grid < H_fringe`` in road-aware mode.
    """
    arr = np.asarray(eigvals, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"eigvals must be 1-D; got shape {arr.shape}")
    arr = np.clip(arr, 0.0, None)
    nonzero = arr[arr > float(tol)]
    if nonzero.size == 0:
        return 0.0
    p = nonzero / float(nonzero.sum())
    p = np.clip(p, 1e-300, None)
    H = float(-(p * np.log(p)).sum())
    if normalise and nonzero.size > 1:
        H /= float(np.log(nonzero.size))
    return H


def normalised_top_k_spectrum(
    eigvals: np.ndarray,
    *,
    k: int = 50,
    drop_zero: bool = True,
    tol: float = 1e-9,
) -> np.ndarray:
    """Return the top-``k`` normalised eigenvalues for cross-graph
    comparison.

    The result is sorted descending, length ``min(k, n_nonzero)``,
    summing to 1.  Used both for the report tables and for σ-matching
    verification (two graphs built with σ-matched parameters should
    have closely-aligned top-``k`` spectra).
    """
    arr = np.asarray(eigvals, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"eigvals must be 1-D; got shape {arr.shape}")
    arr = np.clip(arr, 0.0, None)
    if drop_zero:
        arr = arr[arr > float(tol)]
    if arr.size == 0:
        return np.zeros(0, dtype=float)
    top = np.sort(arr)[::-1][:k]
    s = float(top.sum())
    if s <= 0.0:
        return np.zeros_like(top)
    return top / s


def sigma_matched_spectrum_diff(
    eigvals_a: np.ndarray,
    eigvals_b: np.ndarray,
    *,
    k: int = 50,
) -> float:
    """Total-variation distance between two normalised top-``k`` spectra.

    Used to verify that σ-matching across graph-construction modes
    keeps the two Laplacian spectra in the same ballpark.  TV-distance
    in ``[0, 1]``; small means the modes really are comparing
    structurally similar graphs.
    """
    pa = normalised_top_k_spectrum(eigvals_a, k=k)
    pb = normalised_top_k_spectrum(eigvals_b, k=k)
    n = max(pa.size, pb.size)
    if n == 0:
        return 0.0
    pa_pad = np.zeros(n, dtype=float)
    pb_pad = np.zeros(n, dtype=float)
    pa_pad[: pa.size] = pa
    pb_pad[: pb.size] = pb
    return 0.5 * float(np.abs(pa_pad - pb_pad).sum())


def spectral_diagnostics(eigvals: np.ndarray) -> dict:
    """Bundle the receipt-relevant diagnostics into a JSON-friendly dict.

    Returns
    -------
    dict
        ``{'beta_0', 'spectral_entropy_normalised',
        'spectral_entropy_nats', 'top_k': list, 'n_eigvals'}``.
    """
    arr = np.asarray(eigvals, dtype=float)
    return {
        "n_eigvals": int(arr.size),
        "beta_0": betti_zero(arr),
        "spectral_entropy_nats": spectral_entropy(arr, normalise=False),
        "spectral_entropy_normalised": spectral_entropy(arr, normalise=True),
        "top_k": normalised_top_k_spectrum(arr, k=50).tolist(),
    }


__all__ = [
    "betti_zero",
    "normalised_top_k_spectrum",
    "sigma_matched_spectrum_diff",
    "spectral_diagnostics",
    "spectral_entropy",
]
