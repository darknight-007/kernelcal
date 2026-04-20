"""Canonical spectral-entropy helper shared across kernelcal subpackages.

Before this module existed, three subpackages (``kernelcal.spectral.dynamics``,
``kernelcal.terrain.diagnostics``, ``kernelcal.bio.sleep_eeg``) each carried
their own ``spectral_entropy`` implementation with subtly different
zero-handling behavior. They now all delegate to :func:`spectral_entropy`
defined here so there is a single policy on zeros, a single docstring, and
no risk of formula drift.

The raw entropy of a nonnegative spectrum ``h`` is

    H[h] = - sum_l h_bar_l * log(h_bar_l)        with  h_bar_l = h_l / sum_l' h_l'

which lies in ``[0, log(N)]``. Passing ``normalize=True`` returns
``H[h] / log(N)`` so the value sits in ``[0, 1]`` — useful when comparing
spectra of different sizes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def spectral_entropy(h: ArrayLike, *, normalize: bool = False) -> float:
    """Shannon entropy of a normalized nonnegative spectrum.

    Parameters
    ----------
    h
        Array-like of nonnegative spectral weights (e.g. eigenvalues,
        heat-kernel coefficients, or any positive density). ``NaN`` /
        ``inf`` entries or negative sums are treated as degenerate and
        produce ``0.0``.
    normalize
        If ``True``, divide the result by ``log(N)`` where ``N = h.size``
        so the output lies in ``[0, 1]``. Default is ``False`` (raw
        entropy in nats, in ``[0, log(N)]``).

    Returns
    -------
    float
        Spectral entropy. Exactly ``0.0`` when the total weight is
        nonpositive or nonfinite, and also when ``normalize=True`` and
        ``N <= 1``.

    Notes
    -----
    Zero entries contribute the usual ``0 * log(0) := 0`` via a
    ``log(1) = 0`` masking trick. This matches the conventions
    previously used in both ``kernelcal.spectral.dynamics.spectral_entropy``
    and ``kernelcal.terrain.diagnostics.spectral_entropy`` (filtering
    zeros before normalization is mathematically equivalent, because
    zero entries do not change the denominator).
    """
    h_arr = np.asarray(h, dtype=float)
    total = float(h_arr.sum())
    if total <= 0.0 or not np.isfinite(total):
        return 0.0

    h_bar = h_arr / total
    # Mask nonpositive (and non-finite) entries with 1.0 so log(1) = 0
    # cancels them out of the entropy sum without triggering log(0).
    safe = np.where(np.isfinite(h_bar) & (h_bar > 0.0), h_bar, 1.0)
    raw = float(-np.sum(safe * np.log(safe)))

    if not normalize:
        return raw
    N = int(h_arr.size)
    if N <= 1:
        return 0.0
    return raw / float(np.log(N))


__all__ = ["spectral_entropy"]
