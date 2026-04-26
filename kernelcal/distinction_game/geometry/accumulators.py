"""Per-superquadric running statistics for material/health properties.

Streaming sensor data on the earth-rover is reduced to *running* per-SQ
statistics (Welford's algorithm) so that the on-board memory footprint
is O(1) per (SQ, property) pair regardless of how many observations
each SQ accumulates.  At property-publish time the accumulator emits a
single scalar per property (the running mean), which is then quantized
through the registry in :mod:`.properties` and packed into the wire
trailer.

Why running statistics instead of "last value wins"
---------------------------------------------------

* **Multi-pass aggregation** -- a tree's trunk SQ may be hit by many
  bore-sight rays as the rover passes by.  Each ray contributes a
  spectrum or a temperature reading; the per-SQ mean is robust to
  individual noisy samples.

* **Confidence proxy** -- ``count`` (and optionally ``variance``)
  provide a free confidence estimate that downstream factor-graph
  weights can use without a separate calibration pass.

* **Numerically stable** -- Welford avoids the classic
  ``sum(x^2) - sum(x)^2 / n`` cancellation; works reliably for the
  full int16-quantized range we emit.

* **Compatible with property registry** -- when a property arrives
  on the wire with ``OBSERVATION_COUNT`` and ``CONFIDENCE`` it can be
  *blended* with a server-side accumulator using only those three
  numbers, no per-sample history needed.

Public API
----------

::

    from kernelcal.distinction_game.geometry.accumulators import (
        WelfordAccumulator,
        SuperquadricPropertyStore,
        merge_property_stores,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np

from .properties import PropertyId, PropertySpec, get_spec


# ---------------------------------------------------------------------------
# Welford accumulator (single scalar)
# ---------------------------------------------------------------------------


@dataclass
class WelfordAccumulator:
    """Running mean + variance with optional sample weights.

    Implements Welford's algorithm in its weighted form (West 1979),
    so callers can pass per-sample weights without re-deriving the
    update.  Weights default to 1.0 (uniform).

    Attributes
    ----------
    n
        Number of samples accumulated (count, not weight sum).
    weight_sum
        Sum of sample weights so far.  Equals ``n`` for unit weights.
    mean
        Running weighted mean.
    M2
        Sum of weighted squared deviations from the running mean
        (Welford's ``M2`` quantity).  Used to derive ``variance``.
    min_value, max_value
        Running min/max for sanity checks (cheap to maintain).

    Notes
    -----
    For empty accumulators ``mean`` is ``0.0`` and ``variance`` is
    ``0.0``.  The ``finalize`` accessor returns ``mean`` (always) and
    ``std`` (defined only for ``n >= 2``).
    """

    n: int = 0
    weight_sum: float = 0.0
    mean: float = 0.0
    M2: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf

    def update(self, value: float, weight: float = 1.0) -> None:
        """Incorporate a single sample with optional weight."""
        if not math.isfinite(value):
            return
        if weight <= 0.0 or not math.isfinite(weight):
            return
        self.n += 1
        new_w = self.weight_sum + weight
        delta = value - self.mean
        # Weighted Welford update (West 1979 Algorithm 1)
        self.mean += weight * delta / new_w
        delta2 = value - self.mean
        self.M2 += weight * delta * delta2
        self.weight_sum = new_w
        if value < self.min_value:
            self.min_value = value
        if value > self.max_value:
            self.max_value = value

    def update_batch(self, values: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        """Vectorized update for a numpy array of samples.

        Equivalent to looping over :meth:`update` but uses a single
        Welford merge step (Chan-Golub-LeVeque combine formula) for
        speed.  Discards non-finite samples / weights.
        """
        v = np.asarray(values, dtype=float).ravel()
        if v.size == 0:
            return
        if weights is None:
            w = np.ones_like(v)
        else:
            w = np.asarray(weights, dtype=float).ravel()
            if w.shape != v.shape:
                raise ValueError(
                    f"WelfordAccumulator.update_batch: weights shape "
                    f"{w.shape} != values shape {v.shape}"
                )
        good = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
        v = v[good]
        w = w[good]
        if v.size == 0:
            return

        # Compute batch mean / M2 with weights.
        wsum = float(w.sum())
        if wsum <= 0.0:
            return
        batch_mean = float(np.sum(w * v) / wsum)
        batch_M2 = float(np.sum(w * (v - batch_mean) ** 2))

        # Merge with self (Chan et al. 1983 parallel combination).
        if self.weight_sum == 0.0:
            self.n = int(v.size)
            self.weight_sum = wsum
            self.mean = batch_mean
            self.M2 = batch_M2
        else:
            new_n = self.n + int(v.size)
            new_w = self.weight_sum + wsum
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * wsum / new_w
            new_M2 = self.M2 + batch_M2 + (delta ** 2) * (self.weight_sum * wsum / new_w)
            self.n = new_n
            self.weight_sum = new_w
            self.mean = new_mean
            self.M2 = new_M2

        bmin = float(v.min())
        bmax = float(v.max())
        if bmin < self.min_value:
            self.min_value = bmin
        if bmax > self.max_value:
            self.max_value = bmax

    def variance(self, ddof: int = 1) -> float:
        """Sample variance (default Bessel-corrected, ``ddof=1``)."""
        denom = max(self.weight_sum - ddof, 1e-12)
        if self.n < 2:
            return 0.0
        return float(max(self.M2 / denom, 0.0))

    def std(self, ddof: int = 1) -> float:
        """Sample standard deviation."""
        return float(math.sqrt(self.variance(ddof=ddof)))

    def merge(self, other: "WelfordAccumulator") -> None:
        """In-place merge of another accumulator (Chan combine)."""
        if other.weight_sum <= 0.0:
            return
        if self.weight_sum <= 0.0:
            self.n = other.n
            self.weight_sum = other.weight_sum
            self.mean = other.mean
            self.M2 = other.M2
            self.min_value = other.min_value
            self.max_value = other.max_value
            return
        new_n = self.n + other.n
        new_w = self.weight_sum + other.weight_sum
        delta = other.mean - self.mean
        new_mean = self.mean + delta * other.weight_sum / new_w
        new_M2 = (
            self.M2
            + other.M2
            + (delta ** 2) * (self.weight_sum * other.weight_sum / new_w)
        )
        self.n = new_n
        self.weight_sum = new_w
        self.mean = new_mean
        self.M2 = new_M2
        if other.min_value < self.min_value:
            self.min_value = other.min_value
        if other.max_value > self.max_value:
            self.max_value = other.max_value

    def reset(self) -> None:
        """Clear accumulated state."""
        self.n = 0
        self.weight_sum = 0.0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_value = math.inf
        self.max_value = -math.inf

    def finalize(self) -> Tuple[float, float, int]:
        """Return ``(mean, std, n)``; ``std`` is 0 when ``n < 2``."""
        return float(self.mean), float(self.std()), int(self.n)


# ---------------------------------------------------------------------------
# Per-SQ store: many properties per primitive
# ---------------------------------------------------------------------------


@dataclass
class SuperquadricPropertyStore:
    """All per-SQ accumulators (one Welford per property).

    Caller pattern (earth-rover side)::

        store = SuperquadricPropertyStore(sq_id="sq-abc123")
        for ray, ndvi_value in lidar_intensity_stream:
            store.update(PropertyId.LIDAR_INTENSITY_MEAN, intensity, weight=1.0)
        ...
        # At publish time:
        props = store.finalize_for_packing()  # {PropertyId: float}
        trailer = encode_property_trailer(props)

    Server-side merge (incremental fusion)::

        merged = merge_property_stores(server_store, observation_store)
    """

    sq_id: str
    accumulators: Dict[PropertyId, WelfordAccumulator] = field(default_factory=dict)
    spectrum: Optional["SpectrumAccumulator"] = None  # forward ref; see spectrum.py

    # ---- Single-property update --------------------------------------

    def update(
        self,
        pid: Union[PropertyId, int, str],
        value: float,
        weight: float = 1.0,
    ) -> None:
        """Update a single property accumulator with one sample."""
        spec = get_spec(pid)
        acc = self.accumulators.get(spec.pid)
        if acc is None:
            acc = WelfordAccumulator()
            self.accumulators[spec.pid] = acc
        acc.update(float(value), weight=float(weight))

    def update_batch(
        self,
        pid: Union[PropertyId, int, str],
        values: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """Vectorized batch update for one property."""
        spec = get_spec(pid)
        acc = self.accumulators.get(spec.pid)
        if acc is None:
            acc = WelfordAccumulator()
            self.accumulators[spec.pid] = acc
        acc.update_batch(values, weights=weights)

    def update_many(
        self,
        properties: Mapping[Union[PropertyId, int, str], float],
        weight: float = 1.0,
    ) -> None:
        """Update several properties from a single observation."""
        for pid, value in properties.items():
            self.update(pid, value, weight=weight)

    # ---- Inspection --------------------------------------------------

    def has(self, pid: Union[PropertyId, int, str]) -> bool:
        """Whether this store has any samples for ``pid``."""
        try:
            spec = get_spec(pid)
        except KeyError:
            return False
        acc = self.accumulators.get(spec.pid)
        return acc is not None and acc.n > 0

    def mean(self, pid: Union[PropertyId, int, str], default: float = 0.0) -> float:
        """Running mean for a property (or ``default`` if unobserved)."""
        spec = get_spec(pid)
        acc = self.accumulators.get(spec.pid)
        if acc is None or acc.n == 0:
            return float(default)
        return float(acc.mean)

    def count(self, pid: Union[PropertyId, int, str]) -> int:
        """Number of samples for a property (0 if unobserved)."""
        try:
            spec = get_spec(pid)
        except KeyError:
            return 0
        acc = self.accumulators.get(spec.pid)
        return int(acc.n) if acc is not None else 0

    def aggregate_count(self) -> int:
        """Maximum sample count across all properties (= overall obs count)."""
        if not self.accumulators:
            return 0
        return max(acc.n for acc in self.accumulators.values())

    # ---- Finalization ------------------------------------------------

    def finalize(self) -> Dict[PropertyId, float]:
        """All means, regardless of sample count.

        Returns ``{PropertyId -> mean}`` for every property that has
        any samples.  Useful for inspection / debugging.
        """
        return {
            pid: float(acc.mean)
            for pid, acc in self.accumulators.items()
            if acc.n > 0
        }

    def finalize_for_packing(
        self,
        *,
        min_samples: int = 1,
        include_metadata: bool = True,
    ) -> Dict[PropertyId, float]:
        """Return properties stable enough to put on the wire.

        Parameters
        ----------
        min_samples
            Drop properties with fewer than this many contributing
            samples.  Defaults to ``1`` (include any observed property).
        include_metadata
            If True, also emit ``OBSERVATION_COUNT`` and ``CONFIDENCE``
            (computed from the median sample count and inverse-variance
            blend, respectively).
        """
        out: Dict[PropertyId, float] = {}
        for pid, acc in self.accumulators.items():
            if acc.n < min_samples:
                continue
            out[pid] = float(acc.mean)

        if include_metadata and out:
            counts = [a.n for a in self.accumulators.values() if a.n > 0]
            if counts:
                out[PropertyId.OBSERVATION_COUNT] = float(np.median(counts))
                # Confidence proxy: 1 - (mean relative std across props),
                # clamped to [0, 1].  Robust enough for downstream weighting.
                rel_stds = []
                for acc in self.accumulators.values():
                    if acc.n < 2:
                        continue
                    s = acc.std()
                    denom = max(abs(acc.mean), 1e-3)
                    rel_stds.append(min(s / denom, 1.0))
                if rel_stds:
                    conf = float(np.clip(1.0 - np.mean(rel_stds), 0.0, 1.0))
                else:
                    conf = 1.0  # only single-sample data; trust it provisionally
                out[PropertyId.CONFIDENCE] = conf
        return out

    # ---- Lifecycle ---------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulator state (keeps sq_id, drops samples)."""
        for acc in self.accumulators.values():
            acc.reset()
        if self.spectrum is not None:
            try:
                self.spectrum.reset()
            except AttributeError:
                pass

    def merge(self, other: "SuperquadricPropertyStore") -> None:
        """In-place merge of another store's accumulators (same SQ id)."""
        if other.sq_id and self.sq_id and other.sq_id != self.sq_id:
            raise ValueError(
                f"merge: sq_id mismatch {self.sq_id!r} vs {other.sq_id!r}"
            )
        for pid, other_acc in other.accumulators.items():
            mine = self.accumulators.get(pid)
            if mine is None:
                self.accumulators[pid] = WelfordAccumulator(
                    n=other_acc.n,
                    weight_sum=other_acc.weight_sum,
                    mean=other_acc.mean,
                    M2=other_acc.M2,
                    min_value=other_acc.min_value,
                    max_value=other_acc.max_value,
                )
            else:
                mine.merge(other_acc)
        if other.spectrum is not None:
            if self.spectrum is None:
                self.spectrum = other.spectrum
            else:
                try:
                    self.spectrum.merge(other.spectrum)
                except AttributeError:
                    # Spectrum module may not be importable on the
                    # server in some configurations; tolerate.
                    pass


# ---------------------------------------------------------------------------
# Convenience: server-side merge of many incoming stores
# ---------------------------------------------------------------------------


def merge_property_stores(
    *stores: SuperquadricPropertyStore,
) -> Optional[SuperquadricPropertyStore]:
    """Merge multiple stores for the *same* SQ into a single store.

    Returns ``None`` if no stores are provided.  All input stores must
    share an ``sq_id`` (or all be empty).
    """
    stores = tuple(s for s in stores if s is not None)
    if not stores:
        return None
    head = stores[0]
    out = SuperquadricPropertyStore(sq_id=head.sq_id)
    for s in stores:
        out.merge(s)
    return out


# ---------------------------------------------------------------------------
# Reconstruction helper: arrived-on-wire properties -> store
# ---------------------------------------------------------------------------


def store_from_decoded_trailer(
    sq_id: str,
    decoded: Mapping[PropertyId, float],
    *,
    sample_count_hint: Optional[int] = None,
) -> SuperquadricPropertyStore:
    """Reconstruct a store from a decoded property trailer.

    Trailer-decoded values are *means*; the receiver doesn't know the
    underlying samples.  We synthesize a synthetic single-observation
    accumulator per property so the store can still merge with future
    observations.  If ``sample_count_hint`` is given (or
    ``OBSERVATION_COUNT`` is present in ``decoded``), it is used as
    the synthetic ``n``.
    """
    store = SuperquadricPropertyStore(sq_id=sq_id)
    n_hint = sample_count_hint
    if n_hint is None and PropertyId.OBSERVATION_COUNT in decoded:
        n_hint = max(int(decoded[PropertyId.OBSERVATION_COUNT]), 1)
    n_hint = int(max(n_hint or 1, 1))

    for pid, value in decoded.items():
        if pid == PropertyId.OBSERVATION_COUNT:
            continue  # don't wrap the count into an accumulator
        acc = WelfordAccumulator(
            n=n_hint,
            weight_sum=float(n_hint),
            mean=float(value),
            M2=0.0,
            min_value=float(value),
            max_value=float(value),
        )
        store.accumulators[pid] = acc
    return store


__all__ = [
    "WelfordAccumulator",
    "SuperquadricPropertyStore",
    "merge_property_stores",
    "store_from_decoded_trailer",
]
