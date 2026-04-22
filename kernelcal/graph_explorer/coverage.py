"""Shared visited-area coverage raster for graph-based adaptive mappers.

This is the bishop analog of the drone-DEM explorer's ``visited`` numpy
array (bool mask over the DEM pixel grid).  Bishop has no natural
background raster because its input is a point cloud of rock
centroids, so :class:`CoverageRaster` constructs one over a caller-
provided metric bbox at a caller-chosen resolution.  Both explorers
use the same *square-footprint* coverage update::

    visited[r0:r1, c0:c1] = True

for the drone's patch (DEM: inline numpy; bishop: via
:meth:`CoverageRaster.mark_square`), and the same::

    unseen = 1 - mean(visited[target_patch])

at scoring time.  That means the ``unseen_frac`` term fed into the
shared :func:`~kernelcal.graph_explorer.planner.score_betti_candidate`
rule has identical semantics across both examples: it is the fraction
of the *target footprint's ground area* that has not yet been covered
by any past scan — not a per-feature count.

Coordinate convention
---------------------
``bbox = (xmin, xmax, ymin, ymax)`` in metres.  Row 0 of the internal
mask is the *southern* edge (``y == ymin``) and row grows northward,
so mask indexing with ``[r, c]`` reads ``y = ymin + r · resolution`` /
``x = xmin + c · resolution``.  This differs from the DEM explorer's
image-style ``(row = south→north-down, col = west→east)`` convention
on purpose: bishop's metric axes have y growing north everywhere, and
flipping the mask here keeps it consistent with the rest of the
bishop explorer's plotting.  Callers that want to render the mask
with a standard ``imshow(..., origin='lower')`` get a picture aligned
with the bishop scatter plot.
"""

from __future__ import annotations

import math

import numpy as np


__all__ = ["CoverageRaster"]


class CoverageRaster:
    """Bool visited-mask over a metric bbox at fixed resolution.

    Parameters
    ----------
    bbox
        ``(xmin, xmax, ymin, ymax)`` in metres.  ``xmax > xmin`` and
        ``ymax > ymin`` are required.
    resolution_m
        Side of one mask pixel in metres.  Must be positive.
    """

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        resolution_m: float,
    ) -> None:
        xmin, xmax, ymin, ymax = (float(v) for v in bbox)
        if not (xmax > xmin and ymax > ymin):
            raise ValueError(
                f"bbox must satisfy xmax>xmin and ymax>ymin, got {bbox}"
            )
        res = float(resolution_m)
        if not (res > 0.0):
            raise ValueError(
                f"resolution_m must be > 0, got {resolution_m!r}"
            )

        # Clamp to at least 1 row / col so tiny bboxes still produce a
        # valid mask.  ``math.ceil`` ensures no metric extent is lost.
        ncols = max(1, int(math.ceil((xmax - xmin) / res)))
        nrows = max(1, int(math.ceil((ymax - ymin) / res)))

        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmin + ncols * res  # actual covered extent
        self._ymax = ymin + nrows * res
        self._res = res
        self._mask = np.zeros((nrows, ncols), dtype=bool)

    # -- accessors ----------------------------------------------------------

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Actual covered bbox (may extend beyond the input by < resolution_m)."""
        return (self._xmin, self._xmax, self._ymin, self._ymax)

    @property
    def resolution_m(self) -> float:
        return self._res

    @property
    def mask(self) -> np.ndarray:
        """The bool visited mask.  Shape ``(nrows, ncols)``, row 0 = south."""
        return self._mask

    @property
    def shape(self) -> tuple[int, int]:
        return self._mask.shape

    @property
    def visited_fraction(self) -> float:
        """Fraction of the raster currently marked visited, in ``[0, 1]``."""
        return float(np.mean(self._mask))

    # -- index helpers ------------------------------------------------------

    def _square_slice(
        self, cx: float, cy: float, side_m: float
    ) -> tuple[slice, slice]:
        """``(row_slice, col_slice)`` for a square centred on ``(cx, cy)``.

        Slices are clamped to the raster extent so querying / marking
        near the edge is safe.  A square that falls entirely outside
        the raster returns empty slices (``stop == start``), which is
        a valid slice that yields an empty view.
        """
        half = 0.5 * float(side_m)
        x0 = cx - half
        x1 = cx + half
        y0 = cy - half
        y1 = cy + half
        nrows, ncols = self._mask.shape

        c0 = int(math.floor((x0 - self._xmin) / self._res))
        c1 = int(math.ceil((x1 - self._xmin) / self._res))
        r0 = int(math.floor((y0 - self._ymin) / self._res))
        r1 = int(math.ceil((y1 - self._ymin) / self._res))

        c0 = max(0, min(ncols, c0))
        c1 = max(0, min(ncols, c1))
        r0 = max(0, min(nrows, r0))
        r1 = max(0, min(nrows, r1))
        return slice(r0, r1), slice(c0, c1)

    # -- mutations ----------------------------------------------------------

    def mark_square(self, cx: float, cy: float, side_m: float) -> None:
        """Mark every mask pixel inside the ``side_m × side_m`` square visited.

        Matches the DEM explorer's ``visited[r0:r1, c0:c1] = True`` update.
        Squares that fall entirely outside the raster are a no-op.
        """
        rs, cs = self._square_slice(cx, cy, side_m)
        if rs.stop > rs.start and cs.stop > cs.start:
            self._mask[rs, cs] = True

    # -- queries ------------------------------------------------------------

    def unseen_fraction_at(
        self, cx: float, cy: float, side_m: float
    ) -> float:
        """``1 - mean(visited[target_square])``, matching the DEM explorer.

        A target square that lies entirely outside the raster is treated
        as *fully unseen* (returns ``1.0``) because nothing has been
        visited there; this matches the DEM behaviour where
        ``visited[tr0:tr1, tc0:tc1]`` on an out-of-bounds slice would
        have mean 0.  Callers that want to reject such targets should
        clamp the candidate position to the bbox beforehand.
        """
        rs, cs = self._square_slice(cx, cy, side_m)
        if rs.stop <= rs.start or cs.stop <= cs.start:
            return 1.0
        return 1.0 - float(np.mean(self._mask[rs, cs]))

    # -- plotting helper ----------------------------------------------------

    def extent(self) -> tuple[float, float, float, float]:
        """``(xmin, xmax, ymin, ymax)`` for matplotlib ``imshow(extent=...)``."""
        return (self._xmin, self._xmax, self._ymin, self._ymax)
