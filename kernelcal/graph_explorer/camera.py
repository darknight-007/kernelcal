"""Shared square-footprint camera model for graph-based adaptive mappers.

Both the drone-DEM explorer
(``examples/controller/drone_dem_betti_adaptive_experiment.py``) and the
bishop-rocks explorer
(``examples/bishop/bishop_rocks_graph_explorer.py``) derive the edge of
their square scan footprint from the same nadir-camera geometry::

    footprint_side_m = 2 · altitude_m · tan(fov_deg / 2)

The DEM explorer discretises that footprint on the DEM raster
(``footprint_side_px``); the bishop explorer discretises it on a
:class:`~kernelcal.graph_explorer.coverage.CoverageRaster` built over
the scarp bbox.  The ``resolution_m`` field is the raster pixel size
used for the ``_px`` conversion — DEM pixel spacing for the drone
experiment, a caller-chosen coverage pixel for bishop — and is
**not** interpreted by the camera itself.

Kept in the shared :mod:`kernelcal.graph_explorer` subpackage so both
explorers can import the exact same dataclass and stay in lock-step
when the formula changes (e.g. adding radial distortion or off-nadir
tilt would propagate to both).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = ["CameraModel"]


@dataclass(frozen=True)
class CameraModel:
    """Nadir square-footprint model: altitude + FOV → ground side length.

    Attributes
    ----------
    altitude_m
        Drone altitude above the ground in metres.  Must be positive.
    fov_deg
        Full field-of-view angle in degrees, ``0 < fov_deg < 180``.  The
        footprint edge is ``2·altitude·tan(fov/2)`` so values close to
        180° produce very large (or infinite) footprints — callers
        should keep ``fov_deg`` well under 180° in practice.
    resolution_m
        Raster pixel side in metres used for the ``_px`` conversion.
        For the drone-DEM explorer this is the DEM's native resolution;
        for the bishop explorer it is the ``CoverageRaster``'s pixel.
    """

    altitude_m: float
    fov_deg: float
    resolution_m: float

    @property
    def footprint_side_m(self) -> float:
        """Ground-side length of the square nadir footprint in metres."""
        half_angle = np.deg2rad(self.fov_deg * 0.5)
        return 2.0 * float(self.altitude_m) * float(np.tan(half_angle))

    @property
    def footprint_side_px(self) -> int:
        """Footprint edge in raster pixels (clamped to a minimum of 5)."""
        px = int(round(self.footprint_side_m / float(self.resolution_m)))
        return max(5, px)
