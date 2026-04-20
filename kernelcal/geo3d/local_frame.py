"""Equirectangular lon/lat → local metres projection.

A single, immutable dataclass used whenever kernelcal needs a cheap local
Cartesian frame for a small survey area (< ~50 km across, where the
equirectangular approximation is accurate to well under 1%). Examples
include the Bishop scarp rocks explorer and any future drone / DEM
experiment that has to turn WGS-84 degrees into metres for graph
construction.

Historically, ``LocalFrame`` lived inline in ``bishop_rocks_graph_explorer.py``.
Moving it into the :mod:`kernelcal.geo3d` package avoids duplicating the
formula in every script and gives a single canonical location to extend
(e.g. with UTM / a true ENU tangent-plane frame) without touching callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

# ``111,320 m`` is the conventional equirectangular scale for 1° of latitude;
# the corresponding longitude scale is ``METERS_PER_DEG_LAT * cos(lat0)`` so
# that east-west distances shrink toward the poles the way they should.
METERS_PER_DEG_LAT: float = 111_320.0


@dataclass(frozen=True)
class LocalFrame:
    """Equirectangular projection centred on ``(lon0, lat0)`` in degrees.

    Good for small regions (< ~50 km) where the latitudinal cosine is
    essentially constant. For larger footprints prefer a proper UTM or
    tangent-plane ENU frame.

    Parameters
    ----------
    lon0, lat0
        Origin of the local frame in degrees. ``to_xy(lon0, lat0)``
        returns ``(0, 0)`` by construction.

    Examples
    --------
    >>> f = LocalFrame(lon0=-118.44, lat0=37.45)
    >>> x, y = f.to_xy(-118.44, 38.45)      # one degree north
    >>> int(round(float(y)))
    111320
    """

    lon0: float
    lat0: float

    # ------------------------------------------------------------------
    # Forward projection (lon/lat [deg] -> x/y [m])
    # ------------------------------------------------------------------
    def to_xy(
        self,
        lon: ArrayLike,
        lat: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project degree coordinates to local metres.

        Accepts scalars or arrays; the returned arrays match the input
        broadcasting shape and are always ``float64``.
        """
        lon_arr = np.asarray(lon, dtype=float)
        lat_arr = np.asarray(lat, dtype=float)
        lat_rad = np.deg2rad(self.lat0)
        cos_lat = float(np.cos(lat_rad))
        x = (lon_arr - self.lon0) * METERS_PER_DEG_LAT * cos_lat
        y = (lat_arr - self.lat0) * METERS_PER_DEG_LAT
        return x.astype(float), y.astype(float)

    # ------------------------------------------------------------------
    # Inverse projection (x/y [m] -> lon/lat [deg]) — useful for tests
    # and for rendering map overlays in the original geographic frame.
    # ------------------------------------------------------------------
    def to_lonlat(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of :meth:`to_xy`.

        Accepts scalars or arrays; returned arrays are ``float64``.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        lat_rad = np.deg2rad(self.lat0)
        cos_lat = float(np.cos(lat_rad))
        lon = self.lon0 + x_arr / (METERS_PER_DEG_LAT * cos_lat)
        lat = self.lat0 + y_arr / METERS_PER_DEG_LAT
        return lon.astype(float), lat.astype(float)


__all__ = ["LocalFrame", "METERS_PER_DEG_LAT"]
