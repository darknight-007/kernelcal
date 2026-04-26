"""Parametric 3D primitives for compact scene representation.

This subpackage provides a single unifying primitive — the
**superquadric** — which subsumes cuboids, cylinders, ellipsoids, spheres,
cone-like shapes, and intermediates under one parametric form.

The wire footprint per posed primitive is ~32 bytes for the geometry,
with optional variable-length trailers for derived material/health
properties and a 96-byte UV-VIS-NIR spectrum sidecar -- making the
bandwidth budget for an earth-rover ``->`` server fusion link
~30-120 kbps depending on which trailers are emitted.

Public API
----------

::

    from kernelcal.distinction_game.geometry import (
        Superquadric, fit_superquadric, fit_tree,
        pack_superquadric, unpack_superquadric, packed_size,
        PropertyId, encode_property_trailer, decode_property_trailer,
        WelfordAccumulator, SuperquadricPropertyStore,
        SpectrumPacket, SpectrumAccumulator,
        compress_spectrum, decompress_spectrum,
        SQSpatialIndex,
        LidarIntensityAttributor,
        MicaSenseAttributor,
        OceanOpticsAttributor,
        EPS_MIN, EPS_MAX,
    )

References
----------
Solina, F. & Bajcsy, R. (1990). "Recovery of Parametric Models from
Range Images: The Case for Superquadrics with Global Deformations."
*IEEE TPAMI*, 12(2), 131-147.

Liu, W., Wu, Y., Ruan, S., & Chirikjian, G. S. (2022). "Robust and
Accurate Superquadric Recovery: A Probabilistic Approach." *CVPR 2022*.

Paschalidou, D., Ulusoy, A. O., & Geiger, A. (2019). "Superquadrics
Revisited: Learning 3D Shape Parsing beyond Cuboids." *CVPR 2019*.
"""

from __future__ import annotations

from .superquadric import (
    EPS_MAX,
    EPS_MIN,
    Superquadric,
    superquadric_box,
    superquadric_cylinder,
    superquadric_ellipsoid,
    superquadric_sphere,
)
from .fit import (
    FitDiagnostics,
    SuperquadricFit,
    fit_superquadric,
    fit_tree,
)
from .properties import (
    MAX_PROPERTIES_PER_TRAILER,
    PropertyId,
    PropertySpec,
    all_property_ids,
    all_specs,
    decode_property_trailer,
    encode_property_trailer,
    encoded_size,
    get_spec,
    quantize_round_trip_error,
)
from .accumulators import (
    SuperquadricPropertyStore,
    WelfordAccumulator,
    merge_property_stores,
    store_from_decoded_trailer,
)
from .spectrum import (
    DCT_RETAINED,
    FLAG_HAS_PCA,
    FLAG_LOG_DCT,
    FLAG_NORMALIZED,
    PCA_RETAINED,
    SPECTRUM_PACKET_BYTES,
    SpectrumAccumulator,
    SpectrumPacket,
    compress_spectrum,
    decompress_spectrum,
)
from .attribution import (
    LidarIntensityAttributor,
    MicaSenseAttributor,
    OceanOpticsAttributor,
    SQSpatialIndex,
    evi,
    gndvi,
    ndre,
    ndvi,
)
from .codec import (
    DEFAULT_T_RESOLUTION_M,
    FLAG_HAS_PARENT,
    FLAG_HAS_PROPERTIES,
    FLAG_HAS_SPECTRUM,
    PACKED_BYTES,
    PACKED_PARENT_BYTES,
    PACKED_SPECTRUM_BYTES,
    pack_superquadric,
    packed_size,
    unpack_superquadric,
)
from .frames import (
    FRAME_KINDS,
    FrameSpec,
    R_ENU_TO_NED,
    R_NED_TO_ENU,
    WGS84_A,
    WGS84_B,
    WGS84_E2,
    WGS84_F,
    ecef_to_enu,
    ecef_to_geodetic,
    enu_basis_at,
    enu_to_ecef,
    enu_to_ned,
    geodetic_to_ecef,
    geodetic_to_utm,
    ned_to_enu,
    transform_point,
    transform_pose,
    transform_superquadric,
    transform_superquadrics,
    utm_to_geodetic,
    utm_zone_for,
)

__all__ = [
    # Superquadric primitive
    "EPS_MIN",
    "EPS_MAX",
    "Superquadric",
    "superquadric_box",
    "superquadric_cylinder",
    "superquadric_ellipsoid",
    "superquadric_sphere",
    # Fitting
    "FitDiagnostics",
    "SuperquadricFit",
    "fit_superquadric",
    "fit_tree",
    # Property registry
    "PropertyId",
    "PropertySpec",
    "MAX_PROPERTIES_PER_TRAILER",
    "all_property_ids",
    "all_specs",
    "decode_property_trailer",
    "encode_property_trailer",
    "encoded_size",
    "get_spec",
    "quantize_round_trip_error",
    # Accumulators
    "WelfordAccumulator",
    "SuperquadricPropertyStore",
    "merge_property_stores",
    "store_from_decoded_trailer",
    # Spectrum codec
    "SpectrumPacket",
    "SpectrumAccumulator",
    "compress_spectrum",
    "decompress_spectrum",
    "SPECTRUM_PACKET_BYTES",
    "DCT_RETAINED",
    "PCA_RETAINED",
    "FLAG_HAS_PCA",
    "FLAG_LOG_DCT",
    "FLAG_NORMALIZED",
    # Attribution
    "SQSpatialIndex",
    "LidarIntensityAttributor",
    "MicaSenseAttributor",
    "OceanOpticsAttributor",
    "ndvi",
    "ndre",
    "gndvi",
    "evi",
    # Wire codec
    "PACKED_BYTES",
    "PACKED_PARENT_BYTES",
    "PACKED_SPECTRUM_BYTES",
    "FLAG_HAS_PARENT",
    "FLAG_HAS_PROPERTIES",
    "FLAG_HAS_SPECTRUM",
    "DEFAULT_T_RESOLUTION_M",
    "pack_superquadric",
    "unpack_superquadric",
    "packed_size",
    # Frames (PR-5.7)
    "FrameSpec",
    "FRAME_KINDS",
    "WGS84_A",
    "WGS84_B",
    "WGS84_F",
    "WGS84_E2",
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "enu_basis_at",
    "ecef_to_enu",
    "enu_to_ecef",
    "ned_to_enu",
    "enu_to_ned",
    "R_NED_TO_ENU",
    "R_ENU_TO_NED",
    "utm_to_geodetic",
    "geodetic_to_utm",
    "utm_zone_for",
    "transform_point",
    "transform_pose",
    "transform_superquadric",
    "transform_superquadrics",
]
