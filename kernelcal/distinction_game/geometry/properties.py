"""Material / health / appearance properties for posed superquadrics.

This module defines a *property registry*: a fixed enumeration of
high-level, semantically meaningful scalar properties that can be
attached to a :class:`Superquadric` for transmission over a bandwidth-
constrained link (e.g. 64-100 kbps over 915 MHz / WiFi from an
earth-rover to a fusion server running the kernelcal factor graph).

The trick is to *not* ship raw sensor channels.  Instead the on-board
attributors (see :mod:`.attribution`) reduce per-pixel / per-return
data to **derived material/health properties** -- NDVI, NDRE, surface
temperature, mean LiDAR intensity, albedo proxy, etc. -- each of which
quantizes into 1-2 bytes.  The wire format then carries a tiny
property *trailer* (1 byte length + 1 byte ID + N byte value per
property) appended to the existing 32-byte
:func:`pack_superquadric` payload.

Why this beats raw sensor sidecars
----------------------------------

* **Bandwidth** -- a 5-band MicaSense Altum frame is ~30 MB.  An NDVI
  + NDRE + surface-temp triple is 6 bytes per SQ.  At 200 SQs/sec the
  property trailer is ~1.2 kB/s = ~10 kbps.

* **Semantics** -- properties are pre-derived urban-ecology indicators
  (NDVI for vigor, NDRE for chlorophyll, surface temp for heat-island
  / transpirational cooling, ...).  Downstream consumers don't need
  to know the sensor calibration to interpret them.

* **Multi-sensor reduction** -- a property like ``surface_temp_C`` can
  come from MicaSense LWIR or from a thermal camera; the consumer
  doesn't care.  The on-board attributor decides which sensor feeds
  which property.

* **Welford-friendly** -- because every property is a scalar, the
  per-SQ accumulator (see :mod:`.accumulators`) reduces to a Welford
  running mean+variance in O(1) memory per (SQ, property) pair.

Wire format
-----------

::

    Property trailer (variable length, after the optional parent_hash):

    offset  size  field
    ------  ----  ----------------------------------------------------
    0       1     n_props (uint8, 0-255)
    1...    *     for each prop:
                    1 byte   PropertyId (uint8)
                    1-4 byte value (encoding from registry)

Each :class:`PropertySpec` declares ``bytes_per_value`` in
``{1, 2, 4}`` so the receiver can decode without an external schema:
the ID alone is sufficient because the registry is global and stable.

Adding a new property
---------------------

1. Pick an unused ``PropertyId`` integer.  Reserve ranges below for
   sensor families (``0x00-0x1F`` optical, ``0x20-0x2F`` thermal,
   ``0x30-0x3F`` LiDAR, ``0x40-0x4F`` material, ``0x50-0x5F``
   spectral indices, ``0x60-0x7F`` derived health,
   ``0xF0-0xFF`` statistics).
2. Add a :class:`PropertySpec` row to :data:`_PROPERTY_SPECS`.
3. Tests in ``tests/test_properties.py`` will round-trip the new
   property automatically (parametrized over all registry entries).
4. The earth-rover-side attributor (in :mod:`.attribution`) decides
   how to compute the property from raw sensor data.

Stability guarantee
-------------------

Once an ID is published, it is **never reused** for a different
property.  Removed properties are tombstoned (marked deprecated, but
the slot stays reserved).  Renaming a property is allowed at the
display level; the wire ID is the source of truth.
"""

from __future__ import annotations

import enum
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Registry of property identifiers
# ---------------------------------------------------------------------------


class PropertyId(enum.IntEnum):
    """Globally stable IDs for SQ-attached properties.

    ID ranges (informational, not enforced):

    * ``0x00 - 0x1F``  -- optical / spectral indices (NDVI, NDRE, ...)
    * ``0x20 - 0x2F``  -- thermal (surface temp, emissivity)
    * ``0x30 - 0x3F``  -- LiDAR (intensity, density, returns)
    * ``0x40 - 0x4F``  -- RGB / color
    * ``0x50 - 0x5F``  -- material proxies (albedo, roughness, moisture)
    * ``0x60 - 0x7F``  -- derived health (LAI, canopy water, ...)
    * ``0x80 - 0xEF``  -- reserved
    * ``0xF0 - 0xFF``  -- statistics / metadata (count, age, confidence)
    """

    # ----- Optical / spectral indices ---------------------------------
    NDVI = 0x01            # (NIR - Red) / (NIR + Red), [-1, 1]
    NDRE = 0x02            # (NIR - RedEdge) / (NIR + RedEdge), [-1, 1]
    GNDVI = 0x03           # (NIR - Green) / (NIR + Green), [-1, 1]
    EVI = 0x04             # 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    SAVI = 0x05            # ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    MSI = 0x06             # SWIR / NIR, moisture stress (>1 = dry)
    PRI = 0x07             # photochemical reflectance index

    # ----- Thermal -----------------------------------------------------
    SURFACE_TEMP_C = 0x20  # degC, int16 *100 -> 0.01 degC, +/-327.67 degC
    EMISSIVITY = 0x21      # uint8 [0, 1] *255

    # ----- LiDAR -------------------------------------------------------
    LIDAR_INTENSITY_MEAN = 0x30  # uint8, normalized 0-255
    LIDAR_INTENSITY_STD = 0x31   # uint8
    POINT_DENSITY = 0x32         # uint16, points/m^3 *10
    RETURN_RATIO = 0x33          # uint8, fraction of multi-returns *255

    # ----- RGB / color -------------------------------------------------
    RGB_R_MEAN = 0x40            # uint8
    RGB_G_MEAN = 0x41            # uint8
    RGB_B_MEAN = 0x42            # uint8
    HUE_MEAN = 0x43              # uint8 (0-255 -> 0-360 deg)
    SATURATION_MEAN = 0x44       # uint8 [0, 1] *255
    BRIGHTNESS_MEAN = 0x45       # uint8

    # ----- Material proxies -------------------------------------------
    ALBEDO = 0x50                # uint8 [0, 1] *255
    ROUGHNESS = 0x51             # uint8 [0, 1] *255
    MOISTURE = 0x52              # uint8 [0, 1] *255
    SPECULARITY = 0x53           # uint8 [0, 1] *255

    # ----- Derived health ---------------------------------------------
    LAI = 0x60                   # uint16, leaf area index, [0, 10] *6553.5
    CANOPY_WATER = 0x61          # uint16, normalized [0, 1] *65535
    BIOMASS_PROXY = 0x62         # uint16
    CHLOROPHYLL = 0x63           # uint16, [0, 100] *655.35

    # ----- Statistics / metadata --------------------------------------
    OBSERVATION_COUNT = 0xF0     # uint16, capped at 65535
    AGE_HOURS = 0xF1             # uint16 hours, [0, ~7.5y]
    CONFIDENCE = 0xF2            # uint8 [0, 1] *255
    REVISION = 0xF3              # uint8, modular revision counter


# ---------------------------------------------------------------------------
# Property spec dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropertySpec:
    """How a single property is encoded on the wire and interpreted.

    Attributes
    ----------
    pid
        The :class:`PropertyId` this spec describes.
    name
        Human-readable name (e.g. ``"ndvi"``).
    units
        Display units / domain hint (e.g. ``"unitless [-1, 1]"``,
        ``"degC"``, ``"normalized 0..255"``).
    bytes_per_value
        ``{1, 2, 4}`` -- byte count of the encoded value (after the
        1-byte ID prefix).
    domain
        ``(lo, hi)`` valid range for *decoded* values.  Used for
        validation and for clamping before quantization.
    quantize
        ``f(value: float) -> int``.  Domain ``-> [0, 2**(8*bytes)-1]``
        (or signed equivalent for ``int*`` codes).
    dequantize
        Inverse of ``quantize``.
    description
        Free-form documentation.
    deprecated
        If True, this slot is tombstoned -- value is still decodable
        but new producers should not emit it.
    """

    pid: PropertyId
    name: str
    units: str
    bytes_per_value: int
    domain: Tuple[float, float]
    quantize: Callable[[float], int]
    dequantize: Callable[[int], float]
    description: str = ""
    deprecated: bool = False


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------


def _make_uint_quant(bytes_per_value: int, lo: float, hi: float) -> Tuple[Callable[[float], int], Callable[[int], float]]:
    """Build (quantize, dequantize) for an unsigned linear range."""
    nbits = bytes_per_value * 8
    full = (1 << nbits) - 1

    def quantize(v: float) -> int:
        if not math.isfinite(v):
            return 0
        clamped = float(np.clip(v, lo, hi))
        frac = (clamped - lo) / (hi - lo) if hi > lo else 0.0
        return int(round(np.clip(frac * full, 0, full)))

    def dequantize(c: int) -> float:
        c = int(np.clip(c, 0, full))
        if hi > lo:
            return float(lo + (c / full) * (hi - lo))
        return float(lo)

    return quantize, dequantize


def _make_signed_quant(bytes_per_value: int, lo: float, hi: float) -> Tuple[Callable[[float], int], Callable[[int], float]]:
    """Build (quantize, dequantize) for a centered, signed linear range.

    Values are stored as ``int{N}`` over ``[-(2**(N-1)-1), 2**(N-1)-1]``
    mapped linearly onto ``[lo, hi]``.  Symmetric range avoids the
    sign-bit edge case at the bottom of ``int`` ranges.
    """
    nbits = bytes_per_value * 8
    half = (1 << (nbits - 1)) - 1  # use symmetric -half..+half

    if hi <= lo:
        raise ValueError(f"_make_signed_quant: hi must exceed lo (got {lo}, {hi})")

    def quantize(v: float) -> int:
        if not math.isfinite(v):
            return 0
        # Map [lo, hi] -> [-half, +half]
        clamped = float(np.clip(v, lo, hi))
        center = 0.5 * (lo + hi)
        span = 0.5 * (hi - lo)
        frac = (clamped - center) / span  # in [-1, 1]
        return int(round(np.clip(frac * half, -half, half)))

    def dequantize(c: int) -> float:
        c = int(np.clip(c, -half, half))
        center = 0.5 * (lo + hi)
        span = 0.5 * (hi - lo)
        return float(center + (c / half) * span)

    return quantize, dequantize


def _make_count_quant(bytes_per_value: int, cap: int) -> Tuple[Callable[[float], int], Callable[[int], float]]:
    """Quantizer for non-negative counts, saturating at ``cap``."""
    full = (1 << (bytes_per_value * 8)) - 1
    cap = min(cap, full)

    def quantize(v: float) -> int:
        if not math.isfinite(v) or v < 0:
            return 0
        return int(np.clip(round(v), 0, cap))

    def dequantize(c: int) -> float:
        return float(np.clip(c, 0, cap))

    return quantize, dequantize


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def _build_registry() -> Dict[PropertyId, PropertySpec]:
    """Construct the canonical (pid -> spec) registry.

    All quantization is closed-form and frame-invariant.  See module
    docstring for the ID-range conventions.
    """
    specs: List[PropertySpec] = []

    # ----- Optical / spectral indices: signed [-1, 1] over int8 ------
    for pid, name, desc in [
        (PropertyId.NDVI,  "ndvi",  "(NIR-Red)/(NIR+Red); vegetation vigor"),
        (PropertyId.NDRE,  "ndre",  "(NIR-RedEdge)/(NIR+RedEdge); chlorophyll proxy"),
        (PropertyId.GNDVI, "gndvi", "(NIR-Green)/(NIR+Green); leaf nitrogen proxy"),
        (PropertyId.SAVI,  "savi",  "Soil-adjusted vegetation index"),
        (PropertyId.PRI,   "pri",   "Photochemical reflectance index"),
    ]:
        q, dq = _make_signed_quant(1, -1.0, 1.0)
        specs.append(PropertySpec(
            pid=pid, name=name, units="unitless [-1, 1]",
            bytes_per_value=1, domain=(-1.0, 1.0),
            quantize=q, dequantize=dq, description=desc,
        ))

    # EVI ranges roughly in [-1, 4] but most useful values [0, 1]; keep
    # 2-byte signed for headroom.
    q, dq = _make_signed_quant(2, -1.0, 4.0)
    specs.append(PropertySpec(
        pid=PropertyId.EVI, name="evi", units="unitless",
        bytes_per_value=2, domain=(-1.0, 4.0), quantize=q, dequantize=dq,
        description="Enhanced Vegetation Index; canopy structure-corrected NDVI",
    ))

    # MSI: ratio, > 0; typical [0.4, 2.0]; clamp to [0, 4].
    q, dq = _make_uint_quant(2, 0.0, 4.0)
    specs.append(PropertySpec(
        pid=PropertyId.MSI, name="msi", units="unitless ratio",
        bytes_per_value=2, domain=(0.0, 4.0), quantize=q, dequantize=dq,
        description="Moisture stress index = SWIR / NIR; >1 = drying",
    ))

    # ----- Thermal ----------------------------------------------------
    q, dq = _make_signed_quant(2, -100.0, 100.0)
    specs.append(PropertySpec(
        pid=PropertyId.SURFACE_TEMP_C, name="surface_temp_C", units="degC",
        bytes_per_value=2, domain=(-100.0, 100.0), quantize=q, dequantize=dq,
        description="Surface temperature in Celsius (Altum LWIR or thermal IR)",
    ))
    q, dq = _make_uint_quant(1, 0.0, 1.0)
    specs.append(PropertySpec(
        pid=PropertyId.EMISSIVITY, name="emissivity", units="unitless [0, 1]",
        bytes_per_value=1, domain=(0.0, 1.0), quantize=q, dequantize=dq,
        description="Effective emissivity (used to correct thermal readings)",
    ))

    # ----- LiDAR ------------------------------------------------------
    q, dq = _make_uint_quant(1, 0.0, 255.0)
    specs.append(PropertySpec(
        pid=PropertyId.LIDAR_INTENSITY_MEAN, name="lidar_intensity_mean",
        units="0-255 normalized", bytes_per_value=1, domain=(0.0, 255.0),
        quantize=q, dequantize=dq,
        description="Mean Velodyne return intensity over per-SQ point membership",
    ))
    specs.append(PropertySpec(
        pid=PropertyId.LIDAR_INTENSITY_STD, name="lidar_intensity_std",
        units="0-255 normalized", bytes_per_value=1, domain=(0.0, 255.0),
        quantize=q, dequantize=dq,
        description="Std-dev of LiDAR return intensities per-SQ",
    ))
    q, dq = _make_uint_quant(2, 0.0, 6553.5)
    specs.append(PropertySpec(
        pid=PropertyId.POINT_DENSITY, name="point_density",
        units="points/m^3 *10", bytes_per_value=2, domain=(0.0, 6553.5),
        quantize=q, dequantize=dq,
        description="LiDAR point density (membership_count / SQ.volume)",
    ))
    q, dq = _make_uint_quant(1, 0.0, 1.0)
    specs.append(PropertySpec(
        pid=PropertyId.RETURN_RATIO, name="return_ratio",
        units="fraction [0, 1]", bytes_per_value=1, domain=(0.0, 1.0),
        quantize=q, dequantize=dq,
        description="Multi-return / total-return ratio (canopy permeability proxy)",
    ))

    # ----- RGB / color ------------------------------------------------
    for pid, name in [
        (PropertyId.RGB_R_MEAN, "rgb_r_mean"),
        (PropertyId.RGB_G_MEAN, "rgb_g_mean"),
        (PropertyId.RGB_B_MEAN, "rgb_b_mean"),
        (PropertyId.HUE_MEAN, "hue_mean"),
        (PropertyId.SATURATION_MEAN, "saturation_mean"),
        (PropertyId.BRIGHTNESS_MEAN, "brightness_mean"),
    ]:
        q, dq = _make_uint_quant(1, 0.0, 255.0)
        specs.append(PropertySpec(
            pid=pid, name=name, units="0-255",
            bytes_per_value=1, domain=(0.0, 255.0),
            quantize=q, dequantize=dq,
            description=f"Per-SQ {name.replace('_', ' ')}",
        ))

    # ----- Material proxies -------------------------------------------
    for pid, name, desc in [
        (PropertyId.ALBEDO, "albedo", "Broadband reflectance proxy"),
        (PropertyId.ROUGHNESS, "roughness", "Microfacet roughness proxy"),
        (PropertyId.MOISTURE, "moisture", "Surface moisture proxy"),
        (PropertyId.SPECULARITY, "specularity", "Specular reflection proxy"),
    ]:
        q, dq = _make_uint_quant(1, 0.0, 1.0)
        specs.append(PropertySpec(
            pid=pid, name=name, units="unitless [0, 1]",
            bytes_per_value=1, domain=(0.0, 1.0),
            quantize=q, dequantize=dq, description=desc,
        ))

    # ----- Derived health --------------------------------------------
    q, dq = _make_uint_quant(2, 0.0, 10.0)
    specs.append(PropertySpec(
        pid=PropertyId.LAI, name="lai", units="m^2 leaf / m^2 ground",
        bytes_per_value=2, domain=(0.0, 10.0), quantize=q, dequantize=dq,
        description="Leaf Area Index (canopy density proxy)",
    ))
    q, dq = _make_uint_quant(2, 0.0, 1.0)
    specs.append(PropertySpec(
        pid=PropertyId.CANOPY_WATER, name="canopy_water",
        units="unitless [0, 1]", bytes_per_value=2, domain=(0.0, 1.0),
        quantize=q, dequantize=dq,
        description="Canopy water content (NIR/SWIR ratio derived)",
    ))
    q, dq = _make_uint_quant(2, 0.0, 1.0e3)
    specs.append(PropertySpec(
        pid=PropertyId.BIOMASS_PROXY, name="biomass_proxy",
        units="g/m^2 (proxy)", bytes_per_value=2, domain=(0.0, 1.0e3),
        quantize=q, dequantize=dq,
        description="Above-ground biomass proxy",
    ))
    q, dq = _make_uint_quant(2, 0.0, 100.0)
    specs.append(PropertySpec(
        pid=PropertyId.CHLOROPHYLL, name="chlorophyll",
        units="ug/cm^2 (proxy)", bytes_per_value=2, domain=(0.0, 100.0),
        quantize=q, dequantize=dq,
        description="Chlorophyll concentration proxy from RedEdge",
    ))

    # ----- Statistics / metadata --------------------------------------
    q, dq = _make_count_quant(2, cap=65535)
    specs.append(PropertySpec(
        pid=PropertyId.OBSERVATION_COUNT, name="observation_count",
        units="count", bytes_per_value=2, domain=(0.0, 65535.0),
        quantize=q, dequantize=dq,
        description="Number of independent observations contributing to props",
    ))
    q, dq = _make_count_quant(2, cap=65535)
    specs.append(PropertySpec(
        pid=PropertyId.AGE_HOURS, name="age_hours", units="hours",
        bytes_per_value=2, domain=(0.0, 65535.0), quantize=q, dequantize=dq,
        description="Hours since last observation refreshed this SQ",
    ))
    q, dq = _make_uint_quant(1, 0.0, 1.0)
    specs.append(PropertySpec(
        pid=PropertyId.CONFIDENCE, name="confidence",
        units="unitless [0, 1]", bytes_per_value=1, domain=(0.0, 1.0),
        quantize=q, dequantize=dq,
        description="Aggregate confidence over contributing observations",
    ))
    q, dq = _make_count_quant(1, cap=255)
    specs.append(PropertySpec(
        pid=PropertyId.REVISION, name="revision", units="modular counter",
        bytes_per_value=1, domain=(0.0, 255.0), quantize=q, dequantize=dq,
        description="Modular revision counter (wraps every 256 updates)",
    ))

    return {s.pid: s for s in specs}


_PROPERTY_SPECS: Dict[PropertyId, PropertySpec] = _build_registry()


def get_spec(pid: Union[PropertyId, int, str]) -> PropertySpec:
    """Look up a :class:`PropertySpec` by ID, integer, or name."""
    if isinstance(pid, str):
        for spec in _PROPERTY_SPECS.values():
            if spec.name == pid:
                return spec
        raise KeyError(f"Unknown property name: {pid!r}")
    pid_int = int(pid)
    try:
        pid_enum = PropertyId(pid_int)
    except ValueError as e:
        raise KeyError(f"Unknown property id: 0x{pid_int:02X}") from e
    if pid_enum not in _PROPERTY_SPECS:
        raise KeyError(f"PropertyId {pid_enum.name} has no registered spec")
    return _PROPERTY_SPECS[pid_enum]


def all_specs() -> Tuple[PropertySpec, ...]:
    """Return all registered (non-deprecated) property specs."""
    return tuple(s for s in _PROPERTY_SPECS.values() if not s.deprecated)


def all_property_ids() -> Tuple[PropertyId, ...]:
    """Return all registered :class:`PropertyId`s."""
    return tuple(_PROPERTY_SPECS.keys())


# ---------------------------------------------------------------------------
# Trailer encoding / decoding
# ---------------------------------------------------------------------------

#: Maximum number of properties per SQ trailer (uint8 length byte).
MAX_PROPERTIES_PER_TRAILER: int = 255


def _value_struct_for(spec: PropertySpec) -> struct.Struct:
    """Big-endian fixed-width struct for a property's encoded value.

    Conventions:
      * Signed ranges centered on a midpoint -> ``int8`` / ``int16`` /
        ``int32`` (per ``bytes_per_value``).
      * Unsigned ranges -> ``uint8`` / ``uint16`` / ``uint32``.
    """
    # Signed if domain is centered on or below zero with negative half.
    lo, hi = spec.domain
    is_signed = lo < 0.0
    bv = spec.bytes_per_value
    if is_signed:
        if bv == 1:
            return struct.Struct(">b")
        if bv == 2:
            return struct.Struct(">h")
        if bv == 4:
            return struct.Struct(">i")
    else:
        if bv == 1:
            return struct.Struct(">B")
        if bv == 2:
            return struct.Struct(">H")
        if bv == 4:
            return struct.Struct(">I")
    raise ValueError(
        f"PropertySpec {spec.name}: bytes_per_value={bv} not in (1, 2, 4)"
    )


def encoded_size(properties: Mapping[Union[PropertyId, int, str], float]) -> int:
    """Total byte size of a property trailer for the given properties.

    Returns ``0`` for an empty mapping (no trailer emitted by callers).
    Otherwise: ``1 + sum(1 + spec.bytes_per_value for each prop)``.
    """
    if not properties:
        return 0
    n = 1  # length prefix
    for pid in properties.keys():
        spec = get_spec(pid)
        n += 1 + spec.bytes_per_value
    return n


def encode_property_trailer(
    properties: Mapping[Union[PropertyId, int, str], float],
) -> bytes:
    """Encode ``{pid -> value}`` as a binary property trailer.

    Returns ``b""`` if ``properties`` is empty.

    The output is *not* prefixed with a flag byte; that lives in the
    parent codec's header (see :mod:`.codec`'s ``FLAG_HAS_PROPERTIES``).
    Order is deterministic (sorted by ``PropertyId``).
    """
    if not properties:
        return b""

    # Resolve and dedupe: a name and an id pointing to the same spec
    # collapse to one entry, last-write-wins.
    canonical: Dict[PropertyId, float] = {}
    for raw_pid, value in properties.items():
        spec = get_spec(raw_pid)
        canonical[spec.pid] = float(value)

    if len(canonical) > MAX_PROPERTIES_PER_TRAILER:
        raise ValueError(
            f"encode_property_trailer: {len(canonical)} properties exceeds "
            f"max {MAX_PROPERTIES_PER_TRAILER}."
        )

    out = bytearray()
    out.append(len(canonical) & 0xFF)
    for pid in sorted(canonical.keys(), key=int):
        spec = _PROPERTY_SPECS[pid]
        code = spec.quantize(canonical[pid])
        out.append(int(pid) & 0xFF)
        out += _value_struct_for(spec).pack(code)
    return bytes(out)


def decode_property_trailer(
    data: bytes,
    *,
    offset: int = 0,
) -> Tuple[Dict[PropertyId, float], int]:
    """Inverse of :func:`encode_property_trailer`.

    Parameters
    ----------
    data
        Buffer containing the trailer somewhere starting at ``offset``.
    offset
        Byte offset where the trailer begins (default 0).

    Returns
    -------
    (properties, bytes_consumed)
        ``properties`` is a ``{PropertyId -> float}`` map of decoded
        values; ``bytes_consumed`` is the number of bytes read from
        ``data`` (so the caller can advance past the trailer).

    Raises
    ------
    ValueError
        If the buffer is truncated, or contains an unknown property
        ID (callers can handle the partial decode by catching this).
    """
    if offset >= len(data):
        raise ValueError("decode_property_trailer: empty buffer at offset")
    n_props = data[offset]
    cursor = offset + 1
    out: Dict[PropertyId, float] = {}
    for _ in range(n_props):
        if cursor >= len(data):
            raise ValueError(
                f"decode_property_trailer: buffer truncated reading PID "
                f"({cursor}/{len(data)})"
            )
        pid_byte = data[cursor]
        cursor += 1
        try:
            spec = get_spec(pid_byte)
        except KeyError:
            # Unknown / future property -- we don't know how many bytes
            # to skip without the spec.  Defer to the caller.
            raise ValueError(
                f"decode_property_trailer: unknown PropertyId 0x{pid_byte:02X}; "
                "trailer cannot be safely advanced past unknown IDs."
            )
        s = _value_struct_for(spec)
        if cursor + s.size > len(data):
            raise ValueError(
                f"decode_property_trailer: buffer truncated reading "
                f"{spec.name} (need {s.size} bytes at offset {cursor})"
            )
        (code,) = s.unpack_from(data, cursor)
        out[spec.pid] = float(spec.dequantize(int(code)))
        cursor += s.size
    return out, cursor - offset


def quantize_round_trip_error(pid: Union[PropertyId, int, str], value: float) -> float:
    """Return the quantization error for a single value (decoded - input)."""
    spec = get_spec(pid)
    code = spec.quantize(float(value))
    decoded = spec.dequantize(code)
    return float(decoded - float(value))


__all__ = [
    "PropertyId",
    "PropertySpec",
    "MAX_PROPERTIES_PER_TRAILER",
    "get_spec",
    "all_specs",
    "all_property_ids",
    "encoded_size",
    "encode_property_trailer",
    "decode_property_trailer",
    "quantize_round_trip_error",
]
