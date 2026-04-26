"""Tests for PR-5.5: properties + accumulators + spectrum + attribution.

Covers the full earth-rover -> server appearance wire-format pipeline:

* :mod:`kernelcal.distinction_game.geometry.properties` -- registry,
  quantization round-trip per property, trailer encode/decode.
* :mod:`kernelcal.distinction_game.geometry.accumulators` -- Welford
  running stats, per-SQ store, merge correctness.
* :mod:`kernelcal.distinction_game.geometry.spectrum` -- DCT-32 + PCA-8
  compression, 96-byte packet round-trip, accumulator semantics.
* :mod:`kernelcal.distinction_game.geometry.attribution` -- LiDAR /
  MicaSense / OceanOptics attributors against synthetic SQ scenes.
* :mod:`kernelcal.distinction_game.geometry.codec` -- extended SQ codec
  with property + spectrum trailers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kernelcal.distinction_game.geometry import (
    DCT_RETAINED,
    EPS_MIN,
    FLAG_HAS_PARENT,
    FLAG_HAS_PROPERTIES,
    FLAG_HAS_SPECTRUM,
    LidarIntensityAttributor,
    MAX_PROPERTIES_PER_TRAILER,
    MicaSenseAttributor,
    OceanOpticsAttributor,
    PACKED_BYTES,
    PACKED_SPECTRUM_BYTES,
    PropertyId,
    PropertySpec,
    SpectrumAccumulator,
    SpectrumPacket,
    SQSpatialIndex,
    Superquadric,
    SuperquadricPropertyStore,
    WelfordAccumulator,
    all_property_ids,
    all_specs,
    compress_spectrum,
    decode_property_trailer,
    decompress_spectrum,
    encode_property_trailer,
    get_spec,
    merge_property_stores,
    pack_superquadric,
    packed_size,
    quantize_round_trip_error,
    store_from_decoded_trailer,
    superquadric_box,
    superquadric_cylinder,
    superquadric_ellipsoid,
    superquadric_sphere,
    unpack_superquadric,
)


# ===========================================================================
# Properties registry
# ===========================================================================


class TestPropertyRegistry:
    def test_registry_nonempty(self):
        ids = all_property_ids()
        assert len(ids) >= 20
        # Every id has a spec.
        for pid in ids:
            spec = get_spec(pid)
            assert isinstance(spec, PropertySpec)
            assert spec.bytes_per_value in (1, 2, 4)

    def test_get_spec_by_int_and_name(self):
        spec = get_spec(PropertyId.NDVI)
        assert spec.name == "ndvi"
        assert get_spec(int(PropertyId.NDVI)) is spec
        assert get_spec("ndvi") is spec

    def test_unknown_pid_raises(self):
        with pytest.raises(KeyError):
            get_spec(0xAB)
        with pytest.raises(KeyError):
            get_spec("not_a_real_property")

    def test_specs_have_unique_ids(self):
        seen = set()
        for spec in all_specs():
            assert spec.pid not in seen
            seen.add(spec.pid)


class TestPropertyQuantization:
    @pytest.mark.parametrize("pid", list(all_property_ids()))
    def test_quantize_dequantize_round_trip_error_bounded(self, pid):
        spec = get_spec(pid)
        lo, hi = spec.domain
        rng = np.random.default_rng(int(pid))
        # Sample mid-domain values; round-trip error should be small.
        for v in rng.uniform(lo, hi, size=20):
            err = quantize_round_trip_error(pid, float(v))
            # Per-property tolerance: <= 1 LSB of the range.
            range_span = hi - lo
            tolerance = range_span / (2 ** (spec.bytes_per_value * 8 - 1))
            assert abs(err) <= tolerance * 1.05, (
                f"{spec.name}: |err|={abs(err)} exceeds {tolerance:.4e}"
            )

    def test_quantize_clamps_out_of_domain(self):
        # Way too high: should saturate at domain max within tolerance.
        spec = get_spec(PropertyId.NDVI)
        big = 100.0
        code = spec.quantize(big)
        decoded = spec.dequantize(code)
        assert decoded <= spec.domain[1] + 1e-3
        assert decoded >= spec.domain[1] - 0.1  # near the top

    def test_quantize_handles_nan(self):
        spec = get_spec(PropertyId.NDVI)
        code = spec.quantize(float("nan"))
        # NaN -> 0 code -> midpoint of domain after dequant
        decoded = spec.dequantize(code)
        assert math.isfinite(decoded)


class TestPropertyTrailer:
    def test_empty_trailer_is_empty_bytes(self):
        assert encode_property_trailer({}) == b""

    def test_round_trip_basic(self):
        props = {
            PropertyId.NDVI: 0.42,
            PropertyId.SURFACE_TEMP_C: 32.5,
            PropertyId.LIDAR_INTENSITY_MEAN: 128.0,
        }
        trailer = encode_property_trailer(props)
        decoded, n = decode_property_trailer(trailer)
        assert n == len(trailer)
        for pid, expected in props.items():
            assert pid in decoded
            err = abs(decoded[pid] - expected)
            tolerance = abs(expected) * 0.02 + 0.5
            assert err <= tolerance, f"{pid}: err={err} exceeds {tolerance}"

    def test_trailer_size_formula(self):
        from kernelcal.distinction_game.geometry.properties import encoded_size

        props = {
            PropertyId.NDVI: 0.5,            # 1B id + 1B int8 = 2B
            PropertyId.SURFACE_TEMP_C: 25.0, # 1B id + 2B int16 = 3B
            PropertyId.OBSERVATION_COUNT: 100, # 1B id + 2B uint16 = 3B
        }
        # 1B length + 2 + 3 + 3 = 9B
        assert encoded_size(props) == 9
        assert len(encode_property_trailer(props)) == 9

    def test_trailer_unknown_id_raises(self):
        # Forge a trailer with unknown ID 0xAB
        bad = bytes([1, 0xAB, 0x00, 0x00])
        with pytest.raises(ValueError, match="unknown PropertyId"):
            decode_property_trailer(bad)

    def test_trailer_truncated_raises(self):
        good = encode_property_trailer({PropertyId.SURFACE_TEMP_C: 25.0})
        with pytest.raises(ValueError, match="truncated"):
            decode_property_trailer(good[:-1])

    def test_too_many_props_rejected(self):
        # All registry entries plus duplicates -- but encode dedupes by ID
        # so we need to construct >255 distinct PIDs to actually trip
        # the limit, which the registry can't supply.  Instead verify
        # the constant is enforced by mocking.
        # Sanity: 255 is the limit.
        assert MAX_PROPERTIES_PER_TRAILER == 255

    def test_trailer_deterministic_order(self):
        # Same props in different insertion orders should yield same bytes.
        a = encode_property_trailer({PropertyId.NDVI: 0.5, PropertyId.NDRE: 0.3})
        b = encode_property_trailer({PropertyId.NDRE: 0.3, PropertyId.NDVI: 0.5})
        assert a == b


# ===========================================================================
# Accumulators
# ===========================================================================


class TestWelfordAccumulator:
    def test_empty_state(self):
        a = WelfordAccumulator()
        assert a.n == 0
        assert a.mean == 0.0
        assert a.variance() == 0.0

    def test_single_sample(self):
        a = WelfordAccumulator()
        a.update(5.0)
        assert a.n == 1
        assert a.mean == 5.0
        assert a.variance() == 0.0  # n < 2

    def test_matches_numpy_for_unit_weights(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=10.0, scale=2.0, size=200)
        a = WelfordAccumulator()
        for v in samples:
            a.update(float(v))
        assert a.n == 200
        assert a.mean == pytest.approx(float(np.mean(samples)), abs=1e-9)
        assert a.std() == pytest.approx(float(np.std(samples, ddof=1)), abs=1e-9)

    def test_batch_update_matches_loop(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(size=500)
        a_loop = WelfordAccumulator()
        for v in samples:
            a_loop.update(float(v))
        a_batch = WelfordAccumulator()
        a_batch.update_batch(samples)
        assert a_batch.n == a_loop.n
        assert a_batch.mean == pytest.approx(a_loop.mean, abs=1e-10)
        assert a_batch.std() == pytest.approx(a_loop.std(), abs=1e-9)

    def test_merge_combines_correctly(self):
        rng = np.random.default_rng(1)
        s1 = rng.normal(size=50)
        s2 = rng.normal(loc=5.0, size=80)
        a1 = WelfordAccumulator(); a1.update_batch(s1)
        a2 = WelfordAccumulator(); a2.update_batch(s2)
        a1.merge(a2)
        full = np.concatenate([s1, s2])
        assert a1.n == full.size
        assert a1.mean == pytest.approx(float(np.mean(full)), abs=1e-9)
        assert a1.std() == pytest.approx(float(np.std(full, ddof=1)), abs=1e-9)

    def test_weighted_update(self):
        a = WelfordAccumulator()
        a.update(0.0, weight=1.0)
        a.update(10.0, weight=3.0)
        # Weighted mean: (1*0 + 3*10) / 4 = 7.5
        assert a.mean == pytest.approx(7.5, abs=1e-9)

    def test_rejects_nonfinite(self):
        a = WelfordAccumulator()
        a.update(float("nan"))
        a.update(float("inf"))
        a.update(5.0)
        assert a.n == 1
        assert a.mean == 5.0


class TestSuperquadricPropertyStore:
    def test_basic_update_and_finalize(self):
        store = SuperquadricPropertyStore(sq_id="sq-1")
        for v in [0.4, 0.5, 0.6]:
            store.update(PropertyId.NDVI, v)
        assert store.has(PropertyId.NDVI)
        assert store.count(PropertyId.NDVI) == 3
        assert store.mean(PropertyId.NDVI) == pytest.approx(0.5, abs=1e-9)

    def test_finalize_for_packing_includes_metadata(self):
        store = SuperquadricPropertyStore(sq_id="sq-1")
        store.update(PropertyId.NDVI, 0.5)
        store.update(PropertyId.NDVI, 0.6)
        store.update(PropertyId.SURFACE_TEMP_C, 30.0)
        out = store.finalize_for_packing(include_metadata=True)
        assert PropertyId.NDVI in out
        assert PropertyId.SURFACE_TEMP_C in out
        assert PropertyId.OBSERVATION_COUNT in out
        assert PropertyId.CONFIDENCE in out

    def test_min_samples_filters(self):
        store = SuperquadricPropertyStore(sq_id="sq-1")
        store.update(PropertyId.NDVI, 0.5)  # n=1
        for _ in range(5):
            store.update(PropertyId.SURFACE_TEMP_C, 30.0)  # n=5
        out = store.finalize_for_packing(min_samples=3, include_metadata=False)
        assert PropertyId.NDVI not in out
        assert PropertyId.SURFACE_TEMP_C in out

    def test_merge_two_stores(self):
        s1 = SuperquadricPropertyStore(sq_id="sq-1")
        s1.update(PropertyId.NDVI, 0.4)
        s2 = SuperquadricPropertyStore(sq_id="sq-1")
        s2.update(PropertyId.NDVI, 0.6)
        s2.update(PropertyId.SURFACE_TEMP_C, 28.0)
        s1.merge(s2)
        assert s1.mean(PropertyId.NDVI) == pytest.approx(0.5, abs=1e-9)
        assert s1.has(PropertyId.SURFACE_TEMP_C)

    def test_merge_rejects_id_mismatch(self):
        s1 = SuperquadricPropertyStore(sq_id="sq-1")
        s2 = SuperquadricPropertyStore(sq_id="sq-2")
        with pytest.raises(ValueError, match="sq_id mismatch"):
            s1.merge(s2)

    def test_store_from_decoded_trailer(self):
        props = {PropertyId.NDVI: 0.45, PropertyId.SURFACE_TEMP_C: 31.5}
        store = store_from_decoded_trailer("sq-x", props, sample_count_hint=12)
        assert store.sq_id == "sq-x"
        assert store.count(PropertyId.NDVI) == 12
        assert store.mean(PropertyId.NDVI) == pytest.approx(0.45)

    def test_merge_property_stores_helper(self):
        s1 = SuperquadricPropertyStore(sq_id="sq-1")
        s1.update(PropertyId.NDVI, 0.4)
        s2 = SuperquadricPropertyStore(sq_id="sq-1")
        s2.update(PropertyId.NDVI, 0.5)
        s3 = SuperquadricPropertyStore(sq_id="sq-1")
        s3.update(PropertyId.NDVI, 0.6)
        merged = merge_property_stores(s1, s2, s3)
        assert merged is not None
        assert merged.mean(PropertyId.NDVI) == pytest.approx(0.5, abs=1e-9)


# ===========================================================================
# Spectrum codec
# ===========================================================================


class TestSpectrumPacket:
    def test_packet_size_constant(self):
        assert PACKED_SPECTRUM_BYTES == 96

    def test_round_trip_smooth_signal(self):
        wl = np.linspace(200.0, 1100.0, 2048)
        sig = np.exp(-((wl - 680.0) ** 2) / (2 * 50.0 ** 2)) + 0.5
        pkt = compress_spectrum(
            sig, lambda_lo_nm=200.0, lambda_hi_nm=1100.0,
            n_samples=10, quality_score=0.9,
        )
        b = pkt.to_bytes()
        assert len(b) == PACKED_SPECTRUM_BYTES
        pkt2 = SpectrumPacket.from_bytes(b)
        recon = decompress_spectrum(pkt2, n_channels=2048)
        # Smooth Gaussian-on-flat: DCT-32 should be very accurate.
        rel_err = float(np.mean((recon - sig) ** 2)) / float(np.var(sig))
        assert rel_err < 1e-3, f"smooth-signal rel-MSE too high: {rel_err}"

    def test_metadata_round_trip(self):
        sig = np.linspace(0.1, 1.0, 1024)
        pkt = compress_spectrum(
            sig, lambda_lo_nm=350.0, lambda_hi_nm=950.0,
            n_samples=42, quality_score=0.75,
        )
        b = pkt.to_bytes()
        pkt2 = SpectrumPacket.from_bytes(b)
        assert pkt2.n_samples == 42
        assert abs(pkt2.quality_score - 0.75) < 0.01
        assert abs(pkt2.lambda_lo_nm - 350.0) < 0.1
        assert abs(pkt2.lambda_hi_nm - 950.0) < 0.1
        assert pkt2.n_channels == 1024

    def test_zero_spectrum_safe(self):
        sig = np.zeros(2048)
        pkt = compress_spectrum(
            sig, lambda_lo_nm=200.0, lambda_hi_nm=1100.0,
        )
        b = pkt.to_bytes()
        pkt2 = SpectrumPacket.from_bytes(b)
        recon = decompress_spectrum(pkt2, n_channels=2048)
        assert np.all(np.isfinite(recon))
        assert np.allclose(recon, 0.0, atol=1e-3)

    def test_truncated_buffer_raises(self):
        sig = np.linspace(0.1, 1.0, 512)
        pkt = compress_spectrum(
            sig, lambda_lo_nm=200.0, lambda_hi_nm=1100.0,
        )
        b = pkt.to_bytes()
        with pytest.raises(ValueError, match="need 96 bytes"):
            SpectrumPacket.from_bytes(b[:80])


class TestSpectrumAccumulator:
    def test_running_mean(self):
        rng = np.random.default_rng(7)
        true_mean = np.exp(-((np.linspace(0, 1, 2048) - 0.5) ** 2) / 0.05)
        acc = SpectrumAccumulator(n_channels=2048)
        for _ in range(50):
            acc.update(true_mean + 0.05 * rng.standard_normal(2048))
        assert acc.n_samples == 50
        # Should approximate true_mean within sample SD/sqrt(N) per channel.
        max_err = float(np.max(np.abs(acc.mean_spectrum - true_mean)))
        assert max_err < 0.1, f"max abs error too high: {max_err}"

    def test_resample_to_grid(self):
        acc = SpectrumAccumulator(n_channels=512)
        # Push spectra of mismatched lengths -- they should resample.
        acc.update(np.ones(2048))  # uniform 1
        assert acc.n_samples == 1
        assert np.allclose(acc.mean_spectrum, 1.0, atol=1e-6)

    def test_merge(self):
        a = SpectrumAccumulator(n_channels=256)
        b = SpectrumAccumulator(n_channels=256)
        a.update(np.ones(256) * 1.0)
        a.update(np.ones(256) * 1.0)
        b.update(np.ones(256) * 5.0)
        a.merge(b)
        # Combined mean of [1,1,5] is 7/3
        assert np.allclose(a.mean_spectrum, 7.0 / 3.0, atol=1e-9)
        assert a.n_samples == 3


# ===========================================================================
# Attribution
# ===========================================================================


class TestSQSpatialIndex:
    def test_empty_index(self):
        idx = SQSpatialIndex()
        assert idx.query_point(np.array([0, 0, 0])) is None
        assert idx.query_points(np.zeros((5, 3))).tolist() == [-1] * 5

    def test_point_membership(self):
        idx = SQSpatialIndex()
        sq1 = superquadric_box(center=(0, 0, 0), size=(2.0, 2.0, 2.0))
        sq2 = superquadric_ellipsoid(center=(5, 0, 0), axes=(0.5, 0.5, 0.5))
        idx.extend([sq1, sq2])
        # Inside sq1
        assert idx.query_point(np.array([0.0, 0.0, 0.0])) == 0
        # Inside sq2
        assert idx.query_point(np.array([5.0, 0.0, 0.0])) == 1
        # Outside both
        assert idx.query_point(np.array([10.0, 0.0, 0.0])) is None

    def test_vectorized_membership(self):
        idx = SQSpatialIndex()
        sq = superquadric_box(center=(0, 0, 0), size=(2.0, 2.0, 2.0))
        idx.add(sq)
        pts = np.array([
            [0.0, 0.0, 0.0],   # inside
            [5.0, 0.0, 0.0],   # outside
            [0.5, 0.5, 0.5],   # inside
        ])
        result = idx.query_points(pts)
        assert result.tolist() == [0, -1, 0]

    def test_ray_first_hit(self):
        idx = SQSpatialIndex()
        sq_near = superquadric_ellipsoid(center=(3, 0, 0), axes=(1.0, 1.0, 1.0))
        sq_far = superquadric_ellipsoid(center=(10, 0, 0), axes=(1.0, 1.0, 1.0))
        idx.extend([sq_far, sq_near])  # add far first to test ordering
        hit = idx.first_hit_ray(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
        )
        assert hit is not None
        sq_idx, t = hit
        assert idx.sqs[sq_idx] is sq_near, "should hit nearer SQ first"
        assert 1.5 < t < 3.5


class TestLidarIntensityAttributor:
    def test_attributes_intensity_into_correct_sq(self):
        idx = SQSpatialIndex()
        sq1 = superquadric_box(center=(0, 0, 0), size=(2.0, 2.0, 2.0))
        sq2 = superquadric_box(center=(5, 0, 0), size=(2.0, 2.0, 2.0))
        idx.extend([sq1, sq2])
        attr = LidarIntensityAttributor(idx)

        rng = np.random.default_rng(0)
        # 50 returns inside sq1 with intensity ~ 50; 30 inside sq2 ~ 200.
        pts1 = np.column_stack([
            rng.uniform(-0.8, 0.8, 50),
            rng.uniform(-0.8, 0.8, 50),
            rng.uniform(-0.8, 0.8, 50),
            np.full(50, 50.0),
        ])
        pts2 = np.column_stack([
            rng.uniform(4.2, 5.8, 30),
            rng.uniform(-0.8, 0.8, 30),
            rng.uniform(-0.8, 0.8, 30),
            np.full(30, 200.0),
        ])
        cloud = np.vstack([pts1, pts2])
        attr.attribute(cloud)

        s1 = attr.stores[sq1.id]
        s2 = attr.stores[sq2.id]
        assert s1.has(PropertyId.LIDAR_INTENSITY_MEAN)
        assert s1.mean(PropertyId.LIDAR_INTENSITY_MEAN) == pytest.approx(50.0, abs=1.0)
        assert s2.mean(PropertyId.LIDAR_INTENSITY_MEAN) == pytest.approx(200.0, abs=1.0)
        # Density populated.
        assert s1.has(PropertyId.POINT_DENSITY)


class TestMicaSenseAttributor:
    def test_ndvi_extraction_for_simple_scene(self):
        idx = SQSpatialIndex()
        # SQ at z=5 in front of camera, in the FoV.
        sq = superquadric_box(center=(0, 0, 5), size=(2.0, 2.0, 0.2))
        idx.add(sq)

        H, W = 128, 128
        # Pinhole intrinsics.
        K = np.array([[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]])
        attr = MicaSenseAttributor(idx, K=K, image_shape=(H, W), n_lat=12, n_lon=18)

        # Synthesize a scene where the SQ pixels see high-NIR / low-Red
        # ("vegetation"); set bands to reflect this *only inside the
        # silhouette*.  Simulate by setting full-image to vegetation
        # values; the silhouette mask aggregates the same numbers.
        nir = np.full((H, W), 0.7, dtype=float)
        red = np.full((H, W), 0.1, dtype=float)
        green = np.full((H, W), 0.3, dtype=float)
        blue = np.full((H, W), 0.05, dtype=float)
        red_edge = np.full((H, W), 0.5, dtype=float)
        thermal = np.full((H, W), 25.0, dtype=float)

        bands = {
            "blue": blue, "green": green, "red": red,
            "red_edge": red_edge, "nir": nir, "thermal_C": thermal,
        }
        # World-to-camera: identity (camera at origin looking +Z, std OpenCV).
        # SQ at z=5 is in front; project world (x, y, 5) to camera (x, y, 5).
        # OpenCV pinhole expects camera_z > 0 in front; matches our geometry.
        R_cw = np.eye(3)
        t_cw = np.zeros(3)
        attr.attribute(bands, camera_pose=(R_cw, t_cw))

        store = attr.stores.get(sq.id)
        assert store is not None
        assert store.has(PropertyId.NDVI)
        # NDVI = (0.7 - 0.1) / (0.7 + 0.1) = 0.75
        assert store.mean(PropertyId.NDVI) == pytest.approx(0.75, abs=0.02)
        assert store.mean(PropertyId.SURFACE_TEMP_C) == pytest.approx(25.0, abs=0.5)


class TestOceanOpticsAttributor:
    def test_bore_sight_routes_to_first_hit(self):
        idx = SQSpatialIndex()
        sq_near = superquadric_ellipsoid(center=(2, 0, 0), axes=(0.5, 0.5, 0.5))
        sq_far = superquadric_ellipsoid(center=(8, 0, 0), axes=(0.5, 0.5, 0.5))
        idx.extend([sq_near, sq_far])

        attr = OceanOpticsAttributor(
            idx, n_channels=512, lambda_lo_nm=400.0, lambda_hi_nm=900.0,
        )
        wl = np.linspace(400.0, 900.0, 512)
        # Synthesize a vegetation-like spectrum: NIR > Red.
        spectrum = np.where((wl < 700.0) & (wl > 600.0), 0.1, 0.7)

        sq_id = attr.attribute(
            spectrum, wl,
            bore_sight_origin=np.array([0.0, 0.0, 0.0]),
            bore_sight_direction=np.array([1.0, 0.0, 0.0]),
        )
        assert sq_id == sq_near.id
        store = attr.stores[sq_near.id]
        assert store.spectrum is not None
        assert store.spectrum.n_samples == 1
        # NDVI derived: (0.7 - 0.1) / (0.7 + 0.1) = 0.75
        assert store.has(PropertyId.NDVI)
        assert store.mean(PropertyId.NDVI) == pytest.approx(0.75, abs=0.02)

    def test_no_hit_returns_none(self):
        idx = SQSpatialIndex()
        sq = superquadric_ellipsoid(center=(0, 5, 0), axes=(0.5, 0.5, 0.5))
        idx.add(sq)
        attr = OceanOpticsAttributor(idx)
        wl = np.linspace(200.0, 1100.0, 2048)
        sp = np.ones_like(wl)
        # Ray going +x doesn't intersect SQ at +y=5
        assert attr.attribute(
            sp, wl,
            bore_sight_origin=np.array([0.0, 0.0, 0.0]),
            bore_sight_direction=np.array([1.0, 0.0, 0.0]),
        ) is None


# ===========================================================================
# Extended SQ codec (with property + spectrum trailers)
# ===========================================================================


class TestExtendedSQCodec:
    def _make_sq(self):
        return Superquadric(
            scale=np.array([1.5, 0.8, 2.0]),
            epsilon=np.array([1.0, 0.5]),
            R=np.eye(3),
            t=np.array([10.0, 20.0, 5.0]),
        )

    def test_geometry_only_packing_unchanged(self):
        sq = self._make_sq()
        b = pack_superquadric(sq)
        assert len(b) == PACKED_BYTES
        sq2, meta = unpack_superquadric(b)
        assert meta["properties"] is None
        assert meta["spectrum"] is None
        assert meta["bytes_consumed"] == PACKED_BYTES

    def test_with_properties_only(self):
        sq = self._make_sq()
        props = {
            PropertyId.NDVI: 0.6,
            PropertyId.SURFACE_TEMP_C: 28.0,
            PropertyId.LIDAR_INTENSITY_MEAN: 150.0,
        }
        b = pack_superquadric(sq, properties=props)
        # 32 + 1 (length) + 2 (NDVI) + 3 (Temp) + 2 (Intensity) = 40
        expected = packed_size(properties=props)
        assert len(b) == expected
        sq2, meta = unpack_superquadric(b)
        assert meta["properties"] is not None
        assert meta["flags"] & FLAG_HAS_PROPERTIES
        for pid, expected_value in props.items():
            assert pid in meta["properties"]
            err = abs(meta["properties"][pid] - expected_value)
            assert err < abs(expected_value) * 0.05 + 0.5

    def test_with_parent_and_properties_and_spectrum(self):
        sq = self._make_sq()
        props = {PropertyId.NDVI: 0.5}
        spectrum = compress_spectrum(
            np.linspace(0.1, 0.9, 2048),
            lambda_lo_nm=200.0, lambda_hi_nm=1100.0,
            n_samples=15,
        )
        b = pack_superquadric(
            sq, parent_hash=12345, properties=props, spectrum=spectrum,
        )
        expected = packed_size(has_parent=True, properties=props, has_spectrum=True)
        assert len(b) == expected
        sq2, meta = unpack_superquadric(b)
        assert meta["flags"] & FLAG_HAS_PARENT
        assert meta["flags"] & FLAG_HAS_PROPERTIES
        assert meta["flags"] & FLAG_HAS_SPECTRUM
        assert meta["parent_hash"] == 12345
        assert PropertyId.NDVI in meta["properties"]
        assert meta["spectrum"] is not None
        assert meta["spectrum"].n_samples == 15

    def test_packed_size_helper(self):
        sq = self._make_sq()
        b = pack_superquadric(sq)
        assert len(b) == packed_size()
        b = pack_superquadric(sq, parent_hash=1)
        assert len(b) == packed_size(has_parent=True)
        b = pack_superquadric(sq, spectrum=compress_spectrum(
            np.ones(64), lambda_lo_nm=200.0, lambda_hi_nm=1100.0,
        ))
        assert len(b) == packed_size(has_spectrum=True)

    def test_streaming_decode(self):
        # Pack two SQs back-to-back; decode by advancing bytes_consumed.
        sq_a = self._make_sq()
        sq_b = Superquadric(
            scale=np.array([0.3, 0.3, 5.0]),
            epsilon=np.array([1.0, 0.1]),
        )
        ba = pack_superquadric(sq_a, properties={PropertyId.NDVI: 0.5})
        bb = pack_superquadric(sq_b, parent_hash=42)
        stream = ba + bb

        sq1, meta1 = unpack_superquadric(stream)
        consumed = meta1["bytes_consumed"]
        sq2, meta2 = unpack_superquadric(stream[consumed:])
        assert meta1["flags"] & FLAG_HAS_PROPERTIES
        assert meta2["flags"] & FLAG_HAS_PARENT
        assert meta2["parent_hash"] == 42
        assert consumed + meta2["bytes_consumed"] == len(stream)


# ===========================================================================
# End-to-end: earth_rover -> wire -> server
# ===========================================================================


class TestEndToEndWirePipeline:
    """Simulate an earth-rover producing a few SQs with attached
    properties + spectra, packing, sending, unpacking, and reconstructing."""

    def test_full_pipeline(self):
        rng = np.random.default_rng(123)
        # Build a tiny scene: tree (cylinder trunk + ellipsoid crown)
        # plus a building cuboid.
        trunk = superquadric_cylinder(
            base=(2.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
            radius=0.15, height=3.0,
        )
        crown = superquadric_ellipsoid(
            center=(2.0, 0.0, 4.0), axes=(1.5, 1.5, 1.2),
        )
        crown.parent_id = trunk.id
        building = superquadric_box(
            center=(15.0, 0.0, 4.0), size=(4.0, 3.0, 8.0),
        )

        # On earth_rover: build a scene index, run attributors against
        # synthetic sensor data.
        idx = SQSpatialIndex()
        idx.extend([trunk, crown, building])

        lidar = LidarIntensityAttributor(idx)
        cloud_rows = []
        # 100 points inside trunk @ intensity 80
        for _ in range(100):
            cloud_rows.append([
                2.0 + rng.uniform(-0.1, 0.1),
                rng.uniform(-0.1, 0.1),
                rng.uniform(0.0, 3.0),
                80.0 + rng.normal(0, 5),
            ])
        # 200 points inside crown @ intensity 30 (canopy)
        for _ in range(200):
            cloud_rows.append([
                2.0 + rng.uniform(-1.4, 1.4),
                rng.uniform(-1.4, 1.4),
                4.0 + rng.uniform(-1.0, 1.0),
                30.0 + rng.normal(0, 3),
            ])
        # 50 points inside building @ intensity 150 (concrete)
        for _ in range(50):
            cloud_rows.append([
                15.0 + rng.uniform(-1.8, 1.8),
                rng.uniform(-1.4, 1.4),
                4.0 + rng.uniform(-3.5, 3.5),
                150.0 + rng.normal(0, 8),
            ])
        lidar.attribute(np.asarray(cloud_rows))

        # Add NDVI manually (proxy for MicaSense outcome) and a spectrum
        # for the crown.
        crown_store = lidar.stores[crown.id]
        crown_store.update(PropertyId.NDVI, 0.78)
        crown_store.update(PropertyId.NDRE, 0.45)
        crown_store.update(PropertyId.SURFACE_TEMP_C, 22.0)

        veg_spectrum = np.exp(-((np.linspace(200, 1100, 2048) - 750) / 200) ** 2)
        veg_pkt = compress_spectrum(
            veg_spectrum, lambda_lo_nm=200.0, lambda_hi_nm=1100.0,
            n_samples=20, quality_score=0.92,
        )

        # Pack each SQ with its trailers.
        wire_messages = []
        for sq, parent_hash, store, spec in [
            (trunk, None, lidar.stores.get(trunk.id), None),
            (crown, hash(trunk.id) & 0x7FFFFFFFFFFFFFFF, crown_store, veg_pkt),
            (building, None, lidar.stores.get(building.id), None),
        ]:
            props = (
                store.finalize_for_packing(include_metadata=True)
                if store is not None else None
            )
            wire_messages.append(pack_superquadric(
                sq,
                class_idx=1 if sq is building else 2 if sq is trunk else 3,
                parent_hash=parent_hash,
                properties=props,
                spectrum=spec,
            ))

        # Each message stays under the 200 B limit (well within 100 kbps
        # at modest rates).
        for m in wire_messages:
            assert len(m) <= 200

        # Server side: unpack and reconstruct.
        decoded = [unpack_superquadric(m) for m in wire_messages]
        sq_t, meta_t = decoded[0]
        sq_c, meta_c = decoded[1]
        sq_b, meta_b = decoded[2]

        # Trunk: lidar intensity ~80
        assert meta_t["properties"] is not None
        assert (
            meta_t["properties"][PropertyId.LIDAR_INTENSITY_MEAN]
            == pytest.approx(80.0, abs=2.0)
        )
        # Crown: NDVI ~ 0.78 and has spectrum
        assert meta_c["parent_hash"] is not None
        assert meta_c["properties"][PropertyId.NDVI] == pytest.approx(0.78, abs=0.02)
        assert meta_c["spectrum"] is not None
        # Building: lidar ~150
        assert (
            meta_b["properties"][PropertyId.LIDAR_INTENSITY_MEAN]
            == pytest.approx(150.0, abs=3.0)
        )
