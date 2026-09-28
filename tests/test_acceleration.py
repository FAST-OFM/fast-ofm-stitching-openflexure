"""Tests for bounded correlation and exact precomputed ownership masks."""

from types import SimpleNamespace

import numpy as np
from openflexure_stitching.correlation import displacement_from_crosscorrelation
from openflexure_stitching.loading.image import OFSImage
from openflexure_stitching.settings import CorrelationSettings
from openflexure_stitching.stitching import stitch_images
from openflexure_stitching.types import PairData
from PIL import Image

from fast_ofm_stitching_openflexure.acceleration import (
    AcceleratedMosaicRenderer,
    BoundedArrayCache,
    _full_displacement_from_ffts,
    _strip_displacement,
    overlapping_components,
    position_disconnected_components,
)


def _pair(first: str, second: str) -> PairData:
    """Build a minimal measured overlap for graph-placement tests."""
    return PairData(
        keys=(first, second),
        image_displacement=(0.0, 10.0),
        stage_displacement=(0.0, 10.0),
        fraction_under_threshold={0.9: 0.99},
    )


class _ImageSetForIslands:
    """Small image-set stand-in with acquisition order and stage coordinates."""

    def __init__(self, positions, pairs):
        self._images = {
            key: SimpleNamespace(stage_position_px=np.asarray(value, dtype=float))
            for key, value in positions.items()
        }
        self._pairs = pairs
        self._pair_discrepancies = []
        self._norm_pair_discrepancies = []

    def keys(self):
        """Return acquisition-order keys."""
        return list(self._images)

    def __getitem__(self, key):
        return self._images[key]

    @property
    def pairs(self):
        """Return measured pairs."""
        return self._pairs


def test_overlap_graph_includes_singleton_and_multiple_islands():
    """Unpaired images remain their own stage-anchored component."""
    assert overlapping_components(
        ["a", "b", "c", "d", "e"], [_pair("a", "b"), _pair("c", "d")]
    ) == [["a", "b"], ["c", "d"], ["e"]]


def test_disconnected_islands_keep_local_fit_and_stage_anchor(mocker):
    """Independent dense islands retain local correlation while preserving gaps."""
    image_set = _ImageSetForIslands(
        {
            "a": (0, 0),
            "b": (0, 10),
            "c": (10, 0),
            "d": (1000, 0),
            "e": (1000, 10),
            "f": (1010, 0),
            "g": (2000, 0),
        },
        [
            _pair("a", "b"),
            _pair("a", "c"),
            _pair("b", "c"),
            _pair("d", "e"),
            _pair("d", "f"),
            _pair("e", "f"),
        ],
    )
    mocker.patch(
        "fast_ofm_stitching_openflexure.acceleration.determine_thresholds",
        return_value=(0.5, 100.0),
    )

    def local_fit(view, _peak, _discrepancy):
        offset = 0 if "a" in view._images else 1000
        return {
            key: np.asarray(image_set[key].stage_position_px)
            - np.asarray((offset, 0))
            + np.asarray((2 if key in {"c", "f"} else 0, 0))
            for key in view._images
        }

    fit = mocker.patch(
        "fast_ofm_stitching_openflexure.acceleration.perform_final_position_optimisation",
        side_effect=local_fit,
    )
    positions = position_disconnected_components(image_set)

    assert fit.call_count == 2
    assert set(positions) == set(image_set.keys())
    np.testing.assert_array_equal(positions["g"], (2000, 0))
    np.testing.assert_array_equal(positions["a"], (0, 0))
    np.testing.assert_array_equal(positions["c"], (12, 0))
    np.testing.assert_array_equal(positions["d"], (1000, 0))
    np.testing.assert_array_equal(positions["f"], (1012, 0))


def test_connected_input_preserves_original_optimisation_path(mocker):
    """An ordinary connected scan continues to use the original global fit."""
    image_set = _ImageSetForIslands(
        {"a": (0, 0), "b": (0, 10)}, [_pair("a", "b")]
    )
    mocker.patch(
        "fast_ofm_stitching_openflexure.acceleration.determine_thresholds",
        return_value=(0.5, 100.0),
    )
    expected = {"a": np.asarray((0, 0)), "b": np.asarray((0, 10))}
    fit = mocker.patch(
        "fast_ofm_stitching_openflexure.acceleration.perform_final_position_optimisation",
        return_value=expected,
    )

    assert position_disconnected_components(image_set) is expected
    assert fit.call_args.args[0] is image_set


def test_bounded_array_cache_evicts_and_releases():
    """The LRU obeys its byte ceiling and explicitly drops retained arrays."""
    cache = BoundedArrayCache(16)
    first = np.arange(2, dtype=np.int64)
    second = np.arange(2, dtype=np.int64) + 2

    assert cache.get("first", lambda: first) is first
    assert cache.get("first", lambda: second) is first
    assert cache.get("second", lambda: second) is second

    assert cache.hits == 1
    assert cache.misses == 2
    assert cache.evictions == 1
    assert cache.current_bytes == 16
    cache.clear()
    assert cache.current_bytes == 0


def test_cached_forward_ffts_reproduce_upstream_correlation(tmp_path):
    """Reusing forward FFTs must leave full-frame displacement and quality exact."""
    rng = np.random.default_rng(2)
    base = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    shifted = np.roll(base, shift=(7, -11), axis=(0, 1))
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.fromarray(base).save(first_path)
    Image.fromarray(shifted).save(second_path)
    first = OFSImage(str(first_path))
    second = OFSImage(str(second_path))
    settings = CorrelationSettings(resize=0.5, pad=True)
    expected = (0.0, 0.0)

    upstream = displacement_from_crosscorrelation(
        first,
        second,
        settings,
        expected_displacement_px=expected,
    )
    fft1 = first.fft(resize=settings.resize, pad=True)
    fft2 = second.fft(resize=settings.resize, pad=True)
    from openflexure_stitching.correlation import high_pass_fourier_filter

    filter_array = high_pass_fourier_filter(fft1.shape, settings.high_pass_sigma)
    accelerated = _full_displacement_from_ffts(
        first,
        second,
        fft1,
        fft2,
        settings=settings,
        high_pass_filter=filter_array,
        expected_displacement_px=expected,
    )

    np.testing.assert_array_equal(accelerated[0], upstream[0])
    assert accelerated[1] == upstream[1]


def test_overlap_strip_recovers_stage_predicted_crop_displacement(tmp_path):
    """Strip matching finds a non-wrapped overlap and reuses decoded images."""
    rng = np.random.default_rng(3)
    scene = rng.integers(0, 256, size=(180, 260, 3), dtype=np.uint8)
    first_pixels = scene[40:136, 40:168]
    second_pixels = scene[40:136, 100:228]
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.fromarray(first_pixels).save(first_path)
    Image.fromarray(second_pixels).save(second_path)
    first = OFSImage(str(first_path))
    second = OFSImage(str(second_path))
    cache = BoundedArrayCache(1024 * 1024)

    displacement, quality = _strip_displacement(
        first,
        second,
        CorrelationSettings(resize=1, pad=True),
        (0.0, 60.0),
        cache,
    )
    repeated, _quality = _strip_displacement(
        first,
        second,
        CorrelationSettings(resize=1, pad=True),
        (0.0, 60.0),
        cache,
    )

    np.testing.assert_array_equal(displacement, (0.0, 60.0))
    np.testing.assert_array_equal(repeated, displacement)
    assert quality[0.9] > 0.99
    assert cache.hits == 2
    assert cache.misses == 2


def test_precomputed_masks_preserve_upstream_interior_ownership(tmp_path):
    """Neighbour masks match upstream ownership away from its dropped edges."""
    first = np.zeros((8, 10, 3), dtype=np.uint8)
    first[:, :, 0] = 25
    second = np.zeros((8, 10, 3), dtype=np.uint8)
    second[:, :, 1] = 75
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.fromarray(first).save(first_path)
    Image.fromarray(second).save(second_path)
    top_lefts = np.asarray([[0.0, 0.0], [0.0, 6.0]])
    geometry = SimpleNamespace(
        files=[str(first_path), str(second_path)],
        output_size=(8, 16),
        top_lefts=top_lefts,
        quant_top_lefts=np.ceil(top_lefts).astype(int),
        centres=top_lefts + np.asarray([4.0, 5.0]),
        downsampled_image_size=(8, 10),
        downsample=1,
        indexes=range(2),
        image_shift=lambda _index: (0, 0),
    )
    roi = ((0, 0), (8, 16))

    expected = stitch_images(geometry, region_of_interest=roi)
    renderer = AcceleratedMosaicRenderer(geometry, cache_bytes=1024 * 1024)
    try:
        actual = renderer.render(roi)
    finally:
        renderer.clear()

    np.testing.assert_array_equal(actual[:-1, :-1], expected[:-1, :-1])
    assert np.any(actual[-1, :])
    assert np.any(actual[:, -1])


def test_full_resolution_renderer_preserves_outer_edges(tmp_path):
    """OME rendering keeps the source's final row and column at downsample one."""
    pixels = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    path = tmp_path / "single.png"
    Image.fromarray(pixels).save(path)
    geometry = SimpleNamespace(
        files=[str(path)],
        image_size=(8, 10),
        output_size=(8, 10),
        top_lefts=np.asarray([[0.0, 0.0]]),
        quant_top_lefts=np.asarray([[0, 0]]),
        centres=np.asarray([[4.0, 5.0]]),
        downsampled_image_size=(7, 9),
        downsample=1,
        indexes=range(1),
        image_shift=lambda _index: (0, 0),
    )

    renderer = AcceleratedMosaicRenderer(geometry, cache_bytes=1024 * 1024)
    try:
        actual = renderer.render(((0, 0), (8, 10)))
    finally:
        renderer.clear()

    np.testing.assert_array_equal(actual, pixels)
    assert np.any(actual[-1, :])
    assert np.any(actual[:, -1])
