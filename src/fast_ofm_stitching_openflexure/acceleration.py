"""Bounded-memory correlation and exact nearest-centre mosaic rendering.

SPDX-License-Identifier: LGPL-3.0-only
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from copy import copy
from functools import partial
from typing import Literal, cast

import cv2
import numpy as np
from openflexure_stitching import plotting
from openflexure_stitching.communicate import notify
from openflexure_stitching.correlation import (
    displacement_from_crosscorrelation,
    fraction_under_threshold,
    high_pass_fourier_filter,
    locate_peak,
)
from openflexure_stitching.loading import CorrelatedImageSet
from openflexure_stitching.loading.cache import (
    CachedCorrelatedImageSet,
    load_cached_correlated_image_set,
    save_cached_correlated_image_set,
)
from openflexure_stitching.loading.image import OFSImage, to_gray
from openflexure_stitching.pipeline import (
    determine_thresholds,
    perform_final_position_optimisation,
)
from openflexure_stitching.settings import (
    CorrelationSettings,
    LoadingSettings,
    TilingSettings,
)
from openflexure_stitching.stitching import StitchGeometry
from openflexure_stitching.stitching.utils import (
    arange_from_slice,
    downsample_image,
    overlap_slices,
    regions_overlap,
)
from openflexure_stitching.types import PairData
from PIL import Image

CorrelationMode = Literal["full", "overlap-strip"]


class BoundedArrayCache:
    """A byte-bounded LRU for NumPy arrays with explicit release telemetry."""

    def __init__(self, max_bytes: int) -> None:
        """Create an empty cache whose retained arrays fit within ``max_bytes``."""
        if max_bytes < 0:
            raise ValueError("max_bytes must not be negative")
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._arrays: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str, loader: Callable[[], np.ndarray]) -> np.ndarray:
        """Return an existing array or load it and retain it within the budget."""
        if key in self._arrays:
            self.hits += 1
            array = self._arrays.pop(key)
            self._arrays[key] = array
            return array

        self.misses += 1
        array = loader()
        size = int(array.nbytes)
        if size > self.max_bytes:
            return array
        while self._arrays and self.current_bytes + size > self.max_bytes:
            _old_key, old_array = self._arrays.popitem(last=False)
            self.current_bytes -= int(old_array.nbytes)
            self.evictions += 1
        self._arrays[key] = array
        self.current_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        return array

    def clear(self) -> None:
        """Release every retained array."""
        self._arrays.clear()
        self.current_bytes = 0


def _full_displacement_from_ffts(
    image1: OFSImage,
    image2: OFSImage,
    fft1: np.ndarray,
    fft2: np.ndarray,
    *,
    settings: CorrelationSettings,
    high_pass_filter: np.ndarray,
    expected_displacement_px: tuple[float, float],
) -> tuple[np.ndarray, dict[float, float]]:
    """Reproduce upstream padded correlation while reusing forward FFTs."""
    if not settings.pad:
        return displacement_from_crosscorrelation(
            image1,
            image2,
            correlation_settings=settings,
            precalculated_filter=high_pass_filter,
            expected_displacement_px=expected_displacement_px,
        )

    corr = np.fft.irfft2(np.conj(fft1) * high_pass_filter * fft2)
    mean_corr = np.mean(corr)
    mask_width = 45
    corr[0:mask_width, 0:mask_width] = mean_corr
    corr[-mask_width:, 0:mask_width] = mean_corr
    corr[-mask_width:, -mask_width:] = mean_corr
    corr[0:mask_width, -mask_width:] = mean_corr

    expected = (
        -expected_displacement_px[0] * settings.resize,
        -expected_displacement_px[1] * settings.resize,
    )
    trial_peak = locate_peak(
        corr,
        fractional_threshold=0.02,
        quadrant_swap=True,
        expected_displacement_px=expected,
        mask_radius=settings.corr_mask_radius,
    )
    displacement = -trial_peak / settings.resize
    return displacement, fraction_under_threshold(corr)


def _strip_displacement(
    image1: OFSImage,
    image2: OFSImage,
    settings: CorrelationSettings,
    expected_displacement_px: tuple[float, float],
    image_cache: BoundedArrayCache,
) -> tuple[np.ndarray, dict[float, float]]:
    """Match only the stage-predicted overlap strips with bounded search margin."""

    def grayscale(image: OFSImage) -> np.ndarray:
        return to_gray(image.image_data(resize=settings.resize)).astype(
            np.float32, copy=False
        )

    image1_data = image_cache.get(image1.filepath, lambda: grayscale(image1))
    image2_data = image_cache.get(image2.filepath, lambda: grayscale(image2))
    if image1_data.shape != image2_data.shape:
        raise ValueError("Overlap-strip correlation requires equal image shapes")

    expected = np.rint(
        np.asarray(expected_displacement_px) * settings.resize
    ).astype(int)
    slices = overlap_slices(expected, image1_data.shape)
    if any(value is None for value in slices):
        raise ValueError("Stage-predicted images do not overlap")
    concrete_slices = [value for value in slices if value is not None]
    starts = np.asarray([value.start for value in concrete_slices], dtype=int)
    stops = np.asarray([value.stop for value in concrete_slices], dtype=int)
    sizes = stops - starts
    if np.any(sizes < 16):
        raise ValueError("Stage-predicted overlap strip is too narrow")

    # Keep at least eight pixels in the template's narrow dimension. At resize
    # 0.2 the 48-pixel search radius permits about 240 full-resolution pixels of
    # stage/CSM error while still using only the overlapping strip.
    margins = np.asarray(
        [min(48, max(4, (int(size) - 8) // 2)) for size in sizes], dtype=int
    )
    template_start = starts + margins
    template_stop = stops - margins
    expected_match_start = template_start - expected
    search_start = np.maximum(0, expected_match_start - margins)
    search_stop = np.minimum(
        np.asarray(image2_data.shape), template_stop - expected + margins
    )

    template = image1_data[
        tuple(
            slice(int(template_start[axis]), int(template_stop[axis]))
            for axis in range(2)
        )
    ]
    search = image2_data[
        tuple(
            slice(int(search_start[axis]), int(search_stop[axis]))
            for axis in range(2)
        )
    ]
    if any(search.shape[axis] < template.shape[axis] for axis in range(2)):
        raise ValueError("Overlap-strip search area is smaller than its template")

    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _minimum, score, _minimum_location, maximum_location = cv2.minMaxLoc(result)
    if not np.isfinite(score):
        raise ValueError("Overlap-strip correlation produced a non-finite score")
    matched_offset = np.asarray(
        [maximum_location[1], maximum_location[0]], dtype=int
    )
    displacement = (
        template_start - (search_start + matched_offset)
    ) / settings.resize
    return displacement, {0.9: float(score)}


class AcceleratedCorrelatedImageSet(CorrelatedImageSet):
    """Correlate pairs with bounded reusable spectra or overlap-strip matching."""

    def __init__(
        self,
        folder: str,
        *,
        loading_settings: LoadingSettings | None,
        correlation_settings: CorrelationSettings,
        cached: CachedCorrelatedImageSet | None,
        cache_bytes: int,
        correlation_mode: CorrelationMode,
    ) -> None:
        """Load the image set and correlate it using the selected bounded mode."""
        self._fast_ofm_cache_bytes = cache_bytes
        self._fast_ofm_correlation_mode = correlation_mode
        super().__init__(
            folder,
            loading_settings=loading_settings,
            correlation_settings=correlation_settings,
            cached=cached,
        )

    def _load_fft(self, key: str) -> np.ndarray:
        """Calculate one forward spectrum without retaining it in the OFS image."""
        return self[key].fft(
            resize=self.correlation_settings.resize,
            pad=self.correlation_settings.pad,
            cache_in_memory=False,
            cache_img_in_memory=False,
        )

    def _crosscorrelate_all(
        self, cached: CachedCorrelatedImageSet | None
    ) -> list[PairData]:
        notify(
            "Starting correlation of overlapping image pairs "
            f"with {self._fast_ofm_correlation_mode} mode."
        )
        cached_by_keys = (
            {pair.keys: pair for pair in cached.pairs}
            if isinstance(cached, CachedCorrelatedImageSet)
            else {}
        )
        remaining_pairs = self.find_overlapping_pairs(
            self.correlation_settings.minimum_overlap
        )
        # The final WSI may contain several non-overlapping prescan zones. Pairs
        # inside each zone remain useful; global connectivity is not required.
        pair_data: list[PairData] = []
        array_cache = BoundedArrayCache(self._fast_ofm_cache_bytes)
        high_pass_filter: np.ndarray | None = None

        try:
            for key1, key2 in remaining_pairs:
                cached_pair = cached_by_keys.get((key1, key2))
                if cached_pair is not None:
                    pair_data.append(
                        PairData(
                            keys=cached_pair.keys,
                            image_displacement=cached_pair.image_displacement,
                            stage_displacement=self.stage_displacement_px_between(
                                key2, key1
                            ),
                            fraction_under_threshold=(
                                cached_pair.fraction_under_threshold
                            ),
                        )
                    )
                    self.cache_stats["correlations_loaded_from_cache"] += 1
                    continue

                expected = self.stage_displacement_px_between(key2, key1)
                if self._fast_ofm_correlation_mode == "overlap-strip":
                    displacement, quality = _strip_displacement(
                        self[key1],
                        self[key2],
                        self.correlation_settings,
                        expected,
                        array_cache,
                    )
                else:
                    fft1 = array_cache.get(
                        key1,
                        partial(self._load_fft, key1),
                    )
                    fft2 = array_cache.get(
                        key2,
                        partial(self._load_fft, key2),
                    )
                    if high_pass_filter is None:
                        high_pass_filter = high_pass_fourier_filter(
                            fft1.shape, self.correlation_settings.high_pass_sigma
                        )
                    displacement, quality = _full_displacement_from_ffts(
                        self[key1],
                        self[key2],
                        fft1,
                        fft2,
                        settings=self.correlation_settings,
                        high_pass_filter=high_pass_filter,
                        expected_displacement_px=expected,
                    )
                pair_data.append(
                    PairData(
                        keys=(key1, key2),
                        image_displacement=displacement.tolist(),
                        stage_displacement=expected,
                        fraction_under_threshold=quality,
                    )
                )
                self.cache_stats["correlations_loaded_from_disk"] += 1
        finally:
            peak_bytes = array_cache.peak_bytes
            hits = array_cache.hits
            misses = array_cache.misses
            evictions = array_cache.evictions
            array_cache.clear()
            self.clear_image_memory_cache(retain=None)

        notify(
            "Correlation complete. "
            f"Pairs loaded from cache: {self.cache_stats['correlations_loaded_from_cache']}, "
            f"pairs calculated: {self.cache_stats['correlations_loaded_from_disk']}. "
            f"Bounded array cache hits={hits}, misses={misses}, evictions={evictions}, "
            f"peak={peak_bytes / (1024**2):.1f} MiB; released before rendering."
        )
        return pair_data


def load_correlate_and_position(
    folder: str,
    *,
    artifacts_folder: str | None = None,
    correlation_settings: CorrelationSettings,
    cache_bytes: int,
    correlation_mode: CorrelationMode,
) -> tuple[AcceleratedCorrelatedImageSet, dict[str, np.ndarray]]:
    """Load isolated inputs, cache in the scan directory, and optimise positions."""
    artifacts_folder = artifacts_folder or folder
    notify("Finding a list of all overlapping images")
    cached = (
        load_cached_correlated_image_set(artifacts_folder, correlation_settings)
        if correlation_mode == "full"
        else None
    )
    image_set = AcceleratedCorrelatedImageSet(
        folder,
        loading_settings=None,
        correlation_settings=correlation_settings,
        cached=cached,
        cache_bytes=cache_bytes,
        correlation_mode=correlation_mode,
    )
    if correlation_mode == "full":
        save_cached_correlated_image_set(
            artifacts_folder, image_set.data_for_caching()
        )

    notify("Getting all the image info ready to stitch")
    notify("Plotting the inputs to stitching")
    figure = plotting.plot_inputs(image_set)
    figure.savefig(
        f"{artifacts_folder}/stitching_inputs.png",
        dpi=250,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.clear()
    positions = position_disconnected_components(image_set)
    return image_set, positions


def overlapping_components(
    keys: list[str], pairs: list[PairData]
) -> list[list[str]]:
    """Find all overlap-connected islands, retaining acquisition order."""
    parent = {key: key for key in keys}

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for pair in pairs:
        first, second = pair.keys
        parent[root(second)] = root(first)

    groups: dict[str, list[str]] = {}
    for key in keys:
        groups.setdefault(root(key), []).append(key)
    return list(groups.values())


def position_disconnected_components(
    image_set: AcceleratedCorrelatedImageSet,
) -> dict[str, np.ndarray]:
    """Optimise each island locally and anchor its translation to stage data.

    Relative placement between islands is not observable from their pictures;
    only calibrated stage coordinates can supply it. Small/tree-like islands
    lack redundant correlations for trustworthy QC and remain stage-placed.
    """
    components = overlapping_components(image_set.keys(), image_set.pairs)
    if len(components) == 1:
        thresholds = determine_thresholds(image_set, TilingSettings(), plot=False)
        return perform_final_position_optimisation(image_set, *thresholds)

    notify(
        f"Found {len(components)} disconnected scan zones; "
        "optimising each locally and anchoring zones by calibrated stage positions."
    )
    positions: dict[str, np.ndarray] = {}
    for component_index, keys in enumerate(components, start=1):
        key_set = set(keys)
        pairs = [
            pair for pair in image_set.pairs if set(pair.keys) <= key_set
        ]
        stage_positions = {
            key: np.asarray(image_set[key].stage_position_px, dtype=float)
            for key in keys
        }
        if any(not np.all(np.isfinite(value)) for value in stage_positions.values()):
            raise ValueError("Disconnected zones require finite calibrated stage positions")

        if len(pairs) <= len(keys) - 1:
            notify(
                f"Zone {component_index}: {len(keys)} tiles/{len(pairs)} pairs; "
                "no redundant overlap, using calibrated stage positions."
            )
            positions.update(stage_positions)
            continue

        # Upstream thresholding/LSQR expects a connected image set. A shallow
        # view shares immutable image metadata without copying source images.
        view = copy(image_set)
        view._images = {key: image_set[key] for key in keys}
        view._pairs = pairs
        view._pair_discrepancies = [
            np.asarray(pair.image_displacement)
            - np.asarray(pair.stage_displacement)
            for pair in pairs
        ]
        view._norm_pair_discrepancies = [
            float(np.linalg.norm(delta)) for delta in view._pair_discrepancies
        ]
        try:
            thresholds = determine_thresholds(view, TilingSettings(), plot=False)
            local = perform_final_position_optimisation(view, *thresholds)
            if set(local) != key_set or any(
                not np.all(np.isfinite(local[key])) for key in keys
            ):
                raise ValueError("Local position fit omitted tiles or became non-finite")
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            notify(
                f"Zone {component_index}: local fit refused ({exc}); "
                "using calibrated stage positions."
            )
            positions.update(stage_positions)
            continue

        offsets = np.asarray([stage_positions[key] - local[key] for key in keys])
        anchor = np.median(offsets, axis=0)
        positions.update({key: np.asarray(local[key]) + anchor for key in keys})
        notify(
            f"Zone {component_index}: {len(keys)} tiles/{len(pairs)} pairs, "
            "correlation-fit internally; translation anchored to stage."
        )

    if set(positions) != set(image_set.keys()):
        raise RuntimeError("Disconnected-zone placement omitted one or more tiles")
    return positions


class PackedOwnershipMasks:
    """Precompute exact nearest-centre masks using only actual neighbours."""

    def __init__(self, geometry: StitchGeometry) -> None:
        """Build and pack every immutable nearest-centre ownership mask."""
        # The upstream preview downsampler drops the final source row and
        # column, even when downsample is one.  Full-resolution OME output must
        # retain those pixels or it gains a black bottom/right border.
        if geometry.downsample == 1 and hasattr(geometry, "image_size"):
            self.height, self.width = map(int, geometry.image_size)
        else:
            self.width = int(geometry.downsampled_image_size[1])
            self.height = int(geometry.downsampled_image_size[0])
        self.neighbours = self._build_neighbour_graph(geometry)
        self._packed = [self._build_mask(index, geometry) for index in geometry.indexes]

    @property
    def nbytes(self) -> int:
        """Return the byte size of all packed masks."""
        return sum(int(mask.nbytes) for mask in self._packed)

    def _build_neighbour_graph(self, geometry: StitchGeometry) -> list[list[int]]:
        neighbours: list[list[int]] = [[] for _ in geometry.indexes]
        shape = np.asarray([self.height, self.width])
        for index in geometry.indexes:
            differences = geometry.quant_top_lefts - geometry.quant_top_lefts[index]
            overlapping = np.flatnonzero(
                (np.abs(differences[:, 0]) < shape[0])
                & (np.abs(differences[:, 1]) < shape[1])
            )
            neighbours[index] = [
                int(other) for other in overlapping if int(other) != index
            ]
        return neighbours

    def _build_mask(self, index: int, geometry: StitchGeometry) -> np.ndarray:
        keep = np.ones((self.height, self.width), dtype=bool)
        centre = geometry.centres[index, :]
        top_left = geometry.quant_top_lefts[index, :]
        for other in self.neighbours[index]:
            other_centre = geometry.centres[other, :]
            difference = geometry.quant_top_lefts[other, :] - top_left
            x_range, y_range = overlap_slices(difference, keep.shape)
            if x_range is None or y_range is None:
                continue
            midpoint = (other_centre + centre) / 2.0 - top_left
            x_values = arange_from_slice(x_range)[:, np.newaxis] * difference[0]
            y_values = arange_from_slice(y_range)[np.newaxis, :] * difference[1]
            angle = np.arctan2(difference[0], difference[1])
            if -np.pi / 4 < angle <= 3 * np.pi / 4:
                remove = (x_values + y_values) > np.dot(midpoint, difference)
            else:
                remove = (x_values + y_values) >= np.dot(midpoint, difference)
            keep[x_range, y_range][remove] = False
        return np.packbits(keep, axis=1, bitorder="little")

    def region(self, index: int, slices: tuple[slice, slice]) -> np.ndarray:
        """Unpack only the requested rows, then return the requested columns."""
        rows, columns = slices
        unpacked = np.unpackbits(
            self._packed[index][rows],
            axis=1,
            count=self.width,
            bitorder="little",
        ).astype(bool, copy=False)
        return unpacked[:, columns]

    def clear(self) -> None:
        """Release masks and their neighbour graph."""
        self._packed.clear()
        self.neighbours.clear()


class AcceleratedMosaicRenderer:
    """Reuse decoded images and precomputed packed ownership masks."""

    def __init__(self, geometry: StitchGeometry, cache_bytes: int) -> None:
        """Precompute masks and assign the remaining budget to decoded images."""
        self.geometry = geometry
        self.masks = PackedOwnershipMasks(geometry)
        image_budget = max(0, cache_bytes - self.masks.nbytes)
        self.images = BoundedArrayCache(image_budget)

    def _load_image(self, filename: str, index: int) -> np.ndarray:
        """Decode and position one full-resolution source image."""
        if self.geometry.downsample == 1:
            return np.asarray(Image.open(filename))
        return downsample_image(
            np.asarray(Image.open(filename)),
            self.geometry.downsample,
            shift=self.geometry.image_shift(index),
        )

    def render(
        self, region_of_interest: tuple[tuple[int, int], tuple[int, int]]
    ) -> np.ndarray:
        """Render one output region with the same nearest-centre pixel ownership."""
        canvas_origin = np.asarray(region_of_interest[0], dtype=int)
        canvas_size = np.asarray(region_of_interest[1], dtype=int)
        output = np.zeros(tuple(canvas_size) + (3,), dtype=np.uint8)
        for index, filename in enumerate(self.geometry.files):
            image_roi = (
                tuple(self.geometry.top_lefts[index, :]),
                (self.masks.height, self.masks.width),
            )
            if not regions_overlap(region_of_interest, image_roi):
                continue

            image = self.images.get(
                filename,
                partial(self._load_image, filename, index),
            )
            top_left = self.geometry.quant_top_lefts[index, :] - canvas_origin
            bottom_right = top_left + np.asarray(image.shape[:2])
            canvas_slices = cast(
                tuple[slice, slice],
                tuple(
                    slice(
                        max(0, int(top_left[axis])),
                        min(int(canvas_size[axis]), int(bottom_right[axis])),
                    )
                    for axis in range(2)
                ),
            )
            image_slices = cast(
                tuple[slice, slice],
                tuple(
                    slice(
                        max(0, -int(top_left[axis])),
                        min(int(canvas_size[axis]), int(bottom_right[axis]))
                        - int(top_left[axis]),
                    )
                    for axis in range(2)
                ),
            )
            source = image[image_slices]
            keep = self.masks.region(index, image_slices)
            output[canvas_slices] += source * keep[:, :, np.newaxis]
        return output

    def clear(self) -> None:
        """Release decoded images and masks before libvips builds the pyramid."""
        self.images.clear()
        self.masks.clear()


def position_delta_summary(
    reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]
) -> dict[str, float]:
    """Compare positions after removing their arbitrary global translation."""
    keys = list(reference)
    deltas = np.asarray([candidate[key] - reference[key] for key in keys])
    centered = deltas - np.mean(deltas, axis=0)
    norms = np.linalg.norm(centered, axis=1)
    return {
        "median_px": float(np.median(norms)),
        "p95_px": float(np.percentile(norms, 95)),
        "max_px": float(np.max(norms)),
    }
