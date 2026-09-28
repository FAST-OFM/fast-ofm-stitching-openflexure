"""Build a DZI and lossless pyramidal OME-BigTIFF without a flat JPEG.

SPDX-License-Identifier: LGPL-3.0-only
"""

import argparse
import errno
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.sax.saxutils import escape

import numpy as np
import pyvips
from openflexure_stitching.communicate import notify
from openflexure_stitching.loading import OFSImageSet
from openflexure_stitching.pipeline import (
    choose_final_filename_prefix,
    save_fiji_config,
    save_preview,
)
from openflexure_stitching.settings import CorrelationSettings, OutputSettings
from openflexure_stitching.stitching import StitchGeometry, create_thumbnail
from PIL import Image

from fast_ofm_stitching_openflexure.acceleration import (
    AcceleratedMosaicRenderer,
    load_correlate_and_position,
)
from fast_ofm_stitching_openflexure.settings import StitchingProcessSettings

PYRAMID_TILE_SIZE = 512
INPUT_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _manifest_tiles(manifest_path: Path, images_dir: Path) -> list[Path]:
    """Read exact local tile paths from the already validated process manifest."""
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptors = document.get("tiles")
    if not isinstance(descriptors, list):
        raise ValueError("tile manifest must contain a tiles array")
    tiles: list[Path] = []
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"tiles[{index}] must be an artifact descriptor")
        uri = descriptor.get("uri")
        if not isinstance(uri, str):
            raise ValueError(f"tiles[{index}].uri must be a string")
        parsed = urlparse(uri)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"tiles[{index}].uri must be a local file URI")
        candidate = Path(unquote(parsed.path))
        if candidate.is_symlink():
            raise ValueError(f"tiles[{index}] must not be a symbolic link")
        path = candidate.resolve(strict=True)
        if path.parent != images_dir or not path.is_file():
            raise ValueError(f"tiles[{index}] is not a regular file in images_dir")
        if path.suffix.lower() not in INPUT_SUFFIXES:
            raise ValueError(f"tiles[{index}] has an unsupported image suffix")
        tiles.append(path)
    return tiles


def _default_tiles(images_dir: Path) -> list[Path]:
    """Select conventional OpenFlexure scan frames for direct CLI use."""
    return sorted(
        path
        for path in images_dir.glob("img_*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in INPUT_SUFFIXES
    )


@contextmanager
def isolated_tile_directory(
    images_dir: str, tile_manifest: str | None
) -> Iterator[str]:
    """Expose only declared scan frames to upstream folder-scanning code.

    OpenFlexure Stitching discovers every image in its input directory.  A
    completed OME-TIFF from an earlier run must therefore be kept outside that
    discovery view even though outputs remain in the original scan directory.
    """
    source = Path(images_dir).resolve(strict=True)
    if not source.is_dir():
        raise ValueError("images_dir must be a directory")
    tiles = (
        _manifest_tiles(Path(tile_manifest).resolve(strict=True), source)
        if tile_manifest is not None
        else _default_tiles(source)
    )
    if len(tiles) < 2:
        raise ValueError("at least two scan tiles are required")
    names = [path.name for path in tiles]
    if len(names) != len(set(names)):
        raise ValueError("tile manifest contains duplicate filenames")

    with tempfile.TemporaryDirectory(prefix=".fast-ofm-input-", dir=source) as temporary:
        isolated = Path(temporary)
        for path in tiles:
            destination = isolated / path.name
            try:
                os.link(path, destination)
            except OSError as error:
                if error.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP}:
                    raise
                shutil.copyfile(path, destination)
        yield str(isolated)


def _configure_process(settings: StitchingProcessSettings) -> None:
    """Apply explicit native concurrency and libvips RAM-cache limits."""
    pyvips.concurrency_set(settings.workers)
    pyvips.cache_set_max_mem(settings.ram_cache_mb * 1024 * 1024)


def _require_subifd_support() -> None:
    """Fail before rendering when QuPath-compatible TIFF output is unavailable."""
    libvips_version = (pyvips.version(0), pyvips.version(1))
    if libvips_version < (8, 10):
        raise RuntimeError(
            "pyramidal OME-BigTIFF requires libvips 8.10 or newer for SubIFD "
            "output; upgrade libvips or request DZI-only output"
        )


def _write_lossless_work_tiles(
    directory: Path,
    stitch_geometry: StitchGeometry,
    tile_size: int,
    cache_bytes: int,
) -> list[tuple[int, int, Path]]:
    """Render the mosaic in bounded-memory, lossless TIFF work tiles."""
    output_height, output_width = map(int, stitch_geometry.output_size)
    row_count = math.ceil(output_height / tile_size)
    notify(
        f"Rendering {output_width}x{output_height} mosaic in {row_count} lossless tile rows"
    )
    tiles: list[tuple[int, int, Path]] = []
    renderer = AcceleratedMosaicRenderer(stitch_geometry, cache_bytes)
    try:
        for row_index, y in enumerate(range(0, output_height, tile_size), start=1):
            height = min(tile_size, output_height - y)
            for x in range(0, output_width, tile_size):
                width = min(tile_size, output_width - x)
                pixels = renderer.render(((y, x), (height, width)))
                path = directory / f"{y}_{x}.tif"
                Image.fromarray(pixels).save(path, compression="tiff_lzw")
                tiles.append((x, y, path))
            notify(f"Rendered lossless tile row {row_index}/{row_count}")
    finally:
        renderer.clear()
        notify("Released decoded-image and ownership-mask caches before pyramid writing")
    return tiles


def _assemble_vips_image(
    tiles: Sequence[tuple[int, int, Path]], width: int, height: int
) -> pyvips.Image:
    """Lazily assemble lossless work tiles into one libvips image graph."""
    image = pyvips.Image.black(width, height, bands=3)
    for x, y, path in tiles:
        tile = pyvips.Image.new_from_file(str(path), access="random")
        image = image.insert(tile, x, y)
    return image


def _ome_description(
    name: str, width: int, height: int, pixel_size_um: float | None
) -> str:
    """Return minimal OME-XML for an interleaved uint8 RGB image."""
    escaped_name = escape(name, {'"': "&quot;"})
    physical_size = (
        ""
        if pixel_size_um is None
        else f' PhysicalSizeX="{pixel_size_um}" PhysicalSizeY="{pixel_size_um}"'
        ' PhysicalSizeXUnit="µm" PhysicalSizeYUnit="µm"'
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="{escaped_name}">
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint8"
            SizeX="{width}" SizeY="{height}" SizeZ="1" SizeC="3" SizeT="1"
            Interleaved="true"{physical_size}>
      <Channel ID="Channel:0:0" Name="RGB" SamplesPerPixel="3"/>
      <TiffData IFD="0" PlaneCount="1"/>
    </Pixels>
  </Image>
</OME>"""


def _save_ome_bigtiff(
    image: pyvips.Image, path: Path, *, name: str, pixel_size_um: float | None
) -> None:
    """Write a tiled, lossless, SubIFD pyramidal OME-BigTIFF."""
    _require_subifd_support()
    if pixel_size_um is not None:
        image = image.copy(
            xres=1000 / pixel_size_um,
            yres=1000 / pixel_size_um,
        )
    image.set_type(
        pyvips.GValue.gstr_type,
        "image-description",
        _ome_description(name, image.width, image.height, pixel_size_um),
    )
    options = {
        "tile": True,
        "tile_width": PYRAMID_TILE_SIZE,
        "tile_height": PYRAMID_TILE_SIZE,
        "pyramid": True,
        "bigtiff": True,
        "compression": "deflate",
        "predictor": "horizontal",
        "region_shrink": "mean",
        "subifd": True,
    }
    image.tiffsave(
        str(path),
        **options,
    )


def write_pyramidal_outputs(
    image_set: OFSImageSet,
    positions: Mapping[str, np.ndarray],
    *,
    images_dir: str,
    stitch_tiff: bool,
    stitch_dzi: bool,
    work_tile_size: int,
    process_settings: StitchingProcessSettings,
    output_name: str | None = None,
) -> Path | None:
    """Write preview plus requested full-resolution pyramids from positions."""
    _configure_process(process_settings)
    if stitch_tiff:
        _require_subifd_support()
    output_settings = OutputSettings(output_dir=images_dir)
    position_dict = dict(positions)
    save_fiji_config(position_dict, output_settings=output_settings)
    preview_path = save_preview(
        image_set, position_dict, output_settings=output_settings
    )
    create_thumbnail(preview_path)

    if not (stitch_tiff or stitch_dzi):
        return None

    geometry = StitchGeometry(image_set, positions=position_dict)
    output_height, output_width = map(int, geometry.output_size)
    prefix = output_name or choose_final_filename_prefix(images_dir) + "_stitched"
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", prefix) is None:
        raise ValueError("output_name contains unsupported characters")
    output_directory = Path(images_dir)
    tiff_path: Path | None = None

    with tempfile.TemporaryDirectory(prefix=".pyramid-work-") as temporary:
        tiles = _write_lossless_work_tiles(
            Path(temporary),
            geometry,
            work_tile_size,
            process_settings.ram_cache_mb * 1024 * 1024,
        )
        image = _assemble_vips_image(tiles, output_width, output_height)
        if stitch_dzi:
            notify("Writing Deep Zoom pyramid for the OpenFlexure viewer")
            image.dzsave(
                str(output_directory / prefix),
                tile_size=PYRAMID_TILE_SIZE,
                overlap=1,
                suffix=".jpg[Q=95,optimize_coding]",
            )
        if stitch_tiff:
            notify("Writing lossless pyramidal OME-BigTIFF")
            pixel_size_um = geometry.pixel_size_um
            if (
                pixel_size_um is None
                or pixel_size_um <= 0
                or not math.isfinite(pixel_size_um)
            ):
                notify(
                    "WARNING: no validated pixel size; omitting physical scale from OME metadata"
                )
                pixel_size_um = None
            tiff_path = output_directory / f"{prefix}.ome.tiff"
            _save_ome_bigtiff(
                image,
                tiff_path,
                name=prefix,
                pixel_size_um=pixel_size_um,
            )
    return tiff_path


def stitch_with_correlation(
    images_dir: str,
    *,
    tile_manifest: str | None = None,
    minimum_overlap: float,
    resize: float,
    stitch_tiff: bool,
    stitch_dzi: bool,
    tile_size: int,
    process_settings: StitchingProcessSettings,
    output_name: str | None = None,
) -> Path | None:
    """Find correlated positions, then write direct pyramidal outputs."""
    with isolated_tile_directory(images_dir, tile_manifest) as input_directory:
        image_set, positions = load_correlate_and_position(
            input_directory,
            artifacts_folder=images_dir,
            correlation_settings=CorrelationSettings(
                minimum_overlap=minimum_overlap,
                resize=resize,
            ),
            cache_bytes=process_settings.ram_cache_mb * 1024 * 1024,
            correlation_mode=process_settings.correlation_mode,
        )
        return write_pyramidal_outputs(
            image_set,
            positions,
            images_dir=images_dir,
            stitch_tiff=stitch_tiff,
            stitch_dzi=stitch_dzi,
            work_tile_size=tile_size,
            process_settings=process_settings,
            output_name=output_name,
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run direct pyramidal stitching from the command line."""
    defaults = StitchingProcessSettings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stitch-tiff", action="store_true")
    parser.add_argument("--stitch-dzi", action="store_true")
    parser.add_argument("--tile-size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--ram-cache-mb", type=int, default=defaults.ram_cache_mb)
    parser.add_argument(
        "--correlation-mode",
        choices=("full", "overlap-strip"),
        default=defaults.correlation_mode,
    )
    parser.add_argument("--minimum-overlap", type=float, required=True)
    parser.add_argument("--resize", type=float, required=True)
    parser.add_argument("--output-name")
    parser.add_argument("--tile-manifest")
    parser.add_argument("images_dir")
    args = parser.parse_args(argv)
    stitch_with_correlation(
        args.images_dir,
        tile_manifest=args.tile_manifest,
        minimum_overlap=args.minimum_overlap,
        resize=args.resize,
        stitch_tiff=args.stitch_tiff,
        stitch_dzi=args.stitch_dzi,
        tile_size=args.tile_size,
        process_settings=StitchingProcessSettings(
            workers=args.workers,
            ram_cache_mb=args.ram_cache_mb,
            correlation_mode=args.correlation_mode,
        ),
        output_name=args.output_name,
    )


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    main()
