"""Tests for direct pyramidal WSI output."""

import errno
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fast_ofm_stitching_openflexure import pyramidal as pyramidal_stitch
from fast_ofm_stitching_openflexure.settings import StitchingProcessSettings


def test_isolated_tile_directory_exposes_only_manifest_frames(tmp_path):
    """Existing image outputs cannot be rediscovered as source scan frames."""
    tiles = []
    for index in range(2):
        path = tmp_path / f"img_{index}_0_0.jpeg"
        path.write_bytes(f"tile-{index}".encode())
        tiles.append({"uri": path.resolve().as_uri()})
    (tmp_path / "previous-result.ome.tiff").write_bytes(b"not-a-source-tile")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tiles": tiles}), encoding="utf-8")

    with pyramidal_stitch.isolated_tile_directory(
        str(tmp_path), str(manifest)
    ) as isolated:
        isolated_path = Path(isolated)
        assert sorted(path.name for path in isolated_path.iterdir()) == [
            "img_0_0_0.jpeg",
            "img_1_0_0.jpeg",
        ]
        assert all(
            path.stat().st_ino == (tmp_path / path.name).stat().st_ino
            for path in isolated_path.iterdir()
        )

    assert not any(path.name.startswith(".fast-ofm-input-") for path in tmp_path.iterdir())


def test_isolated_tile_directory_copies_when_hardlinks_are_unsupported(tmp_path, mocker):
    """Manifest isolation remains usable on FAT and across filesystems."""
    tiles = []
    for index in range(2):
        path = tmp_path / f"img_{index}_0_0.png"
        path.write_bytes(f"tile-{index}".encode())
        tiles.append({"uri": path.resolve().as_uri()})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tiles": tiles}), encoding="utf-8")
    mocker.patch.object(
        pyramidal_stitch.os,
        "link",
        side_effect=OSError(errno.EPERM, "hardlinks unsupported"),
    )

    with pyramidal_stitch.isolated_tile_directory(
        str(tmp_path), str(manifest)
    ) as isolated:
        isolated_path = Path(isolated)
        assert sorted(path.read_bytes() for path in isolated_path.iterdir()) == [
            b"tile-0",
            b"tile-1",
        ]
        assert all(
            path.stat().st_ino != (tmp_path / path.name).stat().st_ino
            for path in isolated_path.iterdir()
        )


def test_lossless_work_tiles_cover_non_multiple_geometry(tmp_path, mocker):
    """Edge work tiles preserve the exact output dimensions without padding."""
    geometry = mocker.Mock(output_size=(3, 5))

    def fake_render(region_of_interest):
        height, width = region_of_interest[1]
        return np.zeros((height, width, 3), dtype=np.uint8)

    renderer = mocker.Mock()
    renderer.render.side_effect = fake_render
    mocker.patch.object(
        pyramidal_stitch, "AcceleratedMosaicRenderer", return_value=renderer
    )

    tiles = pyramidal_stitch._write_lossless_work_tiles(tmp_path, geometry, 4, 1024)

    assert [(x, y) for x, y, _path in tiles] == [(0, 0), (4, 0)]
    assert [Image.open(path).size for _x, _y, path in tiles] == [(4, 3), (1, 3)]
    renderer.clear.assert_called_once_with()


def test_process_limits_configure_vips(mocker):
    """The requested process limits reach libvips without allocating the cache."""
    concurrency = mocker.patch.object(pyramidal_stitch.pyvips, "concurrency_set")
    cache = mocker.patch.object(pyramidal_stitch.pyvips, "cache_set_max_mem")

    pyramidal_stitch._configure_process(StitchingProcessSettings())

    concurrency.assert_called_once_with(3)
    cache.assert_called_once_with(4096 * 1024 * 1024)


def test_ome_description_declares_interleaved_uint8_rgb():
    """OME metadata must describe the pixels QuPath will actually read."""
    description = pyramidal_stitch._ome_description("scan", 20, 10, 0.5)

    assert 'Type="uint8"' in description
    assert 'SizeX="20" SizeY="10"' in description
    assert 'SizeC="3"' in description
    assert 'SamplesPerPixel="3"' in description
    assert 'PhysicalSizeX="0.5"' in description
    assert 'PhysicalSizeXUnit="µm"' in description


def test_ome_description_omits_unknown_physical_scale():
    """An unknown calibration must not silently become a one-micron scale bar."""
    description = pyramidal_stitch._ome_description("scan", 20, 10, None)

    assert "PhysicalSize" not in description


def test_bigtiff_refuses_non_subifd_libvips(tmp_path, mocker):
    """Old libvips must not produce a pyramid QuPath sees as one flat level."""
    image = mocker.Mock(width=20, height=10)
    mocker.patch.object(
        pyramidal_stitch.pyvips,
        "version",
        side_effect=lambda index: {0: 8, 1: 9, 2: 1}[index],
    )

    with pytest.raises(RuntimeError, match="libvips 8.10 or newer"):
        pyramidal_stitch._save_ome_bigtiff(
            image, tmp_path / "scan.ome.tiff", name="scan", pixel_size_um=None
        )

    image.tiffsave.assert_not_called()


def test_writer_refuses_old_libvips_before_preview_or_render(tmp_path, mocker):
    """An incompatible OME request fails before any expensive output work."""
    mocker.patch.object(
        pyramidal_stitch.pyvips,
        "version",
        side_effect=lambda index: {0: 8, 1: 9, 2: 1}[index],
    )
    save_preview = mocker.patch.object(pyramidal_stitch, "save_preview")
    render = mocker.patch.object(pyramidal_stitch, "_write_lossless_work_tiles")

    with pytest.raises(RuntimeError, match="libvips 8.10 or newer"):
        pyramidal_stitch.write_pyramidal_outputs(
            mocker.Mock(),
            {"image.jpg": np.array((0, 0))},
            images_dir=str(tmp_path),
            stitch_tiff=True,
            stitch_dzi=True,
            work_tile_size=8,
            process_settings=StitchingProcessSettings(),
        )

    save_preview.assert_not_called()
    render.assert_not_called()


def test_bigtiff_uses_subifd_when_supported(tmp_path, mocker):
    """Modern libvips always writes the QuPath-compatible SubIFD pyramid."""
    image = mocker.Mock(width=20, height=10)
    image.copy.return_value = image
    mocker.patch.object(
        pyramidal_stitch.pyvips,
        "version",
        side_effect=lambda index: {0: 8, 1: 10, 2: 0}[index],
    )

    pyramidal_stitch._save_ome_bigtiff(
        image, tmp_path / "scan.ome.tiff", name="scan", pixel_size_um=None
    )

    assert image.tiffsave.call_args.kwargs["subifd"] is True


def test_writer_creates_dzi_and_lossless_bigtiff_without_flat_jpeg(tmp_path, mocker):
    """The final writer targets only pyramidal outputs plus the small preview."""
    image_set = mocker.Mock()
    geometry = mocker.Mock(output_size=(10, 20), pixel_size_um=0.5)
    mocker.patch.object(pyramidal_stitch, "StitchGeometry", return_value=geometry)
    mocker.patch.object(pyramidal_stitch, "save_fiji_config")
    mocker.patch.object(pyramidal_stitch, "save_preview", return_value="preview.jpg")
    mocker.patch.object(pyramidal_stitch, "create_thumbnail")
    mocker.patch.object(pyramidal_stitch, "_require_subifd_support")
    mocker.patch.object(
        pyramidal_stitch, "choose_final_filename_prefix", return_value="scan"
    )
    mocker.patch.object(pyramidal_stitch, "_write_lossless_work_tiles", return_value=[])
    image = mocker.Mock()
    mocker.patch.object(pyramidal_stitch, "_assemble_vips_image", return_value=image)
    save_tiff = mocker.patch.object(pyramidal_stitch, "_save_ome_bigtiff")

    pyramidal_stitch.write_pyramidal_outputs(
        image_set,
        {"image.jpg": np.array((0, 0))},
        images_dir=str(tmp_path),
        stitch_tiff=True,
        stitch_dzi=True,
        work_tile_size=8,
        process_settings=StitchingProcessSettings(),
    )

    image.dzsave.assert_called_once()
    save_tiff.assert_called_once_with(
        image,
        tmp_path / "scan_stitched.ome.tiff",
        name="scan_stitched",
        pixel_size_um=0.5,
    )
    assert not (tmp_path / "scan_stitched.jpg").exists()
