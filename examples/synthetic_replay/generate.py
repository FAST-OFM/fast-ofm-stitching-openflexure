#!/usr/bin/env python3
"""Generate a deterministic, non-specimen stitching replay.

SPDX-FileCopyrightText: 2026 Alexander Fridman
SPDX-License-Identifier: LGPL-3.0-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_scene() -> Image.Image:
    """Return a deterministic RGB scene with enough texture for correlation."""
    size = 768
    y, x = np.mgrid[:size, :size]
    rng = np.random.default_rng(20260928)
    noise = rng.integers(0, 32, size=(size, size), dtype=np.uint8)
    pixels = np.empty((size, size, 3), dtype=np.uint8)
    pixels[..., 0] = (35 + x * 3 // 5 + noise) % 256
    pixels[..., 1] = (55 + y * 2 // 3 + noise // 2) % 256
    pixels[..., 2] = (80 + (x + y) // 3 + noise) % 256
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse((84, 110, 350, 376), outline=(245, 245, 245), width=18)
    draw.rectangle((410, 76, 692, 304), outline=(18, 30, 45), width=22)
    draw.polygon(((140, 620), (380, 410), (656, 652)), outline=(255, 205, 75), width=18)
    draw.line((40, 490, 720, 450), fill=(220, 65, 85), width=14)
    return image


def generate(output: Path) -> dict[str, object]:
    """Write four overlapping tiles, a strict manifest and synthetic telemetry."""
    output.mkdir(parents=True, exist_ok=False)
    scene = _synthetic_scene()
    tile_size = 512
    positions = ((0, 0), (256, 0), (256, 256), (0, 256))
    tile_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, str]] = []
    for order, (x, y) in enumerate(positions):
        path = output / f"img_{x}_{y}_0.png"
        scene.crop((x, y, x + tile_size, y + tile_size)).save(path, optimize=True)
        manifest_rows.append({"uri": path.resolve().as_uri()})
        tile_rows.append(
            {
                "capture_order": order,
                "filename": path.name,
                "stage_position": {"x": x, "y": y, "z": 0},
                "sha256": _sha256(path),
            }
        )

    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps({"schema": "fast_ofm_synthetic_tiles/v1", "tiles": manifest_rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    telemetry = output / "telemetry.json"
    telemetry.write_text(
        json.dumps(
            {
                "schema": "fast_ofm_synthetic_replay/v1",
                "synthetic": True,
                "contains_specimen_data": False,
                "scene_pixels": [768, 768],
                "tile_pixels": [512, 512],
                "nominal_overlap_pixels": [256, 256],
                "tiles": tile_rows,
                "focus_surface": {
                    "model": "illustrative_plane",
                    "formula": "z_um = 100 + 0.002*x_um - 0.0015*y_um",
                    "anchors": [
                        {"x_um": 0, "y_um": 0, "z_um": 100.0},
                        {"x_um": 256, "y_um": 0, "z_um": 100.512},
                        {"x_um": 256, "y_um": 256, "z_um": 100.128},
                        {"x_um": 0, "y_um": 256, "z_um": 99.616},
                    ],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(output),
        "manifest": str(manifest),
        "telemetry": str(telemetry),
        "tile_count": len(tile_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for generated replay inputs")
    args = parser.parse_args()
    print(json.dumps(generate(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
