# Synthetic stitching replay

This example creates four deterministic 512 × 512 RGB tiles from a generated
768 × 768 scene. It contains no microscope, specimen, patient or laboratory
data and carries no EXIF metadata. The snake-order coordinates are encoded in
ordinary OpenFlexure-compatible filenames.

The generator is original Fast OFM test code licensed under LGPL-3.0-only.
This guide, the reference report and its checksums are licensed under
CC-BY-NC-SA-4.0, as recorded by the repository's `REUSE.toml`. Every input
pixel and telemetry value is generated deterministically by `generate.py`.

Generate the input in a new directory:

```bash
python examples/synthetic_replay/generate.py /tmp/fast-ofm-synthetic
```

Then run the separately installed worker:

```bash
fast-ofm-stitch-openflexure \
  --stitch-tiff \
  --tile-size 128 \
  --workers 3 \
  --ram-cache-mb 512 \
  --correlation-mode full \
  --minimum-overlap 0.4 \
  --resize 1.0 \
  --output-name synthetic_replay \
  --tile-manifest /tmp/fast-ofm-synthetic/manifest.json \
  /tmp/fast-ofm-synthetic
```

The expected result is a small pyramidal
`synthetic_replay.ome.tiff`. `telemetry.json` explicitly marks the data as
synthetic and records the tile hashes plus an illustrative focus plane. The
manifest contains absolute local file URIs and is therefore generated rather
than checked in.

Reference output facts and hashes from the pinned validation environment are
recorded in `reference-report.json` and `CHECKSUMS.sha256`. The report's OME
hash is specific to the listed library versions; the stronger semantic gate is
the decoded level-0 comparison, which must have zero pixel difference from the
generated scene and must retain the bottom and right edges. The reference was
also opened through QuPath 0.5.1/Bio-Formats, which detected the base image and
one SubIFD reduction as two pyramid levels. OME output therefore requires
libvips 8.10 or newer; DZI-only output remains available on older hosts.
