<p align="center">
  <a href="https://github.com/FAST-OFM">
    <img src="https://github.com/FAST-OFM.png?size=200" alt="Fast OFM logo" width="132">
  </a>
</p>

<h1 align="center">Fast OFM Stitching — OpenFlexure worker</h1>

<p align="center"><strong>Replaceable LGPL registration and OME-BigTIFF worker</strong></p>

This independently installed worker contains the Fast OFM extensions that use
or modify `openflexure-stitching`. It is distributed under LGPL-3.0-only so the
LGPL implementation remains replaceable and modifiable independently of the
PolyForm-licensed `fast-ofm-core` process.

The worker provides bounded FFT/image caches, disconnected-zone placement,
nearest-centre mosaic rendering, exact manifest-based input isolation, and
lossless pyramidal OME-BigTIFF output. Manifest isolation prevents an earlier
OME-TIFF or preview in the scan directory from being rediscovered as a source
frame. The worker does not control a microscope, camera, stage, or illumination.

This is prototype research software and is not a medical device.

Pyramidal OME-BigTIFF output requires libvips 8.10 or newer because QuPath and
Bio-Formats expect the reduced levels as SubIFDs. Older libvips versions are
refused for OME output instead of silently writing a multi-page TIFF that a
pathology viewer may expose as one flat resolution. DZI-only output remains
available on older hosts.

The package is separate from both the GPL OpenFlexure server adapter and the
PolyForm Noncommercial Fast OFM Core. Installations may replace or modify this
worker under LGPL-3.0-only. See `AUTHORS.md`, `UPSTREAM.md` and `PROVENANCE.md`
for attribution and source boundaries.

## Reproducible synthetic replay

[`examples/synthetic_replay`](examples/synthetic_replay/README.md) generates a
small non-specimen tile set and builds a real pyramidal OME-BigTIFF. The pinned
ARM64/QEMU reference replay uses libvips 8.18.6, has two pyramid levels,
preserves every level-0 pixel and contains no microscope, specimen, patient or
laboratory data. QuPath 0.5.1/Bio-Formats reports 1× and 2× levels.
