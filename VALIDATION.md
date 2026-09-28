# Validation

Recorded 2026-09-28.

- Linux x86_64 regression suite: 18 passed; Ruff clean.
- The checked-in x86 CI command sequence completed REUSE, Ruff, `18 passed in
  0.89s`, sdist/wheel builds and the installed CLI help smoke check in a fresh
  CPython 3.11 environment. The wheel SHA-256 is
  `6cc4abb187b8a7b833f61f19d52b7625ecec634932b06b50bee83baef509ee05`;
  the sdist SHA-256 is
  `2ecbf1502ef924ba98a3460d568854648161149db75b35b6576872486fc625f1`.
- Linux x86_64 retained replay: 91 JPEG fields, 3 workers, 4 GiB cache.
- Output: 33104×29126 interleaved RGB uint8 OME-BigTIFF, 8 pyramid levels,
  2,552,787,826 bytes.
- Worker time: 242.75 s on libvips 8.9; measured whole-process wall time
  249.44 s and peak RSS 3,990,072 KiB; no swap.
- That timing A/B predated the final-edge correction; its matched level-0
  comparison had maximum and mean absolute differences of zero.
- libvips 8.10+ SubIFD writing and early refusal of incompatible OME requests
  on libvips 8.9 have regression tests. DZI-only output remains supported on
  the older host. The 8.9 timing file is retained as an algorithm/correctness
  benchmark, not as a QuPath-compatible release artifact.
- The old-host CLI refusal gate returned nonzero before preview, DZI, work-tile
  or OME output was created. A separate DZI-only CLI replay on libvips 8.9
  completed in 1.69 s wall time at 201,048 KiB peak RSS and produced the full
  14-file Deep Zoom tree.
- GPL adapter end-to-end replay: two retained 4056×3040 fields passed through
  OpenFlexure → core → this separately installed worker; worker time 8.53 s,
  adapter wall time 10.11 s, peak RSS 380,576 KiB; output OME-BigTIFF was
  81,235,878 bytes and the DZI pyramid was complete.
- Non-editable wheel gate: GPL server, core and worker wheels installed into a
  fresh CPython 3.11 environment and completed the process chain entirely from
  `site-packages`; 5.89 s wall time, 365,392 KiB peak RSS, 64,782,770-byte
  SubIFD OME-BigTIFF plus complete DZI. Level 0 matched the retained reference
  exactly (`max_abs=0`, `mean_abs=0`).
- Repeat-run gate: an unrelated earlier OME-TIFF was deliberately left in the
  scan directory. The worker consumed only the exact manifest tiles and
  completed without treating the prior output as an input frame.
- Public synthetic replay: four generated 512×512 tiles produced a 768×768,
  two-level SubIFD OME-BigTIFF under ARM64/QEMU with libvips 8.18.6.
  Decoded level 0 was pixel-identical to the generated 768×768 scene, including
  the final row and column (`max_abs=0`, `mean_abs=0`). This replay caught and
  now guards against the upstream preview downsampler's one-pixel bottom/right
  border in full-resolution output. QuPath 0.5.1 opened the 1,859,598-byte file
  with Bio-Formats and reported two levels at 1× and 2× downsample. Its SHA-256
  is `b4fbb029def60e808aad0f8af20c898e13b52a971680e9ae076910026c0edb3f`.
  The 72.76-second ARM64/QEMU wall time includes emulation and container startup
  and is an interoperability gate, not a throughput benchmark.
- Installing the final wheel into an isolated ARM64 target directory, with no
  editable markers or source import path, reproduced that exact OME hash and
  byte size under libvips 8.18.6. Installed package metadata identifies
  Alexander Fridman as author/publisher.
- Post-fix 91-tile replay retained the same 158 correlated pairs, 157 accepted
  pairs, 33104×29126 geometry and eight pyramid levels. Compared with the
  pre-fix file, 71,928 of 964,187,104 pixels changed (0.00746%); the mean
  absolute channel delta was 0.01532 and the maximum was 255. The changes are
  the newly retained edge samples and ownership decisions at one-pixel overlap
  boundaries. Validation on external FAT storage took 311.81 s, used 3 workers
  and a 4 GiB cache, peaked at 3,989,772 KiB RSS and used no swap. It is a
  correctness replay, not a replacement for the faster SATA timing benchmark.
- Manifest isolation falls back from hardlinks to byte-only copies on FAT or
  cross-filesystem inputs; the ordinary hardlink path remains the default.

This is software validation of a research prototype, not clinical validation.
