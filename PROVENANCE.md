# Provenance

Audit date: 2026-09-28.

This private clean staging tree was split from the Fast OFM OpenFlexure GPL
integration after a function-level comparison with
`openflexure-stitching==0.3.1`.

- `acceleration.py` originated as the Fast OFM
  `stitch_acceleration.py` candidate. Its correlation-loop and ownership-mask
  adaptations are conservatively treated as LGPL-derived; current SHA-256:
  `3e5a7bc9d6c4aeceabae8f218305cb028b5a060883569dbadf4a8d69132a9f9f`.
- `pyramidal.py` originated as Fast OFM's new pyramidal writer and was rewritten
  to remove every OpenFlexure Microscope Server import; current SHA-256:
  `2d0ce6cb1b256456439a57e10db5657905a0adc10c4ee133ae5f4b06c5aa7fa4`.
- `settings.py`, package metadata and compatibility tests were created for this
  process boundary.
- The pinned upstream dependency remains `openflexure-stitching==0.3.1` under
  LGPL-3.0.

The exact symbol classification and upstream source hashes are maintained in
the release-planning document `STITCHING_LICENSE_AUDIT.md`. This file records
engineering provenance, not a legal opinion or copyright assignment.
