# Stage 04 — Fusion and Calibration

## Goal

Turn independent provider output into review candidates while retaining the
source of every detection.

## Required behavior

- Load a versioned threshold profile.
- Apply thresholds by provider and privacy class, never one global threshold.
- Expand and dilate safety-critical geometry according to the profile.
- Merge overlapping evidence into candidates without losing provenance.
- Mark low-confidence, conflicting, experimental, or unavailable-provider
  cases as mandatory review.
- Record every manual threshold override in the output audit report.

The first checked-in profile is a candidate profile.  It cannot be marked
`release_ready` until the frozen general and India validation sets pass.
