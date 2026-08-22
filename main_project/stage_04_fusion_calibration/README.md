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

The candidate profile has now been calibrated on all 1,576 V2 validation
images. `threshold_profile_v2_validation_calibrated.yaml` remains
`release_ready: false` because the rare privacy classes do not meet the
configured recall/precision floors, and the separate general/India target
validation sets and independent assurance gates are still outstanding.
