# Stage 06 — Evaluation and Release

## Goal

Make a release claim only from frozen, reproducible evidence.

## Required gates

- Global baseline mask mAP does not regress by more than `0.01` from `0.233469`.
- Face, plate, and text candidate recall is at least `95%` in both domains.
- Overall sensitive-pixel recall is at least `95%`.
- Every supported class reaches at least `90%` sensitive-pixel recall.
- Overall sensitive-pixel leakage is at most `5%`.
- Negative-image false-positive rate is at most `15%`.
- Final trained candidates are evaluated over three seeds with confidence intervals.
- Failed or uncertain assurance always blocks automatic export.

## Release artifact

`ConsentGuard v1.0` names the complete reviewed system: code, provider weights,
threshold profile, policy version, dataset manifests, assurance configuration,
test evidence, and model card.  It never names a checkpoint alone.
