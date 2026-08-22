# ConsentGuard validation bundle model card

## Status

This is a validation-only research bundle, not a production release. The
system proposes privacy-sensitive regions for human review; it does not infer
consent, identity, legality, or safe automatic publication.

## Components

- Global Mask R-CNN ResNet-50 FPN v2 trained on the verified Visual Redactions
  V2 negative-inclusive split.
- One-class Mask R-CNN specialists for faces, licence plates, and handwriting.
- Optional YuNet face, LPD-YuNet plate, PP-OCRv3 text geometry, and ZXing-C++
  barcode evidence providers.
- Versioned threshold/fusion, consent, destructive-rendering, and assurance
  code paths.

## Evaluation snapshot

The global V2 validation segmentation mAP is 0.2084. The face specialist
reaches 0.5811 segmentation mAP on all 1,576 V2 validation images. Plate and
handwriting specialists reached 0.0998 and 0.0526 respectively on bounded
300-image audits and remain experimental. The combined evaluator and smoke
reports are in `reports/`.

These numbers do not establish general/India domain recall, low leakage,
acceptable over-redaction, or production safety. The locked test split was not
used.

## Limitations and risks

- No licensed Target-2K general/India pixel-and-annotation release is admitted.
- Three controlled seeds and confidence intervals are not available for the
  current candidate bundle.
- Independent OCR, barcode, face, and plate residual-content attackers are not
  yet implemented as passing assurance checks.
- Small plates, text, handwriting, medicine, signatures, fingerprints, and
  other rare classes can be missed.
- Thresholds are experimental and the profile is explicitly not release-ready.

## Intended use

Local still-image privacy review where a human inspects every proposed region,
approves the mask, and receives an export only after all configured assurance
checks pass.

## Prohibited claims

Do not describe this bundle as guaranteeing privacy, automatically determining
consent, legally compliant, or production-safe.
