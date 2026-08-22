# Stage 03 — Specialist Evidence Providers

## Goal

Use specialist detectors for privacy objects the global model handles poorly.
Every provider produces evidence; no provider decides consent or release.

## Providers

| Provider | Responsibility | V1 behavior |
|---|---|---|
| Mask R-CNN | Broad visual classes and pixel masks | Uses the frozen baseline initially. |
| YuNet | Face localization | Detection only; no identity embeddings. |
| Plate Faster R-CNN | General and Indian registration plates | Separate licensed training data required. |
| PaddleOCR | Printed and handwritten text geometry | Text remains ephemeral. |
| zxing-cpp | QR and barcode geometry | Optional dependency, explicit unavailable state. |
| Metadata | EXIF and container metadata categories | Values are not copied to ordinary reports. |

The broad Mask R-CNN model is now also exposed through the same evidence
contract. It returns original-image boxes and binary-mask RLE with checkpoint
version provenance; Stage 04 still owns thresholds and release decisions.

## Code rule

All providers implement the same `EvidenceProvider.analyze()` contract and
return original-image coordinates with provider/version provenance.  Optional
providers must fail visibly; they may never silently return “nothing found”
when their dependency or weight file is missing.
