# Implementation Status

| Stage | State | Exit condition |
|---|---|---|
| 01 — Data | Implemented | V2 has 8,266 validated records, including 726 negative training images. |
| 02 — Baseline | Implemented | `baseline-v0.1` is verified; the negative-inclusive run is ready. |
| 03 — Specialists | Open providers integrated; target validation pending | YuNet face, LPD-YuNet plate, PP-OCRv3 ONNX text geometry, and zxing-cpp are wired into the evidence pipeline with pinned provenance and explicit unavailable/error handling. LPD-YuNet is trained on Chinese plates; native PaddleOCR remains an isolated experimental runtime and handwriting is unresolved. |
| 04 — Fusion | Validation-calibrated, not release-ready | Per-class Mask R-CNN thresholds were calibrated on all 1,576 V2 validation images; rare classes still miss the recall/precision floors. |
| 05 — Review/export | Integrated, assurance-gated | Consent records, session isolation, evidence registry, end-to-end pipeline, manual review, destructive rendering, policy, and assurance are implemented. |
| 06 — Release | Blocked on data/training | Frozen target tests, trained specialists, and all gates must pass. |

## Honest model status

The moderate-balanced Mask R-CNN checkpoint is the best current baseline, not
a final model.  No file or UI should label the system production-safe until all
Stage 06 gates pass.
