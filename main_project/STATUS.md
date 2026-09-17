# Implementation Status

| Stage | State | Exit condition |
|---|---|---|
| 01 — Data | Implemented | V2 has 8,266 validated records, including 726 negative training images. |
| 02 — Baseline | Implemented | `baseline-v0.1` is verified; the negative-inclusive run is ready. |
| 03 — Specialists | Fine-tuned, experimental | YuNet face, LPD-YuNet plate, PP-OCRv3 ONNX text geometry, zxing-cpp, and one-class learned specialists are wired into the evidence pipeline. The grouped, no-test full-scene plate candidate completed on Kaggle and substantially improved frozen metrics, but missed the Deepak recall promotion gate (0.4118 versus 0.50), so the current website checkpoint remains unchanged. Face validation is promising; plate and handwriting remain experimental. |
| 04 — Fusion | Validation-calibrated, not release-ready | Per-class Mask R-CNN thresholds were calibrated on all 1,576 V2 validation images; rare classes still miss the recall/precision floors. |
| 05 — Review/export | Integrated, assurance-gated | Consent records, session isolation, evidence registry, end-to-end pipeline, manual review, destructive rendering, policy, and assurance are implemented. |
| 06 — Release | Validation bundle complete; release blocked | Strict `release-metrics-v1` gates, model card, data sheet, checkpoint hashes, and a fail-closed bundle manifest are implemented. Target-2K domain evidence, three seeds, fused leakage/FPR confidence bounds, and independent residual-content attacks are still required. |

## Honest model status

The moderate-balanced Mask R-CNN checkpoint is the best current baseline, not
a final model.  No file or UI should label the system production-safe until all
Stage 06 gates pass.

The current validation-only bundle is recorded in
`reports/release_bundle_manifest.json`; the model card and data sheet live in
`stage_06_evaluation_release/`.

Earlier face, plate, handwriting, baseline, CCPD/India, and grouped full-scene
Kaggle experiments are complete. The full-scene plate result and failed
promotion decision are recorded in
`reports/PLATE_FULL_SCENE_KAGGLE_V4_EVALUATION_2026-08-30.md`. Kaggle account
credentials remain outside the repository.
