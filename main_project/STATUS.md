# Implementation Status

| Stage | State | Exit condition |
|---|---|---|
| 01 — Data | Implemented | V2 has 8,266 validated records, including 726 negative training images. |
| 02 — Baseline | Implemented | `baseline-v0.1` is verified; the negative-inclusive run is ready. |
| 03 — Specialists | Fine-tuned, experimental | YuNet face, LPD-YuNet plate, PP-OCRv3 ONNX text geometry, zxing-cpp, and one-class Mask R-CNN fine-tunes for face/plate/handwriting are wired into the evidence pipeline. Face validation is promising; plate and handwriting remain experimental because target-domain licensed data is insufficient. |
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

Kaggle execution is prepared but not submitted from this workspace: the
train/validation data packer and checkpointed multi-component runner live in
`scripts/stage_02_baseline_model/`, with the runbook in
`stage_06_evaluation_release/KAGGLE_TRAINING.md`. Kaggle account credentials are
intentionally not stored in the repository.
