# Implementation Status

| Stage | State | Exit condition |
|---|---|---|
| 01 — Data | Implemented | V2 has 8,266 validated records, including 726 negative training images. |
| 02 — Baseline | Implemented | `baseline-v0.1` is verified; the negative-inclusive run is ready. |
| 03 — Specialists | In progress | Mask R-CNN evidence adapter is implemented and tested; optional specialist weights and licensed training data remain external gates. |
| 04 — Fusion | Implemented, uncalibrated | Class thresholds, fusion, provenance, and tests exist. |
| 05 — Review/export | Prototype implemented | Manual review, rendering, policy, and assurance are coded. |
| 06 — Release | Blocked on data/training | Frozen target tests, trained specialists, and all gates must pass. |

## Honest model status

The moderate-balanced Mask R-CNN checkpoint is the best current baseline, not
a final model.  No file or UI should label the system production-safe until all
Stage 06 gates pass.
