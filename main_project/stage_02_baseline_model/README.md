# Stage 02 — Baseline Model

## Goal

Keep one reproducible comparison point while refusing to call it the final
privacy system.

## Current result

- Model: Mask R-CNN ResNet-50 FPN v2.
- Checkpoint: moderate class balancing, best epoch 5.
- Validation mask mAP: `0.233469`.
- Validation mask AP50: `0.375473`.
- Sensitive-pixel recall at score `0.5`: approximately `0.7639`.

## Canonical files

- [`configs/stage_02_baseline_model/train_maskrcnn_verified_moderate_balance_10ep.yaml`](../configs/stage_02_baseline_model/train_maskrcnn_verified_moderate_balance_10ep.yaml)
- [`src/consentguard/stage_02_baseline_model/models.py`](../src/consentguard/stage_02_baseline_model/models.py)
- [`src/consentguard/stage_02_baseline_model/data_loading.py`](../src/consentguard/stage_02_baseline_model/data_loading.py)
- [`src/consentguard/stage_02_baseline_model/optimization.py`](../src/consentguard/stage_02_baseline_model/optimization.py)
- [`src/consentguard/stage_02_baseline_model/training_loop.py`](../src/consentguard/stage_02_baseline_model/training_loop.py)

## Decisions already made

- Object-centred crops and class-agnostic masks are retained as negative
  experiments, not promoted to the main model.
- The Visual Redactions test split remains locked.
- Further training requires a written failure hypothesis.
