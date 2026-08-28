# CCPD2020 plate model: Indian transfer evaluation

**Date:** 2026-08-28
**Decision:** keep the CCPD2020 model as a research candidate and optional proposal source; do **not** make it the sole production plate detector yet.

## Executive result

The new checkpoint is healthy on its own CCPD2020 validation protocol, but its transfer behaviour on the local Indian INDO-ALPR originals is weak and poorly calibrated. On the same 2,000 Indian images, only **67 images (3.35%)** reached a top detection score of at least 0.50, and the median top score was **0.00032**. The existing Indian Mask R-CNN baseline reached **138 images (6.90%)** at 0.50 and had a median top score of **0.02566**.

This is not an Indian accuracy result: the transfer set has no localization ground truth, and most files are already plate crops rather than phone/vehicle scenes. The measured result says the CCPD model is usable for continued experimentation and as a checkpoint to fine-tune, but it does **not** justify replacing the current model or releasing it as India-ready.

## What was trained

| Model | Training data | Architecture and setup | Training result | Local checkpoint |
|---|---|---|---|---|
| **New CCPD specialist** | Official CCPD2020: 5,769 train records and 1,001 validation records; train/val only, no CCPD test claim | One-class `fasterrcnn_resnet50_fpn_v2`, COCO initialization, small-object anchors, 512/768 resize, crop augmentation, 12 epochs, batch 1, gradient accumulation 4 | Final box mAP **0.8768816**; best box mAP **0.8879097**; 17,316 optimizer steps; successful return code 0 | `artifacts/checkpoints/specialist_plate_ccpd2020_fasterrcnn/best.pt` (345,600,239 bytes; SHA-256 `ff0916d64c9b9130cf82fe63a97d0b7086413b521989fea41ad050a98ea8c72e`) |
| **Existing local baseline** | Indian plate records: 633 train and 1,576 validation records in `data/processed/specialists/plate` | One-class `maskrcnn_resnet50_fpn_v2`, class-agnostic masks, 512/768 resize, 5 epochs | Used as a same-image transfer control here; its previously documented `0.7702` plate mAP belongs to a separate earlier Indian Faster R-CNN run and must not be conflated with this Mask R-CNN checkpoint | `artifacts/checkpoints/specialist_plate_maskrcnn_5ep/last.pt` (366,614,551 bytes; SHA-256 `1c1e74aeafc7130ab7b34739f7c96bfbdb531d3e55ea2a27155d235759271246`) |

The CCPD training source was downloaded from the official Zenodo archive, verified by the recorded official MD5, converted into the project’s one-class records, and trained in the dedicated Kaggle job. Only the selected checkpoint and metadata were downloaded to the laptop; the full image archive is not required for this transfer test.

## Indian/phone-image test set

The test used `data/raw/indo_alpr/original`, with the canonical originals already SHA-256 verified:

| Property | Value |
|---|---:|
| Total images | **2,000** |
| `train` folder | 1,000 |
| `test` folder | 1,000 |
| Local image payload | about **197.51 MiB** |
| Scene-like (`width >= 640` and `height >= 360`) | **54** |
| Crop-like (`width / height >= 2.0`) | **1,734** |
| Localization boxes/IoU labels | **Not available** |

These are Indian license-plate images, but they are not a representative phone-camera/vehicle benchmark. A crop-like image can tell us whether the model reacts to Indian plate appearance, but it cannot tell us whether the detector found the correct plate in a full scene. The 54 scene-like files are a useful sanity slice, not enough for a release claim.

## Same-image transfer results

The evaluator ran both checkpoints over all 2,000 files, one model at a time on the laptop RTX 3050. “Any raw detection” means a model emitted at least one box at score greater than zero; it is intentionally **not** treated as success because both detectors emit low-score proposals.

| Model | Valid images | Any raw box | Top score >= 0.50 | Top score >= 0.70 | Top score >= 0.90 | Median top score | Median top-box area |
|---|---:|---:|---:|---:|---:|---:|---:|
| CCPD Faster R-CNN | 2,000 | 2,000 (100%) | **67 (3.35%)** | **48 (2.40%)** | **22 (1.10%)** | **0.00032** | 7.26% of image |
| Indian Mask R-CNN baseline | 2,000 | 2,000 (100%) | **138 (6.90%)** | **68 (3.40%)** | **23 (1.15%)** | **0.02566** | 4.59% of image |

Useful slices:

- On the 54 scene-like images, the CCPD model’s median top score was about **0.00034** and only **1/54** reached 0.50. The baseline reached **2/54** at 0.50.
- On the 1,734 crop-like images, the CCPD model reached 0.50 on **67** images; the baseline reached **136**.
- The CCPD model’s 0.90 count is essentially tied with the baseline (22 vs 23), but this is still only 1% of the full set and is not precision/recall.

The complete per-image output is retained in `artifacts/evaluations/plate_indian_transfer_2026-08-28.json`. It contains the split, shape category, detection count, maximum confidence, threshold flags, top-box area ratio, errors, and checkpoint hashes for every image. Re-run it with:

```powershell
.venv\Scripts\python.exe main_project\scripts\stage_03_specialists\evaluate_plate_indian_transfer.py --device cuda
```

## Can we use this model?

**Yes, but only in a limited role now.** The CCPD model is useful as:

1. a large, officially sourced plate-localization pretraining checkpoint;
2. a candidate to fine-tune on rights-controlled Indian vehicle/phone scenes;
3. an experimental second proposal source whose output can be compared with the existing detector and LPD-YuNet.

**No, not yet as the only release detector.** The transfer test shows severe confidence drop on Indian images, and the data has no ground truth. We cannot honestly claim Indian recall, precision, AP50/AP75, false-positive rate, or phone-camera robustness from this run.

## Why this model and why YuNet remains separate

- **Faster R-CNN plate specialist:** trained because plate localization is the highest-value learned specialist and CCPD2020 supplies far more verified plate boxes than the old small Indian training run. It predicts a box that can be passed to the review/fusion stage.
- **LPD-YuNet:** kept as a lightweight, independent proposal/safety-net provider. Its released weights were trained on Chinese plates and are not India-validated, so we do not claim it is better; agreement/disagreement with the trained specialist is valuable for recall and review. We are not retraining YuNet in this repository.
- **Fusion:** no provider threshold was changed based on this unlabelled test. Thresholds and mandatory review must be calibrated on labeled Indian validation data.

## What must happen before promotion

1. Collect or obtain a rights-controlled Indian **phone/vehicle** set with boxes: minimum target is a frozen train/validation/test split with plates at different distances, angles, blur, night, glare, occlusion, and vehicles without visible plates.
2. Keep the test split untouched and measure box AP50/AP75, precision, recall, F1, false positives per image, and small-object performance for CCPD, the current baseline, and the fused specialist + LPD-YuNet path.
3. Fine-tune the CCPD checkpoint on the Indian training split, including hard negatives. Do not train directly on the current INDO crops without labels; they are not localization supervision.
4. Recalibrate score thresholds and box expansion on the Indian validation split, then repeat the frozen test once.
5. Keep the old checkpoint and LPD-YuNet as fallbacks until the new model wins the India test at the project’s agreed safety operating point.
6. Record the final checkpoint hash, dataset license/provenance, and training configuration. Do not commit the 345 MB checkpoint or raw datasets to Git; Git contains the evaluator, configuration, report, and small metrics JSON only.

## Files committed for this result

- `main_project/scripts/stage_03_specialists/evaluate_plate_indian_transfer.py` — reproducible two-checkpoint evaluator.
- `main_project/configs/stage_03_specialists/train_plate_ccpd2020_fasterrcnn.yaml` — CCPD model configuration.
- `reports/PLATE_CCPD2020_INDIAN_TRANSFER_EVALUATION_2026-08-28.md` — this decision record.
- `artifacts/evaluations/plate_indian_transfer_2026-08-28.json` — full 2,000-image per-image metrics and hashes.
