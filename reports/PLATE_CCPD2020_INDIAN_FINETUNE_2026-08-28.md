# CCPD2020 to Indian plate fine-tune

**Date:** 2026-08-28
**Decision:** keep this checkpoint as the current Indian plate research candidate; do not promote it as a production phone/vehicle detector until a scene-level, labeled Indian test is available.

## Result in one paragraph

The CCPD2020 plate detector was fine-tuned for five epochs on the downloaded labeled Indian license-plate dataset. On its held-out Indian validation split it reached box mAP **0.773598**, AP50 **0.973758**, AP75 **0.945667**, and box mAR@100 **0.819545**. On the same 2,000-image INDO transfer sanity set, the fine-tuned model produced a score of at least 0.50 on **1,827 images (91.35%)**, compared with **67 images (3.35%)** for the original CCPD checkpoint. This is a strong confidence improvement, but it is not a measured precision/recall result because the INDO files have no localization ground truth and are mostly plate crops.

## Dataset used

Source: Kaggle dataset `kedarsai/indian-license-plates-with-labels`.

| Item | Count / size |
|---|---:|
| Download archive | about **62.8 MB** |
| Extracted raw files | 2,083 images + 2,021 YOLO label files; **64.96 MiB** |
| Converted training records | **1,617 images / 1,728 boxes** |
| Converted validation records | **404 images / 440 boxes** |
| Negative images | **0** |
| Converted metadata | 4 files / **1.80 MiB** |

The converter uses the dataset's YOLO boxes and creates the project's one-class `a108_license_plate_all` records. The train/validation split is deterministic with seed 1337. The Kaggle mirror's declared license/provenance still needs independent review before a public release; this run does not claim ownership of those images.

## Fine-tuning setup

| Setting | Value |
|---|---|
| Initialization | CCPD2020 Faster R-CNN checkpoint, model state only |
| Architecture | `fasterrcnn_resnet50_fpn_v2`, one foreground class |
| Resize | short side 512, max long side 768 |
| Augmentation | 50% context crop, 20% brightness/contrast |
| Optimizer | SGD, learning rate 0.0005, momentum 0.9, weight decay 0.0005 |
| Batch / accumulation | batch 1 / gradient accumulation 4 |
| Epochs / optimizer steps | 5 epochs / **2,025 steps** |
| Runtime | Kaggle Tesla P100 16 GB, PyTorch 2.7.1 + CUDA 11.8 |
| State reset | optimizer, scheduler, and epoch counters reset |

The script constructs the architecture without a network download, then strictly loads the complete CCPD state. This is why the run works on the Kaggle P100 and remains reproducible offline after the runtime is installed.

## Indian validation metrics

The 404-record validation split is labeled and can support mAP/mAR. Final and best values are the epoch-5 evaluation:

| Metric | Value |
|---|---:|
| Box mAP (primary) | **0.773598** |
| Box AP50 | **0.973758** |
| Box AP75 | **0.945667** |
| Box mAR@1 | 0.758864 |
| Box mAR@10 | 0.819318 |
| Box mAR@100 | **0.819545** |
| Medium-object mAP | 0.321279 |
| Small-object mAP | 0.0 |

The zero small-object value is a warning: this dataset does not establish performance on tiny plates in full scenes.

## 2,000-image transfer sanity check

Input: `data/raw/indo_alpr/original` (1,000 train-folder files + 1,000 test-folder files). It contains **1,734 crop-like images** and only **54 scene-like images**. There are no localization annotations, so these numbers describe score stability, not detector accuracy.

| Model | Any raw box | Score >= 0.50 | Score >= 0.70 | Score >= 0.90 | Median top score | Median top-box area |
|---|---:|---:|---:|---:|---:|---:|
| **India-fine-tuned CCPD** | 2,000 (100%) | **1,827 (91.35%)** | **1,686 (84.30%)** | **1,393 (69.65%)** | **0.96946** | 83.18% of image |
| Original CCPD checkpoint | 2,000 (100%) | 67 (3.35%) | 48 (2.40%) | 22 (1.10%) | 0.00032 | 7.26% of image |

On the 54 scene-like files, the fine-tuned model reached 0.50 on **51/54**, 0.70 on **51/54**, and 0.90 on **44/54**. The original CCPD model reached 0.50 on **1/54** and never reached 0.70 or 0.90. Because most INDO files are already crops and the fine-tuned model often proposes a very large box, this result must not be interpreted as phone-camera vehicle recall.

## Checkpoint and reproducibility

| Artifact | Path / identifier |
|---|---|
| Fine-tune configuration | `main_project/configs/stage_03_specialists/train_plate_ccpd2020_india_finetune_5ep.yaml` |
| Fine-tune script | `main_project/scripts/stage_03_specialists/fine_tune_plate_from_checkpoint.py` |
| Transfer evaluator | `main_project/scripts/stage_03_specialists/evaluate_plate_indian_transfer.py` |
| Fine-tuned checkpoint | `artifacts/checkpoints/specialist_plate_ccpd2020_india_finetune_5ep/best.pt` |
| Checkpoint size | **346,511,469 bytes (330.5 MiB)** |
| Fine-tuned SHA-256 | `d8a7a551fe3a9f264bb1ad34066f583d92ae9344503112a582554e29975a81b8` |
| Source CCPD SHA-256 | `ff0916d64c9b9130cf82fe63a97d0b7086413b521989fea41ad050a98ea8c72e` |
| Full transfer JSON | `artifacts/evaluations/plate_indian_transfer_finetuned_2026-08-28.json` |
| Training result | `artifacts/checkpoints/specialist_plate_ccpd2020_india_finetune_5ep/training_result.json` |
| Kaggle job | [ConsentGuard CCPD to Indian plate fine-tune](https://www.kaggle.com/code/nikhil00718/consentguard-ccpd-to-indian-plate-fine-tune) |

The checkpoint and raw datasets remain outside Git. Git stores the code, configuration, and report; the private Kaggle dataset stores the 330 MB initialization checkpoint and the code bundle.

## How it fits the other models

- **Fine-tuned Faster R-CNN:** the primary learned plate proposal source for Indian data. It is now useful for experiments and labeled validation, but needs a scene-level test before release.
- **Original CCPD Faster R-CNN:** retained as a reproducible source checkpoint and fallback comparison; it is not India-calibrated.
- **LPD-YuNet:** remains a lightweight independent proposal/safety-net provider. It is not retrained here; keeping it separate makes disagreement visible for review and fusion.
- **Fusion/review:** no production threshold was changed from this unlabelled sanity set. Thresholds must be calibrated on a frozen Indian validation split with negatives and then checked on an untouched test split.

## Next action

Obtain or create a rights-controlled Indian phone/vehicle dataset with boxes and hard negatives (different distances, angles, blur, night, glare, occlusion, and vehicles without visible plates). Evaluate this checkpoint, the existing detector, and LPD-YuNet on AP50/AP75, precision, recall, F1, false positives per image, and small-object recall. Promote the fine-tuned checkpoint only if it wins that frozen test at the agreed safety operating point.
