# Kaggle Indian plate dataset audit (2026-08-28)

Source: `deepakat002/indian-vehicle-number-plate-yolo-annotation`  
URL: <https://www.kaggle.com/datasets/deepakat002/indian-vehicle-number-plate-yolo-annotation>  
Declared license: **CC0-1.0 (Public Domain)**  
Downloaded archive: about **92.3 MiB** (96,763,237 bytes as reported by Kaggle).

## What was downloaded

| Check | Result |
|---|---:|
| Images | 160 JPG |
| YOLO label files | 160 TXT |
| Classes | 1 (`number_plate`) |
| Annotated boxes | 261 |
| Empty labels | 0 |
| Invalid YOLO rows | 0 |
| Unmatched image/label pairs | 0 |
| Image dimensions | 160 × 1920×1080 |
| Duplicate image hashes | 0 |

Per source video folder:

| Folder | Images | Boxes |
|---|---:|---:|
| `vid-1` | 43 | 68 |
| `vid-2` | 67 | 109 |
| `vid-3` | 50 | 84 |

## Decision

The archive is structurally valid and safe to keep as an external training candidate. It is much smaller than the 1,650-image Roboflow NIVU export and consists of frames from only three videos. Neighboring frames can be near-duplicates, so a random image split would leak scene appearance between train and validation. For a trustworthy experiment, split by `vid-*` folder (or use this dataset only as an additional fine-tuning source) and keep the already-trained checkpoint as the production baseline until the grouped evaluation improves it.

Raw images remain under `data/raw/` and are intentionally ignored by Git; this report and any generated manifests are the reproducible artifacts that belong in the repository.

The preparation script now supports `--group-by-parent`. For this download the reproducible grouped records are **117 train images / 193 boxes** (folders `vid-2` and `vid-3`) and **43 validation images / 68 boxes** (folder `vid-1`). The grouped records are stored locally under `data/processed/external/deepakat_indian_vehicle_number_plate_yolo_grouped/` and are also ignored by Git.

## Existing checkpoint sanity test

The current production checkpoint (`specialist_plate_ccpd2020_india_finetune_5ep/best.pt`) was evaluated on the held-out `vid-1` images at IoU 0.50. At confidence 0.50 it produced **1 true positive, 44 false positives, and 67 missed boxes** (precision 2.2%, recall 1.5%). This is not a defect in the audit: the frames contain very small plates in wide 1920×1080 road scenes, which are a clear domain/scale shift from the current training data. The new dataset is therefore useful as a targeted tiny-plate fine-tuning experiment, but this checkpoint should remain the website default until a retrained model beats its existing validation metrics.
