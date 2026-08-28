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
