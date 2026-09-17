# Online Indian plate dataset audit — 2026-08-28

## Decision

The public Hugging Face parquet was downloaded for inspection, but it is **not
safe to use as a detector-training dataset**. It contains 1,709 embedded plate
crops while every bounding box is expressed in the original-scene coordinate
system. All 1,709 boxes fall outside the corresponding embedded image. The
dataset card also does not declare a license, so it is not added to training or
GitHub.

## Download and QA result

| Field | Result |
|---|---:|
| Source | `zenitsu09/indian-number-plate` |
| File | `data/raw/online_hf/zenitsu09_indian_number_plate/train.parquet` |
| Compressed download | 5,924,512 bytes (5.65 MiB) |
| Embedded image bytes | 6,375,742 bytes (6.08 MiB) |
| Rows | 1,709 |
| Unique original filenames | 1,709 |
| Unique plate strings | 965 |
| Sources | `plates`: 38, `plates1`: 654, `plates2`: 417, `plates3`: 600 |
| State field unknown | 1,109 rows |
| Invalid boxes | 0 |
| Boxes outside embedded image | 1,709 / 1,709 (100%) |
| License | Not declared on the dataset card |

The reproducible audit is `main_project/scripts/stage_03_specialists/audit_online_plate_parquet.py` and its machine-readable output is `artifacts/evaluations/online_hf_zenitsu_plate_audit_2026-08-28.json`.

## Other sources checked

* `thundarstrom/indian-license-plate-detection` advertises 3,742 CC BY 4.0
  images, but its current Hub repository contains only a README and
  `data.yaml` (about 2.4 kB), not the claimed image/label files. It cannot be
  downloaded from that repository as of this audit.
* Roboflow's NIVU Indian License Plate project lists 1,650 images, one class,
  and CC BY 4.0. Its export requires the Roboflow dataset-download flow; no
  export token was available in this environment, so it was not copied into
  the repository.
* The previously downloaded Kaggle set remains the only local Indian
  full-image labeled set used for the current fine-tune. Its Kaggle page does
  not provide a clear reusable license, so keep it for research evaluation
  until provenance is confirmed.

## Effect on ConsentGuard

No model weights or website behavior were changed from this audit dataset.
The Indian fine-tuned Faster R-CNN remains the configured plate provider:

* Held-out Indian validation: bbox mAP 0.773598, AP50 0.973758, AP75 0.945667.
* The website defaults now point to
  `artifacts/checkpoints/specialist_plate_ccpd2020_india_finetune_5ep/best.pt`.
* The local FastAPI/React path was exercised with a real Indian image:
  upload succeeded, `plate-trained` analysis returned 19 raw evidence items
  and one license-plate candidate, with no provider errors.

The correct next data action is to obtain a rights-cleared full-scene export
(for example, the CC BY 4.0 Roboflow source after export access is granted),
then run the same coordinate/duplicate/license audit before training.
