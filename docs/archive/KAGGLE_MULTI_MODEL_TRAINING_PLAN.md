# ConsentGuard Kaggle multi-model training plan

## Model/data matrix

| Component | Architecture | Training source | Why it is separate |
|---|---|---|---|
| Global privacy localizer | Mask R-CNN ResNet-50 FPN v2 | Visual Redactions v1 train/validation | Provides privacy-specific instance masks across nine classes |
| Face specialist | Faster R-CNN ResNet-50 FPN v2 | WIDER FACE train/validation | Face boxes, scale, pose, and occlusion diversity |
| Plate specialist | Faster R-CNN ResNet-50 FPN v2 | Indian License Plates with Labels | India-specific plate geometry; provisional web/pseudo-label provenance |
| Handwriting specialist | Mask R-CNN ResNet-50 FPN v2 | HierText handwritten line polygons | Native polygon supervision for handwritten scene text |
| Printed-text safety net | PP-OCRv3 DB detector | Provider checkpoint, geometry only | OCR text is not the same task as privacy-mask segmentation |
| Barcode/QR safety net | ZXing-C++ | Synthetic robustness fixtures | Algorithmic decoder; there is no neural checkpoint to train |

The locked ConsentGuard test split is not copied, converted, tuned, or opened.

## Prepared artifacts

- `main_project/configs/kaggle/dataset_catalog.yaml` records access, license,
  source, split, and production-eligibility decisions.
- `prepare_external_specialist.py` converts WIDER FACE, YOLO plate labels, and
  HierText polygons into deterministic ConsentGuard records.
- `run_kaggle_training.py` prepares the selected specialist and runs one or
  more seeds with atomic manifests and checkpoints.
- `notebooks/kaggle/consentguard_train.py` is the Kaggle GPU entry point.
- `publish_kaggle_assets.py` stages four independent GPU kernels. Each kernel
  receives only the private/public sources required by its component.

## Recommended session sequence

1. Baseline: seed 1337, then 2027 and 31415 after the run is stable.
2. Face Faster R-CNN on WIDER FACE.
3. Plate Faster R-CNN on the India-specific plate dataset.
4. Handwriting Mask R-CNN on HierText polygons.
5. Download every checkpoint, log, resolved config, and manifest.
6. Rebuild fused validation metrics and recalibrate thresholds on validation.
7. Keep the official test split locked until architecture and thresholds are frozen.

Do not attempt all three seeds for every component in one free Kaggle session.
Checkpoint one component per session so quota loss does not erase all progress.

## Submission boundary

The Kaggle CLI is installed locally, but jobs cannot be pushed until the user
places their own API credential at `C:\Users\atnik\.kaggle\kaggle.json` and
provides their Kaggle username to the publisher. The publisher generates the
four account-specific kernel metadata files. Never commit the token.

## UI preview

Install the optional app dependencies and launch the local research UI:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[app]'
$env:PYTHONPATH='C:\consentGuard\main_project\src'
.\.venv\Scripts\python.exe main_project\scripts\stage_05_review_export\run_demo_app.py
```

Open `http://127.0.0.1:7860`. The preview can select model branches and privacy
groups, but its output still requires the manual review and assurance gate.
