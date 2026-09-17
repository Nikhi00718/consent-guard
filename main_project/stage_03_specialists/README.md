# Stage 03 — Specialist Evidence Providers

## Goal

Use specialist detectors for privacy objects the global model handles poorly.
Every provider produces evidence; no provider decides consent or release.

## Providers

| Provider | Responsibility | V1 behavior |
|---|---|---|
| Mask R-CNN | Broad visual classes and pixel masks | Broad baseline plus one-class face, plate, and handwriting fine-tunes. |
| YuNet | Face localization | OpenCV Zoo MIT weights are checked in; detection only, no identity embeddings. |
| LPD-YuNet | License-plate geometry | OpenCV Zoo Apache-2.0 weights are checked in; trained on Chinese plates, so India validation is still required. |
| Plate Faster R-CNN | General and Indian registration plates | Separate licensed training data required. |
| PP-OCRv3 ONNX | Printed-text geometry | OpenCV Zoo Apache-2.0 detector; recognized text is discarded and handwriting remains unresolved. |
| PaddleOCR | Printed and handwritten text geometry | Optional isolated runtime; native PP-OCRv5 inference is currently blocked on one Windows PaddlePaddle path. |
| zxing-cpp | QR and barcode geometry | Main environment has the pinned optional dependency; missing runtimes remain explicit. |
| Metadata | EXIF and container metadata categories | Values are not copied to ordinary reports. |

The broad Mask R-CNN model is now also exposed through the same evidence
contract. It returns original-image boxes and binary-mask RLE with checkpoint
version provenance; Stage 04 still owns thresholds and release decisions.

## Target-domain fine-tunes

The first trainable specialist pass uses the verified Visual Redactions V2
records already admitted to this repository. `build_specialist_records.py`
creates one-class profiles while retaining negatives; it never reads or writes
the locked test split. The local CUDA runs produce ignored checkpoints at:

```text
artifacts/checkpoints/specialist_face_maskrcnn_5ep/last.pt
artifacts/checkpoints/specialist_plate_maskrcnn_5ep/last.pt
artifacts/checkpoints/specialist_handwriting_maskrcnn_5ep/last.pt
```

The reproducibility record is `reports/specialist_finetune_summary.json`.
The V2 validation pass reached segmentation mAP 0.581 for face on all 1,576
validation images (0.584 on the first 300-image bounded audit), 0.100 for
plate, and 0.053 for handwriting on the bounded audit. Face is a useful
experimental localizer; plate and handwriting are not release-ready and
require additional licensed target-domain examples. The combined pipeline
accepts these optional checkpoints with `--face-checkpoint`, `--plate-checkpoint`, and
`--handwriting-checkpoint`; their evidence is fused under the normal Stage 04
threshold/review gates.

No external image corpus is silently added to training. WIDER FACE and HierText
were used only in isolated research-license Kaggle experiments with frozen
train/validation records; their checkpoints cannot be relabeled for unrestricted
commercial use. IIIT-INDIC-HW-WORDS remains a candidate pending image-level
use-rights records. The earlier Indian LPR source is explicitly not admitted
because its project states that the dataset cannot be publicly released. This
keeps the training manifest honest and prevents an unlicensed dataset from
entering the model.

The later Roboflow `nivu/indian-license-plate-knte7` v1 export is admitted only
as an experimental CC BY 4.0 user-published source. Its original split is not
admitted: the reproducible audit found 17 source/duplicate groups spanning
publisher splits. `prepare_grouped_yolo_plate.py` replaces that split before
`merge_specialist_records.py` builds the no-test full-scene research candidate.
See `reports/PROJECT_AUDIT_AND_EXECUTION_PLAN_2026-08-30.md` for counts, limits,
and the checkpoint promotion rule.

Kaggle version 4 completed the high-resolution full-scene experiment. It
improved merged-validation recall from 0.6832 to 0.7723 and reduced false
positives/image from 0.6557 to 0.1826, but achieved only 0.4118 recall on the
locked Deepak `vid-1` challenge at the precommitted score threshold. The 0.50
promotion requirement therefore failed and the website default was not changed.
See `reports/PLATE_FULL_SCENE_KAGGLE_V4_EVALUATION_2026-08-30.md` and rerun
`verify_plate_full_scene_kaggle_run.py` for the hash-verified decision.

## Code rule

All providers implement the same `EvidenceProvider.analyze()` contract and
return original-image coordinates with provider/version provenance.  Optional
providers must fail visibly; they may never silently return “nothing found”
when their dependency or weight file is missing.

## Checked-in assets and local smoke run

The OpenCV Zoo models and their source/license records live under
`artifacts/specialists/opencv_zoo/`; verify the SHA-256 values in `MANIFEST.json`
before moving them to another machine. The pinned optional Python packages are
listed in `main_project/configs/stage_03_specialists/specialists_requirements_windows.txt`.

From the repository root, the complete local pipeline smoke command is:

```powershell
$env:PYTHONPATH = "C:\consentGuard\main_project\src"
.venv\Scripts\python.exe main_project/scripts/stage_05_review_export/run_analysis_pipeline.py `
  --config main_project/configs/stage_02_baseline_model/train_maskrcnn_moderate_v2_negatives_10ep.yaml `
  --checkpoint artifacts/checkpoints/maskrcnn_moderate_v2_negatives_10ep/last.pt `
  --input data/raw/visual_redactions/images/val2017/2017_10024630.jpg `
  --threshold-profile main_project/configs/stage_04_fusion_calibration/threshold_profile_v2_validation_calibrated.yaml `
  --yunet-model artifacts/specialists/opencv_zoo/face_detection_yunet_2023mar.onnx `
  --plate-yunet-model artifacts/specialists/opencv_zoo/license_plate_detection_lpd_yunet_2023mar.onnx `
  --ppocr-model artifacts/specialists/opencv_zoo/text_detection_en_ppocrv3_2023may.onnx `
  --with-barcode --device cpu `
  --report reports/all_specialists_smoke.json
```

The report must remain `HOLD_FOR_REVIEW` until the Stage 04 profile and Stage 06
release gates are green. No provider is allowed to grant consent by itself.
