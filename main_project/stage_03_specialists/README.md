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

No external image corpus was silently added to training. WIDER FACE and
IIIT-INDIC-HW-WORDS remain candidate sources pending image-level use-rights
records; the Indian LPR source is explicitly not admitted because its project
states that the dataset cannot be publicly released. This keeps the training
manifest honest and prevents an unlicensed dataset from entering the model.

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
