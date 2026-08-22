# Stage 03 — Specialist Evidence Providers

## Goal

Use specialist detectors for privacy objects the global model handles poorly.
Every provider produces evidence; no provider decides consent or release.

## Providers

| Provider | Responsibility | V1 behavior |
|---|---|---|
| Mask R-CNN | Broad visual classes and pixel masks | Uses the frozen baseline initially. |
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
