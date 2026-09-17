# ConsentGuard Kaggle multi-model readiness

Generated: 2026-08-25

## Status

- Kaggle upload/training: **ready but not submitted**. The machine has no
  `C:\Users\atnik\.kaggle\kaggle.json` credential.
- Private baseline data: **materialized and hash-verified** — 5,361 images,
  10,885,100,113 bytes (10.138 GiB), zero hash mismatches, locked test excluded.
- Training code bundle: **verified** — 76 files, SHA-256
  `7b217b6be9293ed772df6ec22461f68c6dcb887c718509462e6313d6fa2199c6`.
- Kaggle jobs: **four independent GPU kernels staged** — baseline, face, plate,
  and handwriting.
- Portable verification: **58 tests passed**. The six new specialist converter
  and detector metric tests also pass.
- Native Windows verification: **blocked by host policy**. Windows Application
  Control blocks TorchVision `_C.pyd`, causing two model-construction tests to
  fail because `torchvision::nms` is unavailable. Kaggle Linux is the target
  native-runtime check.
- Local Gradio UI: **implemented and syntax-checked**, but browser smoke testing
  is blocked by the same Windows policy (pandas/TorchVision native DLLs).

## Training sources and architectures

| Component | Architecture | Source | Current execution state |
|---|---|---|---|
| Global privacy | Mask R-CNN ResNet-50 FPN v2 | Visual Redactions train/validation | Kaggle data staged |
| Face | Faster R-CNN ResNet-50 FPN v2 | WIDER FACE train/validation | Converter/config/kernel staged |
| Plate | Faster R-CNN ResNet-50 FPN v2 | Indian License Plates with Labels | Converter/config/kernel staged |
| Handwriting | Mask R-CNN ResNet-50 FPN v2 | HierText handwritten polygons | Official annotations validated; downloader/config/kernel staged |
| Printed text | PP-OCR detector | Provider checkpoint | Integrated utility; no training in this stack |
| Barcode/QR | ZXing-C++ | Synthetic robustness fixtures | Integrated algorithm; no neural training required |

No new Kaggle-trained checkpoint or accuracy number may be claimed until the
four remote jobs finish and their manifests/checkpoints are downloaded and
validated. Production release remains blocked by licensing and target-set
assurance requirements.

## Required user-owned credential

Download a Kaggle API token from Kaggle account settings, save it privately as
`C:\Users\atnik\.kaggle\kaggle.json`, and provide only the Kaggle username.
Never paste or commit the token.
