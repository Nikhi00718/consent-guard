# From Detection to Safe Release

ConsentGuard is the working folder for a research prototype that localizes
privacy-sensitive visual evidence and produces destructive, newly encoded
redactions. The verified training baseline is Mask R-CNN ResNet-50 FPN v2 for
the nine official **visual** Visual Redactions attributes plus background.
Textual and multimodal attributes require separate OCR/document branches.

The executable codebase is organized for stage-by-stage review under
[`main_project/`](main_project/README.md). Source modules, scripts, configs,
and tests are physically grouped into six numbered stages there.

This repository does **not** infer consent, intent, legality, or identity from
pixels. The current milestone is the perception/localization model required by
the broader consent-state-aware release policy in
`ConsentGuard_Final_Research_Design.md`.

## Architecture

- Primary reproducible baseline: TorchVision `maskrcnn_resnet50_fpn_v2`.
- Local 4 GB profile: 640/1024 aspect-preserving images, small-object anchors,
  batch size 1, gradient accumulation, AMP, and reduced RPN proposals.
- Controlled 12–16 GB profile: 800/1333 full images and standard anchors.
- Optional comparison after the baseline: Mask2Former on a larger GPU.
- Engineering evaluation: pycocotools box and mask AP@[.50:.95]/AR, with
  compressed-RLE accumulation and object-size breakdowns. The paper's official
  thresholded pixel precision-recall AP is a separate metric and must be used
  for direct comparison with published results.

The loader transforms polygons before rasterization, so large source images do
not allocate one full-resolution bitmap per annotation.

## Environment (Windows)

When restoring the project from GitHub, use Git LFS and recursive submodules so
the versioned best checkpoint and optional third-party reference code are
available:

```powershell
git lfs install
git clone --recurse-submodules https://github.com/Nikhi00718/consent-guard.git
git lfs pull
```

Use Python 3.11. The setup script installs a matched CUDA 12.6 PyTorch pair,
all evaluation/test dependencies, the editable package, and runs a preflight.
Large wheels and the official COCO Mask R-CNN initialization are downloaded
resumably into `data/cache`, with size/hash checks where the publisher provides
them.

```powershell
Set-Location C:\consentGuard
powershell -ExecutionPolicy Bypass -File main_project\scripts\stage_02_baseline_model\setup_environment.ps1
```

CPU-only setup is available for data/test development:

```powershell
powershell -ExecutionPolicy Bypass -File main_project\scripts\stage_02_baseline_model\setup_environment.ps1 -CpuOnly
```

Official references: [PyTorch installation](https://pytorch.org/get-started/locally/),
[TorchVision detection tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html),
and [Mask R-CNN v2](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html).

## Data lifecycle

Raw archives are never modified. A split is usable only after exact-size and
full gzip/tar validation, safe extraction, manifest rebuilding, preprocessing,
geometry verification, modality filtering, and record validation. A dimension
mismatch is accepted only when annotation and decoded image aspect ratios agree
within 1%; crops, stitches, and rotations are quarantined.

```powershell
.\.venv\Scripts\python.exe main_project\scripts\stage_01_data\finalize_vispr_data.py --split val --extract --rebuild-records
.\.venv\Scripts\python.exe main_project\scripts\stage_01_data\audit_visual_redactions_alignment.py
.\.venv\Scripts\python.exe main_project\scripts\stage_01_data\preprocess_visual_redactions_verified.py --profile visual
.\.venv\Scripts\python.exe main_project\scripts\stage_01_data\validate_processed_records.py `
  --data data\processed\visual_redactions_verified_visual `
  --report reports\processed_records_verified_visual_validation.json
.\.venv\Scripts\python.exe main_project\scripts\stage_01_data\audit_split_leakage.py
```

`data/processed/visual_redactions/` and the v1/v2 configs are retained only to
reproduce the failed legacy runs. Do not use them for new training.

Never tune on `records_test2017.jsonl`; the official Visual Redactions test
split remains locked until the final experiment.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main_project\scripts\stage_02_baseline_model\preflight_environment.py
.\.venv\Scripts\python.exe main_project\scripts\stage_02_baseline_model\train_maskrcnn.py --config main_project\configs\stage_02_baseline_model\train_smoke.yaml
```

The smoke run performs a real Mask R-CNN forward pass, backward pass, optimizer
step, validation inference, COCO metric update, and atomic checkpoint write on
real processed VISPR data.

## Train

Laptop RTX 3050 (4 GB):

```powershell
.\main_project\scripts\stage_02_baseline_model\start_maskrcnn_verified_visual.ps1
```

Controlled 12–16 GB GPU baseline:

```powershell
.\.venv\Scripts\python.exe main_project\scripts\stage_02_baseline_model\train_maskrcnn.py --config main_project\configs\stage_02_baseline_model\train_maskrcnn_baseline.yaml
```

Resume without losing optimizer, scheduler, scaler, epoch, loader/sampler RNG,
or CUDA RNG state:

```powershell
.\.venv\Scripts\python.exe main_project\scripts\stage_02_baseline_model\train_maskrcnn.py --config main_project\configs\stage_02_baseline_model\train_maskrcnn_4gb.yaml --resume artifacts\checkpoints\maskrcnn_4gb\last.pt
```

Each run writes the resolved configuration, environment details, JSONL metrics,
TensorBoard events, `last.pt`, per-epoch checkpoints, and `best.pt` selected by
segmentation mAP. Immutable checkpoint aliases use NTFS hard links when
available, avoiding repeated copies of the same large checkpoint.

## Evaluate and redact

```powershell
.\.venv\Scripts\python.exe main_project\scripts\stage_02_baseline_model\evaluate_maskrcnn.py `
  --config main_project\configs\stage_02_baseline_model\train_maskrcnn_4gb.yaml `
  --checkpoint artifacts\checkpoints\maskrcnn_4gb\best.pt

.\.venv\Scripts\python.exe main_project\scripts\stage_05_review_export\infer_maskrcnn.py `
  --config main_project\configs\stage_02_baseline_model\train_maskrcnn_4gb.yaml `
  --checkpoint artifacts\checkpoints\maskrcnn_4gb\best.pt `
  --input path\to\input.jpg `
  --output outputs\redacted\result.jpg
```

Inference unions accepted masks, dilates boundaries, applies solid replacement,
encodes a fresh JPEG/PNG/WebP without copying source metadata, reopens the
result, and writes a sidecar audit report containing hashes and geometry only.

## Main files

- `configs/train_maskrcnn_verified_visual.yaml` - corrected local profile.
- `configs/train_maskrcnn_verified_overfit.yaml` - mandatory learning gate.
- `scripts/preprocess_visual_redactions_verified.py` - geometry-safe visual records.
- `docs/V3_PREPROCESSING_FIX_AND_TRAINING_REPORT.md` - root cause and fix evidence.

- `configs/train_maskrcnn_4gb.yaml` — runnable local profile.
- `configs/train_maskrcnn_baseline.yaml` — scientific baseline.
- `configs/train_smoke.yaml` — one-step end-to-end verification.
- `src/consentguard/perception/dataset.py` — polygon-aware data pipeline.
- `src/consentguard/perception/trainer.py` — AMP/checkpoint/resume loop.
- `docs/TRAINING_ARCHITECTURE.md` — architecture rationale and constraints.
- `TRAINING_SETUP_REPORT.md` — measured readiness evidence and final commands.
- `ConsentGuard_Final_Research_Design.md` — complete research and safety plan.

Dataset media, model checkpoints, and generated outputs are intentionally
ignored by version control and must not be redistributed without their original
licenses and research-use terms.
