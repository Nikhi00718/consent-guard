# ConsentGuard

A local tool that erases private content from a photo before you share it.

Drop in a photo or screenshot. It finds faces, number plates, text, handwriting,
QR codes and barcodes, covers them with solid black, strips the GPS/camera data,
and gives you a clean copy. Nothing leaves your machine.

**It will miss things.** Detection is automatic but never complete, so look at
the result before you share it, and fix anything it missed with the brush. The
app labels the content types it is bad at instead of pretending otherwise.

## Run it

```powershell
powershell -ExecutionPolicy Bypass -File main_project\scripts\stage_05_review_export\start_consentguard.ps1
```

That loads the detectors (about a minute) and opens `http://127.0.0.1:7860`.
For a desktop shortcut, run
`main_project\scripts\stage_05_review_export\create_desktop_shortcut.ps1` once.

First-time setup, if the Python environment does not exist yet:

```powershell
powershell -ExecutionPolicy Bypass -File main_project\scripts\stage_02_baseline_model\setup_environment.ps1
```

## How it works

```text
photo  ->  normalize (fix orientation, drop metadata, isolated session)
       ->  detectors: 9-class Mask R-CNN, face, plate, text, QR/barcode, EXIF
       ->  second pass over overlapping tiles, so small regions are found
       ->  fuse: per-class thresholds, area sanity caps, merge overlaps
       ->  solid black fill of the mask shape, grown a few pixels
       ->  encode a fresh file, re-open it, re-scan it for leftovers
       ->  download (same format as the input)
```

You either press **Erase and save** and get the file, or open the review screen
first to paint over misses and un-erase anything covered by mistake. Working
copies are deleted when you close the page; your original is never modified.

## What it catches

| Content | How well it works |
|---|---|
| Faces | Good |
| Whole people | Good, and switched **off** by default because it erases most of a photo |
| QR codes and barcodes | Reliable (a decoder, not a model) |
| GPS/camera metadata | Always removed; the output is written from scratch |
| Printed text | Decent, and it erases signs and logos too |
| Number plates | Catches many, misses small and distant ones |
| Handwriting, signatures, fingerprints, medicine, documents | Weak — the app says "check manually" |

Measured evidence lives in `reports/`. Reproduce the per-class numbers:

```powershell
.\.venv\Scripts\python.exe main_project\scripts\stage_06_evaluation_release\evaluate_fused_validation.py `
  --config main_project\configs\stage_02_baseline_model\train_maskrcnn_moderate_v2_negatives_10ep.yaml `
  --checkpoint artifacts\checkpoints\maskrcnn_moderate_v2_negatives_10ep\last.pt `
  --threshold-profile main_project\configs\stage_04_fusion_calibration\threshold_profile_personal_max_coverage.yaml `
  --device cuda --max-images 120 --output reports\personal_profile_validation_scorecard.json
```

`measure_mask_coverage.py` answers the narrower question of whether annotated
regions actually end up under the burnt-in mask, by driving the running app.

## What it will not do

- Guarantee anything. It reduces risk; it does not certify a photo as safe.
- Decide whether anyone consented. You decide what to keep.
- Work on video, or help with a photo you already shared.
- Send anything anywhere. There is no server and no telemetry.

## Checks

```powershell
powershell -ExecutionPolicy Bypass -File main_project\scripts\ci_local.ps1
```

Python tests, the frozen baseline hash check, the frontend build, frontend unit
tests, and the browser end-to-end flow. Add `-IncludeModels` for a real
one-photo run through every detector. There is no hosted CI: the tests need the
local GPU, the checkpoints and the dataset records, none of which leave this
machine.

## Layout

- `main_project/` — the code, in six numbered stages (data, baseline model,
  specialist detectors, fusion, review/export, evaluation and release).
- `main_project/scripts/stage_05_review_export/` — the app and its launcher.
- `main_project/configs/stage_04_fusion_calibration/` — threshold profiles. The
  personal profile is what the app uses; the validation-calibrated one is
  stricter and precision-oriented.
- `reports/` — measured evidence, one file per run.
- `docs/archive/` — earlier research reports, failure registers and download
  guides, with an index explaining what is still true.
- `research/` — training and Kaggle machinery. Not needed for the app.

Datasets, checkpoints and outputs stay out of version control and must not be
redistributed without their original licences.

## Honest status

The detectors are single-seed research checkpoints. The threshold profile is
tuned, not certified: `release_ready: false` is in that file on purpose, and the
app shows it as a warning rather than blocking the download. The strict release
gates in `main_project/stage_06_evaluation_release/` (95% recall in two domains,
three seeds, independent attackers, a 2,000-photo target set) are **not met**
and are not being pursued. A person reviewing the result is the safety net
instead.

The research background, including why the project is scoped this way, is in
[`ConsentGuard_Final_Research_Design.md`](ConsentGuard_Final_Research_Design.md).
