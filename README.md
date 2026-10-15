# ConsentGuard

A local tool that erases private content from a photo before you share it.

**[consent-guard project page](https://nikhi00718.github.io/consent-guard/)** · screenshots, what it catches, and how to run it.

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

The trained weights are not in this repository. Download them into the repo
root before the first run (see [Models](#models)):

```powershell
hf auth login
hf download Nikhil00718/consentguard-weights --local-dir .
```

## Models

The app's trained checkpoints live in a private Hugging Face repository:
**[Nikhil00718/consentguard-weights](https://huggingface.co/Nikhil00718/consentguard-weights)**.
Files keep the same `artifacts/checkpoints/...` paths the app expects, so the
download command above puts them in place.

| Checkpoint | Role in the app | Architecture | Trained on | Training | SHA-256 |
|---|---|---|---|---|---|
| `maskrcnn_moderate_v2_negatives_10ep/last.pt` | Main 9-class detector | Mask R-CNN ResNet-50 FPN v2 (torchvision, COCO-pretrained) | Visual Redactions masks on VISPR images, verified visual classes plus clean negatives | 10 epochs, SGD lr 0.0025, class-balanced sampling, small-object anchors, seed 1337; best val mAP 0.187 | `f3a47cd3bcbc52bcbaddd9094fb0fae8171d0753d70220ddc83dfa515f5411d8` |
| `specialist_face_maskrcnn_5ep/last.pt` | Face specialist | Mask R-CNN ResNet-50 FPN v2, one class | Face regions from the same Visual Redactions train split (3,785 records, 3,992 faces) | 5 epochs, SGD lr 0.0025, 512 px crops; best val mAP 0.561 | `0b4084480cee3efd6034ab03a4bc0b15c94cbb83cbfa60d6ecf61ab29d92a929` |
| `specialist_handwriting_maskrcnn_5ep/last.pt` | Handwriting specialist | Mask R-CNN ResNet-50 FPN v2, one class | Handwriting regions from the same Visual Redactions train split (750 records, 1,040 regions) | 5 epochs, SGD lr 0.0025, 512 px crops, no flips | `1614534292bdc7fe2771defba4fe9e1994bbb0a3d40b509402f42ef1447d2f0b` |
| `specialist_plate_ccpd2020_india_finetune_5ep/best.pt` | Number-plate specialist | Faster R-CNN ResNet-50 FPN v2, one class | Pre-trained on CCPD2020 ([Zenodo 15647076](https://zenodo.org/records/15647076)), then fine-tuned on Kaggle `kedarsai/indian-license-plates-with-labels` (1,617 train images) | 5 epochs on Kaggle GPU, SGD lr 0.0005; val bbox mAP 0.774, mAP@50 0.974 | `d8a7a551fe3a9f264bb1ad34066f583d92ae9344503112a582554e29975a81b8` |

The 9 classes of the main detector: face, license plate, whole person, nudity,
handwriting, physical disability, medicine, fingerprint, signature
(`a105`, `a108`, `a109`, `a110`, `a26`, `a39`, `a43`, `a7`, `a8` in the
Visual Redactions attribute set).

Also used, and already in this repository:

- `artifacts/checkpoints/maskrcnn_verified_class_balanced_5ep/best.pt` (Git LFS).
- OpenCV Zoo ONNX models in `artifacts/specialists/opencv_zoo/`: YuNet face
  detection, YuNet license-plate detection, PP-OCRv3 English text detection.
- QR codes and barcodes are found by a decoder, not a trained model.
- NudeNet, optional, only if installed (AGPL-3.0, see `THIRD_PARTY.md`).

Sizes and the per-run configs are recorded in each run's
`resolved_config.json` and `training_result.json`, and in the reports under
`reports/`. The weights are kept private because the training datasets do not
allow redistribution; see `THIRD_PARTY.md` for their terms.

### Datasets used

None of these are in the repository; download them from their sources.

| Dataset | Used for |
|---|---|
| [VISPR](https://tribhuvanesh.github.io/vpa/) images | Main detector and face/handwriting specialists |
| [Visual Redactions](https://resources.mpi-inf.mpg.de/d2/orekondy/redactions/) masks | Main detector and face/handwriting specialists |
| [CCPD / CCPD2020](https://zenodo.org/records/15647076) | Plate specialist pre-training |
| Kaggle `kedarsai/indian-license-plates-with-labels` | Plate specialist fine-tune |
| HierText, IIIT-INDIC-HW, Indo-ALPR, Roboflow `nivu` Indian plates, Kaggle `deepakat` Indian plates, HF `zenitsu09` Indian plates | Experiments and evaluation, not the shipped weights |

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

Measured on 120 validation photos the models never trained on: **83% of all
labelled private pixels get covered** (95% CI 78–87%).

**Expect erasure even on photos with nothing private in them.** The tool erases
*all* text by design, and on text-heavy photos — labels, adverts, documents,
signs — that is most of the picture. On 30 validation photos with no private
visual content, every one had something erased, 45% of each photo on average.
A diagnostic on six of them traced ~40 points of that to text (the detector was
right: it was real text) and ~6 points to stray plate boxes on photos with no
plates. Switch text off under **What to erase** when you only want faces and
plates gone. Full report:
[`reports/PERSONAL_PROFILE_SCORECARD_2026-09-18.md`](reports/PERSONAL_PROFILE_SCORECARD_2026-09-18.md).

| Content | Private pixels covered | Whole regions caught | Verdict |
|---|---:|---:|---|
| Faces | 99.8% | 95% | Good |
| Whole people | 94.8% | 83% | Good, and **off** by default — it erases most of a photo |
| Number plates | 95.4% | 56% | Catches many; small and distant ones are missed |
| Nudity | 93.4% | 33% | Check manually |
| Signatures | 65.8% | 23% | Weak |
| Medicine and documents | 52.1% | 7% | Weak |
| Handwriting | 48.5% | 50% | Weak |
| Other details (fingerprints, visible disability) | — | — | **Best effort only.** No working detector for either; whatever the broad model flags is still erased, but they are not offered as protected categories |
| QR codes and barcodes | — | — | Reliable: a decoder, not a model |
| GPS/camera metadata | — | — | Always removed; the file is written from scratch |

"Whole regions caught" counts regions where a mask of that type covered at least
half the region, which is the number that matters when one small plate is the
thing you needed hidden. Those figures are from the single-pass configuration.

The app also runs a **tiled second pass**, which is what rescues small regions.
Measured on 20 real road photos holding 40 annotated number plates: it covers
**33 of 40 plates (82.5%)** against 24 of 40 (60%) single-pass, for 1.7 points
more of the image erased.

Reproduce them:

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
