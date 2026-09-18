# Personal-profile coverage scorecard — 18 September 2026

Machine-readable source: `reports/personal_profile_validation_scorecard.json`.

## What was measured

120 Visual Redactions V2 **validation** images (deterministic sampling, seed
1337, zero skipped). The locked test split was not touched. Providers: broad
Mask R-CNN (moderate-balance V2-negatives checkpoint), face specialist, plate
full-scene v4, handwriting specialist, YuNet, LPD-YuNet, PP-OCRv3 text and
ZXing. Threshold profile: `threshold-profile-personal-max-coverage`
(`release_ready: false`). Every provider was available; no provider errors.

This is the **single-pass** configuration. The tiled second pass the app runs by
default is not exercised by this evaluator, so real app coverage on small
regions is better than the numbers below — measured separately with
`measure_mask_coverage.py`.

## Headline

| Metric | Value | 95% bootstrap CI |
|---|---:|---|
| Sensitive-pixel recall (fraction of labelled private pixels covered) | **0.830** | 0.784 – 0.873 |
| ~~Negative-image false-positive rate~~ — **withdrawn**, see correction below | ~~0.058~~ | — |
| Mean review candidates per image | 7.2 | — |

For reference, the frozen broad-model comparator (`baseline-v0.1`, single global
threshold 0.5) reached sensitive-pixel recall 0.764. Fusing the specialists
under the personal profile raises that to 0.830.

## Per class

`pixel_recall` is union-based: were those pixels covered by *any* accepted mask.
`instance_recall` is class-matched: did a candidate **of that class** cover at
least half of the instance. The gap between them is the interesting part.

| Class | Instances | Pixel recall | Instance recall | Pixel leakage |
|---|---:|---:|---:|---:|
| Face | 66 | 0.998 | 0.955 | 0.002 |
| Licence plate | 34 | 0.954 | 0.559 | 0.046 |
| Person body | 154 | 0.948 | 0.831 | 0.052 |
| Nudity | 27 | 0.934 | 0.333 | 0.066 |
| Fingerprint | 11 | 0.954 | **0.000** | 0.046 |
| Signature | 47 | 0.658 | 0.234 | 0.342 |
| Disability evidence | 15 | 0.562 | 0.667 | 0.438 |
| Medicine / document | 98 | 0.521 | 0.071 | 0.479 |
| Handwriting | 129 | 0.485 | 0.496 | 0.515 |

## How to read this

- **Faces are solid.** 99.8% of face pixels covered, 95% of individual faces.
- **Plates: pixels yes, instances no.** 95% of plate *pixels* are covered
  because large plates dominate the pixel count, while 44% of individual plates
  are not covered by a plate-classed mask. Small and distant plates are the
  failure, which is exactly what the tiled pass targets.
- **Fingerprint's 0.000 instance recall with 0.954 pixel recall** means those
  pixels were covered incidentally, by a hand or body mask, and never by
  fingerprint detection. There is effectively no working fingerprint detector;
  do not rely on it.
- **Handwriting, medicine, signature and disability leak 34–52% of their
  pixels.** The interface labels these "check manually" for this reason.
- **False alarms are not low.** See the correction below.

## What this does not establish

Single seed, one validation split, one threshold profile, no India-domain or
general-domain slices, and no independent attacker implementations. These are
usable operating numbers for a personal tool with a human reviewing every
export; they are not release evidence, and nothing here unlocks the Stage 06
gates.

## Reproduce

```powershell
.\.venv\Scripts\python.exe main_project\scripts\stage_06_evaluation_release\evaluate_fused_validation.py `
  --config main_project\configs\stage_02_baseline_model\train_maskrcnn_moderate_v2_negatives_10ep.yaml `
  --checkpoint artifacts\checkpoints\maskrcnn_moderate_v2_negatives_10ep\last.pt `
  --threshold-profile main_project\configs\stage_04_fusion_calibration\threshold_profile_personal_max_coverage.yaml `
  --face-checkpoint artifacts\checkpoints\specialist_face_maskrcnn_5ep\last.pt `
  --plate-config main_project\configs\stage_03_specialists\train_plate_full_scene_research_v1_highres_5ep.yaml `
  --plate-checkpoint artifacts\kaggle\remote-runs\plate-full-scene-v4\consentguard\artifacts\checkpoints\specialist_plate_full_scene_research_v1_highres_5ep\best.pt `
  --handwriting-checkpoint artifacts\checkpoints\specialist_handwriting_maskrcnn_5ep\last.pt `
  --yunet-model artifacts\specialists\opencv_zoo\face_detection_yunet_2023mar.onnx `
  --plate-yunet-model artifacts\specialists\opencv_zoo\license_plate_detection_lpd_yunet_2023mar.onnx `
  --ppocr-model artifacts\specialists\opencv_zoo\text_detection_en_ppocrv3_2023may.onnx `
  --with-barcode --device cuda --max-images 120 `
  --output reports\personal_profile_validation_scorecard.json
```

## The tiled second pass, measured on road photos

`measure_mask_coverage.py` drives the running app over 20 Indian road frames
(Deepak `vid-2` **training** side; the `vid-1` challenge stays locked) holding 40
annotated number plates. A plate counts as covered when at least half of its
box ends up under the mask the user is about to burn in.

| Configuration | Plates covered | Rate | Mean image erased |
|---|---:|---:|---:|
| All providers, tiled second pass on | 33 / 40 | **0.825** | 10.9% |
| Single pass only | 24 / 40 | 0.600 | 9.2% |

Tiling buys 22 percentage points of plate coverage for 1.7 points of extra
erasure, on the failure mode the single pass is worst at. It costs seconds per
photo, which is the trade this tool was configured to take.

Source: `reports/plate_mask_coverage_2026-09-18.json`.

## Correction — false-alarm rate (added the same day)

The 0.058 "negative-image false-positive rate" above was wrong, and it was
repeated in the README and the status page as "only 5.8% of clean photos get
anything erased".

**Cause.** `evaluate_fused_validation.py` appended a flag for *every* image but
only ever set it for negatives, so the rate was divided by all 120 images
instead of by the negatives. The report's own numbers show the true value:
0.058 × 120 = 7 negative images in the sample, and `images_with_candidates` was
120 of 120, so **all 7 negatives produced candidates — 7/7, not 7/120.** The
evaluator is fixed (`negative_false_positive_rate`, with a regression test), and
a sample containing no negatives now reports the rate as unknown, not zero.

**What false alarms actually look like.** Measured through the running app on
30 validation photos with no private *visual* content (tiled pass on): all 30
had something erased, averaging 45% of the photo. A six-photo diagnostic
attributed the erased area as follows (shares overlap, so they do not sum):

| Provider / class | Mean share of photo |
|---|---:|
| PP-OCR printed text, tiled | 42.0% |
| PP-OCR printed text, single pass | 39.4% |
| LPD-YuNet plate | 6.5% |
| Plate Faster R-CNN v4 (tiled / single) | 6.4% / 6.3% |
| Handwriting specialist | 3.8% |
| Everything else | < 1.2% each |

The text detections were inspected and are **real text**, precisely boxed, with
scores of 0.92–0.998: cartridge labels, a printed advertisement. "Negative" in
Visual Redactions means no private *visual* attribute, and many such photos are
text-heavy. So the dominant cost is the chosen policy — erase all text — not a
misfiring detector. The plate share, on photos that contain no plates, is a
genuine false alarm, partly on labels that resemble plates.
