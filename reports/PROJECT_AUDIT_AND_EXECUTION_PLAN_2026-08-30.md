# ConsentGuard repository audit and execution plan — 2026-08-30

## Decision

The repository is a credible research prototype, but it is not a completed or
release-safe product. The architecture is sensible: broad Mask R-CNN evidence,
specialist providers, deterministic fusion, manual consent/review, destructive
redaction, and fail-closed export. The largest remaining problem is not the
website. It is target-domain evidence: several classes have too little licensed
data, only one random seed has been run for the specialists, and the independent
residual-content attacks required by Stage 06 are not complete.

The correct immediate target is one controlled plate experiment, not replacing
every model at once. The current website continues to use the CCPD-to-India
checkpoint until the new full-scene candidate beats it on frozen validation and
the separate Deepak road-video challenge.

## What is actually in the system

| Branch | Runtime model/provider | Training data represented in the repository | Measured state | Decision |
|---|---|---|---|---|
| Broad visual privacy | Mask R-CNN ResNet-50 FPN v2, nine Visual Redactions visual classes | Visual Redactions V2: 8,266 validated records; 726 negative training images | Best research checkpoint remains below the release gate; rare-class recall is weak | Keep as broad evidence provider; do not call it production-safe |
| Face geometry | OpenCV YuNet pretrained detector | YuNet weights are upstream pretrained weights; ConsentGuard does not retrain YuNet | Fast independent box evidence; no identity inference | Keep. Retraining YuNet is not the next priority |
| Face specialist | Mask R-CNN on Visual Redactions; separate Faster R-CNN WIDER FACE experiment | Local Visual Redactions specialist plus WIDER FACE run with 12,880 train and 3,226 validation images | Local mask mAP 0.5811 on 1,576 validation images; WIDER box mAP 0.2947 and small-object mAP 0.2232 | Useful experimental face evidence; needs three seeds and target-domain assurance |
| Plate geometry | OpenCV LPD-YuNet pretrained detector | Upstream model is mainly Chinese-plate geometry | Useful independent proposal source, but India domain transfer is not established | Keep only as corroborating evidence; never let it grant release |
| Plate specialist | Faster R-CNN ResNet-50 FPN v2 | CCPD2020 → Indian adaptation → new full-scene research bundle | Existing India checkpoint box mAP 0.7736 on its validation, but only 1 TP/44 FP/67 FN at confidence 0.5 on the held-out Deepak road-video challenge | Train the new candidate; keep current website checkpoint until it wins |
| Printed text | PP-OCRv3 ONNX geometry | Upstream pretrained detector; recognized text is discarded | Geometry provider works; it is not a handwriting model | Keep as independent printed-text evidence |
| Handwriting | Mask R-CNN plus optional PaddleOCR geometry | Local Visual Redactions and Kaggle HierText: 8,281 train and 1,724 validation images | Local bounded mask mAP 0.0526; HierText mask mAP 0.1577 | Not release-ready; next major data/model target after plate |
| QR/barcode | zxing-cpp | No learned ConsentGuard dataset | Deterministic decoder/geometry provider | No training is required |
| Metadata | EXIF/container inspection | No learned dataset | Rule-based evidence | No training is required |

The nine broad learned classes are face, license plate, person body, nudity,
handwriting, disability evidence, medicine, fingerprint, and signature. There
are no separate trainable models for person body, nudity, disability, medicine,
fingerprint, or signature yet; those remain responsibilities of the broad
Mask R-CNN and are among the reasons Target-2K is required.

## Why YuNet exists and whether we should train it

YuNet is present because a second, independently implemented face detector can
catch faces missed by the broad segmentation model and can provide disagreement
signals to the manual-review policy. ConsentGuard uses detection only; it does
not calculate face embeddings or identities.

We are not currently training YuNet. Doing that is technically possible, but it
would require a correctly licensed, target-domain face-box dataset, conversion
to the YuNet training recipe, multiple seeds, and a frozen external evaluation.
The existing local face Mask R-CNN already reaches 0.5811 mask mAP, while the
WIDER Faster R-CNN experiment supplies better target-scale evidence. Therefore
YuNet retraining is lower value than fixing plate and handwriting evidence.

## New Roboflow Indian plate dataset audit

Source: Roboflow Universe project `nivu/indian-license-plate-knte7`, version 1.
The publisher page declares CC BY 4.0. This is a user-published source, so that
declaration and attribution record must stay attached to any research use; it
is not equivalent to an institutional provenance guarantee.

| Item | Result |
|---|---:|
| Downloaded archive | 44,581,183 bytes (42.5 MiB) |
| Archive SHA-256 | `14d89c1ccd50341279e2cfa154e3b253247a92cc0704655bef24dfd26cc04db7` |
| Images / label files | 1,650 / 1,650 |
| Decodable images | 1,650 |
| Missing labels / label errors | 0 / 0 |
| Boxes | 1,641 |
| Classes | one: `indian_licence_plate` |
| Publisher split | 1,156 train / 330 valid / 164 test |
| Cross-publisher-split duplicate/source groups | 17 |
| Publisher split admitted | **No** |

The important defect is leakage. Frames from the same source asset or numbered
video occur in more than one publisher split. Reporting results on that split
would overstate generalization. The data is still useful after regrouping.

At the publisher's 416×416 preprocessing size, only 7 boxes are below 32 pixels
wide, 108 are 32–64 pixels, 954 are 64–128 pixels, and 572 are at least 128
pixels. Median width is 113 pixels. This dataset is therefore useful for Indian
appearance transfer but, by itself, does not solve distant road-scene plates.

## Leakage-safe replacement split and merged candidate

The audit pipeline groups exact images, original source assets, numbered video
frames, and very-near perceptual duplicates before splitting. The publisher's
train/valid/test assignments are discarded for model selection.

| Safe Roboflow split | Images | Positives | Negatives | Boxes |
|---|---:|---:|---:|---:|
| Train | 1,154 | 1,145 | 9 | 1,145 |
| Validation | 248 | 248 | 0 | 248 |
| Locked test | 248 | 248 | 0 | 248 |

The locked 248-image split is written for future final evaluation but is not
used by the current training or calibration configuration.

The full-scene candidate merges the safe Roboflow train/validation data with
Visual Redactions plate positives/negatives and 117 grouped Deepak `vid-2`
training road frames. Deepak `vid-1` remains a separate external challenge.

| Candidate split | Images | Positives | Negatives | Boxes |
|---|---:|---:|---:|---:|
| Train | 1,904 | 1,473 | 431 | 1,708 |
| Validation | 1,824 | 331 | 1,493 | 382 |

Exact image-hash leakage between candidate train and validation is zero. The
large negative validation set is deliberate: plate privacy is harmed by both
misses and false redactions, so false-alarm behavior must be measured.

## Storage and compute decision

The new Roboflow download needs only about 88.3 MB locally when retaining both
the zip and extraction. The merged candidate references 4.674 GB of existing
image bytes and a 346.5 MB initialization checkpoint. The Kaggle staging tree
uses NTFS hard links, so it does not duplicate those bytes on disk.

Do not put datasets in ordinary Git. Git stores source, manifests, hashes,
licenses, configs, and reports; Kaggle private datasets store transport payloads;
Git LFS stores only deliberately versioned winning checkpoints. Putting raw
datasets in Git would increase clone size, create licensing risk, and not save
local disk unless local copies were later removed after independent recovery
verification.

The laptop's RTX 3050 has 4 GB VRAM. Both candidate configurations pass model
and data preflight. Real standard- and high-resolution CUDA training steps
completed at losses 0.0471 and 0.0352 respectively without an out-of-memory
error. Full five-epoch training and 1,824-image validation still belong on
Kaggle's larger GPU because local throughput is much lower. The Kaggle candidate
uses 800/1333 resize and 768-pixel object-context crops because tiny/full-scene
plates are the present failure mode. The 512/768 configuration remains the
faster laptop fallback.

## Execution order

1. Publish the private, no-test Kaggle transport and deterministic code bundle.
2. Run the high-resolution five-epoch plate candidate from the current
   CCPD-to-India checkpoint.
3. Download the completed checkpoint and logs; verify their hashes and run
   frozen validation plus the independent Deepak `vid-1` challenge with
   `evaluate_plate_detection_challenge.py`.
4. Replace the website default only if the candidate improves target-domain
   recall without an unacceptable false-positive increase. Otherwise retain
   the current checkpoint and record the failed experiment.
5. Run two additional seeds only for a candidate that clears the first gate.
6. After plate, address handwriting with better target-domain data and an OCR
   detector/segmentation comparison. Face is the next assurance task, not the
   next model rewrite.
7. Build Target-2K through rights-cleared collection/outsourced annotation.
   Target-2K, three seeds, fused leakage/FPR confidence bounds, and independent
   residual-content attacks remain mandatory before any release-safe claim.

## Repository engineering findings

Good:

- Stage boundaries, evidence-provider contracts, geometry-safe preprocessing,
  checkpoint compatibility checks, test-lock rules, manual review, and
  fail-closed export are all implemented.
- Dataset/checkpoint payloads are ignored by ordinary Git.
- The React production build, Python tests, and self-contained Playwright flow
  are executable without a separately started ML backend.
- A real held-out road-frame smoke loaded the broad, face, plate, handwriting,
  YuNet, LPD-YuNet, PP-OCR, and barcode paths together on the 4 GB laptop GPU;
  no provider was marked unavailable and policy correctly returned
  `HOLD_FOR_CONSENT` with export disabled.

Still blocked:

- The release manifest is validation-only and must not be treated as current
  production evidence.
- Target-2K does not exist yet.
- Most model branches have only one seed.
- Plate and handwriting target-domain assurance is below the required level.
- Independent residual-content attack providers are incomplete, so export
  correctly remains fail-closed.

This is not wasted work. It is a functional, auditable research system whose
remaining gaps are now explicit and testable.

## Frozen website-promotion gate

The gate was fixed before the new checkpoint was available. At score threshold
0.50 and IoU 0.50, the current website checkpoint establishes these baselines:

| Frozen set | Recall | Precision | False positives/image |
|---|---:|---:|---:|
| Merged validation: 1,824 images / 382 boxes | 0.6832 | 0.1791 | 0.6557 |
| Deepak `vid-1`: 43 images / 68 boxes | 0.0147 | 0.0222 | 1.0233 |

The new candidate may replace the experimental website default only if all of
the following hold: the Kaggle run succeeds with no test use; merged-validation
recall is at least 0.6832 with false positives/image no greater than 0.6557;
Deepak `vid-1` recall is at least 0.50 with false positives/image no greater
than 1.0233. This promotion does not make the system release-ready; the stricter
Stage 06 95% recall, multi-seed, confidence-bound, and attack gates still apply.
