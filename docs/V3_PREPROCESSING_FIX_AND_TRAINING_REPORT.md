# ConsentGuard v3 preprocessing fix and training report

Date: 2026-08-12

## Outcome

The v1/v2 model code and GPU loop were operational, but the training set was
not geometrically valid. The legacy preprocessor independently multiplied X
and Y annotation coordinates by the decoded/source dimension ratios for every
image. That operation is valid for an aspect-preserving resize; it is invalid
for a crop, stitch, changed image, or unhandled rotation. Consequently, many
privacy masks supervised unrelated pixels.

The unsafe records are preserved for old-checkpoint reproducibility but are no
longer used. v3 trains a new Mask R-CNN from COCO initialization on a separate,
geometry-verified visual-only dataset.

## Root-cause evidence

The decoded-image versus annotation-space audit at 1% aspect-ratio tolerance
found:

| Geometry status | All | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Aligned resize | 1,434 | 597 | 359 | 478 |
| 90-degree candidate | 654 | 302 | 107 | 245 |
| Other geometry mismatch | 6,384 | 2,974 | 1,145 | 2,265 |
| Missing image | 1 | 0 | 0 | 1 |

The four-image overlay montage demonstrates that an aligned person's face/body
masks follow the subjects, while mismatched masks land on unrelated tree,
portrait, and product-label pixels:

`reports/visual_redactions_alignment_overlays.jpg`

Three live Flickr source URLs were also downloaded again. Their SHA-256 hashes
matched the VISPR archive files exactly, so the local extraction did not select
the wrong duplicate and the downloads were complete. The current source image
itself does not match the released segmentation coordinate space in those
cases.

## Architecture correction

The official project groups the evaluated attributes into eight textual, nine
visual, and seven multimodal attributes, and ignores four additional released
labels. A single 28-class visual detector was therefore the wrong baseline.
v3 uses Mask R-CNN only for the nine visual attributes:

- face, licence plate, person body, nudity;
- handwriting, physical disability, medicine;
- fingerprint and signature.

The next branches are OCR plus text classification/NER for textual attributes,
and document detection plus OCR/context classification for multimodal
attributes. Their masks will be fused only after each branch passes its own
validation gate.

Primary sources:

- [CVPR 2018 paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Orekondy_Connecting_Pixels_to_CVPR_2018_paper.pdf)
- [Official Visual Redactions repository and evaluation procedure](https://github.com/tribhuvanesh/visual_redactions)
- [Official taxonomy and ignored-label configuration](https://github.com/tribhuvanesh/visual_redactions/blob/master/config.py)
- [TorchVision instance-segmentation target contract](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial)

## Implemented safeguards

- `src/consentguard/data_quality.py` defines the official modality taxonomy,
  ignored attributes, 625-pixel official evaluation cutoff, EXIF-aware display
  dimensions, and symmetric aspect-ratio checks.
- `scripts/audit_visual_redactions_alignment.py` writes a complete geometry
  audit and visual overlays.
- `scripts/preprocess_visual_redactions_verified.py` writes new records only;
  every rejected image is listed in `quarantined_geometry.jsonl`.
- The preprocessor confirms every accepted candidate using the same OpenCV
  decoder used during training.
- `scripts/validate_processed_records.py` accepts an explicit dataset/report
  path and rejects a non-verified geometry status.
- New v3 and controlled-overfit configs use a 10-class head (background plus
  nine visual attributes). Legacy 29-class checkpoints cannot be resumed into
  v3 because the class-map compatibility check rejects them.
- `scripts/watch_maskrcnn.ps1` provides overall and epoch progress bars, ETA,
  losses, validation metrics, and GPU health.

## Verified dataset

After geometry and visual-modality filtering:

| Split | Images | Instances |
|---|---:|---:|
| Train | 501 | 2,191 |
| Validation | 294 | 1,190 |
| Test (locked) | 400 | 1,703 |
| Total | 1,195 | 5,084 |

All nine classes are present in train and validation. The train imbalance is
92.75x: person body has 1,113 instances while physical disability has 12.
Class-balanced sampling is capped at 5x; rare-class AP must be reported and
claims for those classes remain limited.

## Verification results

- Processed-record validator: 1,195 records, 5,084 instances, zero invalid
  records, duplicate IDs, or duplicate paths.
- Automated tests: 12 passed.
- CUDA/model preflight: RTX 3050 Laptop GPU, 4 GB; 501 train and 294 validation
  images; 45,923,467 parameters.
- Attached 15-step production-recipe smoke: passed; peak reserved VRAM 31.8%.
- Fixed eight-image overfit gate: best COCO mask AP@[.50:.95] 0.303; final mask
  AP@0.50 0.441; loss fell from about 4.1 to about 0.25.

The overfit result is a pipeline diagnostic, not a scientific test result.

## Active v3 run

- Config: `configs/train_maskrcnn_verified_visual.yaml`
- Output: `artifacts/checkpoints/maskrcnn_verified_visual_v3`
- Schedule: 30 epochs, 126 optimizer steps per epoch, 3,780 total steps.
- Input: 640 short side, 1,024 maximum long side, batch 1, four-step gradient
  accumulation, AMP, small anchors, capped class-balanced sampling.
- Checkpoints: atomic `last.pt`, per-epoch checkpoints, and `best.pt` selected by
  validation segmentation AP.
- Epoch 1 completed the full train/evaluate/save cycle and epoch 2 started.
  Epoch-1 COCO mask AP was 0.000064 after a 100-step warm-up; this is recorded
  as an early baseline, not treated as a convergence result.

After epoch 1 included full validation, the estimated total runtime was about
2.5-3 hours on the local GPU. The live monitor continuously recalibrates this.

## Metric warning

Current training reports COCO box/mask AP@[.50:.95], AP@0.50, and AR. The paper
uses a different pixel-level precision-recall procedure: prediction thresholds
are swept, curves are corrected, and AP is integrated. Published paper numbers
must not be compared directly with the current COCO AP. A final experiment must
export predictions to the official format and report both metric families with
unambiguous names.

## Remaining limitations

- v3 is a valid visual-branch baseline, not the complete 24-attribute system.
- 7,040 image records remain quarantined. Rotation candidates must not be
  auto-rotated without proving the corresponding polygon transform; arbitrary
  mismatches require recovery of the exact annotated image version.
- Some verified classes have too few examples for strong generalization.
- The locked test split remains unused until architecture and thresholds are
  frozen.
