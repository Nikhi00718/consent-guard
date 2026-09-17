# ConsentGuard dataset and implementation status

**Verified:** 12 August 2026  
**Workspace:** `C:\consentGuard`

## Outcome

The official Visual Redactions v1 train, validation, and test image archives are
fully downloaded, byte-length verified, completely readable, extracted, and
matched only to annotations from the same release and split. The corrected
pipeline is ready for a short baseline training run.

VISPR pixels and the public VPD video repository are not used by this pipeline.
The raw Visual Redactions release remains unchanged.

## Official release validation

| Split | Archive bytes | Release images | Model-ready records |
|---|---:|---:|---:|
| Train | 7,816,171,895 | 3,873 | 3,059 |
| Validation | 3,158,549,744 | 1,611 | 1,271 |
| Test (locked evaluation) | 5,633,037,591 | 2,989 | 2,351 |
| **Total** | **16,607,759,230** | **8,473** | **6,681** |

All 8,473 release images decode successfully and have unique image IDs. The
model-ready visual profile contains 26,992 valid mask instances across nine
privacy classes.

## Safety decisions

- 202 EXIF/orientation-risk images are quarantined instead of receiving unsafe
  mask scaling.
- Four train-side exact or perceptual duplicates of validation/test scenes are
  reproducibly excluded by `configs/cross_split_leakage_quarantine.json`.
- Validation and test references are preserved; no raw image is deleted or
  modified.
- 1,585 images without a valid instance in the selected nine-class profile are
  omitted.
- Test records exist for final evaluation but must remain untouched while model
  and threshold choices are being made on train/validation only.

## Passed gates

- Strict archive and same-release identity validation: **passed**.
- Processed record validation: **6,681 records / 26,992 instances, zero invalid
  records, duplicate IDs, or duplicate paths**.
- Complete exact SHA-256 and perceptual-hash split audit: **passed**, zero
  cross-split candidates at pHash Hamming distance <= 5.
- Automated tests: **14 passed**.
- Data-loader smoke test: **passed**.
- Tiny learning smoke test: loss decreased from **1.258 to 0.573**.
- CUDA Mask R-CNN smoke test on RTX 3050: **passed**.
- Eight-image Mask R-CNN overfit gate: mask mAP **0.564**, AP50 **0.860**, AP75
  **0.689**.

## Next action

Run a short baseline on the 3,059 training records and select checkpoints using
the 1,271 validation records. Do not use the 2,351 test records until the model,
privacy threshold, and utility evaluation protocol are frozen.

Key machine-readable evidence is stored in:

- `reports/visual_redactions_release_validation.json`
- `reports/processed_records_validation.json`
- `reports/same_release_split_leakage_audit_clean.json`
- `data/processed/visual_redactions_verified_visual/preprocess_summary.json`
