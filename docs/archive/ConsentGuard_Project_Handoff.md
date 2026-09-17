# Project handoff

**Status:** Superseded  
**Date:** 8 August 2026

The earlier VPD-first, learned multimodal-fusion plan has been retired after a literature, dataset-access, architecture, and failure-mode audit.

Use this document as the only research source of truth:

- [Final research plan](ConsentGuard_Final_Research_Design.md)

The former implementation report is retained only as historical failure evidence:

- [Invalidated historical setup report](TRAINING_SETUP_REPORT.md)
- [Current dataset status and recovery gate](DATASET_DOWNLOAD_STATUS.md)

## Current decision in one paragraph

The core project is a still-image pre-share assistant that localizes privacy-sensitive evidence, processes explicit scoped consent assertions, applies a deterministic and auditable policy, and chooses `ALLOW_PIXELS_UNCHANGED`, `ALLOW_REDACTED`, `HOLD_FOR_CONSENT`, or `MANUAL_REVIEW`. Even the first action creates a new metadata-free raster. The system attacks that output for residual OCR, barcode, metadata, and recognition leakage. It does not infer consent, intent, guilt, or legality from pixels.

## Critical scope changes

- VPD-100K is not on the critical path because the currently visible public repository contains videos without the paper's visible image/box annotations.
- Video, audio privacy, biometric identity matching, generative inpainting, and automated reporting are future work.
- Learned image-caption-consent fusion is an optional baseline, not the release controller.
- Visual Redactions is the core localization dataset and requires its own separate
  image release. VISPR may support a later independent ablation, but its pixels
  must never be joined to Visual Redactions masks by matching ID strings.
- The first implementation milestone is an oracle-mask destructive renderer plus recovery tests, not detector training.
- `ConsentGuard` remains only the historical workspace name because that name is already used by unrelated products.

## Dataset operations

- [Download guide](DATASET_DOWNLOAD_GUIDE.md)
- [Download status](DATASET_DOWNLOAD_STATUS.md)
- `scripts\download_datasets.ps1`
- `scripts\check_dataset_status.ps1`
- `scripts\validate_dataset_downloads.py`

Do not use the old handoff from backups or chat excerpts to make architecture decisions.
