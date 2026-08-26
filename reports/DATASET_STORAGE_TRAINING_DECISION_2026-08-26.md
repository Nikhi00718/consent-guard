# ConsentGuard dataset, storage, and training decision

Date: 2026-08-26

## Decision

Do not delete or replace the current Visual Redactions corpus before a new
global model has beaten the frozen comparator. Keep datasets separated by
role instead of merging them into one undifferentiated training set.

Full training should run on Kaggle GPUs. The laptop's RTX 3050 has 4 GB VRAM
and is appropriate for conversion, integrity checks, loader smoke tests,
inference, calibration, and very small overfit runs. It is not the preferred
machine for full WIDER FACE, HierText, CCPD, or Mask R-CNN training.

Dataset archives and checkpoints remain outside Git. Git stores code,
configuration, checksums, manifests, metrics, and reports; it is not a safe or
space-efficient substitute for dataset storage.

## Current local storage

Measurements taken before the new acquisition wave:

| Area | Local volume | Decision |
|---|---:|---|
| `data/raw/visual_redactions` | 34.08 GiB | Keep until replacement passes the frozen comparator |
| `data/raw/hiertext` | 0.07 GiB | Annotation files only; full images already used on Kaggle |
| `data/processed` | 0.06 GiB | Keep reproducible record manifests |
| `artifacts/checkpoints` before cleanup | 35.54 GiB | Prune redundant epoch snapshots |
| Redundant `epoch-*.pt` removed | 27.98 GiB / 85 files | Removed only when `best.pt` or `last.pt` remained |
| Reusable model cache | 0.17 GiB | Keep; deletion would force a redownload |
| Kaggle train/validation package | 10.15 GiB | Keep until the remote copy is verified |

Free space increased from 83.51 GiB to 105.34 GiB after checkpoint cleanup.

After the staged dataset downloads and the reusable Faster R-CNN cache were
installed, the C: volume had 97.86 GiB free. The temporary one-step CCPD2020
training checkpoint was deleted after verification, so this figure does not
include an unnecessary smoke-test model.

## Laptop training smoke validation

The CCPD2020 plate configuration passed preflight on the local NVIDIA GeForce
RTX 3050 Laptop GPU (4 GiB VRAM). A one-optimizer-step CUDA run also passed:

- 5,769 train images loaded; validation was intentionally skipped.
- Peak allocated GPU memory: 2.33 GB; peak reserved: 2.37 GB of 4.29 GB.
- Faster R-CNN COCO initialization loaded from the verified TorchVision cache.
- The temporary smoke checkpoint was removed after the run.

This validates the local data/model path, not full-model quality. Full training
remains a Kaggle job; use the laptop for conversion, checks, smoke tests, and
small overfit runs.

## Model-by-model data plan

| Component | What runs now | Current data and measured scale | Main problem | New data/decision | Train it? |
|---|---|---|---|---|---|
| Global privacy localizer | Mask R-CNN ResNet-50 FPN v2 | Visual Redactions V2, about 8,266 project records; 34.08 GiB local source tree | Small/old research domain, restricted rights, class imbalance, weak India/mobile coverage | Preserve as comparator. Use BIV support only after protocol review; keep all 1,056 BIV query images untouched for external evaluation. Later add a rights-audited Open Images subset and Target-2K. | Yes, after the new training manifest is frozen |
| Face primary specialist | Faster R-CNN ResNet-50 FPN v2 | WIDER FACE Kaggle run: 12,880 train images/156,985 boxes and 3,226 validation images/39,111 boxes | Almost no negative training images; research-restricted license; benchmark-to-phone domain gap; current mAP 0.2947 | Add staged hard negatives and Target-2K face slices. Do not add identity-recognition corpora. | Yes, retrain/fine-tune after hard negatives exist |
| Face safety net | OpenCV YuNet | Provider checkpoint originally trained outside this repository, using WIDER FACE lineage | It is a generic face localizer and is not India/mobile release evidence | Keep as an independent lightweight recall safety net. Calibrate threshold and box expansion on ConsentGuard validation. | No. We are not training YuNet in this repository |
| Plate primary specialist | Faster R-CNN ResNet-50 FPN v2 | Current Kaggle source: 1,617 train images/1,728 boxes and 404 validation images/440 boxes; no negative images; final mAP 0.7702 | Small corpus, weak/pseudo-label provenance, no negatives, not sufficient release evidence | Pretrain/fine-tune with official CCPD, then validate on rights-controlled Indian Target-2K. Add vehicle/no-plate hard negatives. INDO-ALPR is excluded from direct detector training because its portal assets are predominantly plate crops without localization boxes. | Yes; this is the highest-value retraining target |
| Plate safety net | LPD-YuNet/provider checkpoint | Fixed provider weights; upstream training volume is not pinned in this repository | Chinese-domain bias and unclear direct India evidence | Keep only as an independent proposal source until target validation. | Not now; train the Faster R-CNN branch instead |
| Handwriting specialist | Mask R-CNN ResNet-50 FPN v2 | HierText: 8,281 train scenes/34,200 handwritten regions and 1,724 validation scenes/6,667 regions; current mask mAP 0.1577 | Small/thin text regions, many negative scenes, limited Indic representation | Keep HierText for scene localization. Add IIIT Indic word crops for representation/synthetic scene generation, then Target-2K phone scenes. | Yes, after Indic data and scene synthesis are prepared |
| Printed-text safety net | PP-OCRv3 DB detector | Fixed ONNX provider checkpoint | Upstream training corpus/version is not fully pinned; may miss tiny, rotated, blurred, or Indic text | Evaluate first. Download TextOCR only if geometry tests show a real deficit; preserve Open Images rights metadata. | Do not retrain yet |
| Barcode/QR | ZXing-C++ | Deterministic decoder | Neural training would add complexity without solving the current need | Generate synthetic corruption fixtures with fictional payloads. | No neural training |
| Metadata scrubber | Pillow/ExifTool | Deterministic inspection and fresh encoding | Needs assurance coverage, not learned weights | Generate EXIF/XMP/IPTC/GPS/thumbnail fixtures. | No neural training |

## Acquisition sizes and roles

| Dataset | Published/verified scale | Transfer/storage plan | Project role |
|---|---:|---|---|
| BIV-Priv-Seg | 1,072 images total; 932 annotated private instances | Installed and SHA-256 verified; 0.944 GiB compressed locally | Support set is candidate training data; query set is external evaluation only |
| CCPD2020 | 11,776 images; 907,711,344 bytes (0.85 GiB archive) | Installed, official MD5 verified, and extracted; 5,769 train + 1,001 validation records converted | Plate pretraining and difficult-condition robustness; never Indian validation |
| CCPD2019 | 13,164,924,944 bytes (12.26 GiB archive) | Download compressed and verify; do not extract all until a subset is selected | Large plate pretraining pool (>300,000 images across CCPD releases) |
| INDO-ALPR | Landing page says 6,174 images; official API exposes 1,000 original train + 1,000 original test files | 197.51 MiB canonical originals installed and 2,000/2,000 SHA-256 verified. The 7,631,061,443-byte full ZIP was rejected because it is dominated by pre-generated augmentations. | Quarantined plate-crop representation/synthetic source; not direct localization training |
| IIIT-INDIC-HW-WORDS | Roughly 1.09 million word crops across 10 scripts | Devanagari installed first: 69,853 measured train images + 1,000 validation images; 1.475 GiB compressed and 1.49 GiB extracted | Indic representation and synthetic-scene source, not full-scene validation |
| TextOCR | 28,134 images and 903,069 word annotations | 7,072,297,970-byte (6.59 GiB) image archive; conditional, not immediate | Printed-text retraining only if PP-OCR evaluation fails |
| HierText images | 11,639 scenes total | 3.65 GiB compressed for train/validation/test; avoid duplicating locally unless a new run needs it | Existing handwriting scene training/reference |
| Open Images V7 | About 1.9 million dense images; hundreds of GB in full | Never download all. Select 10k-30k rights-audited images with a 10-15 GiB cap | Hard negatives and carefully mapped global classes |
| Target-2K | Planned 2,000 consented/staged/licensed images | Must be collected; estimated size depends on capture resolution | Only authoritative general + India deployment evidence |

## Staged download plan

1. Complete and validate BIV-Priv-Seg.
2. Download and validate CCPD2020, then extract it for converter tests.
3. Download CCPD2019 as a compressed, resumable archive; select a 25k-50k
   sample before extraction/training.
4. Acquire only canonical INDO-ALPR originals from its DOI record and audit
   their geometry before admitting them. The audit found predominantly plate
   crops and no localization boxes, so the set remains outside detector data.
5. Acquire the most relevant IIIT Indic script first instead of all scripts at
   once. Expand only if the first representation experiment helps.
6. Do not duplicate WIDER FACE or HierText locally just because the Kaggle run
   used them; their completed run artifacts are already present.
7. Download TextOCR or an Open Images subset only after provider evaluation
   demonstrates the specific gap they are intended to fix.
8. Collect Target-2K before making release-quality recall claims.

## Training sequence

1. Run data converters and 10-50-image laptop overfit tests.
2. Train the plate Faster R-CNN on CCPD2020; do not mix INDO plate crops into
   detection training without a separately reviewed compositing protocol.
3. Fine-tune and validate the plate branch on rights-controlled Indian data.
4. Retrain the face Faster R-CNN only after negative/hard-case data is ready;
   retain YuNet as a separate fallback.
5. Add Indic representation to the handwriting branch, then fine-tune on scene
   polygons and staged phone captures.
6. Retrain the global Mask R-CNN last, because its ontology, mask mappings,
   negative sampling, and leakage checks require the most careful manifest.
7. Recalibrate the fused system on frozen validation and open Target-2K test
   only after architectures and thresholds are frozen.

## Current blockers

- Full Kaggle submission requires the user's Kaggle API credential at
  `C:\Users\atnik\.kaggle\kaggle.json`. The token must never be committed.
- INDO-ALPR and IIIT may require an interactive repository/portal session.
- Target-2K has not yet been collected.
- The earlier global Kaggle baseline is not still training; it failed because
  one transported Visual Redactions image did not match the expected SHA-256.
  That image/package must be repaired before resubmitting the global run.

## Verified acquisition hashes

| File | SHA-256 |
|---|---|
| BIV `support_images.zip` | `3A37B93DAAD15905FB2FFC25D76CCCAA9D88C57D5FC23E2F5AC66DAC7D3B3E2F` |
| BIV `support_set.json` | `3936B12169813DA19659A8099484C13FD1692412659244444E0458425589476D` |
| BIV `query_images.zip` | `A5206BA41E65A92733346E92195FDFF0C13E2BB5DF19EA87598A6C336B4D797F` |
| BIV `query_set.json` | `E37227B33D22857FA52ADB0AD5A13C889C3D1C8F9BEA027FAAE3208FBF438161` |
| CCPD `CCPD2020.zip` | `7CE77266D2FA216E1903DF47BE800B25FA1F408B985920DBB5E15BC6CFD72271` |
| IIIT `devanagari.zip` | `CAACAA4976E5E70DC4393E8F5562A91C6190813D4DB3EB11A058D75853CDE8CA` |
| IIIT `validationset.zip` | `63A426EBFCFDD6E2F2EDABB3E44322D2483BC67189C4E3E8846644610F894C7E` |
