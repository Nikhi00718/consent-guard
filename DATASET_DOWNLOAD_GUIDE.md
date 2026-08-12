# ConsentGuard Dataset Download Guide

**Prepared:** 8 August 2026; corrected after forensic audit on 12 August 2026  
**Workspace:** `C:\consentGuard`  

## Before downloading

After the 12 August cleanup, the C: drive has approximately **200.6 GiB free**.
Recheck it before every large download.

Known public download sizes:

| Dataset | Public download | Approximate compressed size |
|---|---|---:|
| VISPR annotations | Three small archives | Less than 2 MB total |
| VISPR images | Train, validation, and test | About 47 GB |
| Visual Redactions annotations | Masks plus weak annotations | About 300 MB |
| Visual Redactions images | Separate train, validation, and test archives | 15.467 GiB |
| Current VPD Hugging Face repository | 2,462 public videos | About 33 GB |
| Full VPD-100K image benchmark | Not currently visible in public repository | Unknown |

Extraction, preprocessing, checkpoints, and caches require additional space. Keep at least twice the compressed dataset size available before extraction.

## Downloader script

Use:

```powershell
Set-Location C:\consentGuard
.\scripts\download_datasets.ps1
```

The script does nothing unless an explicit switch is supplied.

While background downloads are running, check live progress with:

```powershell
.\scripts\check_dataset_status.ps1
```

Download logs are stored under `data\download_logs\`.

## Step 1: Download VISPR annotations

```powershell
Set-Location C:\consentGuard
.\scripts\download_datasets.ps1 -VisprAnnotations
```

This downloads the official train, validation, and test annotation archives to:

```text
C:\consentGuard\data\raw\vispr\
```

Do this first and inspect the label structure before downloading images.

## Step 2: Get Visual Redactions annotations

```powershell
.\scripts\download_datasets.ps1 -VisualRedactionsAnnotations
```

This downloads the official Train/Val/Test COCO-like mask JSON files plus the
Google Cloud Vision weak-annotation ZIP files, then extracts the weak annotations under:

```text
C:\consentGuard\data\raw\visual_redactions\annotations-extra\
```

The project page currently returns HTTP 403, but its official dataset host is accessible.

> **Critical correction:** do not pair these masks with VISPR pixels. The repeated
> `2017_xxxxxxxx` names are not a reliable cross-release image key. A forensic review
> proved that many matching IDs describe completely different scenes.

The official Visual Redactions image archives are separate downloads:

| Split | URL | Compressed size |
|---|---|---:|
| Train | `https://datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/images/train2017.tar.gz` | 7.279 GiB |
| Validation | `https://datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/images/val2017.tar.gz` | 2.942 GiB |
| Test | `https://datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/images/test2017.tar.gz` | 5.246 GiB |

The current 200.6 GiB free space is sufficient for the 15.467 GiB archives,
extraction, processed copies, checkpoints, and validation workspace. Use resumable
downloads and verify exact byte lengths before extraction.

The downloaded `v1` masks contain 8,473 image records and 48,846 instances across
28 distinct attribute IDs. Loaders must resolve each record only inside the matching
Visual Redactions release and split, then verify image identity before accepting it.

## Step 3: Download Visual Redactions images (required next)

Download the three separate archives listed above. Train and validation are
needed for development; test must remain locked until the final protocol is frozen.
Do not preprocess anything until archive integrity and sample image identity pass.

## Step 4: Download VISPR images (optional, separate experiments only)

VISPR pixels are not required for the Visual Redactions localizer and must never
be paired with Visual Redactions masks. Skip this step for the current project stage.

The images are approximately 47 GB compressed.

```powershell
.\scripts\download_datasets.ps1 `
  -VisprImages `
  -AcceptLargeDownloads
```

The script uses resumable `curl` downloads. If the connection stops, run the same command again.

## Step 5: Extract VISPR (optional)

Check free space before extracting:

```powershell
Get-PSDrive C
```

Then run:

```powershell
.\scripts\download_datasets.ps1 -ExtractVispr
```

Do not delete the archives until the extracted files and sample annotations have been verified.

## Step 6: Install the Hugging Face CLI

The current machine has Python, but `hf` is not installed yet.

```powershell
python -m pip install --upgrade huggingface_hub
```

Open a new PowerShell window if `hf` is not immediately recognized, then verify:

```powershell
hf version
```

The modern command is `hf`, not the deprecated `huggingface-cli`.

## Step 7: Inspect VPD before downloading

```powershell
Set-Location C:\consentGuard
.\scripts\download_datasets.ps1 -VpdInspect
```

This checks the Dataset Viewer and writes an `hf download --dry-run` inventory under:

```text
C:\consentGuard\data\raw\vpd_public\vpd_download_dry_run.txt
```

As of 8 August 2026, the public repository exposes 2,462 video rows and about 33 GB. It does not visibly provide the 100,000 images and bounding-box annotations described by the paper.

## Step 8: Download the public VPD videos only if needed

These videos are useful for later video demonstrations, but they are not a confirmed detector-training package.

```powershell
.\scripts\download_datasets.ps1 `
  -VpdPublicVideos `
  -AcceptLargeDownloads
```

Before training a VPD detector, confirm that you have:

- 100,000 image files;
- official train/validation/test splits;
- 33-class category definitions;
- bounding-box annotations;
- current dataset license.

If those files are not present, contact the VPD authors or monitor the official project page instead of inventing annotations.

## Step 9: Create a checksum manifest

After downloads finish:

```powershell
.\scripts\download_datasets.ps1 -ComputeHashes
```

This writes:

```text
C:\consentGuard\data\raw\sha256_manifest.txt
```

Hashing tens of gigabytes will take time.

## Dataset that cannot be downloaded

The ConsentGuard consent/context dataset does not exist publicly. We will create it later using fictional scenarios and, if ethically approved, safely staged adult-volunteer examples.

## Licenses and source pages

- VISPR: https://tribhuvanesh.github.io/vpa/
- Visual Redactions: https://resources.mpi-inf.mpg.de/d2/orekondy/redactions/
- VPD-100K: https://vpd-100k.github.io/
- Current VPD repository: https://huggingface.co/datasets/XiaoyuSunANU/Visual_Privacy_Dataset

VISPR, Visual Redactions, and the current VPD repository are marked for non-commercial research use. Verify each current license and the original image licenses before publication, redistribution, or product use.

## Recommended order summary

```text
VISPR annotations
    -> Visual Redactions annotations
    -> separate Visual Redactions image archives
    -> verify byte lengths, decode, split paths, and image identity
    -> build a same-release manifest
    -> manually approve overlays
    -> run staged learning gates
    -> optional VISPR/VPD experiments only after the core baseline is valid
```
