# ConsentGuard Main Project

This is the real, stage-by-stage ConsentGuard codebase. Source modules,
scripts, configurations, and tests are physically grouped here. Datasets,
checkpoints, reports, and third-party sources remain at repository root as
shared artifacts.

## How to review the project

Review one numbered folder at a time:

1. [`stage_01_data`](stage_01_data/README.md) — trusted records, negative images, and provenance.
2. [`stage_02_baseline_model`](stage_02_baseline_model/README.md) — frozen Mask R-CNN baseline and diagnostics.
3. [`stage_03_specialists`](stage_03_specialists/README.md) — face, plate, text, barcode, and metadata evidence.
4. [`stage_04_fusion_calibration`](stage_04_fusion_calibration/README.md) — class thresholds, evidence fusion, and uncertainty.
5. [`stage_05_review_export`](stage_05_review_export/README.md) — human review, policy, destructive redaction, and assurance.
6. [`stage_06_evaluation_release`](stage_06_evaluation_release/README.md) — release metrics, gates, model card, and locked tests.

[`STATUS.md`](STATUS.md) records what is implemented, what is being worked on,
and what still depends on external data or model weights.

## Folder rule

- `src/consentguard/stage_*/` contains canonical reusable code.
- `scripts/stage_*/` contains matching command-line entry points.
- `configs/stage_*/` contains matching behavior and training configuration.
- `tests/stage_*/` mirrors each stage's reusable code.
- `reports/` contains generated evidence, never executable logic.
- `artifacts/` contains model checkpoints and other large generated outputs.
- `main_project/` is both the executable project and its human review route.

Do not copy a Python module into a stage folder.  Duplication makes fixes land
in one copy but not another.

## First verification command

```powershell
Set-Location C:\consentGuard
.\.venv\Scripts\python.exe -m pytest -q
```
