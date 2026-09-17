# Stage 01 — Data and Provenance

## Goal

Produce geometry-safe, split-safe training records with explicit background
examples.  A model cannot learn realistic false-positive behavior if every
training image contains a privacy object.

## Canonical files

- [`scripts/stage_01_data/preprocess_visual_redactions_verified.py`](../scripts/stage_01_data/preprocess_visual_redactions_verified.py)
- [`src/consentguard/stage_01_data/dataset.py`](../src/consentguard/stage_01_data/dataset.py)
- [`scripts/stage_01_data/validate_processed_records.py`](../scripts/stage_01_data/validate_processed_records.py)
- [`tests/stage_01_data/test_dataset.py`](../tests/stage_01_data/test_dataset.py)

## Data contract

- A positive record has one or more validated `instances`.
- A genuine background record has `instances: []` and
  `negative_for_profile: true`.
- A record whose selected annotations failed geometry validation is omitted;
  it must never be relabelled as background.
- Raw archives and the official test split are never modified.

## Run order

```powershell
.\.venv\Scripts\python.exe main_project\scripts\stage_01_data\preprocess_visual_redactions_verified.py --profile visual
.\.venv\Scripts\python.exe main_project\scripts\stage_01_data\validate_processed_records.py `
  --data data\processed\visual_redactions_verified_visual `
  --report reports\processed_records_verified_visual_validation.json
.\.venv\Scripts\python.exe main_project\scripts\stage_01_data\profile_training_records.py `
  --records data\processed\visual_redactions_verified_visual\records_train2017.jsonl `
  --output reports\training_data_profile.json
```

## Review checklist

- Confirm negative records are counted separately.
- Confirm invalid selected annotations are not treated as negatives.
- Confirm each report contains the input-record SHA-256 and generation time.
- Confirm split leakage and image-path rules still pass.
