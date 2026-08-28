# Kaggle training runbook

Kaggle is the next compute step for ConsentGuard, but remote submission requires
the user's own Kaggle account/API credentials. No Kaggle credential is stored
in this repository or in chat.

## Prepare the private dataset

Run the dry-run manifest locally:

```powershell
$env:PYTHONPATH='C:\consentGuard\main_project\src'
.venv\Scripts\python.exe main_project/scripts/stage_02_baseline_model/prepare_kaggle_data.py
```

The manifest covers 5,361 unique baseline train/validation images (about 10.1 GiB)
and no test records. Specialist models deliberately use separate sources. To materialize the package for a
private Kaggle Dataset, add `--copy --dataset-dir <directory>` and upload that
directory as a private Dataset. Do not upload the locked test split.

## Run one component per session

Upload the repository code and the private Dataset, then run from the Kaggle
Notebook terminal. The Dataset mount must contain `data/processed`:

```bash
pip install -e .
export PYTHONPATH=/kaggle/working/consent-guard/main_project/src
python main_project/scripts/stage_02_baseline_model/run_kaggle_training.py \
  --component baseline \
  --seeds 1337 \
  --data-root /kaggle/input/consentguard-v2-trainval
```

Repeat with `face`, `plate`, and `handwriting`. Supply WIDER FACE with
`--face-root`, the Indian plate Kaggle dataset with `--plate-root`, and the
official HierText train/validation files with `--handwriting-root`. The helper
`notebooks/kaggle/consentguard_train.py` attaches/downloads these sources.
For finalist evidence, repeat
each component with seeds `1337 2027 31415` in separate checkpointed sessions.
The runner writes an atomic run manifest, per-run logs, and checkpoints under
`artifacts/`.

`--component all` exists for deliberate experiments, but it is not the default:
the handbook requires one justified experiment per free GPU session and
checkpoint/resume support.

The prepared publisher creates separate `baseline`, `face`, `plate`, and
`handwriting` kernels so a Kaggle time limit cannot erase every component in a
single run:

```powershell
.venv\Scripts\python.exe main_project\scripts\stage_02_baseline_model\publish_kaggle_assets.py `
  --username YOUR_KAGGLE_USERNAME --upload-datasets --push-kernel
```

The command requires a private `C:\Users\atnik\.kaggle\kaggle.json` credential;
never add that file to the repository.

## First new experiment: official CCPD2020 plate detector

The first replacement experiment is isolated from the legacy Indian plate
mirror. Build the current code bundle, stage the official CCPD2020 archive,
and publish one private dataset plus one GPU kernel:

```powershell
.venv\Scripts\python.exe main_project/scripts/stage_02_baseline_model/prepare_kaggle_bundle.py `
  --config main_project/configs/stage_03_specialists/train_plate_ccpd2020_fasterrcnn.yaml `
  --output artifacts/kaggle/consentguard-training-code-ccpd2020.zip
.venv\Scripts\python.exe main_project/scripts/stage_02_baseline_model/publish_ccpd2020_kaggle_job.py `
  --username nikhil00718 --upload --push
```

The kernel is `nikhil00718/consentguard-plate-ccpd2020-training`. It uses only
CCPD2020 train/validation records (5,769/1,001 images) and does not attach the
old Indian plate dataset. If Kaggle is still indexing the new private dataset,
wait for it to become ready and rerun the same publisher with `--push` only.
Do not start a second
unrelated component while this plate run is active.

## Important limits

- Kaggle GPU access and quotas are account-dependent; configure credentials
  privately rather than committing an API key.
- The current project code uses one GPU. T4×2 does not automatically double
  throughput without a multi-GPU launcher.
- Target-2K general/India data still requires licensing and an approved frozen
  manifest before it can enter release evidence.
- WIDER FACE and HierText are research-license sources in this plan. Their
  checkpoints cannot be relabeled as production-ready models.
