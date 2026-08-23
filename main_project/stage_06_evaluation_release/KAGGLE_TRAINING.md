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

The manifest covers 5,361 unique train/validation images (about 10.1 GiB),
all specialist records, and no test records. To materialize the package for a
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

Repeat with `face`, `plate`, and `handwriting`. For finalist evidence, repeat
each component with seeds `1337 2027 31415` in separate checkpointed sessions.
The runner writes an atomic run manifest, per-run logs, and checkpoints under
`artifacts/`.

`--component all` exists for deliberate experiments, but it is not the default:
the handbook requires one justified experiment per free GPU session and
checkpoint/resume support.

## Important limits

- Kaggle GPU access and quotas are account-dependent; configure credentials
  privately rather than committing an API key.
- The current project code uses one GPU. T4×2 does not automatically double
  throughput without a multi-GPU launcher.
- Target-2K general/India data still requires licensing and an approved frozen
  manifest before it can enter release evidence.
