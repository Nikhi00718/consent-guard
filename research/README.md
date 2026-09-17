# research/

Training and experiment machinery. **None of this is needed to run the tool** —
see the top-level [README](../README.md) for that.

## What lives here

- `notebooks/kaggle/` — the Kaggle GPU job entry points used to train the
  specialist detectors (face, plate, handwriting) on a rented P100.
- `notebooks/ConsentGuard_Full_Data_Model_Audit.ipynb` — the dataset/model audit
  notebook.

## What deliberately did not move here

The training scripts, configs and tests stay in `main_project/scripts/stage_*`
and `main_project/configs/stage_*`. Two reasons:

1. `baselines/baseline-v0.1.json` pins the frozen comparator by **path and
   SHA-256** (checkpoint, training config, class map, validation records).
   Moving those files breaks the hash check that proves which artifacts produced
   the recorded numbers.
2. The dated reports in `reports/` record the exact commands that were run.
   Rewriting paths inside them would make the evidence trail describe something
   that never happened.

So the boundary is documented rather than physical: anything under
`stage_02_baseline_model` (except `setup_environment.ps1` and
`preflight_environment.py`) and the `prepare_*`/`publish_*`/`queue_*`/`train_*`
scripts under `stage_03_specialists` are training-time tools. The app never
imports them.

## If you do train again

1. Read `docs/archive/TRAINING_FAILURE_MODES.md` first — it lists the stop
   conditions that were learned the hard way.
2. Write the failure hypothesis down before changing a config.
3. Keep the Visual Redactions test split locked; select on validation only.
4. Record the result in `reports/`, whether it worked or not.
