# ConsentGuard experiment versioning

The repository stores reproducible code, configurations, validation reports,
and small run summaries. Raw media, processed image records, caches, and
generated TensorBoard files stay local and are excluded by `.gitignore`.

## Branch and tag policy

- `main` is the reviewed project baseline.
- Use `agent/<experiment-name>` for each new model or data experiment.
- Merge only after tests, preflight, and the relevant training gates pass.
- Tag completed milestones as `v<major>.<minor>.<patch>-<experiment>`.
- Keep each experiment in its own output directory so checkpoints are never
  silently mixed.

## Model recovery

The best checkpoint for each completed experiment is tracked with Git LFS. To
recover a model after cloning:

```powershell
git lfs install
git clone https://github.com/Nikhi00718/consent-guard.git
git checkout <experiment-tag>
git lfs pull
```

The matching resolved configuration, environment snapshot, metrics, and run
summary must be committed beside the checkpoint. This makes a checkpoint
usable only with the architecture and class map that produced it.

## Current experiments

| Version | Branch | Experiment | Status | Primary result |
|---|---|---|---|---|
| `v0.1.0-verified-baseline` | `main` | Uniform five-epoch Mask R-CNN | Complete | Segmentation mAP 0.1865; AP50 0.3041 |
| `v0.2.0-class-balanced-5ep` | `agent/class-balanced-ablation` | Capped inverse-square-root image sampling | Running | To be measured on the same validation split |

Never evaluate or select a model using `test2017` before the experiment and
privacy threshold are frozen.
