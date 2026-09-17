# docs/archive/

Historical documents. They record decisions and measurements that were true when
written, and several are explicitly superseded. Nothing here is needed to run
the tool.

Kept because they are the project's evidence trail — reasons, failures and
numbers that would otherwise have to be rediscovered.

| Document | What it is | Still true? |
|---|---|---|
| `CONSENTGUARD_ISSUES_AND_RESEARCH_CHECKLIST.md` | The open problems list: class imbalance, tiny regions, weak rare classes | Mostly, as a description of the model's limits |
| `TRAINING_FAILURE_MODES.md` | Failure register and stop conditions for training runs | Yes — read before training anything |
| `TRAINING_ARCHITECTURE.md` | Why Mask R-CNN ResNet-50 FPN v2, and the 4 GB profile | Yes |
| `EXPERIMENT_VERSIONING.md` | Branch/tag policy and the experiment results table | Historical |
| `NEXT_EXPERIMENT_DECISION.md` | Which ablation to run next, and why | Superseded: the model chase stopped |
| `V3_PREPROCESSING_FIX_AND_TRAINING_REPORT.md` | The geometry bug that invalidated v1/v2 training, and its fix | Yes — explains why the data pipeline is strict |
| `END_TO_END_IMPLEMENTATION_PLAN.md` | The six-stage build plan | Historical |
| `KAGGLE_MULTI_MODEL_TRAINING_PLAN.md` | How the specialist models were trained on Kaggle | Historical |
| `TRAINING_SETUP_REPORT.md` | Readiness evidence for the first full run | Invalidated by the geometry fix above |
| `ConsentGuard_Project_Handoff.md` | Earlier handoff; points at the research design | Superseded |
| `DATASET_DOWNLOAD_GUIDE.md` | How the datasets were acquired and validated | Yes, if you ever re-download |
| `DATASET_DOWNLOAD_STATUS.md` | Download/validation state as of August 2026 | Historical snapshot |
| `PROJECT_LAYOUT.md` | The originally proposed folder layout | Superseded by the real layout |

The research plan itself stays at the top level:
[`ConsentGuard_Final_Research_Design.md`](../../ConsentGuard_Final_Research_Design.md).
It describes a stricter, publication-oriented system than the tool that was
built; the differences are listed under "Honest status" in the main README.
