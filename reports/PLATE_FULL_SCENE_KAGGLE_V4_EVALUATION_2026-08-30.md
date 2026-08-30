# Full-scene Indian plate candidate: Kaggle v4

## Decision

Kaggle version 4 completed successfully and produced a useful research
checkpoint, but it is **not promoted to the website default**. It passed the
frozen merged-validation recall and false-positive requirements and passed the
Deepak false-positive requirement. It missed the precommitted Deepak `vid-1`
recall requirement: 0.4118 observed versus 0.50 required at score threshold
0.50 and IoU 0.50.

Changing the score threshold after seeing the locked challenge would invalidate
the gate. The existing India checkpoint therefore remains the website default,
while the new checkpoint is retained locally and in the private Kaggle kernel
output for the next research iteration.

## Why this model was trained

The existing plate specialist was adapted from CCPD2020 and performed well on
its narrow validation set, but transferred poorly to complete Indian road
scenes. On the independent Deepak `vid-1` challenge it found only 1 of 68 plate
boxes at score 0.50 and produced 44 false positives. The full-scene candidate
was designed to add Indian plate appearances, real road context, and many
negative images without using the challenge or any test split for training.

This is a trainable Faster R-CNN specialist. YuNet remains a separately wired
OpenCV model used for complementary proposals; the project does not have the
original YuNet training recipe and landmark labels, so this experiment did not
pretend to retrain YuNet.

## Audited data

The merged research records contain:

| Split | Images | Positive images | Negative images | Plate boxes |
|---|---:|---:|---:|---:|
| Train | 1,904 | 1,473 | 431 | 1,708 |
| Validation | 1,824 | 331 | 1,493 | 382 |

Sources were the leakage-safe grouped Roboflow `nivu/indian-license-plate-knte7`
v1 export, Visual Redactions plate positives and negatives, and 117 grouped
Deepak `vid-2`/`vid-3` road frames in training. Deepak `vid-1` stayed outside
training as an independent 43-image, 68-box challenge. Exact-image hash overlap
between train and validation is zero, and no test records entered the Kaggle
transport.

The original Roboflow publisher split was not used because its audit found 17
source/duplicate groups crossing publisher split boundaries. The grouped
replacement has no group overlap. The source is publisher-marked CC BY 4.0;
Visual Redactions and every other source retain their own original terms.

Kaggle transport verification:

- 3,728 unique train/validation images
- 4,674,069,651 image bytes
- train-record SHA-256 `91e9f28fe2e99f6dace2f4ae3feb5a4ced973f3f4671969bfbffafb435993ced`
- validation-record SHA-256 `d0e9c5969201f15a7cc0a0a5c473ed6ae731748700e105cab77295a9625ccc5f`
- cross-split hash leakage: 0
- test split used: false

## Training

The candidate used Faster R-CNN ResNet-50 FPN v2 with small-object anchors,
800-pixel short-side / 1,333-pixel maximum resize, 768-pixel context crops,
batch size 1, gradient accumulation 4, and five epochs. It initialized from the
existing Indian-adapted plate checkpoint:

`d8a7a551fe3a9f264bb1ad34066f583d92ae9344503112a582554e29975a81b8`

Kaggle assigned a Tesla P100. Its default PyTorch 2.10 CUDA 12.8 build did not
contain P100 (`sm_60`) kernels, so the launcher installed the official
PyTorch 2.7.1 CUDA 11.8 build and verified a real CUDA tensor allocation before
training. The completed run used 2,380 optimizer steps and peaked at about
10.66 GB allocated CUDA memory.

Training output:

- box mAP: 0.58097
- box mAP@0.50: 0.73942
- small-object box mAP: 0.05667
- medium-object box mAP: 0.53324
- large-object box mAP: 0.76816
- candidate checkpoint SHA-256: `45181310c47940361128aa2478da523829e7b3a9fa5f24d6a2b5335884ec6bcd`
- candidate checkpoint size: 346,511,469 bytes

## Frozen comparisons

All figures below use score threshold 0.50 and IoU 0.50.

| Frozen set / model | TP | FP | FN | Precision | Recall | F1 | FP/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Merged validation — current | 261 | 1,196 | 121 | 0.1791 | 0.6832 | 0.2838 | 0.6557 |
| Merged validation — candidate | 295 | 333 | 87 | 0.4697 | 0.7723 | 0.5842 | 0.1826 |
| Deepak `vid-1` — current | 1 | 44 | 67 | 0.0222 | 0.0147 | 0.0177 | 1.0233 |
| Deepak `vid-1` — candidate | 28 | 7 | 40 | 0.8000 | 0.4118 | 0.5437 | 0.1628 |

The candidate is a major improvement: on Deepak it changes 1 TP / 44 FP into
28 TP / 7 FP. It still misses 40 of 68 plates at the frozen operating point,
which is why it cannot replace the website model under the existing rule.

## What is integrated

- The Kaggle launcher now handles Kaggle-expanded code datasets and incompatible
  P100 runtimes deterministically.
- Both current and candidate checkpoints were evaluated with the same frozen
  evaluator, records, score thresholds, and IoU rule.
- The verification report re-hashes the records, initialization checkpoint,
  candidate checkpoint, and evaluator checkpoint references.
- The website retains the current India checkpoint. No hidden threshold change
  or partial promotion was made.

The 346.5 MB binary is intentionally not committed to ordinary Git because it
exceeds GitHub's normal file limit. It remains recoverable from the private
Kaggle kernel `nikhil00718/consentguard-plate-full-scene-training` and is also
stored locally under the ignored `artifacts/kaggle/remote-runs/plate-full-scene-v4`
tree. Git stores the code, configuration, hashes, metrics, and decision evidence.

## Next plate target

Do not tune against Deepak `vid-1`; it is now a locked challenge. The next
iteration should add new, rights-cleared Indian road scenes with genuinely tiny,
blurred, oblique, night, and partially occluded plates plus plate-like hard
negatives. Create a separate calibration split before training, select the score
threshold only on that split, and evaluate once on a new untouched challenge.
The main weakness visible in this run is small-object mAP (0.0567), not a lack of
capacity on medium and large plates.

## Reproduction and evidence

```powershell
.venv\Scripts\python.exe main_project/scripts/stage_03_specialists/verify_plate_full_scene_kaggle_run.py
```

Machine-readable evidence:

- `reports/plate_full_scene_kaggle_v4_verification.json`
- `reports/plate_full_scene_v4_merged_validation.json`
- `reports/plate_full_scene_v4_deepak_challenge.json`
- `reports/roboflow_nivu_indian_plate_v1_audit.json`
