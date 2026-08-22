# ConsentGuard: Issues and Research Checklist

**Project:** ConsentGuard visual privacy redaction  
**Dataset:** Official Visual Redactions v1 release  
**Current model:** TorchVision Mask R-CNN ResNet-50 FPN v2  
**Current best experiment:** Class-balanced five-epoch run  
**Status:** Dataset integrity is controlled; performance and safety research remain open

## Executive conclusion

The dataset is not fundamentally broken and should not be replaced immediately.
The current difficulties are mainly:

1. Extreme long-tail class imbalance.
2. Very small privacy regions.
3. Low validation support for rare classes.
4. Tradeoffs caused by aggressive class balancing.
5. Incomplete privacy coverage because the current model handles only nine
   visual classes.
6. Missing calibration and end-to-end privacy-utility evaluation.

The original data-identity failure has been repaired. The next work should focus
on training recipes, small-object handling, rare-class evaluation, and safety
thresholds before changing datasets or architectures.

## 1. Dataset integrity issues

These caused the original failed training run and are now controlled.

### Wrong image/annotation pairing — fixed

The previous pipeline paired Visual Redactions annotations with unrelated VISPR
pixels. This produced contradictory supervision and invalid model results.

The corrected pipeline now joins only images and annotations from the same
Visual Redactions release and matching split.

### Cross-split duplicate images — fixed

Exact and perceptual duplicate scenes were found across train, validation, and
test. Four train-side records were removed while validation and test references
were preserved.

The cleaned leakage audit reports zero exact duplicates and zero perceptual-hash
candidates at the configured threshold.

### Geometry and orientation mismatch — controlled

The release contains images whose decoded dimensions do not safely match the
annotation geometry because of orientation metadata. These records are
quarantined instead of receiving unsafe mask scaling.

- 202 orientation-risk images are quarantined.
- 8,267 images pass the aligned-resize geometry rule.

Future research question: independently verify whether some quarantined images
can be recovered with a correct orientation transform.

### Omitted images — expected but worth documenting

1,585 images do not contain a valid instance from the selected nine-class visual
profile and are omitted from model-ready records. This is acceptable for the
current profile, but it means the training corpus is not identical to the full
release.

## 2. Active data problems

### Extreme class imbalance

The repaired training split contains:

| Class | Train instances | Validation instances |
|---|---:|---:|
| Person body | 5,852 | 2,442 |
| Face | 3,992 | 1,671 |
| Handwriting | 1,040 | 450 |
| License plate | 370 | 134 |
| Nudity | 316 | 185 |
| Signature | 238 | 85 |
| Medicine | 116 | 121 |
| Fingerprint | 78 | 11 |
| Disability | 73 | 21 |

The most common class has roughly 80 times more training instances than some
tail classes. This causes head classes to dominate the classifier and mask
losses.

Relevant research:

- [Balanced Group Softmax for long-tail object detection](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Overcoming_Classifier_Imbalance_for_Long-Tail_Object_Detection_With_Balanced_Group_Softmax_CVPR_2020_paper.html)
- [Pairwise Class Balance for long-tailed instance segmentation](https://openaccess.thecvf.com/content/CVPR2022/html/He_Relieving_Long-Tailed_Instance_Segmentation_via_Pairwise_Class_Balance_CVPR_2022_paper.html)
- [Classification Equilibrium for long-tailed object detection](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_Exploring_Classification_Equilibrium_in_Long-Tailed_Object_Detection_ICCV_2021_paper.html)

### Very small validation support

Some validation classes are too small for stable conclusions:

- Fingerprint: 11 instances.
- Disability: 21 instances.
- Medicine: 121 instances across only 16 images.
- License plate: 134 instances.

A few correct or incorrect predictions can change their mAP substantially.
Research should report per-class confidence intervals or repeat experiments with
multiple seeds.

### Tiny privacy regions

Fingerprints, signatures, handwriting, medicine labels, and license plates can
occupy very few pixels. Current small-object segmentation mAP is approximately
0.0845.

Research directions:

- Higher-resolution inputs.
- Object-centric crops.
- Copy-Paste augmentation.
- Larger or higher-resolution FPN features.
- Dedicated small-object or document branches.
- Separate OCR/document models.

Relevant research:

- [Simple Copy-Paste for instance segmentation](https://openaccess.thecvf.com/content/CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html)

### Possible annotation ambiguity

The current audits validate structure and geometry, but they do not prove that
every semantic boundary is consistent. Potentially ambiguous cases include:

- Face versus person body.
- Nudity versus body.
- Medicine object versus text printed on medicine.
- Signature versus handwriting.
- Overlapping privacy regions.
- Partially occluded or cropped private objects.

Research action: manually review rare-class masks and, if possible, measure
inter-annotator agreement on a small sample.

### Dataset domain coverage

The Visual Redactions paper describes an 8.5k-image dataset derived from a
subset of images with at most five people. It is diverse, but it may not fully
represent:

- Crowded images.
- Modern mobile-camera images.
- Low-light images.
- Video frames.
- Scanned documents.
- Screenshots and social-media compression.
- Production images from the intended application.

The dataset is still suitable for the current visual baseline. This is a domain
coverage limitation, not evidence that the release is invalid.

Reference: [Connecting Pixels to Privacy and Utility](https://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Orekondy_Connecting_Pixels_to_CVPR_2018_paper.pdf)

## 3. Model and training issues

### Aggressive class balancing creates a head-versus-tail tradeoff

The uniform five-epoch run reached:

- Segmentation mAP: 0.1865.
- AP50: 0.3041.

The class-balanced five-epoch run reached:

- Segmentation mAP: 0.2207.
- AP50: 0.3575.

However, class-balanced sampling reduced some head-class scores while improving
tail classes:

| Class | Uniform mAP | Balanced mAP |
|---|---:|---:|
| Face | 0.6148 | 0.5801 |
| Person body | 0.5537 | 0.5298 |
| Disability | 0.0679 | 0.1783 |
| Medicine | 0.0269 | 0.1197 |
| Fingerprint | 0.0047 | 0.1799 |

The next sampling experiment should test moderate balancing instead of
immediately changing the architecture.

### Training is approaching a plateau

The class-balanced continuation reached approximately:

- Five-epoch mAP: 0.2207.
- Epoch-seven mAP: 0.2237.

This is still a small improvement, not a complete failure. Possible causes are
learning-rate decay, limited rare-class information, sampler variance, and
small-object difficulty.

### No object-centric crop training in the main run

The main experiments use full-image training with `crop_probability: 0.0`.
This preserves context but leaves small privacy regions small. Crop training
should be tested as an isolated ablation.

### COCO pretraining domain gap

Mask R-CNN starts from COCO weights, which is a sound transfer-learning
baseline. However, COCO does not provide strong visual features for:

- Fingerprints.
- Signatures.
- Medicine labels.
- Privacy-specific handwriting.
- Document fields.

This may require longer fine-tuning, targeted augmentation, specialist branches,
or a stronger privacy-specific pretrained model.

### Small effective batch size

The local RTX 3050 profile uses batch size 1 with four-step gradient
accumulation and AMP. This fits the hardware but creates noisy optimization and
limits high-resolution training.

Possible research directions:

- More gradient accumulation.
- Larger GPU for higher resolution.
- Gradient checkpointing.
- Specialist models trained on cropped regions.

### Tail oversampling may cause memorization

Class-balanced sampling repeatedly shows the same rare images to the model.
This can improve recall while reducing generalization. Research should record:

- Unique images sampled per epoch.
- Per-class sampling frequency.
- Train-versus-validation performance for each rare class.
- Whether tail improvements survive a new random seed.

## 4. Evaluation and privacy-safety issues

### COCO mAP is not the complete privacy metric

A good detection score does not guarantee that sensitive pixels are never missed.
Privacy evaluation must also measure:

- Per-class recall.
- False-negative rate.
- Worst-case examples.
- Mask coverage of sensitive regions.
- Confidence thresholds.
- Redaction effectiveness.

### Confidence calibration is missing

The model currently produces scores, but thresholds have not yet been calibrated
for a privacy-risk operating point. A production redactor should distinguish:

- Confidently safe.
- Confidently sensitive.
- Uncertain and requiring human review.

Relevant research:

- [Uncertainty estimation in instance segmentation](https://openaccess.thecvf.com/content/WACV2024/html/Siddiqui_Uncertainty_Estimation_in_Instance_Segmentation_With_Star-Convex_Shapes_WACV2024_paper.html)

### No unseen-privacy detection

The model knows only the nine trained visual classes. It may miss a new privacy
category while appearing confident.

Future safeguards:

- Unknown-risk or abstention output.
- Human-review queue.
- Open-set testing.
- Out-of-distribution validation.

### No complete privacy-utility evaluation yet

The system still needs to measure:

- How much sensitive information remains after redaction.
- How much useful visual information is destroyed.
- Whether redaction boundaries leak clues.
- Whether the result remains useful to the user.

The original Visual Redactions work treats automatic redaction as a
privacy-utility tradeoff, not only as a segmentation task. See the [official
project page](https://resources.mpi-inf.mpg.de/d2/orekondy/redactions/).

### Current model covers only visual classes

The current model covers nine visual attributes. It does not yet cover:

- OCR-based names, addresses, usernames, and text.
- Passport and identity-document fields.
- Contextual or relationship privacy.
- Video and temporal privacy.

The complete system therefore requires additional OCR, document, multimodal,
and possibly video branches.

## 5. Reproducibility and experiment issues

### Only limited seeds have been tested

One main seed is not enough for rare-class conclusions. At least three seeds
should be used for final comparisons.

### Rare-class confidence intervals are missing

Per-class mAP should be reported with uncertainty, especially for classes with
fewer than 25 validation images.

### Test split must remain locked

The test split must not be used for sampler selection, crop selection,
threshold calibration, or architecture decisions. It should be used only after
the experiment is frozen.

### Model comparison must use identical protocols

Uniform sampling, balanced sampling, crop training, and architecture changes
must use the same:

- Dataset version.
- Validation split.
- Class map.
- Resolution protocol.
- Evaluation code.
- Test-lock policy.

## 6. Recommended research order

### Priority 1: Finish the current continuation

Complete the class-balanced ten-epoch run and record whether mAP meaningfully
improves beyond 0.2207.

### Priority 2: Moderate balancing

Test:

```yaml
class_balance_power: 0.25
max_class_balance_ratio: 3.0
```

Keep every other setting fixed.

### Priority 3: Small-object crop ablation

Test object-centric crops while keeping the model and sampler fixed. Measure
small-object mAP and per-class recall, especially for handwriting, signatures,
fingerprints, medicine, and plates.

### Priority 4: Copy-Paste augmentation

Use only training masks, never validation/test masks. Compare rare-class recall
and overall mAP against the moderate-balancing run.

### Priority 5: Calibration and privacy metrics

Choose per-class thresholds using validation data, then measure privacy recall,
false negatives, redaction coverage, and utility preservation.

### Priority 6: Specialist branches

Add OCR/document and small-object specialist models after the visual baseline is
stable.

### Priority 7: Architecture comparison

Only after the data and training ablations should Mask2Former or another
instance-segmentation architecture be compared. A model switch alone will not
solve label scarcity or domain coverage.

### Priority 8: Dataset expansion

Consider VPD-100K or another dataset only after verifying that its annotations,
image modality, privacy classes, and license match the intended experiment.
VPD-100K is a possible future domain-expansion resource, not a replacement for
the currently verified still-image benchmark.

## Current decision

Keep the official Visual Redactions dataset and Mask R-CNN baseline.

Do not switch to VISPR pixels or unverified VPD data.

Do not evaluate the locked test split yet.

Focus research on moderate long-tail balancing, small-object crops,
Copy-Paste augmentation, calibration, and privacy-utility evaluation.
