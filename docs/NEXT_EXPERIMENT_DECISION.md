# Next experiment decision: object-centric crops

## Evidence

The repaired uniform five-epoch baseline reached mask mAP 0.1865. Capped
inverse-square-root sampling reached 0.2207, an 18.4% relative improvement.
The aggressive class-balanced run improved from 0.2207 at epoch five to 0.2244
at epoch ten. This is a small absolute gain of 0.0036, so the aggressive
sampler is near a plateau rather than clearly continuing to improve.

Balancing improved disability, medicine, fingerprint, signature, and plate,
but reduced face, body, nudity, and handwriting performance. This is a measured
head-versus-tail tradeoff, not evidence for changing architecture yet.

## Completed continuation

The ten-epoch continuation preserved the head-class performance reasonably:
face mAP rose from 0.5801 to 0.5914 and person-body mAP changed from 0.5298 to
0.5277. The test split remains locked.

## Completed moderate-balancing ablation

The moderate run used `class_balance_power: 0.25` from COCO weights and reached
best validation segmentation mAP **0.2335** at epoch five, versus **0.2244**
for aggressive balancing. AP50 was 0.3755 and bbox mAP was 0.2576. Moderate
balancing improved face, license plate, handwriting, fingerprint, and
signature relative to the aggressive run, while medicine and disability were
slightly lower. This is the best current validation checkpoint.

## Selected next run

Run an object-centric crop ablation using the moderate sampler and the same
model, split, and optimizer. The current small-region segmentation mAP remains
only 0.0751, so improving spatial resolution for rare/small regions is a more
direct next test than changing the mask head or classifier loss.

## Decision gate after object-centric crops

- Keep the crop recipe if small-region mAP and tail recall improve without a
  material drop in head-class recall.
- If crops do not help, then test the class-agnostic mask head as an isolated
  architecture ablation.
- Calibrate per-class confidence thresholds and inspect validation overlays
  before any test-set evaluation.
