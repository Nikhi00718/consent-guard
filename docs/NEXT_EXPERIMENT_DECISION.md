# Next experiment decision: moderate class balancing

## Evidence

The repaired uniform five-epoch baseline reached mask mAP 0.1865. Capped
inverse-square-root sampling reached 0.2207, an 18.4% relative improvement.
The balanced curve improved from 0.2207 at epoch five to 0.2244 at epoch ten.
This is a small absolute gain of 0.0036, so the aggressive sampler is near a
plateau rather than clearly continuing to improve.

Balancing improved disability, medicine, fingerprint, signature, and plate,
but reduced face, body, nudity, and handwriting performance. This is a measured
head-versus-tail tradeoff, not evidence for changing architecture yet.

## Completed continuation

The ten-epoch continuation preserved the head-class performance reasonably:
face mAP rose from 0.5801 to 0.5914 and person-body mAP changed from 0.5298 to
0.5277. Tail results improved for nudity, disability, medicine, and license
plate, while fingerprint, signature, and handwriting remain weak. The test
split remains locked.

## Selected next run

Run `configs/train_maskrcnn_verified_moderate_balance_10ep.yaml` from COCO
weights with `class_balance_power: 0.25`. This isolates whether the aggressive
inverse-square-root sampler is over-correcting the head/tail tradeoff.

After this run, compare uniform, moderate, and aggressive sampling using
overall segmentation mAP, per-class mAP/recall, and small-object metrics before
changing the mask head or classifier loss.

## Decision gate after moderate balancing

- Keep moderate balancing if it retains most tail gains with improved head
  recall or better overall mAP.
- If it does not improve the tradeoff, test object-centric crops for small
  privacy regions.
- Calibrate per-class confidence thresholds and inspect validation overlays
  before any test-set evaluation.
- Calibrate per-class confidence thresholds and inspect validation overlays
  before any test-set evaluation.
