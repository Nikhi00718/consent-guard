# Mask R-CNN v1 diagnostic

## Conclusion

The v1 run is operationally healthy but underfits. The processed VISPR data,
mask rasterization, model output, and COCO segmentation metric are valid. Do
not continue the v1 recipe; train `configs/train_maskrcnn_4gb_v2.yaml` after
its smoke test passes.

## Evidence

| Check | Result | Interpretation |
| --- | ---: | --- |
| Processed-record validation | 8,472 images / 48,824 instances; pass | All boxes and polygons are valid; no duplicate IDs or paths. |
| Loader smoke test | 100/100 samples; pass | Image tensors, boxes and masks are non-empty and in bounds. |
| Perfect-prediction COCO test | 1.0 mask mAP; pass | The evaluator measures masks correctly. |
| Learning smoke test | loss 0.8664 to 0.7371 | Gradients flow from rasterized masks. |
| v1 validation, epoch 11 best | 0.00237 mask mAP | Far below the project gate. |
| v1 best checkpoint on 100 training images | 0.01179 mask mAP | v1 has weak learning even on seen data. |
| Original-recipe fixed-set ablation | 0.13458 best mask mAP | The original recipe can learn, but slowly and weakly. |
| Improved fixed-set ablation | 0.24458 best mask mAP | Stronger settings improve learnability by 82%. |

## Root causes

1. Only one backbone stage was trainable, which is too little adaptation for
   privacy attributes such as faces, names, documents and text.
2. Training used 75% instance-centred square crops, while validation used full
   images. That changes object scale and context between training and scoring.
3. The initial learning rate was lower and decayed by 10x after epoch 8 while
   validation was still effectively flat.
4. The max-weight class-balanced sampler can over-repeat rare-label images and
   reduce useful coverage of frequent privacy labels. The first fair baseline
   should use uniform image sampling; per-class analysis can drive a later
   targeted sampler.
5. v1 used a smaller 512/768 full-image resolution. v2 raises this to 640/1024
   while retaining batch size one for the 4 GB GPU.

## v2 changes

`configs/train_maskrcnn_4gb_v2.yaml` uses full-image training, 640/1024
resolution, three trainable backbone stages, uniform sampling, SGD at 0.0025,
short warmup, delayed learning-rate decay, and reduced RPN proposal counts.
The v1 configuration and checkpoints are unchanged for reproducibility.

## Guardrails

- Run a v2 one-step GPU smoke test before the full run.
- Evaluate only on validation while choosing the best checkpoint; keep the
  test split sealed until final evaluation.
- At epochs 3-5, require a clear rising validation mAP. At epoch 8, stop and
  diagnose if mask mAP remains below 0.01.
- Never use the fixed-set ablation score as a reported validation result.
