# Training architecture decision

## Core model

The reproducible visual baseline is Torchvision `maskrcnn_resnet50_fpn_v2`
initialized from COCO weights and adapted to the nine official visual
attributes plus background. It directly predicts instance masks required by
the destructive renderer. The eight textual attributes belong to an OCR/text
branch; the seven multimodal document attributes belong to a detector plus OCR
branch. Four released labels are ignored by the official protocol. The
official Torchvision tutorial defines the required target fields and head
replacement pattern, while the model documentation reports the improved v2
training recipe and COCO mask AP.

- https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html

## Local 4 GB profile

The local RTX 3050 profile uses batch size 1, four-step gradient accumulation,
AMP, three trainable backbone layers, 640/1024 aspect-preserving input, and
smaller RPN anchors. Polygon coordinates are transformed before rasterization,
so a large source image cannot allocate a full-resolution mask per instance.

The geometry-verified visual training records have a 92.75x instance-count
ratio between the most and least frequent classes. The local profile therefore
uses capped inverse-square-root image sampling. A final study must report a
uniform-versus-balanced ablation rather than silently changing the baseline.

Only ordinary resizes are eligible for training. At a 1% aspect-ratio
tolerance, the release contains 597 aligned train images, 359 aligned
validation images, and 478 aligned test images before modality filtering.
Rotation candidates and arbitrary geometry mismatches remain quarantined until
their coordinate transform or original annotated image is independently
verified.

The standard baseline configuration remains separate and unchanged for a
12–16 GB controlled GPU. This separation prevents hardware accommodations from
being mistaken for the scientific baseline.

## Optional stronger comparison

Mask2Former is the optional higher-capacity comparison because it natively
supports instance segmentation. It is not the local default: overlapping
privacy masks require direct mask-label inputs, and the model has materially
higher memory and dependency cost. It is added only after Mask R-CNN training,
evaluation, rendering, and assurance are working end to end.

- https://huggingface.co/docs/transformers/model_doc/mask2former

SAM/SAM2 is not an automatic privacy detector by itself because it requires a
prompt. It may later refine a detector-proposed region, but it cannot replace
the class-aware localizer in the core experiment.

## Evaluation

Engineering validation uses COCO-style box and segmentation AP@[.50:.95]/AR,
including small, medium, and large regions. Dense masks are converted to compressed COCO RLE
immediately after each prediction, preventing validation memory from growing as
the number of full-resolution masks. The official pycocotools evaluator then
computes bbox and segmentation metrics separately so their area buckets remain
correct. Test data remains locked until the final report.

These COCO metrics are not numerically comparable to the Visual Redactions
paper's AP. The official code thresholds pixel predictions at 50 values,
constructs precision-recall curves, and integrates corrected curves. Both
metric families must be labeled and reported separately.

- https://github.com/cocodataset/cocoapi

## Reproducibility and mixed precision

Every run records seeds, package versions, device information, configuration,
and checkpoint state. AMP uses `torch.amp.autocast` and `torch.amp.GradScaler`.
Deterministic mode is available for debugging, with the documented performance
trade-off.

- https://docs.pytorch.org/docs/stable/amp.html
- https://docs.pytorch.org/docs/stable/notes/randomness
