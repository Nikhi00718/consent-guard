# Training failure-mode register

This register covers the localizer training milestone. System-level consent,
policy, renderer, and assurance risks remain in
`ConsentGuard_Final_Research_Design.md`.

| Failure | Detection signal | Mitigation implemented | Residual action |
|---|---|---|---|
| Two downloaders write one archive | More than one process targets the same split; gzip error | Exclusive per-split file lock; duplicate launch exits with code 3 | Preserve contaminated file, restart clean, perform full tar read |
| Archive has expected size but corrupt bytes | `zlib`/tar failure during sequential read | Two-pass safe finalizer validates every member before extraction | Never extract or preprocess that archive |
| Path traversal or links in archive | Member resolves outside `data/interim/vispr`, or is a link | Reject member before extraction; Python data extraction filter | Stop and audit source release |
| Partial or missing images | Manifest row has no path | Pending-record file and strict config preflight | Continue only smoke/train-only work; no final evaluation |
| Annotation/image geometry differs | X/Y scales disagree, inverse aspect ratio matches, or overlay misses the object | Accept only aspect-preserving resizes within 1%; quarantine crop/stitch/rotation cases | Recover the exact annotated image or validate an explicit transform before re-entry |
| Invalid polygon or empty rasterized mask | Zero area, malformed coordinates, or empty mask | Exclude the bad instance with reason; fail if an image has no target | Report exclusions; never silently relabel background |
| Full-resolution mask memory explosion | RAM spike on large/many-instance images | Resize/crop polygon coordinates before mask rasterization | Lower crop/long-side limits if needed |
| Train/validation/test leakage | Same Visual Redactions ID in multiple task splits | Select records by official Visual Redactions split; test remains locked | Freeze IDs and hashes before final experiments |
| Duplicate content under different IDs | Exact or near-identical pixels cross official splits | SHA-256 plus 64-bit perceptual-hash cross-split audit | Manually review every near-duplicate candidate before the frozen run |
| Long-tail class collapse | Rare-class recall near zero; 92.75x verified-visual train imbalance | Capped inverse-square-root image sampler | Report uniform/balanced ablation and per-class AP/recall |
| Tiny-object miss rate | Low `map_small`/`mar_small` | FPN, 16 px first anchor, 512 px instance-centred crops | Add OCR/face safety nets later; do not claim complete coverage |
| Wrong modality in one detector | Text/document classes remain near zero despite stable losses | Mask R-CNN is restricted to nine visual attributes; OCR and document branches are separate | Evaluate each modality and ensemble only after branch-level gates pass |
| COCO head class mismatch | State-dict/head shape error or labels out of range | Replace box/mask heads for 10 classes; contiguous class-map gate | Reject mismatched checkpoints |
| CUDA/PyTorch binary mismatch | Missing NMS/C++ ops or CUDA unavailable | Pin matched official torch/torchvision wheels; execute NMS/CUDA preflight | Recreate `.venv`; do not mix wheel indexes |
| RTX 3050 out of memory | `torch.OutOfMemoryError` | 4 GB profile, batch 1, accumulation 4, AMP, 512 crops, reduced proposals | Close GPU apps or reduce crop/proposals; log changed config |
| Non-finite loss or exploding gradient | NaN/Inf loss | Explicit finite check and gradient clipping | Stop run and inspect offending image/config |
| Interrupted checkpoint write | Truncated `.pt` file | Write temporary checkpoint and atomically replace destination | Resume from last valid epoch checkpoint |
| Incorrect resume state | Learning-rate jump or class mismatch | Save/load optimizer, scheduler, scaler, epoch, RNG, class map | Refuse a class-map mismatch |
| Resume silently changes sample order | Balanced/random sampler restarts from its initial RNG state | Save and restore loader and sampler generator tensors | Check resumed step/epoch trace in smoke test |
| Malicious/untrusted checkpoint | Deserialization could execute unsafe payloads | Load only tensor/primitive checkpoint content with `weights_only=True` | Never accept arbitrary third-party `.pt` files |
| Evaluation augmentation contaminates metrics | Run-to-run variation | Validation dataset disables crop/flip/photometric transforms | Keep test untouched until frozen final run |
| Validation masks exhaust host RAM | Dense masks accumulate for every prediction and target | Convert each batch immediately to compressed COCO RLE | Keep standard 100 detections/image and monitor host memory |
| Majority classes hide failures | Good aggregate AP but rare-class failure | Box+mask AP/AR by class and size; severity analysis in final study | Publish per-class support and confidence intervals |
| Threshold tuned on test | Inflated final result | Tune on validation only; inference threshold is explicit | One locked test evaluation after protocol freeze |
| Mask boundary leaks visible content | OCR/re-identification after redaction | Mask dilation and solid replacement, not blur, in inference path | Run independent leakage attacks before safe-release claims |
| Original metadata survives output | GPS/EXIF disclosure | Fresh OpenCV encoding; no metadata copied; independent decode check | Add independent EXIF scanner in assurance milestone |
| Random untrained model appears to “work” | Smoke image is redacted but AP is meaningless | Smoke report is labeled engineering-only; checkpoint selection uses val mask AP | Never report smoke metrics as scientific performance |

## Stop conditions

Do not begin a full scientific run when any of these is true:

- validation records are absent or an archive fails sequential integrity;
- any training record is not marked `geometry_status: aligned_resize`;
- CUDA, TorchVision NMS, pycocotools, or the real-data preflight fails;
- test IDs have been used for threshold or hyperparameter selection;
- the smoke run cannot complete forward, backward, optimizer, evaluation,
  checkpoint, resume, and inference stages;
- free disk cannot hold the planned checkpoints and temporary atomic copies.

## Required evidence before the first full run

1. `pytest` passes in the pinned Python 3.11 environment.
2. `reports/training_environment_preflight.json` has `passed: true`.
3. processed-record validation has no invalid or duplicate IDs.
4. a fixed eight-image Mask R-CNN overfit gate reaches meaningful mask AP.
5. inference creates and reopens a newly encoded redacted image.
6. the resolved full-run config and environment snapshot are archived.
