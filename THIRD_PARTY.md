# Third-party data and software

## Datasets

- VISPR images: use the terms and citation on the
  [Visual Privacy Advisor project page](https://tribhuvanesh.github.io/vpa/).
- Visual Redactions masks: use the terms and citation on the
  [Visual Redactions project page](https://resources.mpi-inf.mpg.de/d2/orekondy/redactions/).
- VPD public videos are optional and outside the image-localizer critical path.
  Their presence does not imply that the unavailable VPD-100K image-box release
  is present.

Do not commit or redistribute downloaded media from this repository. Verify the
upstream terms with the institution before training or publication.

## Software

- PyTorch and TorchVision — BSD-style license.
- OpenCV — Apache 2.0.
- pycocotools — BSD-style license.
- Hugging Face Transformers (optional Mask2Former comparison) — Apache 2.0.
- NudeNet (optional intimate-content detector, off unless installed) —
  **AGPL-3.0**. Local personal use is fine; enabling it in a hosted service
  extends the AGPL source-offer obligation to that service.

Exact installed versions are captured in each run's `environment.json`; retain
the corresponding upstream license notices when distributing software.
