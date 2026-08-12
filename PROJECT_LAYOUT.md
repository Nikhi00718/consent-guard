# ConsentGuard workspace layout

The raw downloads are preserved under `data/raw` and are never used as scratch space.

```text
data/
  raw/          Original archives, extracted source images, annotations, and VPD files
  interim/      Temporary normalized files produced during processing
  processed/    Model-ready data and converted masks
  manifests/    Image/annotation indexes and audit manifests
  splits/       Frozen leakage-safe train/validation/test manifests
  cache/        Rebuildable caches only

src/consentguard/
  ingest/       Secure image decoding and metadata inspection
  perception/   Privacy localization and face safety net
  ocr/          Local OCR and structured-value validators
  policy/       Consent schema, scope matching, and release decisions
  redaction/    Destructive rendering and metadata-free export
  assurance/    Independent post-export privacy checks

artifacts/      Checkpoints, metrics, and visual diagnostics
outputs/        Redacted/review images and run logs
configs/        Reproducible experiment configuration
tests/          Unit, renderer, policy, and end-to-end tests
reports/        Audit and experiment reports
```

The first proof-of-concept milestone is a validated image/annotation manifest and a
renderer that can safely export an image using a known mask. Model training comes
after that gate.
