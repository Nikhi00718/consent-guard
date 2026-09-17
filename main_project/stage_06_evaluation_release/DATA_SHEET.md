# ConsentGuard validation bundle data sheet

## Admitted data

The current trainable bundle uses the same-release Visual Redactions V2
verified records. The train split contains 3,785 images, including 726
negative images; the validation split contains 1,576 images. The test split is
structurally present but locked and is not used for training, threshold tuning,
qualitative selection, or the fused validation evaluator.

Specialist profiles are one-class projections of the verified V2 records and
retain sampled negative images. Their manifests record source-record hashes,
class mappings, and `test_split_used: false`.

## Data quality and provenance

The Visual Redactions image archives were downloaded from the official release,
decoded, and checked against annotation dimensions and split identity. The
processed validation reports and same-release leakage audit are retained in
`reports/`. Raw images and processed JSONL records remain local ignored data
artifacts rather than being copied into Git.

## Missing target domains

The release gates require general and India domain recall for faces, plates, and
text. The current Visual Redactions records do not carry those domain labels,
and no licensed Target-2K general/India manifest has been admitted. Therefore
the release candidate is blocked rather than filled with guessed domain labels.

## Privacy and retention

Images may contain faces, plates, handwriting, medical material, signatures,
fingerprints, and other sensitive content. Keep raw data local, minimize
access, retain only what the experiment needs, and support withdrawal. Dataset
licences do not replace participant or jurisdiction-specific ethics review.
