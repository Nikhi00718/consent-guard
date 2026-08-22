# Stage 05 — Human Review and Safe Export

## Goal

Give the reviewer final control and prevent unchecked downloads.

## Workflow

1. Normalize a supported still image and apply orientation inside an isolated session.
2. Analyze it with all configured evidence providers.
3. Display fused candidates and uncertainty.
4. Let the reviewer add, erase, or expand masks.
5. Resolve explicit, scope-bound consent and record review completion with policy reasons.
6. Render solid redaction into a new RGB buffer.
7. Encode a fresh JPEG, PNG, or WebP without source metadata.
8. Independently inspect the output.
9. Enable export only if mandatory assurance checks pass.

Blur and pixelation are not safe defaults.  Missing provider dependencies or
unresolved consent never become permission.

## End-to-end service

`src/consentguard/stage_05_review_export/pipeline.py` composes normalization,
the sequential specialist orchestrator, provenance registry, thresholded
fusion, destructive rendering, independent assurance, and deterministic policy.
The framework-independent command is
`scripts/stage_05_review_export/run_analysis_pipeline.py`. It can emit evidence
and review candidates without exporting, or accept a reviewer-approved mask for
a newly encoded output. Missing specialist providers remain explicit and keep
the decision fail-closed.
