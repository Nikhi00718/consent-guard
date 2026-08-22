# Stage 05 — Human Review and Safe Export

## Goal

Give the reviewer final control and prevent unchecked downloads.

## Workflow

1. Normalize a supported still image and apply orientation.
2. Analyze it with all configured evidence providers.
3. Display fused candidates and uncertainty.
4. Let the reviewer add, erase, or expand masks.
5. Record review completion and policy reasons.
6. Render solid redaction into a new RGB buffer.
7. Encode a fresh JPEG, PNG, or WebP without source metadata.
8. Independently inspect the output.
9. Enable export only if mandatory assurance checks pass.

Blur and pixelation are not safe defaults.  Missing provider dependencies or
unresolved consent never become permission.
