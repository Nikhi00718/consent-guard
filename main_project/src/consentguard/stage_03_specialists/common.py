"""Small shared helpers for specialist evidence providers."""

from __future__ import annotations

import hashlib
import json


def stable_evidence_id(provider: str, provider_version: str, payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    suffix = hashlib.sha256(encoded).hexdigest()[:20]
    return f"{provider}-{provider_version}-{suffix}"
