"""Verify every file referenced by a frozen ConsentGuard baseline manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for artifact in payload.get("artifacts", []):
        path = Path(artifact["path"])
        path = path if path.is_absolute() else ROOT / path
        result = {
            "role": artifact["role"],
            "path": str(path),
            "exists": path.is_file(),
            "bytes_match": False,
            "sha256_match": False,
        }
        if path.is_file():
            result["bytes_match"] = path.stat().st_size == int(artifact["bytes"])
            result["sha256_match"] = sha256_file(path) == str(artifact["sha256"])
        result["passed"] = all(
            (result["exists"], result["bytes_match"], result["sha256_match"])
        )
        results.append(result)
    return {
        "baseline_id": payload.get("baseline_id"),
        "manifest": str(manifest_path),
        "artifacts": results,
        "passed": bool(results) and all(result["passed"] for result in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("baselines/baseline-v0.1.json"))
    args = parser.parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    result = verify_manifest(manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
