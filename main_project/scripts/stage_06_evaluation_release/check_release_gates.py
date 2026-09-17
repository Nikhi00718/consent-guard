"""Evaluate a frozen metrics JSON file against ConsentGuard v1 gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from consentguard.stage_06_evaluation_release.release_gates import evaluate_release_gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_release_gates(json.loads(args.metrics.read_text(encoding="utf-8")))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["release_candidate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
