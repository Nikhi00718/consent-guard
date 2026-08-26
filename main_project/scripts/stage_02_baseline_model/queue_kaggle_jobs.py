"""Queue remaining ConsentGuard Kaggle GPU jobs without exceeding two sessions."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
USERNAME = "nikhil00718"
POLL_SECONDS = 45
TIMEOUT_SECONDS = 8 * 60 * 60
INITIAL_ACTIVE = {
    "plate_v7": f"{USERNAME}/consentguard-plate-model-training/7",
    "handwriting_v1": f"{USERNAME}/consentguard-handwriting-model-training/1",
}
PENDING = ["face", "baseline"]
RETRY_ON_ERROR = {"handwriting_v1": "handwriting"}


def emit(event: str, **values: object) -> None:
    payload = {
        "time": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **values,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def kaggle(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "kaggle", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def kernel_status(reference: str) -> str:
    result = kaggle("kernels", "status", reference)
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", result.stdout)
    status = match.group(1) if match else "UNKNOWN"
    emit("status", reference=reference, status=status, output=result.stdout.strip())
    return status


def push(component: str) -> bool:
    stage = ROOT / "artifacts" / "kaggle" / "kernels" / component
    result = kaggle("kernels", "push", "-p", str(stage))
    accepted = result.returncode == 0 and "successfully pushed" in result.stdout
    emit(
        "push",
        component=component,
        accepted=accepted,
        return_code=result.returncode,
        output=result.stdout.strip(),
    )
    return accepted


def main() -> None:
    started = time.monotonic()
    tracked = dict(INITIAL_ACTIVE)
    pending = list(PENDING)
    while pending:
        if time.monotonic() - started > TIMEOUT_SECONDS:
            emit("timeout", pending=pending)
            raise SystemExit(2)
        statuses = {name: kernel_status(ref) for name, ref in tracked.items()}
        for name, status in statuses.items():
            retry = RETRY_ON_ERROR.get(name)
            if status == "ERROR" and retry and retry not in pending:
                pending.insert(0, retry)
                emit("retry_queued", source=name, component=retry)
        # Treat an API/network outage as conservatively active.  A false
        # inactive result can cause duplicate GPU submissions while Kaggle is
        # still running the original sessions.
        active = {
            name: ref
            for name, ref in tracked.items()
            if statuses[name] in {"RUNNING", "UNKNOWN"}
        }
        tracked = active
        if len(active) < 2:
            component = pending[0]
            if push(component):
                pending.pop(0)
                tracked[component] = f"{USERNAME}/consentguard-{component}-model-training"
        emit("queue", active=sorted(tracked), pending=pending)
        if pending:
            time.sleep(POLL_SECONDS)
    emit("all_submitted")


if __name__ == "__main__":
    main()
