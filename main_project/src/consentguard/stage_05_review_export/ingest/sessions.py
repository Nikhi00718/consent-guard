"""Randomized, TTL-bound staging sessions for image uploads."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class SessionHandle:
    session_id: str
    root: Path
    created_at: float
    expires_at: float

    def public_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


class SessionStore:
    """Keep staged bytes in per-session directories with no user path reuse."""

    def __init__(self, root: str | Path, *, ttl_seconds: int = 3600) -> None:
        self.root = Path(root).resolve()
        self.ttl_seconds = int(ttl_seconds)
        if self.ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, *, now: float | None = None) -> SessionHandle:
        created = time.time() if now is None else float(now)
        session_id = f"session-{uuid4().hex}"
        session_root = self.root / session_id
        session_root.mkdir(mode=0o700)
        return SessionHandle(session_id, session_root, created, created + self.ttl_seconds)

    def stage_bytes(self, handle: SessionHandle, payload: bytes, *, suffix: str = ".bin") -> Path:
        self._assert_owned(handle)
        if not payload:
            raise ValueError("cannot stage empty payload")
        safe_suffix = suffix if suffix.startswith(".") and suffix[1:].isalnum() else ".bin"
        filename = f"upload-{hashlib.sha256(payload).hexdigest()[:24]}{safe_suffix.lower()}"
        destination = handle.root / filename
        destination.write_bytes(payload)
        return destination

    def cleanup_expired(self, *, now: float | None = None) -> tuple[str, ...]:
        moment = time.time() if now is None else float(now)
        removed: list[str] = []
        for entry in self.root.iterdir():
            if not entry.is_dir() or not entry.name.startswith("session-"):
                continue
            age = moment - entry.stat().st_mtime
            if age > self.ttl_seconds:
                self._assert_inside_root(entry)
                shutil.rmtree(entry)
                removed.append(entry.name)
        return tuple(sorted(removed))

    def delete(self, handle: SessionHandle) -> None:
        """Delete one owned session and all of its staged artifacts."""

        self._assert_owned(handle)
        self._assert_inside_root(handle.root)
        shutil.rmtree(handle.root)

    def _assert_owned(self, handle: SessionHandle) -> None:
        self._assert_inside_root(handle.root)
        if handle.root.name != handle.session_id or not handle.session_id.startswith("session-"):
            raise ValueError("invalid session handle")
        if time.time() >= handle.expires_at:
            raise ValueError("session has expired")
        if not handle.root.is_dir():
            raise FileNotFoundError("session directory does not exist")

    def _assert_inside_root(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved == self.root or self.root not in resolved.parents:
            raise ValueError("path escapes session store root")
