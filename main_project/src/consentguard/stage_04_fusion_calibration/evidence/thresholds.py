"""Versioned per-provider and per-class threshold profiles."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from consentguard.stage_04_fusion_calibration.domain import ThresholdRule


@dataclass(frozen=True)
class ThresholdProfile:
    profile_id: str
    release_ready: bool
    rules: tuple[ThresholdRule, ...]
    source_path: Path
    source_sha256: str


class ThresholdRegistry:
    def __init__(self, profile: ThresholdProfile) -> None:
        self.profile = profile
        self._rules = {(rule.provider, rule.privacy_class): rule for rule in profile.rules}
        if len(self._rules) != len(profile.rules):
            raise ValueError("Threshold profile contains duplicate provider/class rules")

    @classmethod
    def load(cls, path: str | Path) -> "ThresholdRegistry":
        path = Path(path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload.get("profile_id"):
            raise ValueError("Threshold profile requires profile_id")
        raw_rules = payload.get("rules")
        if not isinstance(raw_rules, list) or not raw_rules:
            raise ValueError("Threshold profile requires a non-empty rules list")
        rules = tuple(ThresholdRule(**raw) for raw in raw_rules)
        profile = ThresholdProfile(
            profile_id=str(payload["profile_id"]),
            release_ready=bool(payload.get("release_ready", False)),
            rules=rules,
            source_path=path.resolve(),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        return cls(profile)

    def get(self, provider: str, privacy_class: str) -> ThresholdRule:
        for key in (
            (provider, privacy_class),
            (provider, "*"),
            ("*", privacy_class),
            ("*", "*"),
        ):
            if key in self._rules:
                return self._rules[key]
        raise KeyError(f"No threshold rule for provider={provider!r}, class={privacy_class!r}")
