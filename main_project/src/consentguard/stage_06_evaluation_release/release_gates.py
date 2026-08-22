"""Machine-readable ConsentGuard v1 release gates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    observed: float | bool
    requirement: str


def evaluate_release_gates(metrics: dict) -> dict:
    domain_recall = metrics.get("domain_instance_recall", {})
    class_recall = metrics.get("class_sensitive_pixel_recall", {})
    checks = [
        GateResult("baseline_map", float(metrics.get("baseline_mask_map", -1)) >= 0.223469, float(metrics.get("baseline_mask_map", -1)), ">= 0.223469"),
        GateResult("overall_pixel_recall", float(metrics.get("sensitive_pixel_recall", -1)) >= 0.95, float(metrics.get("sensitive_pixel_recall", -1)), ">= 0.95"),
        GateResult("overall_pixel_leakage", float(metrics.get("sensitive_pixel_leakage", 2)) <= 0.05, float(metrics.get("sensitive_pixel_leakage", 2)), "<= 0.05"),
        GateResult("negative_false_positive_rate", float(metrics.get("negative_image_false_positive_rate", 2)) <= 0.15, float(metrics.get("negative_image_false_positive_rate", 2)), "<= 0.15"),
        GateResult("three_seeds", int(metrics.get("seed_count", 0)) >= 3, int(metrics.get("seed_count", 0)), ">= 3"),
        GateResult("assurance_fail_closed", metrics.get("assurance_fail_closed") is True, bool(metrics.get("assurance_fail_closed", False)), "true"),
    ]
    for domain in ("general", "india"):
        for privacy_class in ("face", "license_plate", "text"):
            observed = float(domain_recall.get(domain, {}).get(privacy_class, -1))
            checks.append(GateResult(f"{domain}_{privacy_class}_recall", observed >= 0.95, observed, ">= 0.95"))
    for privacy_class, observed_value in sorted(class_recall.items()):
        observed = float(observed_value)
        checks.append(GateResult(f"class_{privacy_class}_pixel_recall", observed >= 0.90, observed, ">= 0.90"))
    results = [result.__dict__ for result in checks]
    return {"release_candidate": bool(results) and all(item["passed"] for item in results), "gates": results}
