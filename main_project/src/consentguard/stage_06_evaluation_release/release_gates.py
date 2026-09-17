"""Machine-readable ConsentGuard v1 release gates.

The release evaluator is deliberately fail-closed. A metric that is absent,
malformed, or missing its confidence interval is failed rather than silently
omitted from the gate list.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


SCHEMA_VERSION = "release-metrics-v1"
REQUIRED_DOMAINS = ("general", "india")
REQUIRED_DOMAIN_CLASSES = ("face", "license_plate", "text")


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    observed: float | bool
    requirement: str
    point_estimate: float | None = None


def _number(value: Any, default: float) -> float:
    """Return a finite real number while rejecting bool and malformed input."""

    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _validate_rate(errors: list[str], path: str, value: Any) -> None:
    number = _number(value, float("nan"))
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        errors.append(f"{path} must be a finite number in [0, 1]")


def _validate_interval(errors: list[str], path: str, value: Any, point: Any) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path} must contain lower and upper confidence bounds")
        return
    lower = _number(value.get("lower"), float("nan"))
    upper = _number(value.get("upper"), float("nan"))
    estimate = _number(point, float("nan"))
    if not all(isfinite(item) and 0.0 <= item <= 1.0 for item in (lower, upper)):
        errors.append(f"{path}.lower and {path}.upper must be finite numbers in [0, 1]")
    elif lower > upper:
        errors.append(f"{path}.lower must not exceed {path}.upper")
    elif isfinite(estimate) and not lower <= estimate <= upper:
        errors.append(f"{path} must contain its point estimate")


def validate_release_metrics(metrics: Any) -> list[str]:
    """Validate the frozen release-metric contract and return all errors."""

    if not isinstance(metrics, dict):
        return ["metrics must be a JSON object"]
    errors: list[str] = []
    allowed_top_level = {
        "schema_version",
        "confidence_level",
        "baseline_mask_map",
        "sensitive_pixel_recall",
        "sensitive_pixel_leakage",
        "negative_image_false_positive_rate",
        "seed_count",
        "assurance_fail_closed",
        "supported_classes",
        "domain_instance_recall",
        "class_sensitive_pixel_recall",
        "confidence_bounds",
    }
    unknown = sorted(set(metrics) - allowed_top_level)
    if unknown:
        errors.append(f"unknown top-level fields: {unknown}")
    if metrics.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must equal {SCHEMA_VERSION!r}")

    for name in (
        "baseline_mask_map",
        "sensitive_pixel_recall",
        "sensitive_pixel_leakage",
        "negative_image_false_positive_rate",
    ):
        _validate_rate(errors, name, metrics.get(name))
    if not isinstance(metrics.get("assurance_fail_closed"), bool):
        errors.append("assurance_fail_closed must be a boolean")
    seed_count = metrics.get("seed_count")
    if isinstance(seed_count, bool) or not isinstance(seed_count, int) or seed_count < 0:
        errors.append("seed_count must be a non-negative integer")

    supported = metrics.get("supported_classes")
    if (
        not isinstance(supported, list)
        or not supported
        or any(not isinstance(name, str) or not name for name in supported)
        or len(set(supported)) != len(supported)
    ):
        errors.append("supported_classes must be a non-empty list of unique class names")
        supported_names: tuple[str, ...] = ()
    else:
        supported_names = tuple(supported)

    domain_recall = metrics.get("domain_instance_recall")
    if not isinstance(domain_recall, dict):
        errors.append("domain_instance_recall must be an object")
        domain_recall = {}
    elif set(domain_recall) != set(REQUIRED_DOMAINS):
        errors.append(f"domain_instance_recall keys must be {list(REQUIRED_DOMAINS)}")
    for domain in REQUIRED_DOMAINS:
        values = domain_recall.get(domain)
        if not isinstance(values, dict):
            errors.append(f"domain_instance_recall.{domain} must be an object")
            values = {}
        elif set(values) != set(REQUIRED_DOMAIN_CLASSES):
            errors.append(
                f"domain_instance_recall.{domain} keys must be {list(REQUIRED_DOMAIN_CLASSES)}"
            )
        for privacy_class in REQUIRED_DOMAIN_CLASSES:
            _validate_rate(
                errors,
                f"domain_instance_recall.{domain}.{privacy_class}",
                values.get(privacy_class),
            )

    class_recall = metrics.get("class_sensitive_pixel_recall")
    if not isinstance(class_recall, dict):
        errors.append("class_sensitive_pixel_recall must be an object")
        class_recall = {}
    elif supported_names and set(class_recall) != set(supported_names):
        errors.append("class_sensitive_pixel_recall keys must exactly match supported_classes")
    for privacy_class in supported_names:
        if privacy_class not in class_recall:
            errors.append(f"class_sensitive_pixel_recall.{privacy_class} is required")
        else:
            _validate_rate(
                errors,
                f"class_sensitive_pixel_recall.{privacy_class}",
                class_recall[privacy_class],
            )

    confidence = metrics.get("confidence_bounds")
    if not isinstance(confidence, dict):
        errors.append("confidence_bounds must be an object")
        confidence = {}
    elif set(confidence) != {
        "sensitive_pixel_recall",
        "sensitive_pixel_leakage",
        "negative_image_false_positive_rate",
        "domain_instance_recall",
        "class_sensitive_pixel_recall",
    }:
        errors.append("confidence_bounds contains missing or unknown fields")
    if _number(metrics.get("confidence_level"), -1.0) != 0.95:
        errors.append("confidence_level must equal 0.95")
    for name in (
        "sensitive_pixel_recall",
        "sensitive_pixel_leakage",
        "negative_image_false_positive_rate",
    ):
        _validate_interval(errors, f"confidence_bounds.{name}", confidence.get(name), metrics.get(name))

    domain_bounds = confidence.get("domain_instance_recall")
    if not isinstance(domain_bounds, dict):
        errors.append("confidence_bounds.domain_instance_recall must be an object")
        domain_bounds = {}
    elif set(domain_bounds) != set(REQUIRED_DOMAINS):
        errors.append(
            f"confidence_bounds.domain_instance_recall keys must be {list(REQUIRED_DOMAINS)}"
        )
    for domain in REQUIRED_DOMAINS:
        bounds = domain_bounds.get(domain)
        if not isinstance(bounds, dict):
            errors.append(f"confidence_bounds.domain_instance_recall.{domain} must be an object")
            bounds = {}
        elif set(bounds) != set(REQUIRED_DOMAIN_CLASSES):
            errors.append(
                "confidence_bounds.domain_instance_recall."
                f"{domain} keys must be {list(REQUIRED_DOMAIN_CLASSES)}"
            )
        points = domain_recall.get(domain, {})
        points = points if isinstance(points, dict) else {}
        for privacy_class in REQUIRED_DOMAIN_CLASSES:
            _validate_interval(
                errors,
                f"confidence_bounds.domain_instance_recall.{domain}.{privacy_class}",
                bounds.get(privacy_class),
                points.get(privacy_class),
            )

    class_bounds = confidence.get("class_sensitive_pixel_recall")
    if not isinstance(class_bounds, dict):
        errors.append("confidence_bounds.class_sensitive_pixel_recall must be an object")
        class_bounds = {}
    elif supported_names and set(class_bounds) != set(supported_names):
        errors.append(
            "confidence_bounds.class_sensitive_pixel_recall keys must exactly match supported_classes"
        )
    for privacy_class in supported_names:
        _validate_interval(
            errors,
            f"confidence_bounds.class_sensitive_pixel_recall.{privacy_class}",
            class_bounds.get(privacy_class),
            class_recall.get(privacy_class),
        )
    return errors


def _bound(container: Any, *keys: str, side: str, default: float) -> float:
    value = container
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    if not isinstance(value, dict):
        return default
    return _number(value.get(side), default)


def evaluate_release_gates(metrics: Any) -> dict:
    """Evaluate a validated metrics document, failing incomplete evidence."""

    errors = validate_release_metrics(metrics)
    metrics = metrics if isinstance(metrics, dict) else {}
    domain_recall = metrics.get("domain_instance_recall", {})
    domain_recall = domain_recall if isinstance(domain_recall, dict) else {}
    class_recall = metrics.get("class_sensitive_pixel_recall", {})
    class_recall = class_recall if isinstance(class_recall, dict) else {}
    supported = metrics.get("supported_classes", [])
    supported = supported if isinstance(supported, list) else []
    confidence = metrics.get("confidence_bounds", {})
    confidence = confidence if isinstance(confidence, dict) else {}

    baseline = _number(metrics.get("baseline_mask_map"), -1.0)
    pixel_recall = _number(metrics.get("sensitive_pixel_recall"), -1.0)
    pixel_recall_lcb = _bound(confidence, "sensitive_pixel_recall", side="lower", default=-1.0)
    leakage = _number(metrics.get("sensitive_pixel_leakage"), 2.0)
    leakage_ucb = _bound(confidence, "sensitive_pixel_leakage", side="upper", default=2.0)
    negative_fpr = _number(metrics.get("negative_image_false_positive_rate"), 2.0)
    negative_fpr_ucb = _bound(
        confidence, "negative_image_false_positive_rate", side="upper", default=2.0
    )
    seed_count = metrics.get("seed_count", 0)
    seed_count = seed_count if isinstance(seed_count, int) and not isinstance(seed_count, bool) else 0

    checks = [
        GateResult("metrics_schema", not errors, not errors, f"valid {SCHEMA_VERSION}"),
        GateResult("baseline_map", baseline >= 0.223469, baseline, ">= 0.223469"),
        GateResult(
            "overall_pixel_recall",
            0.0 <= pixel_recall <= 1.0 and pixel_recall_lcb >= 0.95,
            pixel_recall_lcb,
            "95% lower bound >= 0.95",
            pixel_recall,
        ),
        GateResult(
            "overall_pixel_leakage",
            0.0 <= leakage <= 1.0 and leakage_ucb <= 0.05,
            leakage_ucb,
            "95% upper bound <= 0.05",
            leakage,
        ),
        GateResult(
            "negative_false_positive_rate",
            0.0 <= negative_fpr <= 1.0 and negative_fpr_ucb <= 0.15,
            negative_fpr_ucb,
            "95% upper bound <= 0.15",
            negative_fpr,
        ),
        GateResult("three_seeds", seed_count >= 3, seed_count, ">= 3"),
        GateResult(
            "assurance_fail_closed",
            metrics.get("assurance_fail_closed") is True,
            metrics.get("assurance_fail_closed") is True,
            "true",
        ),
    ]
    for domain in REQUIRED_DOMAINS:
        points = domain_recall.get(domain, {})
        points = points if isinstance(points, dict) else {}
        for privacy_class in REQUIRED_DOMAIN_CLASSES:
            point = _number(points.get(privacy_class), -1.0)
            lower = _bound(
                confidence,
                "domain_instance_recall",
                domain,
                privacy_class,
                side="lower",
                default=-1.0,
            )
            checks.append(
                GateResult(
                    f"{domain}_{privacy_class}_recall",
                    0.0 <= point <= 1.0 and lower >= 0.95,
                    lower,
                    "95% lower bound >= 0.95",
                    point,
                )
            )
    for privacy_class in sorted(name for name in supported if isinstance(name, str)):
        point = _number(class_recall.get(privacy_class), -1.0)
        lower = _bound(
            confidence,
            "class_sensitive_pixel_recall",
            privacy_class,
            side="lower",
            default=-1.0,
        )
        checks.append(
            GateResult(
                f"class_{privacy_class}_pixel_recall",
                0.0 <= point <= 1.0 and lower >= 0.90,
                lower,
                "95% lower bound >= 0.90",
                point,
            )
        )
    results = [result.__dict__ for result in checks]
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_valid": not errors,
        "validation_errors": errors,
        "release_candidate": not errors and bool(results) and all(item["passed"] for item in results),
        "gates": results,
    }
