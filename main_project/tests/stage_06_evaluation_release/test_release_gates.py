from copy import deepcopy

from consentguard.stage_06_evaluation_release.release_gates import evaluate_release_gates


def _interval(point: float, *, radius: float = 0.005) -> dict[str, float]:
    return {"lower": max(0.0, point - radius), "upper": min(1.0, point + radius)}


def _passing_metrics():
    domain = {
        name: {privacy_class: 0.97 for privacy_class in ("face", "license_plate", "text")}
        for name in ("general", "india")
    }
    classes = {"face": 0.96, "license_plate": 0.92}
    return {
        "schema_version": "release-metrics-v1",
        "confidence_level": 0.95,
        "baseline_mask_map": 0.23,
        "sensitive_pixel_recall": 0.96,
        "sensitive_pixel_leakage": 0.04,
        "negative_image_false_positive_rate": 0.10,
        "seed_count": 3,
        "assurance_fail_closed": True,
        "supported_classes": list(classes),
        "domain_instance_recall": domain,
        "class_sensitive_pixel_recall": classes,
        "confidence_bounds": {
            "sensitive_pixel_recall": _interval(0.96),
            "sensitive_pixel_leakage": _interval(0.04),
            "negative_image_false_positive_rate": _interval(0.10),
            "domain_instance_recall": {
                name: {privacy_class: _interval(0.97) for privacy_class in values}
                for name, values in domain.items()
            },
            "class_sensitive_pixel_recall": {
                name: _interval(value) for name, value in classes.items()
            },
        },
    }


def test_all_release_gates_must_pass() -> None:
    assert evaluate_release_gates(_passing_metrics())["release_candidate"] is True
    failing = _passing_metrics()
    failing["domain_instance_recall"]["india"]["license_plate"] = 0.8
    failing["confidence_bounds"]["domain_instance_recall"]["india"]["license_plate"] = _interval(0.8)
    result = evaluate_release_gates(failing)
    assert result["release_candidate"] is False
    assert next(gate for gate in result["gates"] if gate["name"] == "india_license_plate_recall")["passed"] is False


def test_missing_supported_class_is_an_explicit_schema_failure() -> None:
    metrics = _passing_metrics()
    metrics["supported_classes"].append("handwriting")
    result = evaluate_release_gates(metrics)
    assert result["schema_valid"] is False
    assert "class_sensitive_pixel_recall.handwriting is required" in result["validation_errors"]
    assert result["release_candidate"] is False


def test_missing_required_metric_fails_without_raising() -> None:
    metrics = _passing_metrics()
    del metrics["sensitive_pixel_recall"]
    result = evaluate_release_gates(metrics)
    assert result["schema_valid"] is False
    assert any(error.startswith("sensitive_pixel_recall must") for error in result["validation_errors"])
    assert next(gate for gate in result["gates"] if gate["name"] == "overall_pixel_recall")["passed"] is False


def test_non_object_metrics_fail_without_raising() -> None:
    result = evaluate_release_gates([])
    assert result["schema_valid"] is False
    assert result["validation_errors"] == ["metrics must be a JSON object"]
    assert result["release_candidate"] is False


def test_release_uses_confidence_bound_not_only_point_estimate() -> None:
    metrics = _passing_metrics()
    metrics["confidence_bounds"]["sensitive_pixel_recall"] = {"lower": 0.90, "upper": 0.97}
    result = evaluate_release_gates(metrics)
    gate = next(gate for gate in result["gates"] if gate["name"] == "overall_pixel_recall")
    assert gate["point_estimate"] == 0.96
    assert gate["observed"] == 0.90
    assert gate["passed"] is False


def test_three_distinct_seed_runs_are_required_by_count() -> None:
    metrics = _passing_metrics()
    metrics["seed_count"] = 2
    result = evaluate_release_gates(metrics)
    assert next(gate for gate in result["gates"] if gate["name"] == "three_seeds")["passed"] is False
    assert result["release_candidate"] is False


def test_malformed_intervals_and_non_finite_values_fail_closed() -> None:
    metrics = deepcopy(_passing_metrics())
    metrics["baseline_mask_map"] = float("nan")
    metrics["confidence_bounds"]["sensitive_pixel_leakage"] = {"lower": 0.05, "upper": 0.01}
    result = evaluate_release_gates(metrics)
    assert result["schema_valid"] is False
    assert any("baseline_mask_map" in error for error in result["validation_errors"])
    assert any("must not exceed" in error for error in result["validation_errors"])
    assert result["release_candidate"] is False
