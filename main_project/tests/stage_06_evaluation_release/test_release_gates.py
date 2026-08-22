from consentguard.stage_06_evaluation_release.release_gates import evaluate_release_gates


def _passing_metrics():
    return {
        "baseline_mask_map": 0.23,
        "sensitive_pixel_recall": 0.96,
        "sensitive_pixel_leakage": 0.04,
        "negative_image_false_positive_rate": 0.10,
        "seed_count": 3,
        "assurance_fail_closed": True,
        "domain_instance_recall": {
            domain: {name: 0.96 for name in ("face", "license_plate", "text")}
            for domain in ("general", "india")
        },
        "class_sensitive_pixel_recall": {"face": 0.96, "license_plate": 0.92},
    }


def test_all_release_gates_must_pass() -> None:
    assert evaluate_release_gates(_passing_metrics())["release_candidate"] is True
    failing = _passing_metrics()
    failing["domain_instance_recall"]["india"]["license_plate"] = 0.8
    result = evaluate_release_gates(failing)
    assert result["release_candidate"] is False
    assert next(gate for gate in result["gates"] if gate["name"] == "india_license_plate_recall")["passed"] is False
