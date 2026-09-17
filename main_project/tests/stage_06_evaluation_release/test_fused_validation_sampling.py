import importlib.util
from pathlib import Path


def _evaluator_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "main_project" / "scripts" / "stage_06_evaluation_release" / "evaluate_fused_validation.py"
    spec = importlib.util.spec_from_file_location("consentguard_fused_validation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bounded_fused_sample_is_deterministic_and_includes_negatives() -> None:
    evaluator = _evaluator_module()
    records = [
        {"image_id": str(index), "instances": [] if index % 4 == 0 else [{"class_id": index % 3 + 1}]}
        for index in range(20)
    ]
    first, metadata = evaluator._select_records(records, max_images=6, seed=7)
    second, second_metadata = evaluator._select_records(records, max_images=6, seed=7)
    assert [record["image_id"] for record in first] == [record["image_id"] for record in second]
    assert metadata == second_metadata
    assert metadata["strategy"] == "stratified_ground_truth_class_and_negative"
    assert metadata["selected"] == 6
    assert metadata["bucket_counts"]["negative"] == 1
