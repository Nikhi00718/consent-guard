import json
from pathlib import Path


def test_versioned_artifact_schemas_are_valid_json_and_have_stable_ids() -> None:
    root = Path(__file__).parents[2] / "schemas"
    names = {"consent-v1.schema.json", "evidence-v1.schema.json", "decision-v1.schema.json"}
    documents = [json.loads((root / name).read_text(encoding="utf-8")) for name in names]
    assert all(document["$schema"].endswith("2020-12/schema") for document in documents)
    assert {document["$id"] for document in documents} == {
        "https://consentguard.local/schemas/consent-v1.schema.json",
        "https://consentguard.local/schemas/evidence-v1.schema.json",
        "https://consentguard.local/schemas/decision-v1.schema.json",
    }
