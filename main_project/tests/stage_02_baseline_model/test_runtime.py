from __future__ import annotations

import torch

from consentguard.shared.runtime import atomic_json_dump, atomic_link_or_copy, atomic_torch_save


def test_atomic_artifact_writes(tmp_path) -> None:
    json_path = tmp_path / "report.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    atomic_json_dump({"passed": True}, json_path)
    atomic_torch_save({"tensor": torch.tensor([1, 2, 3])}, checkpoint_path)
    assert '"passed": true' in json_path.read_text(encoding="utf-8")
    loaded = torch.load(checkpoint_path, weights_only=True)
    assert loaded["tensor"].tolist() == [1, 2, 3]

    published_path = tmp_path / "published.pt"
    atomic_link_or_copy(checkpoint_path, published_path)
    published = torch.load(published_path, weights_only=True)
    assert published["tensor"].tolist() == [1, 2, 3]
