from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from consentguard.redaction.prediction_renderer import write_metadata_free_redaction


def test_redaction_is_newly_encoded_and_reopens(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    destination = tmp_path / "redacted.png"
    bgr = np.full((24, 32, 3), 200, dtype=np.uint8)
    assert cv2.imwrite(str(source), bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[5:15, 7:20] = 255
    report = write_metadata_free_redaction(source, destination, rgb, mask)
    reopened = cv2.imread(str(destination), cv2.IMREAD_COLOR)
    assert reopened is not None
    assert np.all(reopened[5:15, 7:20] == 0)
    assert report["newly_encoded"] is True
    assert report["source_sha256"] != report["output_sha256"]


def test_redaction_refuses_to_overwrite_source(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    image = np.full((8, 8, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(source), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    mask = np.ones((8, 8), dtype=np.uint8) * 255
    with pytest.raises(ValueError, match="overwrite the source"):
        write_metadata_free_redaction(source, source, image, mask)
