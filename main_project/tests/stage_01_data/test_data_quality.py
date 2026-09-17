from __future__ import annotations

import pytest

from consentguard.stage_01_data.data_quality import (
    IGNORED_ATTRIBUTE_IDS,
    MULTIMODAL_ATTRIBUTE_IDS,
    OFFICIAL_ATTRIBUTE_IDS,
    TEXTUAL_ATTRIBUTE_IDS,
    VISUAL_ATTRIBUTE_IDS,
    geometry_status,
)


def test_official_taxonomy_has_disjoint_modes_and_24_evaluated_attributes() -> None:
    modes = (TEXTUAL_ATTRIBUTE_IDS, VISUAL_ATTRIBUTE_IDS, MULTIMODAL_ATTRIBUTE_IDS)
    assert tuple(len(mode) for mode in modes) == (8, 9, 7)
    assert len(OFFICIAL_ATTRIBUTE_IDS) == len(set(OFFICIAL_ATTRIBUTE_IDS)) == 24
    assert not set(OFFICIAL_ATTRIBUTE_IDS).intersection(IGNORED_ATTRIBUTE_IDS)


def test_geometry_status_accepts_only_aspect_preserving_resize() -> None:
    status, direct_error, _rotated_error = geometry_status(3000, 2000, 900, 600)
    assert status == "aligned_resize"
    assert direct_error == pytest.approx(0.0)

    status, _direct_error, rotated_error = geometry_status(3000, 2000, 600, 900)
    assert status == "rotation_candidate"
    assert rotated_error == pytest.approx(0.0)

    status, _direct_error, _rotated_error = geometry_status(1552, 2592, 640, 640)
    assert status == "geometry_mismatch"


def test_geometry_status_rejects_invalid_dimensions_and_tolerance() -> None:
    with pytest.raises(ValueError, match="positive"):
        geometry_status(0, 100, 20, 20)
    with pytest.raises(ValueError, match="tolerance"):
        geometry_status(100, 100, 20, 20, tolerance=0.0)
