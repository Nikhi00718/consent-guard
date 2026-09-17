import numpy as np

from consentguard.stage_05_review_export.review import editor_layers_to_mask


def test_editor_layers_are_unioned_into_binary_mask() -> None:
    first = np.zeros((10, 12, 4), dtype=np.uint8)
    second = np.zeros((10, 12, 4), dtype=np.uint8)
    first[1:4, 2:5, 3] = 255
    second[6:9, 7:11, 3] = 128
    mask = editor_layers_to_mask({"layers": [first, second]}, width=12, height=10)
    assert mask[2, 3] == 255
    assert mask[7, 8] == 255
    assert mask[0, 0] == 0
