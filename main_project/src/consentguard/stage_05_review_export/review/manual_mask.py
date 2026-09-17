"""Convert Gradio ImageEditor layers into one deterministic binary mask."""

from __future__ import annotations

import cv2
import numpy as np


def editor_layers_to_mask(editor_value: dict, *, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for layer in editor_value.get("layers", []) if isinstance(editor_value, dict) else []:
        array = np.asarray(layer)
        if array.ndim != 3:
            continue
        if array.shape[:2] != (height, width):
            array = cv2.resize(array, (width, height), interpolation=cv2.INTER_NEAREST)
        if array.shape[2] >= 4:
            active = array[:, :, 3] > 0
        else:
            active = np.any(array[:, :, :3] != 0, axis=2)
        mask[active] = 255
    return mask
