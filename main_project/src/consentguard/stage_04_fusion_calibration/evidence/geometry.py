"""Lossless binary-mask encoding and geometry rasterization helpers."""

from __future__ import annotations

import cv2
import numpy as np

from consentguard.stage_04_fusion_calibration.domain import EvidenceGeometry, ThresholdRule


def encode_binary_mask(mask: np.ndarray) -> tuple[int, ...]:
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    flat = np.asarray(mask > 0, dtype=np.uint8).reshape(-1)
    runs: list[int] = []
    current = 0
    length = 0
    for value in flat.tolist():
        if value == current:
            length += 1
        else:
            runs.append(length)
            current = value
            length = 1
    runs.append(length)
    return tuple(runs)


def decode_binary_mask(runs: tuple[int, ...], height: int, width: int) -> np.ndarray:
    if any(run < 0 for run in runs) or sum(runs) != height * width:
        raise ValueError("Invalid row-major mask RLE")
    values: list[np.ndarray] = []
    bit = 0
    for run in runs:
        if run:
            values.append(np.full(run, bit, dtype=np.uint8))
        bit = 1 - bit
    flat = np.concatenate(values) if values else np.zeros(height * width, dtype=np.uint8)
    return flat.reshape(height, width)


def geometry_to_mask(geometry: EvidenceGeometry, rule: ThresholdRule) -> np.ndarray:
    height, width = geometry.height, geometry.width
    if geometry.mask_rle:
        mask = decode_binary_mask(geometry.mask_rle, height, width)
    else:
        mask = np.zeros((height, width), dtype=np.uint8)
        if geometry.polygon_xy:
            points = np.asarray(geometry.polygon_xy, dtype=np.float32)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        elif geometry.box_xyxy is not None:
            left, top, right, bottom = geometry.box_xyxy
            box_width, box_height = right - left, bottom - top
            expand_x = box_width * rule.expansion_fraction
            expand_y = box_height * rule.expansion_fraction
            x0 = max(0, int(np.floor(left - expand_x)))
            y0 = max(0, int(np.floor(top - expand_y)))
            x1 = min(width, int(np.ceil(right + expand_x)))
            y1 = min(height, int(np.ceil(bottom + expand_y)))
            mask[y0:y1, x0:x1] = 1
    if rule.dilation_pixels and np.any(mask):
        size = 2 * rule.dilation_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return np.asarray(mask > 0, dtype=np.uint8)
