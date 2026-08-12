"""Render destructive redactions from instance-segmentation predictions."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from consentguard.runtime import atomic_json_dump


def load_rgb_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not decode input image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_for_inference(image: np.ndarray, short_side: int, max_long_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = short_side / float(min(height, width))
    if max(height, width) * scale > max_long_side:
        scale = max_long_side / float(max(height, width))
    new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))
    if (new_height, new_width) == (height, width):
        return image, 1.0
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation), scale


@torch.inference_mode()
def predict_union_mask(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: torch.device,
    *,
    short_side: int,
    max_long_side: int,
    score_threshold: float,
    mask_threshold: float = 0.5,
    class_ids: set[int] | None = None,
    dilation_pixels: int = 3,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be a uint8 HxWx3 array")
    if short_side < 32 or max_long_side < short_side:
        raise ValueError("Require max_long_side >= short_side >= 32")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1]")
    if not 0.0 <= mask_threshold <= 1.0:
        raise ValueError("mask_threshold must be in [0, 1]")
    if dilation_pixels < 0:
        raise ValueError("dilation_pixels must be non-negative")
    resized, _ = resize_for_inference(image_rgb, short_side, max_long_side)
    tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float().div_(255.0).to(device)
    model.eval()
    prediction = model([tensor])[0]
    keep = prediction["scores"] >= score_threshold
    if class_ids is not None:
        allowed = torch.tensor(
            sorted(class_ids),
            dtype=prediction["labels"].dtype,
            device=prediction["labels"].device,
        )
        keep &= torch.isin(prediction["labels"], allowed)
    indices = torch.nonzero(keep, as_tuple=False).flatten()
    resized_union = np.zeros(resized.shape[:2], dtype=np.uint8)
    detections: list[dict[str, Any]] = []
    for index in indices.tolist():
        mask = prediction["masks"][index, 0].detach().cpu().numpy() >= mask_threshold
        resized_union[mask] = 255
        detections.append(
            {
                "class_id": int(prediction["labels"][index].item()),
                "score": round(float(prediction["scores"][index].item()), 6),
                "box_resized_xyxy": [round(float(value), 3) for value in prediction["boxes"][index].tolist()],
                "mask_pixels_resized": int(mask.sum()),
            }
        )
    if dilation_pixels > 0 and np.any(resized_union):
        size = 2 * dilation_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        resized_union = cv2.dilate(resized_union, kernel, iterations=1)
    if resized_union.shape != image_rgb.shape[:2]:
        union = cv2.resize(
            resized_union,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        union = resized_union
    return union, detections


def write_metadata_free_redaction(
    source_path: str | Path,
    destination_path: str | Path,
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    fill_rgb: tuple[int, int, int] = (0, 0, 0),
    jpeg_quality: int = 95,
) -> dict[str, Any]:
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("Refusing to overwrite the source image; choose a distinct output path")
    if destination_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError("Output must use .jpg, .jpeg, .png, or .webp")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be a uint8 HxWx3 array")
    if mask.shape != image_rgb.shape[:2]:
        raise ValueError("mask dimensions must match image dimensions")
    if len(fill_rgb) != 3 or any(not 0 <= int(value) <= 255 for value in fill_rgb):
        raise ValueError("fill_rgb must contain three values in [0, 255]")
    if not 1 <= int(jpeg_quality) <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    output_rgb = image_rgb.copy()
    output_rgb[mask > 0] = np.asarray(fill_rgb, dtype=np.uint8)
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    extension = destination_path.suffix.lower()
    parameters: list[int] = []
    if extension in {".jpg", ".jpeg"}:
        parameters = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    success, encoded = cv2.imencode(extension, output_bgr, parameters)
    if not success:
        raise RuntimeError(f"OpenCV could not encode output as {extension}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + f".tmp-{os.getpid()}")
    try:
        temporary.write_bytes(encoded.tobytes())
        os.replace(temporary, destination_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    reopened = cv2.imread(str(destination_path), cv2.IMREAD_COLOR)
    if reopened is None or reopened.shape[:2] != image_rgb.shape[:2]:
        raise RuntimeError("Encoded redaction failed independent decode/dimension verification")
    return {
        "source_sha256": source_sha256,
        "output_sha256": hashlib.sha256(destination_path.read_bytes()).hexdigest(),
        "width": int(image_rgb.shape[1]),
        "height": int(image_rgb.shape[0]),
        "redacted_pixels": int(np.count_nonzero(mask)),
        "redacted_fraction": float(np.count_nonzero(mask) / mask.size),
        "newly_encoded": True,
        "pixel_decode_verified": True,
    }


def write_inference_report(report: dict[str, Any], destination: str | Path) -> None:
    atomic_json_dump(report, destination)
