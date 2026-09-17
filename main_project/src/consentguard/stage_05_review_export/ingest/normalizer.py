"""Decode a still image into an oriented, opaque RGB buffer with bounded size."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class IngestLimits:
    max_bytes: int = 25 * 1024 * 1024
    max_pixels: int = 40_000_000
    #: MPO is a JPEG container holding more than one frame, produced by many
    #: phone and camera modes. Rejecting it makes ordinary photos unopenable,
    #: so the primary frame is used and the extra frames are dropped by the
    #: re-encode, which removes whatever they contained.
    allowed_formats: tuple[str, ...] = ("JPEG", "PNG", "WEBP", "MPO")
    #: Formats where extra frames mean animation rather than a second still.
    multi_frame_formats: tuple[str, ...] = ("MPO",)


@dataclass(frozen=True)
class NormalizedImage:
    source_path: Path
    source_sha256: str
    pixel_sha256: str
    pixels_rgb: np.ndarray
    source_format: str
    metadata_categories: tuple[str, ...]
    orientation_applied: bool

    @property
    def width(self) -> int:
        return int(self.pixels_rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels_rgb.shape[0])


def normalize_image(path: str | Path, limits: IngestLimits = IngestLimits()) -> NormalizedImage:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")
    payload = path.read_bytes()
    if not payload or len(payload) > limits.max_bytes:
        raise ValueError(f"Input image byte size must be in [1, {limits.max_bytes}]")
    try:
        with Image.open(BytesIO(payload)) as source:
            source_format = str(source.format or "").upper()
            if source_format not in limits.allowed_formats:
                raise ValueError(f"Unsupported image format: {source_format or 'unknown'}")
            frames = int(getattr(source, "n_frames", 1))
            if frames != 1:
                if source_format not in limits.multi_frame_formats:
                    raise ValueError("Animated or multi-frame images are not supported")
                source.seek(0)
            if source.width * source.height > limits.max_pixels:
                raise ValueError(f"Decoded image exceeds {limits.max_pixels} pixels")
            exif = source.getexif()
            orientation_applied = bool(exif.get(274, 1) != 1)
            metadata_categories = []
            if exif:
                metadata_categories.append("exif")
            if frames != 1:
                metadata_categories.append("extra_frames_dropped")
            for key in ("icc_profile", "xmp", "comment", "dpi"):
                if key in source.info:
                    metadata_categories.append(key)
            oriented = ImageOps.exif_transpose(source)
            if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                rgb = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                rgb = oriented.convert("RGB")
            pixels = np.asarray(rgb, dtype=np.uint8).copy()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Could not safely decode input image: {error}") from error
    pixel_digest = hashlib.sha256()
    pixel_digest.update(pixels.shape[1].to_bytes(8, "little"))
    pixel_digest.update(pixels.shape[0].to_bytes(8, "little"))
    pixel_digest.update(pixels.tobytes(order="C"))
    return NormalizedImage(
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        pixel_sha256=pixel_digest.hexdigest(),
        pixels_rgb=pixels,
        source_format=source_format,
        metadata_categories=tuple(sorted(metadata_categories)),
        orientation_applied=orientation_applied,
    )
