"""Dataset contracts shared by Visual Redactions preprocessing and audits."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image


TEXTUAL_ATTRIBUTE_IDS = (
    "a106_address_current_all",
    "a107_address_home_all",
    "a111_name_all",
    "a24_birth_date",
    "a49_phone",
    "a73_landmark",
    "a82_date_time",
    "a90_email",
)

VISUAL_ATTRIBUTE_IDS = (
    "a105_face_all",
    "a108_license_plate_all",
    "a109_person_body",
    "a110_nudity_all",
    "a26_handwriting",
    "a39_disability_physical",
    "a43_medicine",
    "a7_fingerprint",
    "a8_signature",
)

MULTIMODAL_ATTRIBUTE_IDS = (
    "a30_credit_card",
    "a31_passport",
    "a32_drivers_license",
    "a33_student_id",
    "a35_mail",
    "a37_receipt",
    "a38_ticket",
)

IGNORED_ATTRIBUTE_IDS = (
    "a70_education_history",
    "a29_ausweis",
    "a18_ethnic_clothing",
    "a85_username",
)

OFFICIAL_ATTRIBUTE_IDS = (
    TEXTUAL_ATTRIBUTE_IDS + VISUAL_ATTRIBUTE_IDS + MULTIMODAL_ATTRIBUTE_IDS
)
OFFICIAL_MIN_PIXELS = 25**2

PROFILE_ATTRIBUTE_IDS = {
    "official": OFFICIAL_ATTRIBUTE_IDS,
    "textual": TEXTUAL_ATTRIBUTE_IDS,
    "visual": VISUAL_ATTRIBUTE_IDS,
    "multimodal": MULTIMODAL_ATTRIBUTE_IDS,
}


def geometry_status(
    source_width: int,
    source_height: int,
    decoded_width: int,
    decoded_height: int,
    tolerance: float = 0.01,
) -> tuple[str, float, float]:
    """Classify whether annotation coordinates can be scaled to an image.

    Independent X/Y scaling is safe only for an ordinary aspect-preserving
    resize. A 90-degree candidate is reported but never accepted implicitly.
    """

    dimensions = (source_width, source_height, decoded_width, decoded_height)
    if any(int(value) < 1 for value in dimensions):
        raise ValueError(f"image dimensions must be positive: {dimensions}")
    if not 0 < tolerance < 0.25:
        raise ValueError("tolerance must be between 0 and 0.25")
    source_ratio = source_width / source_height
    decoded_ratio = decoded_width / decoded_height
    direct_error = abs(math.log(decoded_ratio / source_ratio))
    rotated_error = abs(math.log(decoded_ratio * source_ratio))
    limit = math.log1p(tolerance)
    if direct_error <= limit:
        return "aligned_resize", direct_error, rotated_error
    if rotated_error <= limit:
        return "rotation_candidate", direct_error, rotated_error
    return "geometry_mismatch", direct_error, rotated_error


def image_display_size(path: str | Path) -> tuple[int, int]:
    """Return width/height after applying the EXIF orientation contract.

    OpenCV's normal color decoder applies EXIF orientation. Reading the tag
    here lets geometry audits avoid decoding every high-resolution image.
    """

    with Image.open(path) as image:
        width, height = image.size
        orientation = int(image.getexif().get(274, 1))
    if orientation in {5, 6, 7, 8}:
        width, height = height, width
    return int(width), int(height)
