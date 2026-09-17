"""Fuse accepted evidence into connected review candidates with provenance."""

from __future__ import annotations

from uuid import uuid4

import cv2
import numpy as np

from consentguard.stage_04_fusion_calibration.domain import Evidence, ReviewCandidate, ReviewCandidateSet
from consentguard.stage_04_fusion_calibration.evidence.geometry import encode_binary_mask, geometry_to_mask
from consentguard.stage_04_fusion_calibration.evidence.thresholds import ThresholdRegistry


class EvidenceFusion:
    def __init__(self, thresholds: ThresholdRegistry) -> None:
        self.thresholds = thresholds

    def combine(
        self,
        evidence: list[Evidence],
        *,
        width: int,
        height: int,
        unavailable_providers: tuple[str, ...] = (),
    ) -> ReviewCandidateSet:
        accepted: list[tuple[Evidence, np.ndarray, bool, tuple[str, ...]]] = []
        rejected: list[str] = []
        union = np.zeros((height, width), dtype=np.uint8)
        for item in evidence:
            if item.geometry.width != width or item.geometry.height != height:
                raise ValueError(f"Evidence {item.evidence_id} has mismatched image dimensions")
            rule = self.thresholds.get(item.provider, item.privacy_class)
            if item.confidence < rule.score_threshold:
                rejected.append(item.evidence_id)
                continue
            mask = geometry_to_mask(item.geometry, rule)
            if int(mask.sum()) < rule.min_area_pixels:
                rejected.append(item.evidence_id)
                continue
            flags = tuple(sorted(set(item.uncertainty_flags)))
            mandatory = rule.mandatory_review or rule.experimental or bool(flags)
            accepted.append((item, mask, mandatory, flags))
            union |= mask

        candidates: list[ReviewCandidate] = []
        component_count, component_labels = cv2.connectedComponents(union, connectivity=8)
        for component_id in range(1, component_count):
            component = np.asarray(component_labels == component_id, dtype=np.uint8)
            contributors = [entry for entry in accepted if np.any(entry[1] & component)]
            flags = sorted({flag for _, _, _, item_flags in contributors for flag in item_flags})
            candidates.append(
                ReviewCandidate(
                    candidate_id=f"candidate-{uuid4().hex}",
                    width=width,
                    height=height,
                    mask_rle=encode_binary_mask(component),
                    evidence_ids=tuple(sorted(item.evidence_id for item, *_ in contributors)),
                    privacy_classes=tuple(sorted({item.privacy_class for item, *_ in contributors})),
                    providers=tuple(sorted({item.provider for item, *_ in contributors})),
                    mandatory_review=any(entry[2] for entry in contributors),
                    uncertainty_flags=tuple(flags),
                )
            )

        return ReviewCandidateSet(
            width=width,
            height=height,
            candidates=tuple(candidates),
            threshold_profile_id=self.thresholds.profile.profile_id,
            threshold_profile_release_ready=self.thresholds.profile.release_ready,
            unavailable_providers=tuple(sorted(set(unavailable_providers))),
            rejected_evidence_ids=tuple(sorted(rejected)),
        )
