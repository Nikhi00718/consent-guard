"""Evidence-provider contracts and fusion utilities."""

from consentguard.stage_04_fusion_calibration.evidence.base import EvidenceProvider, ProviderUnavailableError
from consentguard.stage_04_fusion_calibration.evidence.fusion import EvidenceFusion
from consentguard.stage_04_fusion_calibration.evidence.thresholds import ThresholdProfile, ThresholdRegistry

__all__ = [
    "EvidenceFusion",
    "EvidenceProvider",
    "ProviderUnavailableError",
    "ThresholdProfile",
    "ThresholdRegistry",
]
