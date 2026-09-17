"""Stage 03: specialist face, plate, text, and barcode providers."""
from consentguard.stage_03_specialists.barcode_zxing import ZXingBarcodeProvider
from consentguard.stage_03_specialists.box_detector import BoxDetectorEvidenceProvider
from consentguard.stage_03_specialists.face_yunet import YuNetFaceProvider
from consentguard.stage_03_specialists.plate_yunet import LPDYuNetPlateProvider
from consentguard.stage_03_specialists.ppocr_onnx import PPOCRTextGeometryProvider
from consentguard.stage_03_specialists.text_paddleocr import PaddleOCRTextProvider

__all__ = [
    "BoxDetectorEvidenceProvider",
    "LPDYuNetPlateProvider",
    "PPOCRTextGeometryProvider",
    "PaddleOCRTextProvider",
    "YuNetFaceProvider",
    "ZXingBarcodeProvider",
]
