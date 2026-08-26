"""Launch a local automatic ConsentGuard image-analysis preview UI.

This interface is intentionally a research preview. It runs only the providers
selected by the user, fuses their evidence, and creates a destructive redaction
preview. It does not bypass the separate manual-review and assurance gate.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from consentguard.shared.paths import project_path
from consentguard.shared.runtime import select_device
from consentguard.stage_02_baseline_model.config import load_training_config
from consentguard.stage_03_specialists.barcode_zxing import ZXingBarcodeProvider
from consentguard.stage_03_specialists.face_yunet import YuNetFaceProvider
from consentguard.stage_03_specialists.plate_yunet import LPDYuNetPlateProvider
from consentguard.stage_03_specialists.ppocr_onnx import PPOCRTextGeometryProvider
from consentguard.stage_04_fusion_calibration.domain import ConsentState
from consentguard.stage_04_fusion_calibration.evidence import ThresholdRegistry
from consentguard.stage_04_fusion_calibration.evidence.geometry import decode_binary_mask
from consentguard.stage_05_review_export.pipeline import ReviewExportService
from consentguard.stage_05_review_export.provider_factory import load_torchvision_provider


PROVIDER_LABELS = {
    "global": "Global privacy segmentation",
    "face-trained": "Trained face specialist",
    "plate-trained": "Trained plate specialist",
    "handwriting-trained": "Trained handwriting specialist",
    "yunet-face": "YuNet face safety net",
    "lpd-yunet": "LPD-YuNet plate safety net",
    "ppocr-text": "PP-OCR text geometry",
    "zxing-barcode": "ZXing barcode / QR",
}

PRIVACY_GROUPS = {
    "Face": {"a105_face_all", "face"},
    "License plate": {"a108_license_plate_all", "license_plate"},
    "Person / body": {"a109_person_body"},
    "Nudity": {"a110_nudity_all"},
    "Text / handwriting": {"a26_handwriting", "handwriting", "printed_text"},
    "Physical disability": {"a39_disability_physical"},
    "Medicine": {"a43_medicine"},
    "Fingerprint": {"a7_fingerprint"},
    "Signature": {"a8_signature"},
    "Barcode / QR": {"barcode"},
}

COLORS = (
    (12, 169, 165),
    (255, 107, 107),
    (133, 108, 189),
    (250, 190, 60),
    (47, 190, 220),
    (50, 184, 120),
)


def _version(path: Path) -> str:
    return f"{path.name}:{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"


def _selected_classes(groups: list[str] | None) -> set[str]:
    return {item for group in (groups or []) for item in PRIVACY_GROUPS.get(group, set())}


def _candidate_masks(candidate_set, selected_classes: set[str]) -> list[tuple[object, np.ndarray]]:
    selected = []
    for candidate in candidate_set.candidates:
        if selected_classes and not selected_classes.intersection(candidate.privacy_classes):
            continue
        mask = decode_binary_mask(candidate.mask_rle, candidate.height, candidate.width)
        selected.append((candidate, mask.astype(bool)))
    return selected


def _overlay(image: np.ndarray, candidates: list[tuple[object, np.ndarray]]) -> np.ndarray:
    output = image.copy()
    for index, (_, mask) in enumerate(candidates):
        color = np.asarray(COLORS[index % len(COLORS)], dtype=np.float32)
        output[mask] = np.clip(output[mask] * 0.45 + color * 0.55, 0, 255).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if xs.size:
            cv2.rectangle(
                output,
                (int(xs.min()), int(ys.min())),
                (int(xs.max()), int(ys.max())),
                tuple(int(value) for value in color),
                2,
            )
    return output


def _redact(image: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    output = image.copy()
    if not mask.any():
        return output
    if mode == "Solid fill":
        output[mask] = (20, 27, 45)
    elif mode == "Pixelate":
        height, width = output.shape[:2]
        small = cv2.resize(output, (max(1, width // 24), max(1, height // 24)), interpolation=cv2.INTER_AREA)
        pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        output[mask] = pixelated[mask]
    else:
        blurred = cv2.GaussianBlur(output, (0, 0), sigmaX=18, sigmaY=18)
        output[mask] = blurred[mask]
    return output


class DemoRuntime:
    def __init__(self, providers: dict[str, object], thresholds: ThresholdRegistry) -> None:
        self.providers = providers
        self.thresholds = thresholds

    def analyze(
        self,
        image_path: str | None,
        provider_labels: list[str] | None,
        privacy_groups: list[str] | None,
        redaction_mode: str,
    ):
        if not image_path:
            raise ValueError("Upload an image first")
        reverse_labels = {label: key for key, label in PROVIDER_LABELS.items()}
        provider_keys = [reverse_labels[label] for label in (provider_labels or []) if label in reverse_labels]
        selected_providers = tuple(self.providers[key] for key in provider_keys if key in self.providers)
        if not selected_providers:
            raise ValueError("Select at least one available model provider")
        service = ReviewExportService(selected_providers, self.thresholds)
        result = service.run(
            image_path,
            consent_state=ConsentState.UNKNOWN,
            review_completed=False,
        )
        selected_classes = _selected_classes(privacy_groups)
        candidates = _candidate_masks(result.candidates, selected_classes)
        union = np.zeros((result.image.height, result.image.width), dtype=bool)
        for _, mask in candidates:
            union |= mask
        overlay = _overlay(result.image.pixels_rgb, candidates)
        redacted = _redact(result.image.pixels_rgb, union, redaction_mode)
        summary = {
            "research_preview": True,
            "release_ready": False,
            "selected_providers": provider_keys,
            "providers_with_evidence": sorted({item.provider for item in result.evidence.evidence}),
            "unavailable_providers": list(result.evidence.unavailable_providers),
            "provider_errors": result.provider_errors,
            "raw_evidence_count": len(result.evidence.evidence),
            "fused_candidate_count": len(result.candidates.candidates),
            "displayed_candidate_count": len(candidates),
            "selected_privacy_groups": privacy_groups or [],
            "redacted_pixels": int(union.sum()),
            "threshold_profile": result.candidates.threshold_profile_id,
            "manual_review_required": True,
        }
        status = (
            f"Detected **{len(candidates)} displayed region(s)** from "
            f"**{len(result.evidence.evidence)} raw evidence item(s)**. "
            "This is a research preview; use the review/export gate before sharing an image."
        )
        return Image.fromarray(overlay), Image.fromarray(redacted), summary, status


def _add_checkpoint_provider(
    providers: dict[str, object],
    *,
    key: str,
    config_path: Path | None,
    checkpoint_path: Path | None,
    provider_name: str,
    device,
) -> None:
    if config_path is None or checkpoint_path is None or not checkpoint_path.is_file():
        return
    config = load_training_config(config_path, require_validation_data=False)
    resolved_provider_name = provider_name
    if str(config.section("model")["name"]).startswith("fasterrcnn"):
        if key.startswith("face"):
            resolved_provider_name = "face_fasterrcnn"
        elif key.startswith("plate"):
            resolved_provider_name = "plate_fasterrcnn"
    providers[key] = load_torchvision_provider(
        config,
        checkpoint_path,
        device,
        provider_name=resolved_provider_name,
    )


def build_runtime(args: argparse.Namespace) -> DemoRuntime:
    device = select_device(args.device)
    providers: dict[str, object] = {}
    _add_checkpoint_provider(
        providers,
        key="global",
        config_path=project_path(args.config),
        checkpoint_path=project_path(args.checkpoint),
        provider_name="maskrcnn",
        device=device,
    )
    for key, config_value, checkpoint_value, provider_name in (
        ("face-trained", args.face_config, args.face_checkpoint, "face_maskrcnn"),
        ("plate-trained", args.plate_config, args.plate_checkpoint, "plate_maskrcnn"),
        ("handwriting-trained", args.handwriting_config, args.handwriting_checkpoint, "handwriting_maskrcnn"),
    ):
        if checkpoint_value:
            _add_checkpoint_provider(
                providers,
                key=key,
                config_path=project_path(config_value),
                checkpoint_path=project_path(checkpoint_value),
                provider_name=provider_name,
                device=device,
            )
    for key, path_value, factory in (
        ("yunet-face", args.yunet_model, lambda path: YuNetFaceProvider(path, version=_version(path))),
        ("lpd-yunet", args.plate_yunet_model, lambda path: LPDYuNetPlateProvider(path, version=_version(path))),
        ("ppocr-text", args.ppocr_model, lambda path: PPOCRTextGeometryProvider(path, version=_version(path))),
    ):
        if path_value:
            path = project_path(path_value)
            if path.is_file():
                providers[key] = factory(path)
    if args.with_barcode:
        providers["zxing-barcode"] = ZXingBarcodeProvider()
    return DemoRuntime(providers, ThresholdRegistry.load(project_path(args.threshold_profile)))


def build_app(runtime: DemoRuntime):
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Install the UI dependencies with: pip install -e .[app]") from error

    available_provider_labels = [PROVIDER_LABELS[key] for key in PROVIDER_LABELS if key in runtime.providers]
    with gr.Blocks(
        title="ConsentGuard research preview",
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="amber"),
        css=".cg-note {border-left: 4px solid #0ca9a5; padding-left: 12px}",
    ) as app:
        gr.Markdown(
            "# ConsentGuard · image privacy preview\n"
            "Upload your own image, choose only the model branches you need, and inspect the fused redaction preview."
        )
        gr.Markdown(
            "**Research-only:** the current model bundle is not production certified and every result requires manual review.",
            elem_classes="cg-note",
        )
        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Image(
                    label="Your image",
                    type="filepath",
                    sources=["upload", "webcam", "clipboard"],
                )
                provider_select = gr.CheckboxGroup(
                    choices=available_provider_labels,
                    value=available_provider_labels,
                    label="Model branches to run",
                )
                privacy_select = gr.CheckboxGroup(
                    choices=list(PRIVACY_GROUPS),
                    value=list(PRIVACY_GROUPS),
                    label="Privacy regions to redact",
                )
                redaction_mode = gr.Radio(
                    ["Solid fill", "Pixelate", "Blur"],
                    value="Solid fill",
                    label="Preview redaction",
                )
                analyze = gr.Button("Analyze and create preview", variant="primary")
                status = gr.Markdown()
            with gr.Column(scale=2):
                with gr.Row():
                    overlay = gr.Image(label="Detected regions", interactive=False)
                    redacted = gr.Image(label="Redacted preview", interactive=False)
                details = gr.JSON(label="Evidence summary")
        analyze.click(
            runtime.analyze,
            inputs=[source, provider_select, privacy_select, redaction_mode],
            outputs=[overlay, redacted, details, status],
            concurrency_limit=1,
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="main_project/configs/stage_02_baseline_model/train_maskrcnn_moderate_v2_negatives_10ep.yaml")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/maskrcnn_moderate_v2_negatives_10ep/last.pt")
    parser.add_argument("--face-config", default="main_project/configs/stage_03_specialists/train_face_maskrcnn_5ep.yaml")
    parser.add_argument("--face-checkpoint", default="artifacts/checkpoints/specialist_face_maskrcnn_5ep/last.pt")
    parser.add_argument("--plate-config", default="main_project/configs/stage_03_specialists/train_plate_maskrcnn_5ep.yaml")
    parser.add_argument("--plate-checkpoint", default="artifacts/checkpoints/specialist_plate_maskrcnn_5ep/last.pt")
    parser.add_argument("--handwriting-config", default="main_project/configs/stage_03_specialists/train_handwriting_maskrcnn_5ep.yaml")
    parser.add_argument("--handwriting-checkpoint", default="artifacts/checkpoints/specialist_handwriting_maskrcnn_5ep/last.pt")
    parser.add_argument("--threshold-profile", default="main_project/configs/stage_04_fusion_calibration/threshold_profile_v2_validation_calibrated.yaml")
    parser.add_argument("--yunet-model", default="artifacts/specialists/opencv_zoo/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--plate-yunet-model", default="artifacts/specialists/opencv_zoo/license_plate_detection_lpd_yunet_2023mar.onnx")
    parser.add_argument("--ppocr-model", default="artifacts/specialists/opencv_zoo/text_detection_en_ppocrv3_2023may.onnx")
    parser.add_argument("--with-barcode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    runtime = build_runtime(args)
    build_app(runtime).queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
    )


if __name__ == "__main__":
    main()
