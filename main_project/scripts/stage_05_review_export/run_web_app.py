"""Launch the local React/FastAPI ConsentGuard reviewer workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from consentguard.shared.paths import project_path
from consentguard.stage_05_review_export.api import create_app
from run_demo_app import PRIVACY_GROUPS, PROVIDER_LABELS, build_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="main_project/configs/stage_02_baseline_model/train_maskrcnn_moderate_v2_negatives_10ep.yaml")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/maskrcnn_moderate_v2_negatives_10ep/last.pt")
    parser.add_argument("--face-config", default="main_project/configs/stage_03_specialists/train_face_maskrcnn_5ep.yaml")
    parser.add_argument("--face-checkpoint", default="artifacts/checkpoints/specialist_face_maskrcnn_5ep/last.pt")
    parser.add_argument(
        "--plate-config",
        default="main_project/configs/stage_03_specialists/train_plate_ccpd2020_india_finetune_5ep.yaml",
        help="Indian fine-tuned plate detector config (override to compare another checkpoint).",
    )
    parser.add_argument(
        "--plate-checkpoint",
        default="artifacts/checkpoints/specialist_plate_ccpd2020_india_finetune_5ep/best.pt",
        help="Indian fine-tuned plate detector checkpoint (override to compare another checkpoint).",
    )
    parser.add_argument("--handwriting-config", default="main_project/configs/stage_03_specialists/train_handwriting_maskrcnn_5ep.yaml")
    parser.add_argument("--handwriting-checkpoint", default="artifacts/checkpoints/specialist_handwriting_maskrcnn_5ep/last.pt")
    parser.add_argument("--threshold-profile", default="main_project/configs/stage_04_fusion_calibration/threshold_profile_v2_validation_calibrated.yaml")
    parser.add_argument("--yunet-model", default="artifacts/specialists/opencv_zoo/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--plate-yunet-model", default="artifacts/specialists/opencv_zoo/license_plate_detection_lpd_yunet_2023mar.onnx")
    parser.add_argument("--ppocr-model", default="artifacts/specialists/opencv_zoo/text_detection_en_ppocrv3_2023may.onnx")
    parser.add_argument("--with-barcode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--session-root", default="outputs/reviewer-sessions")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = build_runtime(args)
    frontend_dist = project_path("main_project/frontend/dist")
    app = create_app(
        runtime.providers,
        runtime.thresholds,
        provider_labels=PROVIDER_LABELS,
        privacy_groups=PRIVACY_GROUPS,
        session_root=project_path(args.session_root),
        frontend_dist=frontend_dist,
    )
    if not Path(frontend_dist).is_dir():
        raise RuntimeError("Frontend build missing. Run: npm --prefix main_project/frontend run build")
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
