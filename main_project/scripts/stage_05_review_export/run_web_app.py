"""Launch the local ConsentGuard redaction workspace (React + FastAPI).

Defaults are the personal-tool configuration: every detector on, low
thresholds, a tiled second pass for small regions, and the person at the
keyboard as the release gate. Pass ``--policy-mode research`` to get the
original consent-and-calibration-gated behaviour back.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from consentguard.shared.paths import project_path
from consentguard.stage_03_specialists.tiled import TiledEvidenceProvider
from consentguard.stage_05_review_export.api import create_app
from consentguard.stage_05_review_export.policy import PERSONAL_MODE, RESEARCH_MODE
from run_demo_app import PRIVACY_GROUPS, PROVIDER_LABELS, build_runtime


#: Honest per-group reliability, from the 120-image validation scorecard in
#: reports/PERSONAL_PROFILE_SCORECARD_2026-09-18.md. "reliable" needs both a
#: high pixel recall and a high instance recall; plates cover 95% of plate
#: pixels but only 56% of individual plates, which is why they are not
#: "reliable" here. Fingerprint detection never fires at all (0% instance
#: recall). Anything not "reliable" is surfaced as "check this yourself".
GROUP_RELIABILITY = {
    "Face": "reliable",
    "Person / body": "reliable",
    "Barcode / QR": "reliable",
    "License plate": "check_manually",
    "Text / handwriting": "check_manually",
    "Nudity": "check_manually",
    "Physical disability": "check_manually",
    "Medicine": "check_manually",
    "Fingerprint": "check_manually",
    "Signature": "check_manually",
}

#: Erase everything by default except whole bodies: blacking out a body removes
#: most of a photo, so it stays an explicit opt-in.
DEFAULT_OFF_GROUPS = ("Person / body",)

#: Providers worth re-running over tiles. Faces, plates and text are the classes
#: whose failures are concentrated in small regions.
TILED_PROVIDER_KEYS = ("global", "face-trained", "plate-trained", "ppocr-text")

#: The v4 full-scene plate checkpoint stays outside ordinary Git (346 MB). It
#: beats the previous website default on every measured number, so it is the
#: default here while remaining a research artifact.
PLATE_V4_CHECKPOINT = (
    "artifacts/kaggle/remote-runs/plate-full-scene-v4/consentguard/artifacts/checkpoints/"
    "specialist_plate_full_scene_research_v1_highres_5ep/best.pt"
)
PLATE_V4_CONFIG = "main_project/configs/stage_03_specialists/train_plate_full_scene_research_v1_highres_5ep.yaml"
PLATE_FALLBACK_CHECKPOINT = "artifacts/checkpoints/specialist_plate_ccpd2020_india_finetune_5ep/best.pt"
PLATE_FALLBACK_CONFIG = "main_project/configs/stage_03_specialists/train_plate_ccpd2020_india_finetune_5ep.yaml"


def _plate_defaults() -> tuple[str, str]:
    if project_path(PLATE_V4_CHECKPOINT).is_file():
        return PLATE_V4_CONFIG, PLATE_V4_CHECKPOINT
    return PLATE_FALLBACK_CONFIG, PLATE_FALLBACK_CHECKPOINT


def build_parser() -> argparse.ArgumentParser:
    plate_config, plate_checkpoint = _plate_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="main_project/configs/stage_02_baseline_model/train_maskrcnn_moderate_v2_negatives_10ep.yaml")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/maskrcnn_moderate_v2_negatives_10ep/last.pt")
    parser.add_argument("--face-config", default="main_project/configs/stage_03_specialists/train_face_maskrcnn_5ep.yaml")
    parser.add_argument("--face-checkpoint", default="artifacts/checkpoints/specialist_face_maskrcnn_5ep/last.pt")
    parser.add_argument(
        "--plate-config",
        default=plate_config,
        help="Plate detector config (defaults to the full-scene v4 candidate when present).",
    )
    parser.add_argument(
        "--plate-checkpoint",
        default=plate_checkpoint,
        help="Plate detector checkpoint (defaults to the full-scene v4 candidate when present).",
    )
    parser.add_argument("--handwriting-config", default="main_project/configs/stage_03_specialists/train_handwriting_maskrcnn_5ep.yaml")
    parser.add_argument("--handwriting-checkpoint", default="artifacts/checkpoints/specialist_handwriting_maskrcnn_5ep/last.pt")
    parser.add_argument(
        "--threshold-profile",
        default="main_project/configs/stage_04_fusion_calibration/threshold_profile_personal_max_coverage.yaml",
    )
    parser.add_argument("--yunet-model", default="artifacts/specialists/opencv_zoo/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--plate-yunet-model", default="artifacts/specialists/opencv_zoo/license_plate_detection_lpd_yunet_2023mar.onnx")
    parser.add_argument("--ppocr-model", default="artifacts/specialists/opencv_zoo/text_detection_en_ppocrv3_2023may.onnx")
    parser.add_argument("--with-barcode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--nudenet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the pretrained NudeNet detector for intimate content when it is installed (AGPL-3.0).",
    )
    parser.add_argument(
        "--policy-mode",
        choices=(PERSONAL_MODE, RESEARCH_MODE),
        default=PERSONAL_MODE,
        help="personal: you are the release gate. research: consent and calibration gates block export.",
    )
    parser.add_argument(
        "--tiled-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-run detectors over overlapping tiles so small regions are found.",
    )
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--max-tiles", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--session-root", default="outputs/reviewer-sessions")
    parser.add_argument(
        "--session-ttl-seconds",
        type=int,
        default=900,
        help="Backstop cleanup for session copies; the interface also deletes on close.",
    )
    return parser


def add_tiled_providers(providers: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    """Add a `<key>-tiled` companion for each provider worth a second pass."""

    extended = dict(providers)
    for key in TILED_PROVIDER_KEYS:
        provider = providers.get(key)
        if provider is None:
            continue
        extended[f"{key}-tiled"] = TiledEvidenceProvider(
            provider,
            tile_size=args.tile_size,
            max_tiles=args.max_tiles,
        )
    return extended


def main() -> None:
    args = build_parser().parse_args()
    runtime = build_runtime(args)
    providers = add_tiled_providers(runtime.providers, args) if args.tiled_pass else runtime.providers
    labels = dict(PROVIDER_LABELS)
    for key in providers:
        if key.endswith("-tiled"):
            labels[key] = f"{labels.get(key[: -len('-tiled')], key)} · tiled pass"
    frontend_dist = project_path("main_project/frontend/dist")
    if not Path(frontend_dist).is_dir():
        raise RuntimeError("Frontend build missing. Run: npm --prefix main_project/frontend run build")
    app = create_app(
        providers,
        runtime.thresholds,
        provider_labels=labels,
        privacy_groups=PRIVACY_GROUPS,
        session_root=project_path(args.session_root),
        frontend_dist=frontend_dist,
        ttl_seconds=args.session_ttl_seconds,
        policy_mode=args.policy_mode,
        default_privacy_groups=tuple(
            group for group in PRIVACY_GROUPS if group not in DEFAULT_OFF_GROUPS
        ),
        group_reliability=GROUP_RELIABILITY,
    )
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
