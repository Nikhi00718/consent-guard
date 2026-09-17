"""Launch the localhost-only ConsentGuard manual review and safe-export UI."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2

from consentguard.stage_05_review_export.assurance import AssuranceService, RenderedAsset
from consentguard.stage_05_review_export.ingest import normalize_image
from consentguard.stage_05_review_export.redaction.prediction_renderer import write_metadata_free_redaction
from consentguard.stage_05_review_export.review import editor_layers_to_mask


def build_app():
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Install the app dependencies with: pip install -e .[app]") from error

    def load_source(path: str):
        image = normalize_image(path)
        state = {"path": str(image.source_path), "width": image.width, "height": image.height}
        return {"background": image.pixels_rgb, "layers": [], "composite": image.pixels_rgb}, state, "Image normalized. Paint every region that must be hidden."

    def review_and_render(editor_value: dict, state: dict, review_complete: bool):
        if not review_complete:
            return None, None, "Export blocked: reviewer confirmation is required."
        image = normalize_image(state["path"])
        mask = editor_layers_to_mask(editor_value, width=image.width, height=image.height)
        if not mask.any():
            return None, None, "Export blocked: no approved redaction mask exists."
        output_dir = Path(tempfile.mkdtemp(prefix="consentguard-reviewed-"))
        output = output_dir / "reviewed-redaction.png"
        report = write_metadata_free_redaction(image.source_path, output, image.pixels_rgb, mask)
        assurance = AssuranceService().inspect(
            RenderedAsset(output, image.width, image.height, report)
        )
        audit_path = output.with_suffix(".audit.json")
        audit = {
            "review_completed": True,
            "export": report,
            "assurance_status": assurance.status.value,
            "checks": [
                {"name": check.name, "status": check.status.value, "reason_code": check.reason_code}
                for check in assurance.checks
            ],
        }
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if assurance.status.value != "PASS":
            return str(output), None, f"Preview created, but download blocked: assurance is {assurance.status.value}."
        return str(output), str(output), "Reviewed redaction passed assurance and is ready to download."

    with gr.Blocks(title="ConsentGuard Review") as app:
        gr.Markdown("# ConsentGuard — local privacy review")
        state = gr.State({})
        source = gr.File(label="Source JPEG/PNG/WebP", file_types=["image"], type="filepath")
        editor = gr.ImageEditor(label="Approved redaction mask", type="numpy", brush=gr.Brush(colors=["#000000"], default_size=24))
        review_complete = gr.Checkbox(label="I reviewed the full image and approved this mask")
        render = gr.Button("Render and verify", variant="primary")
        preview = gr.Image(label="Redacted preview", interactive=False)
        download = gr.File(label="Sanitized export")
        status = gr.Markdown()
        source.upload(load_source, inputs=source, outputs=[editor, state, status])
        render.click(review_and_render, inputs=[editor, state, review_complete], outputs=[preview, download, status], concurrency_limit=1)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, share=False
    )


if __name__ == "__main__":
    main()
