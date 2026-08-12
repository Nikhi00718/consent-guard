"""Build the final ConsentGuard full data/model audit PDF with ReportLab."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "reports" / "full_audit"
SUMMARY_PATH = AUDIT_DIR / "audit_summary.json"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "ConsentGuard_Full_Data_and_Model_Audit.pdf"
PAGE_WIDTH, PAGE_HEIGHT = A4

NAVY = HexColor("#102A43")
BLUE = HexColor("#1677FF")
LIGHT_BLUE = HexColor("#EAF2FF")
RED = HexColor("#C62828")
LIGHT_RED = HexColor("#FFF0F0")
AMBER = HexColor("#B76E00")
LIGHT_AMBER = HexColor("#FFF7E6")
GREEN = HexColor("#1B7F3A")
LIGHT_GREEN = HexColor("#EDF8F0")
GRAY = HexColor("#536777")
LIGHT_GRAY = HexColor("#F4F6F8")
BORDER = HexColor("#D6DEE5")
INK = HexColor("#172B3A")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def metric(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def _font_setup() -> tuple[str, str, str]:
    candidates = [
        Path("C:/Windows/Fonts/aptos.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    bold_candidates = [
        Path("C:/Windows/Fonts/aptos-bold.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ]
    mono_candidates = [
        Path("C:/Windows/Fonts/cascadiamono.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
    ]
    regular = next((path for path in candidates if path.is_file()), None)
    bold = next((path for path in bold_candidates if path.is_file()), None)
    mono = next((path for path in mono_candidates if path.is_file()), None)
    if regular and bold:
        pdfmetrics.registerFont(TTFont("AuditSans", str(regular)))
        pdfmetrics.registerFont(TTFont("AuditSansBold", str(bold)))
        if mono:
            pdfmetrics.registerFont(TTFont("AuditMono", str(mono)))
        return "AuditSans", "AuditSansBold", "AuditMono" if mono else "Courier"
    return "Helvetica", "Helvetica-Bold", "Courier"


REGULAR_FONT, BOLD_FONT, MONO_FONT = _font_setup()


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "AuditTitle",
            parent=base["Title"],
            fontName=BOLD_FONT,
            fontSize=27,
            leading=31,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "AuditSubtitle",
            parent=base["Normal"],
            fontName=REGULAR_FONT,
            fontSize=12,
            leading=17,
            textColor=GRAY,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "AuditH1",
            parent=base["Heading1"],
            fontName=BOLD_FONT,
            fontSize=19,
            leading=23,
            textColor=NAVY,
            spaceBefore=4,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "AuditH2",
            parent=base["Heading2"],
            fontName=BOLD_FONT,
            fontSize=13.5,
            leading=17,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "AuditBody",
            parent=base["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=9.3,
            leading=13.2,
            textColor=INK,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "AuditSmall",
            parent=base["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=7.6,
            leading=10.3,
            textColor=GRAY,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "AuditCaption",
            parent=base["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=7.5,
            leading=10,
            textColor=GRAY,
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=3,
        ),
        "callout": ParagraphStyle(
            "AuditCallout",
            parent=base["BodyText"],
            fontName=BOLD_FONT,
            fontSize=11,
            leading=15,
            textColor=RED,
            spaceAfter=0,
        ),
        "table": ParagraphStyle(
            "AuditTable",
            parent=base["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=7.8,
            leading=10.2,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "AuditTableHead",
            parent=base["BodyText"],
            fontName=BOLD_FONT,
            fontSize=7.8,
            leading=10.2,
            textColor=colors.white,
        ),
        "mono": ParagraphStyle(
            "AuditMono",
            parent=base["BodyText"],
            fontName=MONO_FONT,
            fontSize=7.2,
            leading=9.4,
            textColor=INK,
        ),
        "reference": ParagraphStyle(
            "AuditReference",
            parent=base["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=7.4,
            leading=10,
            textColor=INK,
            leftIndent=10,
            firstLineIndent=-10,
            spaceAfter=4,
        ),
    }


ST = styles()


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, ST[style])


def bullet(text: str, *, level: int = 0) -> Paragraph:
    style = ParagraphStyle(
        f"bullet-{level}",
        parent=ST["body"],
        leftIndent=12 + level * 12,
        firstLineIndent=-8,
        bulletIndent=0 + level * 12,
        spaceAfter=3,
    )
    return Paragraph(text, style, bulletText="-")


def table(
    rows: list[list[Any]],
    widths: list[float],
    *,
    header: bool = True,
    font_size: float = 7.8,
    row_backgrounds: Iterable[colors.Color] | None = None,
) -> Table:
    converted: list[list[Any]] = []
    for row_index, row in enumerate(rows):
        style_name = "table_head" if header and row_index == 0 else "table"
        converted.append([cell if hasattr(cell, "wrap") else Paragraph(str(cell), ST[style_name]) for cell in row])
    result = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.45, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTNAME", (0, 0), (-1, -1), REGULAR_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), BOLD_FONT),
            ]
        )
    start = 1 if header else 0
    backgrounds = list(row_backgrounds or [colors.white, LIGHT_GRAY])
    for row_index in range(start, len(rows)):
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), backgrounds[(row_index - start) % len(backgrounds)]))
    result.setStyle(TableStyle(commands))
    return result


def callout(text: str, color: colors.Color = RED, background: colors.Color = LIGHT_RED) -> Table:
    paragraph_style = ParagraphStyle("DynamicCallout", parent=ST["callout"], textColor=color)
    content = Table([[Paragraph(text, paragraph_style)]], colWidths=[PAGE_WIDTH - 34 * mm])
    content.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 1.0, color),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return content


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def figure(path: Path, caption: str, *, max_height: float = 178 * mm) -> list[Any]:
    return [
        scaled_image(path, PAGE_WIDTH - 34 * mm, max_height),
        P(caption, "caption"),
    ]


def section_page(title: str, intro: str | None = None) -> list[Any]:
    prefix, separator, remainder = title.partition(". ")
    if separator and prefix.isdigit():
        title = remainder
    items: list[Any] = [PageBreak(), P(title, "h1"), HRFlowable(width="100%", thickness=1, color=BLUE), Spacer(1, 5)]
    if intro:
        items.append(P(intro))
    return items


def page_header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(17 * mm, PAGE_HEIGHT - 13 * mm, PAGE_WIDTH - 17 * mm, PAGE_HEIGHT - 13 * mm)
    canvas.setFont(REGULAR_FONT, 7.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(17 * mm, PAGE_HEIGHT - 10 * mm, "ConsentGuard - Full Data and Model Audit")
    canvas.drawRightString(PAGE_WIDTH - 17 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


class AuditDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs: Any) -> None:
        super().__init__(filename, **kwargs)
        frame = Frame(
            17 * mm,
            15 * mm,
            PAGE_WIDTH - 34 * mm,
            PAGE_HEIGHT - 31 * mm,
            id="normal",
            leftPadding=0,
            rightPadding=0,
            topPadding=4 * mm,
            bottomPadding=2 * mm,
        )
        self.addPageTemplates(PageTemplate(id="audit", frames=[frame], onPage=page_header_footer))


def build_story(summary: dict[str, Any]) -> list[Any]:
    story: list[Any] = []
    processed = summary["processed_data"]
    geometry = summary["geometry"]
    model = summary["model"]
    verdict = summary["verdict"]
    downloads = summary["download_integrity"]
    manual = pd.read_csv(AUDIT_DIR / "manual_visual_review.csv", dtype={"image_id": str})
    root_evidence = pd.read_csv(AUDIT_DIR / "root_cause_identity_evidence.csv", dtype={"image_id": str})
    support = pd.read_csv(AUDIT_DIR / "class_support.csv")
    shifts = pd.read_csv(AUDIT_DIR / "train_validation_shift.csv")

    # Cover
    story.extend(
        [
            Spacer(1, 14 * mm),
            P("ConsentGuard", "subtitle"),
            P("Full Data and Model Forensic Audit", "title"),
            P("Why the completed Mask R-CNN run failed, what must change, and the exact gate before retraining", "subtitle"),
            Spacer(1, 7 * mm),
            callout("STOP: Do not train another model on the current processed records.", RED, LIGHT_RED),
            Spacer(1, 8 * mm),
        ]
    )
    cards = [
        [P("Confirmed root cause", "table_head"), P("Evidence", "table_head")],
        [P("Wrong image identity", "table"), P("Visual Redactions masks were joined to different VISPR pixels through colliding ID strings.", "table")],
        [P("Manual audit", "table"), P(f"At least {processed['manual_stratified_review']['result_counts'].get('clearly_mismatched', 0)} of {processed['manual_stratified_review']['images_reviewed']} stratified examples are clearly mismatched.", "table")],
        [P("Full-train result", "table"), P(f"Mask mAP {metric(model['full_train_mask_map'])}; validation mAP {metric(model['validation_mask_map'])}.", "table")],
        [P("Pipeline control", "table"), P(f"Eight-image overfit reaches mask mAP {model['eight_image_overfit_best_mask_map']:.3f}, proving the training/evaluator path can learn coherent supervision.", "table")],
    ]
    story.append(table(cards, [49 * mm, 124 * mm], row_backgrounds=[colors.white, LIGHT_BLUE]))
    story.extend(
        [
            Spacer(1, 8 * mm),
            P("Final decision", "h2"),
            P("The current result is primarily a <b>data identity failure</b>, not evidence that Mask R-CNN is the wrong architecture. Correct the pixels first; then compare models under controlled conditions."),
            P("Prepared from the executed notebook, full source/processed-data audits, completed 30-epoch run, and full 501-image training evaluation. Date: 12 August 2026.", "small"),
            Spacer(1, 10 * mm),
            P("Deliverables", "h2"),
            bullet("Executed notebook: notebooks/ConsentGuard_Full_Data_Model_Audit.ipynb"),
            bullet("Machine-readable audit: reports/full_audit/audit_summary.json and CSV tables"),
            bullet("This PDF: output/pdf/ConsentGuard_Full_Data_and_Model_Audit.pdf"),
        ]
    )

    # Executive diagnosis
    story.extend(section_page("1. Executive diagnosis"))
    story.append(callout(f"Primary blocker: {verdict['primary_blocker']}.", RED, LIGHT_RED))
    story.extend(
        [
            Spacer(1, 5 * mm),
            P("What is healthy", "h2"),
            bullet("All VISPR archives, Visual Redactions annotation files, and the local VPD public repository completed their download inventories."),
            bullet("All 1,195 selected records decode; record dimensions match; 5,084 instances pass structural validation; there are zero out-of-bounds polygon points."),
            bullet("The official full split leakage audit found no exact or near cross-split duplicate groups."),
            bullet("CUDA, TorchVision detection operations, checkpointing, COCO evaluation, and the eight-image overfit control work."),
            P("What is invalid", "h2"),
            bullet("The masks are from Visual Redactions 2018, but the image pixels are from the VISPR 2017 archives."),
            bullet("The repeated `2017_xxxxxxxx` naming pattern was treated as a cross-release primary key. Direct weak-metadata and pixel inspection proves that this assumption is false for audited examples."),
            bullet("Aspect ratio equality is only a geometry test. Different images frequently share the same aspect ratio, so the v3 filter retained semantically unrelated pairs."),
            P("What the model result means", "h2"),
            bullet(f"Full-train mask mAP is {metric(model['full_train_mask_map'])}, so the model cannot fit the contradictory 501-image supervision well."),
            bullet(f"Validation mask mAP is {metric(model['validation_mask_map'])}, only {pct(model['validation_to_train_map_ratio'], 2)} of train mAP, adding a severe generalization collapse."),
            bullet("A stronger model would learn the same wrong target-pixel associations more expensively. Architecture changes must wait."),
        ]
    )

    # Root cause
    story.extend(section_page("2. Root cause: an invalid cross-dataset identity join", "Five direct examples compare the original Visual Redactions weak metadata with the VISPR pixels used during training."))
    rows = [["Image ID", "Visual Redactions source description", "VISPR pixels used", "Finding"]]
    for row in root_evidence.itertuples(index=False):
        rows.append([row.image_id, row.visual_redactions_weak_metadata, row.current_vispr_pixels, row.conclusion])
    story.append(table(rows, [30 * mm, 55 * mm, 43 * mm, 45 * mm]))
    story.extend(
        [
            Spacer(1, 6 * mm),
            P("Why the previous checks passed", "h2"),
            bullet("The decoded image and annotation dimensions were different but had the same ratio for 1,433 source records. That proves a resize transform is geometrically possible; it does not prove the image is the annotated image."),
            bullet("Some incorrect masks intersect real edges by chance, especially in crowded scenes. Automated edge support can prioritize review but cannot certify semantic identity."),
            bullet("The eight-image overfit gate used a small fixed subset. Memorization of a few noisy samples is possible and does not certify the full corpus."),
            P("Correct data source", "h2"),
            P("The official Visual Redactions host exposes separate image archives. They were not included in the earlier downloader because the project guide incorrectly assumed that VISPR pixels could be reused."),
        ]
    )
    archive_rows = [["Split", "Compressed size", "Official archive"]]
    total = 0
    for split, entry in downloads["visual_redactions_required_image_archives"].items():
        total += int(entry["bytes"])
        archive_rows.append([split, f"{entry['bytes'] / 1024**3:.3f} GiB", entry["url"]])
    archive_rows.append(["Total", f"{total / 1024**3:.3f} GiB", "Download only after freeing safe extraction space."])
    story.append(table(archive_rows, [25 * mm, 31 * mm, 117 * mm]))
    story.extend(
        [
            Spacer(1, 5 * mm),
            callout("Storage warning: only about 31.3 GB was free during this audit. The 15.467 GiB compressed archives plus extraction and atomic validation need substantially more headroom. Free space before downloading.", AMBER, LIGHT_AMBER),
        ]
    )

    # Figures: geometry, class, sizes
    story.extend(section_page("3. Dataset inventory and geometry"))
    story.extend(figure(AUDIT_DIR / "01_download_and_geometry.png", "Figure 1. Downloads are complete, but only 16.9% of source image/annotation pairs pass the aspect-ratio geometry screen. That screen did not certify image identity.", max_height=145 * mm))
    story.append(P("Interpretation", "h2"))
    story.append(P(f"Of {summary['scope']['source_manifest_records']:,} source records, {geometry['aligned_resize']:,} were classified as aligned resizes, {geometry['rotation_candidate']:,} as rotation candidates, {geometry['geometry_mismatch']:,} as geometry mismatches, and {geometry['missing_image']:,} as missing. The selected model corpus contains 501 train, 294 validation, and 400 locked-test images."))

    story.extend(section_page("4. Class support and long-tail risk"))
    story.extend(figure(AUDIT_DIR / "02_class_support.png", "Figure 2. Common classes dominate. Six train classes have fewer than 30 images or fewer than 50 instances."))
    rare_rows = [["Class", "Train images", "Train instances", "Validation images", "Validation instances"]]
    train_support = support[support["split"] == "train2017"].set_index("class_short")
    val_support = support[support["split"] == "val2017"].set_index("class_short")
    for item in processed["rare_train_classes"]:
        name = item["class"]
        rare_rows.append([name, int(train_support.loc[name, "images"]), int(train_support.loc[name, "instances"]), int(val_support.loc[name, "images"]), int(val_support.loc[name, "instances"])])
    story.append(table(rare_rows, [55 * mm, 28 * mm, 30 * mm, 30 * mm, 30 * mm]))
    story.append(P("Even with correct pixels, these rare-class metrics will have high variance. For classes with fewer than 25 images, review every image and report confidence intervals or descriptive results rather than strong generalization claims.", "small"))

    story.extend(section_page("5. Small-object difficulty"))
    story.extend(figure(AUDIT_DIR / "03_object_sizes.png", "Figure 3. Object areas at the 640/1024 model input. A total of 124 train/validation instances have less than a 16 x 16-pixel equivalent area."))
    story.append(P("Mask R-CNN with FPN and 16-pixel anchors is a reasonable baseline, but tiny handwriting, signatures, plates, and faces need higher-resolution crops or specialist detectors. This remains a secondary issue after data identity repair."))

    # Image quality and annotations
    story.extend(section_page("6. Image-quality and split comparison"))
    story.extend(figure(AUDIT_DIR / "04_image_quality_shift.png", "Figure 4. Train and validation have similar brightness, contrast, sharpness, and resolution distributions. Ordinary image quality is not the main cause of the near-zero AP."))
    shift_rows = [["Measure", "Train mean", "Validation mean", "Standardized difference", "KS statistic"]]
    for row in shifts.itertuples(index=False):
        if row.metric == "class_distribution_jensen_shannon":
            continue
        shift_rows.append([row.metric, f"{row.train_mean:.4g}", f"{row.val_mean:.4g}", f"{row.standardized_difference_val_minus_train:.3f}", f"{row.ks_statistic:.3f}"])
    story.append(table(shift_rows, [50 * mm, 30 * mm, 32 * mm, 34 * mm, 27 * mm], font_size=7.2))

    story.extend(section_page("7. Annotation alignment proxies"))
    story.extend(figure(AUDIT_DIR / "05_annotation_quality.png", "Figure 5. Structural checks pass, while the boundary-edge proxy exposes a suspicious low-support tail. Edge support is a review heuristic, not proof of correctness.", max_height=168 * mm))

    story.extend(section_page("8. Class co-occurrence"))
    story.extend(figure(AUDIT_DIR / "06_class_cooccurrence.png", "Figure 6. Person and face labels dominate co-occurrence. Rare labels often share images with dominant classes, complicating image-level balanced sampling.", max_height=172 * mm))

    # Manual review
    story.extend(section_page("9. Manual ground-truth review", "The representative grid selects one median edge-lift example per class and split. This avoids choosing only the most obviously bad cases."))
    story.extend(figure(AUDIT_DIR / "09_representative_ground_truth_grid.jpg", "Figure 7. Stratified representative ground-truth overlays. Sensitive nudity regions are dark-filled in report previews.", max_height=200 * mm))

    story.extend(section_page("10. Manual review outcome"))
    review_counts = manual["review_result"].value_counts().to_dict()
    story.append(callout(f"At least {review_counts.get('clearly_mismatched', 0)} of {len(manual)} stratified examples ({pct(review_counts.get('clearly_mismatched', 0) / len(manual))}) are clearly paired with unrelated pixels.", RED, LIGHT_RED))
    story.extend(
        [
            Spacer(1, 5 * mm),
            P("Review categories", "h2"),
        ]
    )
    review_rows = [["Focus class", "Split", "Image", "Judgment", "Reason"]]
    for row in manual.itertuples(index=False):
        review_rows.append([row.focus_class, row.split.replace("2017", ""), row.image_id, row.review_result.replace("_", " "), row.review_note])
    story.append(table(review_rows, [31 * mm, 15 * mm, 28 * mm, 28 * mm, 71 * mm], font_size=6.7))

    story.extend(section_page("11. Low edge-support review set"))
    story.extend(figure(AUDIT_DIR / "10_low_edge_support_review_grid.jpg", "Figure 8. Images selected by low median boundary edge lift. Many masks visibly float over unrelated objects; handwriting examples illustrate that low support can also occur on valid low-contrast boundaries.", max_height=185 * mm))

    # Training curves and metrics
    story.extend(section_page("12. Training behavior"))
    story.extend(figure(AUDIT_DIR / "07_training_curves.png", "Figure 9. Training loss falls, but validation localization remains near zero over all 30 epochs. Low detection loss can coexist with background-dominated or contradictory supervision.", max_height=168 * mm))
    story.append(P("The checkpoint retained 100 low-threshold predictions per validation image, so near-zero AP is not caused by an overly aggressive external confidence threshold."))

    story.extend(section_page("13. Full-train versus validation evaluation"))
    story.extend(figure(AUDIT_DIR / "08_model_train_vs_validation.png", "Figure 10. The final checkpoint was evaluated on all 501 deterministic training images for this audit; validation metrics come from the completed final epoch.", max_height=152 * mm))
    metric_rows = [
        ["Metric", "Full train", "Validation", "Interpretation"],
        ["Mask mAP", metric(model["full_train_mask_map"]), metric(model["validation_mask_map"]), "Severe train underfit plus generalization collapse"],
        ["Mask AP50", metric(model["full_train_mask_map_50"]), metric(model["validation_mask_map_50"]), "Almost no usable validation overlap"],
        ["Eight-image overfit best mAP", f"{model['eight_image_overfit_best_mask_map']:.3f}", "same images", "Core pipeline can memorize coherent small subsets"],
        ["Eight-image overfit final AP50", f"{model['eight_image_overfit_final_mask_map_50']:.3f}", "same images", "Metric and mask path are operational"],
    ]
    story.append(table(metric_rows, [42 * mm, 30 * mm, 30 * mm, 71 * mm]))

    story.extend(section_page("14. Qualitative final-checkpoint predictions"))
    story.extend(figure(AUDIT_DIR / "11_final_model_predictions_grid.jpg", "Figure 11. Green contours are ground truth; translucent colored regions and boxes are model predictions. Multiple ground truths themselves target unrelated content, and predictions are low-confidence or dominated by person/body classes.", max_height=205 * mm))

    # Sampler and model decisions
    story.extend(section_page("15. Sampler and optimization risks"))
    sampler = summary["sampler"]
    story.extend(
        [
            P("The capped inverse-square-root image sampler draws with replacement. Its expected behavior per 501-draw epoch is:"),
            table(
                [
                    ["Sampler quantity", "Value"],
                    ["Expected unique images", f"{sampler['expected_unique_images_per_epoch']:.1f} / 501"],
                    ["Expected duplicate draws", f"{sampler['expected_duplicate_draws_per_epoch']:.1f} / 501"],
                    ["Maximum/minimum image probability", f"{sampler['max_to_min_probability_ratio']:.2f}x"],
                ],
                [80 * mm, 45 * mm],
            ),
            Spacer(1, 5 * mm),
            P("This means only about 276 unique images are expected in a nominal epoch. The sampler is not the confirmed root cause, but after data repair it must be compared with uniform sampling. Oversampling six physical-disability images or seven medicine images can memorize noise and distort epoch semantics."),
            P("Recommended recipe ablation", "h2"),
            bullet("Uniform sampler baseline, same seed and resolution."),
            bullet("Balanced sampler with replacement, current cap."),
            bullet("Class-aware batch/crop strategy without discarding common-class diversity."),
            bullet("Report per-class support, AP, recall, and bootstrap intervals; never rely only on aggregate mAP."),
        ]
    )

    story.extend(section_page("16. Architecture decision after data repair"))
    architecture_rows = [
        ["Component", "Recommended role", "Decision now", "Reason"],
        ["Mask R-CNN ResNet50-FPN v2", "Reproducible pixel-mask baseline for nine visual labels", "Keep as first repaired baseline", "The current run cannot judge it because targets are invalid; it passes the overfit control."],
        ["Mask2Former", "Higher-capacity instance-segmentation comparison", "Defer", "Valid comparison after clean data, but higher VRAM/dependency cost and no protection against bad labels."],
        ["Face/person specialist", "High-recall safety branch", "Add after baseline", "Common privacy objects benefit from mature specialist pretraining and separate thresholds."],
        ["PaddleOCR text detector/recognizer", "Text boxes plus recognition for names, addresses, IDs, handwriting", "Required separate branch", "A visual mask classifier alone does not cover textual privacy semantics."],
        ["Document/layout detector + OCR/NER", "Passports, cards, mail, receipts, tickets", "Required separate branch", "Document identity and field semantics are multimodal."],
        ["SAM 2", "Refine detector/OCR boxes into masks", "Optional refinement", "SAM 2 is promptable segmentation, not a privacy-class detector by itself."],
        ["VPD-100K", "Potential box-detector pretraining/benchmark", "Do not use local videos as image labels", "The local repository has videos but no verified 100K-image annotation package."],
    ]
    story.append(table(architecture_rows, [34 * mm, 46 * mm, 35 * mm, 58 * mm], font_size=7.1))
    story.extend(
        [
            Spacer(1, 5 * mm),
            callout("Data first, baseline second, architecture comparison third. Switching models before repairing identity would repeat the same failure.", BLUE, LIGHT_BLUE),
        ]
    )

    story.extend(section_page("17. Recovery plan and go/no-go gates"))
    steps = [
        ("0 - Quarantine", "Mark the current processed records and v3 checkpoint invalid for scientific use. Preserve them only for root-cause reproduction."),
        ("1 - Storage", "Free enough disk for 15.467 GiB compressed archives, extraction, validation copies, processed records, and checkpoints. Keep a safety reserve."),
        ("2 - Correct download", "Download train/validation/test image archives from the `orekondy18cvpr/v1/images/` host with resume locks; verify byte lengths and full sequential tar reads."),
        ("3 - Same-release preprocessing", "Pair Visual Redactions JSON with Visual Redactions image archives by split/path. Never resolve pixels from VISPR by ID."),
        ("4 - Identity gate", "Require decode dimensions, exact file inventory, weak-metadata sanity, and stratified overlays. Review at least 25 images per common class and every image for very rare classes."),
        ("5 - Leakage gate", "Freeze official splits and run SHA-256 plus perceptual-hash cross-split checks on the corrected files."),
        ("6 - Learning ladder", "One image -> eight images -> 32 images. Require meaningful mask AP/AP50 before any full run."),
        ("7 - Short baseline", "Run five epochs with uniform sampling. Check train and validation AP, per-class recall, prediction scores, and overlays."),
        ("8 - Controlled ablations", "Compare uniform vs balanced sampling, resolution/crops, and only then Mask R-CNN vs Mask2Former or specialist branches."),
        ("9 - Locked test", "Use test once after architecture, thresholds, and redaction policy are frozen."),
    ]
    gate_rows = [["Gate", "Required evidence before proceeding"]] + [[name, description] for name, description in steps]
    story.append(table(gate_rows, [38 * mm, 135 * mm], font_size=7.4))
    story.extend(
        [
            Spacer(1, 5 * mm),
            P("Engineering learning gates (not publication promises)", "h2"),
            bullet("Eight-image repaired subset: mask mAP >= 0.25 and AP50 >= 0.40."),
            bullet("32-image repaired subset: clear train-AP growth and visually correct masks across every represented class."),
            bullet("First full repaired baseline: full-train mAP materially above 0.10 and validation mAP materially above current 0.000256 before spending on long architecture sweeps."),
            bullet("Safety claim: AP alone is insufficient. Run redaction leakage tests, OCR/re-identification attacks, mask dilation checks, and metadata stripping verification."),
        ]
    )

    story.extend(section_page("18. Failure register"))
    risk_rows = [
        ["Failure", "Status", "Signal", "Mitigation"],
        ["Cross-dataset ID collision", "CONFIRMED", "Weak metadata and pixels describe different content", "Use same-release Visual Redactions image archives; no VISPR fallback"],
        ["Aspect ratio treated as identity", "CONFIRMED", "Wrong overlays pass geometry", "Aspect ratio is necessary only; add content/overlay gate"],
        ["Corrupt/incomplete downloads", "NOT OBSERVED", "Byte inventories and record decodes pass", "Retain sequential archive validation and checksums"],
        ["Rare-class instability", "OPEN", "Six classes below 30 train images or 50 instances", "Review all rare examples; acquire/annotate more; report uncertainty"],
        ["Small-object misses", "OPEN", "124 instances below 16 x 16 area; small AP zero", "Higher-resolution crops and specialist OCR/face/plate branches"],
        ["Sampler over-repetition", "OPEN", "About 225 duplicate draws per epoch expected", "Uniform-vs-balanced ablation"],
        ["Metric/evaluator bug", "UNLIKELY", "Eight-image overfit reaches 0.303 mAP", "Retain fixed overfit regression gate"],
        ["Model architecture mismatch", "SECONDARY", "One visual branch cannot cover text/doc semantics", "Modular visual, OCR, and document branches"],
        ["VPD videos mistaken for VPD-100K boxes", "PREVENTED", "No local JSON/XML/YAML annotations", "Keep VPD out until official image annotations are verified"],
        ["Premature test tuning", "PREVENTED", "No final-checkpoint test evaluation in this audit", "Keep test locked until protocol freeze"],
    ]
    status_colors = [colors.white, LIGHT_RED, LIGHT_RED, LIGHT_GREEN, LIGHT_AMBER, LIGHT_AMBER, LIGHT_AMBER, LIGHT_GREEN, LIGHT_AMBER, LIGHT_GREEN, LIGHT_GREEN]
    risk_table = table(risk_rows, [45 * mm, 24 * mm, 48 * mm, 56 * mm], font_size=6.9, row_backgrounds=[colors.white])
    risk_table.setStyle(TableStyle([("BACKGROUND", (0, index), (-1, index), status_colors[index]) for index in range(1, len(risk_rows))]))
    story.append(risk_table)

    story.extend(section_page("19. Method, limitations, and references"))
    story.extend(
        [
            P("Audit method", "h2"),
            bullet("Decoded all 1,195 selected files; checked dimensions and basic quality statistics."),
            bullet("Profiled all 5,084 polygons for model-input size, clipping, border contact, mask density, and edge-boundary support."),
            bullet("Measured train-validation distribution shifts, class support, co-occurrence, split leakage evidence, and expected sampler repetition."),
            bullet("Parsed all 30 epochs of loss and validation metrics."),
            bullet("Evaluated the final checkpoint on all 501 deterministic training images with the same COCO evaluator used in training."),
            bullet("Inspected a stratified 18-image overlay grid and low-edge-support review grid; judgments are recorded in a CSV."),
            P("Limitations", "h2"),
            bullet("Manual review is a stratified diagnostic sample, not a complete human relabel of 1,195 images. It is sufficient to invalidate the corpus because even a few confirmed identity mismatches violate supervision integrity."),
            bullet("Boundary edge lift is not an annotation truth metric. It is reported only as a triage signal."),
            bullet("Current COCO metrics are not directly comparable with the Visual Redactions paper's threshold-swept pixel AP procedure."),
            bullet("No model was evaluated on the locked test split during this audit."),
            P("Primary sources", "h2"),
            P('1. Orekondy, Fritz, and Schiele. <link href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Orekondy_Connecting_Pixels_to_CVPR_2018_paper.pdf">Connecting Pixels to Privacy and Utility</link>, CVPR 2018.', "reference"),
            P('2. Official Visual Redactions code and evaluation: <link href="https://github.com/tribhuvanesh/visual_redactions">github.com/tribhuvanesh/visual_redactions</link>.', "reference"),
            P('3. TorchVision Mask R-CNN ResNet50-FPN v2 documentation: <link href="https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html">docs.pytorch.org/vision</link>.', "reference"),
            P('4. Cheng et al. <link href="https://arxiv.org/abs/2112.01527">Masked-attention Mask Transformer for Universal Image Segmentation (Mask2Former)</link>, CVPR 2022.', "reference"),
            P('5. Meta AI. <link href="https://ai.meta.com/research/sam2/">Segment Anything Model 2</link> - promptable image/video segmentation.', "reference"),
            P('6. PaddleOCR official documentation: <link href="https://www.paddleocr.ai/latest/en/index.html">paddleocr.ai</link>.', "reference"),
            P('7. VPD-100K official project page: <link href="https://vpd-100k.github.io/">vpd-100k.github.io</link>. The local public repository inventory does not currently contain the image annotation package described there.', "reference"),
            Spacer(1, 5 * mm),
            callout("Bottom line: replace the pixels, rebuild and verify the records, then rerun the baseline. Do not change models first.", GREEN, LIGHT_GREEN),
        ]
    )
    return story


def build_pdf(output: Path) -> Path:
    summary = load_json(SUMMARY_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = ROOT / "tmp" / "pdfs" / (output.stem + ".building.pdf")
    output_tmp.parent.mkdir(parents=True, exist_ok=True)
    doc = AuditDocTemplate(
        str(output_tmp),
        pagesize=A4,
        title="ConsentGuard Full Data and Model Forensic Audit",
        author="ConsentGuard research audit",
        subject="Data integrity, Mask R-CNN diagnosis, and recovery plan",
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=16 * mm,
        bottomMargin=15 * mm,
    )
    doc.build(build_story(summary))
    os.replace(output_tmp, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = build_pdf(output)
    print(json.dumps({"pdf": str(result), "bytes": result.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
