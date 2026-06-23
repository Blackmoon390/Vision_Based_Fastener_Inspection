from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Image,
    Spacer,
    HRFlowable,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from io import BytesIO
import os
import cv2
from reportlab.lib.utils import ImageReader

# ── Color palette ──────────────────────────────────────────────────────────────
PRIMARY   = colors.HexColor("#1A3A5C")   # dark navy
ACCENT    = colors.HexColor("#2E86C1")   # steel blue
LIGHT_BG  = colors.HexColor("#EBF5FB")  # pale blue
ALT_ROW   = colors.HexColor("#F4F6F7")  # light grey
GOOD      = colors.HexColor("#1E8449")  # green  (confidence high)
WARN      = colors.HexColor("#B7950B")  # amber  (confidence medium)
BAD       = colors.HexColor("#922B21")  # red    (confidence low)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")



def cv2_to_rl_image(img, width, height):
    """
    Converts an OpenCV (numpy.ndarray) image directly into
    a ReportLab Image without saving to disk.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode image.")

    bio = BytesIO(buf.tobytes())
    return Image(bio, width=width, height=height)


def _confidence_color(confidence: float) -> colors.Color:
    if confidence >= 95:
        return GOOD
    elif confidence >= 80:
        return WARN
    return BAD


def _make_styles() -> dict:
    base = getSampleStyleSheet()

    custom = {
        "ReportTitle": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=20,
            textColor=PRIMARY,
            spaceAfter=4,
            alignment=TA_CENTER,
        ),
        "SectionHeader": ParagraphStyle(
            "SectionHeader",
            parent=base["Heading2"],
            fontSize=12,
            textColor=PRIMARY,
            spaceBefore=14,
            spaceAfter=4,
            borderPad=2,
        ),
        "BoltID": ParagraphStyle(
            "BoltID",
            parent=base["BodyText"],
            fontSize=11,
            textColor=ACCENT,
            alignment=TA_CENTER,
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER,
        ),
        "DescBody": ParagraphStyle(
            "DescBody",
            parent=base["BodyText"],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#222222"),
        ),
        # FIX 2: Added missing ISONote style
        "ISONote": ParagraphStyle(
            "ISONote",
            parent=base["BodyText"],
            fontSize=8,
            textColor=colors.HexColor("#444444"),
            leading=12,
        ),
    }

    for name, style in custom.items():
        base.add(style)

    return base


def _divider(color=ACCENT, thickness=1.2):
    return HRFlowable(
        width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=2
    )


def _report_output_path(output_pdf: str) -> str:
    """Return a PDF path inside the local reports folder."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = os.path.basename(output_pdf)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    return os.path.join(REPORTS_DIR, filename)


def _measurement_table(measurement_data: dict, styles: dict):
    """Actual vs ISO comparison table for physical measurements."""
    header_style = ParagraphStyle(
        "MeasTH", parent=styles["BodyText"], fontSize=9,
        textColor=colors.white, alignment=TA_CENTER, fontName="Helvetica-Bold"
    )
    cell_style = ParagraphStyle(
        "MeasTD", parent=styles["BodyText"], fontSize=9, alignment=TA_CENTER
    )

    rows = [
        [
            Paragraph("Measurement", header_style),
            Paragraph("Measured Value (mm)", header_style),
        ],
        ["Bolt Height",       measurement_data.get("bolt_height_mm", "—")],
        ["Thread Height",     measurement_data.get("bolt_thread_height_mm", "—")],
        ["Total Width",       measurement_data.get("bolt_total_width_mm", "—")],
        ["Center Width",      measurement_data.get("bolt_center_width_mm", "—")],
        ["Bottom Width",      measurement_data.get("bolt_bottom_width_mm", "—")],
    ]

    # Convert non-header cells to Paragraphs for consistent styling
    for i in range(1, len(rows)):
        rows[i] = [
            Paragraph(str(rows[i][0]), cell_style),
            Paragraph(str(rows[i][1]), cell_style),
        ]

    t = Table(rows, colWidths=[3.5 * inch, 2.2 * inch])
    t.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",   (0, 0), (-1, 0), PRIMARY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _iso_table(iso_data: dict, styles: dict):
    """ISO / DB reference values table."""
    header_style = ParagraphStyle(
        "ISOTH", parent=styles["BodyText"], fontSize=9,
        textColor=colors.white, alignment=TA_CENTER, fontName="Helvetica-Bold"
    )
    cell_c = ParagraphStyle(
        "ISOTDc", parent=styles["BodyText"], fontSize=9, alignment=TA_CENTER
    )
    cell_l = ParagraphStyle(
        "ISOTDl", parent=styles["BodyText"], fontSize=9, alignment=TA_LEFT
    )

    confidence = iso_data.get("confidence", 0)
    conf_color = _confidence_color(confidence)

    # FIX 1: hexval() returns "#xRRGGBB" so strip 2 chars (not 1) to get "RRGGBB"
    hex_str = conf_color.hexval()[2:]

    rows = [
        [
            Paragraph("ISO Parameter",          header_style),
            Paragraph("Standard / DB Value",    header_style),
            Paragraph("Description",            header_style),
        ],
        [
            Paragraph("Bolt Size (ISO)",        cell_l),
            Paragraph(str(iso_data.get("bolt_size", "—")), cell_c),
            Paragraph("Metric thread designation per ISO 261", cell_l),
        ],
        [
            Paragraph("Thread Diameter (d<sub>b</sub>)", cell_l),
            Paragraph(f"{iso_data.get('thread_diameter_db', '—')} mm", cell_c),
            Paragraph("Nominal major thread diameter (ISO 724)", cell_l),
        ],
        [
            Paragraph("Head Height (k)",        cell_l),
            Paragraph(f"{iso_data.get('head_height_db', '—')} mm", cell_c),
            Paragraph("Hex bolt head height per ISO 4014 / 4017", cell_l),
        ],
        [
            Paragraph("Across Corners (a<sub>c</sub>)", cell_l),
            Paragraph(f"{iso_data.get('ac_db', '—')} mm", cell_c),
            Paragraph("Distance across corners of hex head (ISO 4014)", cell_l),
        ],
        [
            Paragraph("Detection Confidence",  cell_l),
            Paragraph(f"<font color='#{hex_str}'><b>{confidence}%</b></font>", cell_c),
            Paragraph("Model classification confidence score", cell_l),
        ],
    ]

    col_widths = [2.0 * inch, 1.8 * inch, 3.0 * inch]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    return t


def create_bolt_report(
    measurement_data: dict,
    iso_data: dict,
    bolt_id: str,
    original_image: str,
    measurement_image: str,
    output_pdf: str,
):
    output_pdf = _report_output_path(output_pdf)
    styles = _make_styles()
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    elements = []

    # ── Report header ───────────────────────────────────────────────────────────
    elements.append(Paragraph("Bolt Measurement Report", styles["ReportTitle"]))
    elements.append(Paragraph(f"Bolt ID: <b>{bolt_id}</b>", styles["BoltID"]))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(_divider(PRIMARY, thickness=2))

    # ── Images ──────────────────────────────────────────────────────────────────
    elements.append(Paragraph("Visual Inspection", styles["SectionHeader"]))
    elements.append(_divider())

    if hasattr(original_image, "shape"):
        img1 = cv2_to_rl_image(original_image, 2.8 * inch, 2.8 * inch)
    else:
        img1 = Image(original_image, width=2.8 * inch, height=2.8 * inch)
    if hasattr(measurement_image, "shape"):
        img2 = cv2_to_rl_image(measurement_image, 2.8 * inch, 2.8 * inch)
    else:
        img2 = Image(measurement_image, width=2.8 * inch, height=2.8 * inch)

    cap1 = Paragraph("Original Frame", styles["Caption"])
    cap2 = Paragraph("Measurement Overlay", styles["Caption"])

    image_table = Table(
        [[img1, img2], [cap1, cap2]],
        colWidths=[3.1 * inch, 3.1 * inch],
    )
    image_table.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",          (0, 0), (-1, 0),  0.5, colors.HexColor("#CCCCCC")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ]))
    elements.append(image_table)
    elements.append(Spacer(1, 0.25 * inch))

    # ── Physical measurements ───────────────────────────────────────────────────
    elements.append(Paragraph("Physical Measurements", styles["SectionHeader"]))
    elements.append(_divider())
    elements.append(
        Paragraph(
            "The table below presents dimensional measurements captured from the "
            "vision system. All values are expressed in millimetres (mm).",
            styles["DescBody"],
        )
    )
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(_measurement_table(measurement_data, styles))
    elements.append(Spacer(1, 0.25 * inch))

    # ── ISO / DB reference ──────────────────────────────────────────────────────
    elements.append(Paragraph("ISO Standard Reference Data", styles["SectionHeader"]))
    elements.append(_divider())

    bolt_size = iso_data.get("bolt_size", "Unknown")
    confidence = iso_data.get("confidence", 0)
    elements.append(
        Paragraph(
            f"The vision model classified this fastener as an <b>{bolt_size}</b> metric hex bolt "
            f"with a detection confidence of <b>{confidence}%</b>. "
            "The reference dimensions listed below are sourced from the ISO fastener database "
            "and correspond to the identified bolt size. These values serve as the ground-truth "
            "baseline for tolerance verification and quality control.",
            styles["DescBody"],
        )
    )
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(_iso_table(iso_data, styles))
    elements.append(Spacer(1, 0.12 * inch))

    # ISO note
    elements.append(
        Paragraph(
            "<b>Standards referenced:</b> ISO 261 (metric screw threads — general plan), "
            "ISO 724 (metric screw threads — basic dimensions), "
            "ISO 4014 / ISO 4017 (hexagon head bolts — dimensions).",
            styles["ISONote"],
        )
    )
    elements.append(Spacer(1, 0.25 * inch))

    # ── Footer rule ─────────────────────────────────────────────────────────────
    elements.append(_divider(PRIMARY, thickness=1))
    elements.append(
        Paragraph(
            "This report is generated automatically by the bolt measurement system. "
            "All measurements are for quality-assurance purposes only.",
            styles["Caption"],
        )
    )

    try:
        doc.build(elements)
        # print(f"Report saved → {output_pdf}")
    except Exception as e:
        import traceback
        print("\nPDF GENERATION FAILED")
        traceback.print_exc()


# ── Sample usage ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    measurement_data = {
        "bolt_height_mm":        29.49,
        "bolt_thread_height_mm": 25.27,
        "bolt_total_width_mm":   12.00,
        "bolt_center_width_mm":   5.69,
        "bolt_bottom_width_mm":   5.52,
    }

    iso_data = {
        "bolt_size":          "M6",
        "confidence":         98.7,
        "thread_diameter_db":  6.0,
        "head_height_db":      3.9,
        "ac_db":              11.547,
    }

    create_bolt_report(
        measurement_data=measurement_data,
        iso_data=iso_data,
        bolt_id="BOLT_0001",
        original_image=r"C:\Users\VISHNU\Downloads\files (1)\saved_frames\frame_00003_obj_0.png",
        measurement_image=r"C:\Users\VISHNU\Downloads\files (1)\saved_frames\frame_00003_obj_0.png",
        output_pdf="BOLT_0001.pdf",
    )
