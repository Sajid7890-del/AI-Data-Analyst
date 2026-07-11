"""
report_generator.py
--------------------
Builds downloadable reports in three formats:
  - CSV     : cleaned dataset + summary stats
  - Excel   : multi-sheet workbook (data, summary, correlation, outliers)
  - PDF     : formatted executive report with charts, using ReportLab

All functions return raw `bytes`, so app.py can feed them straight into
`st.download_button` without touching the filesystem.
"""

import io
from datetime import datetime

import pandas as pd
from openpyxl.styles import Font, PatternFill
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

from utils import get_basic_info, get_statistical_summary, get_correlation_matrix, detect_outliers
from ai_engine import generate_key_findings, generate_recommendations, generate_executive_summary


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------
def generate_csv_report(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Excel report
# ---------------------------------------------------------------------------
def generate_excel_report(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    header_fill = PatternFill(start_color="4C78A8", end_color="4C78A8", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Data", index=False)

        info = get_basic_info(df)
        summary_df = pd.DataFrame(
            [{"Metric": k, "Value": v} for k, v in info.items() if k != "column_types"]
        )
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        stats = get_statistical_summary(df)
        if not stats.empty:
            stats.to_excel(writer, sheet_name="Statistics")

        corr = get_correlation_matrix(df)
        if not corr.empty:
            corr.to_excel(writer, sheet_name="Correlation")

        outliers = detect_outliers(df)
        if outliers:
            pd.DataFrame(outliers).T.to_excel(writer, sheet_name="Outliers")

        # Style header rows on every sheet
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
            for col_cells in ws.columns:
                max_len = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 3, 40)

    return buffer.getvalue()


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
def generate_pdf_report(df: pd.DataFrame, dataset_name: str = "Dataset", chart_images: dict = None) -> bytes:
    """
    chart_images: optional dict of {title: png_bytes} to embed in the report
    (produced via charts.fig_to_png_bytes).
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], textColor=colors.HexColor("#4C78A8"))
    heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading2"], textColor=colors.HexColor("#333333"))
    body_style = styles["BodyText"]

    story = []

    # --- Cover ---
    story.append(Paragraph("AI Data Analyst Report", title_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Dataset: {dataset_name}", body_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(Spacer(1, 0.8 * cm))

    # --- Executive summary ---
    story.append(Paragraph("Executive Summary", heading_style))
    info = get_basic_info(df)
    exec_lines = [
        f"Rows: {info['rows']:,} | Columns: {info['columns']} | "
        f"Missing values: {info['missing_values']:,} | Duplicate rows: {info['duplicate_rows']:,}"
    ]
    for line in exec_lines:
        story.append(Paragraph(line, body_style))
    story.append(Spacer(1, 0.5 * cm))

    # --- Key findings ---
    story.append(Paragraph("Key Findings", heading_style))
    for line in generate_key_findings(df).split("\n"):
        story.append(Paragraph(line.replace("**", ""), body_style))
    story.append(Spacer(1, 0.5 * cm))

    # --- Recommendations ---
    story.append(Paragraph("Recommendations", heading_style))
    for line in generate_recommendations(df).split("\n"):
        story.append(Paragraph(line.replace("**", ""), body_style))
    story.append(Spacer(1, 0.5 * cm))

    # --- Statistical summary table ---
    stats = get_statistical_summary(df)
    if not stats.empty:
        story.append(Paragraph("Statistical Summary", heading_style))
        table_data = [["Column"] + list(stats.columns)] + [
            [idx] + [f"{v:.2f}" for v in row] for idx, row in stats.iterrows()
        ]
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C78A8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.5 * cm))

    # --- Charts ---
    if chart_images:
        story.append(PageBreak())
        story.append(Paragraph("Charts", heading_style))
        for title, png_bytes in chart_images.items():
            if not png_bytes:
                continue
            story.append(Paragraph(title, styles["Heading3"]))
            story.append(Image(io.BytesIO(png_bytes), width=16 * cm, height=9 * cm))
            story.append(Spacer(1, 0.4 * cm))

    doc.build(story)
    return buffer.getvalue()
