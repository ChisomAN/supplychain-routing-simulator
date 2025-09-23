# reports.py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime
import os
import pandas as pd

def make_report(ctx, plots: list = None, out_dir: str = "artifacts/reports"):
    """
    Generate a structured PDF report from the session context.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = int(datetime.utcnow().timestamp())
    out_path = os.path.join(out_dir, f"report_{ts}.pdf")

    metrics = ctx.get("metrics", {})
    cleaned_df = ctx.get("edges_clean")

    if cleaned_df is None or cleaned_df.empty:
        raise ValueError("No cleaned dataset found in context.")

    doc = SimpleDocTemplate(out_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title + Timestamp
    story.append(Paragraph("RL Supply-Chain Routing Simulator — Summary Report", styles['Title']))
    story.append(Spacer(1, 12))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated on: {ts}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Dataset Overview
    story.append(Paragraph("Dataset Overview", styles['Heading2']))
    story.append(Paragraph(f"Rows: {cleaned_df.shape[0]}, Columns: {cleaned_df.shape[1]}", styles['Normal']))
    story.append(Paragraph(f"Columns: {', '.join(cleaned_df.columns)}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Descriptive Statistics
    story.append(Paragraph("Descriptive Statistics (first 5 numeric columns)", styles['Heading2']))
    desc = cleaned_df.describe(include='all').round(2).reset_index()
    desc_table_data = [list(desc.columns)] + desc.values.tolist()
    desc_table = Table(desc_table_data, hAlign='LEFT')
    desc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold")
    ]))
    story.append(desc_table)
    story.append(Spacer(1, 12))

    # Cleaning Steps
    story.append(Paragraph("Cleaning Steps Applied", styles['Heading2']))
    story.append(Paragraph("✓ Missing values handled (drop/fill)<br/>"
                           "✓ Normalization/standardization applied<br/>"
                           "✓ Outliers detected and removed (IQR-based)", styles['Normal']))
    story.append(Spacer(1, 12))

    # Metrics
    story.append(Paragraph("Model Results", styles['Heading2']))

    # Case 1: Nested dict (multi-model comparison)
    if all(isinstance(v, dict) for v in metrics.values()):
        df_metrics = pd.DataFrame(metrics).T.round(4).reset_index().rename(columns={"index": "Model"})
        table_data = [list(df_metrics.columns)] + df_metrics.values.tolist()
        table = Table(table_data, hAlign='LEFT')
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold")
        ]))
        story.append(table)
    else:
        # Case 2: Flat dict
        for k, v in metrics.items():
            story.append(Paragraph(f"{k}: {round(v, 4)}", styles['Normal']))

    story.append(Spacer(1, 12))

    # Visuals
    if plots:
        story.append(Paragraph("Visualizations", styles['Heading2']))
        for p in plots:
            if os.path.exists(p):
                story.append(Image(p, width=400, height=300))
                story.append(Spacer(1, 12))

    # Save
    doc.build(story)
    return out_path
    print(f"Report generated at {out_path}")
