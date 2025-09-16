import os
import time
from typing import Dict

try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    _HAS_REPORTLAB = True
except Exception:
    _HAS_REPORTLAB = False


def make_report(context: Dict, outdir: str = "artifacts/reports") -> str:
    os.makedirs(outdir, exist_ok=True)
    ts = int(time.time())
    if _HAS_REPORTLAB:
        path = os.path.join(outdir, f"report_{ts}.pdf")
        c = canvas.Canvas(path, pagesize=LETTER)
        c.setFont("Helvetica", 12)
        c.drawString(72, 750, "RL Supply-Chain Simulator — Summary Report")
        y = 720
        for k, v in (context.get("metrics", {})).items():
            c.drawString(72, y, f"{k}: {v}")
            y -= 16
        c.showPage()
        c.save()
        return path
    else:
        path = os.path.join(outdir, f"report_{ts}.txt")
        lines = ["RL Supply-Chain Simulator — Summary Report\n", "\n"]
        for k, v in (context.get("metrics", {})).items():
            lines.append(f"{k}: {v}\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return path
