# Test Report — First Iteration (Sample Outcomes)

- File load (sample_edges.csv): PASS — preview visible (8 rows).
- URL load: SKIPPED (offline). Manual test in connected env recommended.
- Schema warning: PASS — warning shows for missing/non-numeric columns (simulated by removing `distance_km`).
- EDA (Plotly): PASS — histogram & scatter render.
- EDA (Matplotlib): CONDITIONAL — if MPL not installed, app shows warning (expected). With MPL installed, plots render.
- Cleaning: PASS — drop NA reduces rows; IQR capping clamps outliers; MinMax scales 0–1.
- Baseline A\*: PASS — returns path and weighted_length > 0 on synthetic graph.
- Report (ReportLab installed): PASS — PDF created. If not installed: PASS — TXT fallback created.
- Pipeline: PASS — synthetic → clean → baseline → report completes; metrics JSON shown.

Artifacts verified in `artifacts/` and logs appended to `artifacts/logs/runs.jsonl`.
