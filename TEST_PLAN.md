# Test Plan — First Iteration

## Scope

Covers ingest (file/URL), EDA plots, cleaning transforms, baseline model (A\*), report generation, and pipeline runner.

## Test Types

- **Unit tests**: cleaning outlier capping; schema validation; baseline path computation with small graphs.
- **Integration tests**: synth → clean → baseline → metrics → report.
- **UX smoke**: Streamlit widgets respond without exceptions; warnings show for schema and missing MPL/ReportLab/RL deps.

## Environments

- Python 3.10+ on macOS/Windows/Linux. Plotly only required. Matplotlib/ReportLab/RL libs optional.

## Cases

1. **File load success**: `sample_edges.csv` renders preview.
2. **URL load success**: any small CSV URL (optional offline).
3. **Schema warning**: CSV missing `distance_km` → warning appears; EDA still works.
4. **EDA**: histogram/scatter render; backend toggle shows MPL if installed or warns otherwise.
5. **Cleaning**: Drop NA, IQR cap (1.5x), MinMax normalize; distributions change.
6. **Baseline**: A\* returns path and weighted_length.
7. **Report**: with/without ReportLab; output path exists; download works.
8. **Pipeline**: one-click run finishes and writes metrics + report.
