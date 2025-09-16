# Help — RL Supply-Chain Routing Simulator

## What this product does

An end-to-end, browser-based data product for logistics routing. It loads CSV/URL data, supports exploration and cleaning, runs a baseline A\* route model (with optional RL stubs), visualizes results, and exports a report.

## How to run

1. `pip install -r requirements.txt`
2. `streamlit run app.py`
3. Use the sidebar to **Load Data** (file upload or URL) or generate **Synthetic Data**.

## Tabs & Functions

- **Data**: Preview your dataset; schema issues are listed here.
- **Explore**: Summary table; Histogram & Scatter. Toggle **Use Matplotlib** in sidebar if you want MPL plots.
- **Clean**: Options: Drop NAs, IQR capping, MinMax normalization. Saves a cleaned CSV into `artifacts/datasets`.
- **Model**: Run **A\*** baseline (distance/time/fuel weight). RL (DQN) is optional and loads only on demand.
- **Results**: Shows your KPIs (baseline path weighted length; RL mean reward if run).
- **Reports**: Generates a **PDF** when ReportLab is installed, else a **TXT** report. You can download the cleaned CSV here too.
- **Pipeline**: One-click demo: synthetic → clean → baseline → metrics → report.

## Data Requirements

For modeling: `origin_id, dest_id, distance_km, travel_time_est, fuel_rate`. For EDA/cleaning only, any CSV will work.

## Troubleshooting

- URL fails → save file locally and upload.
- Schema warning → fix missing/non-numeric columns.
- No PDF? Install ReportLab (`pip install reportlab`) or use the TXT report fallback.
- RL import error → install `gymnasium`, `stable-baselines3`, `torch`, or skip RL.

## Credits

Milestone 3 by Chisom Atulomah. Plotly by default; Matplotlib optional.
