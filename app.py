# app.py — RL Supply-Chain Routing Simulator (Milestone 3/4)
# Streamlit UI: Data | Explore | Clean | Model | Results | Reports | Pipeline | Help

# ---------------------------- Imports ----------------------------
import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st
import yaml
import importlib

# UX/ plotting helpers for non-technical stakeholders
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# ---------------------------- Dependency Audit ----------------------------
def check_dependency(module_name, pip_name=None, critical=False):
    """
    Try to import a module. If missing, warn (or stop if critical).
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        pkg = pip_name if pip_name else module_name
        if critical:
            st.error(f"Critical dependency `{module_name}` is missing. "
                     f"Please install with `pip install {pkg}`. App cannot run.")
            st.stop()
        else:
            st.warning(f"Optional dependency `{module_name}` not found. "
                       f"Some features may be disabled. Install with: pip install {pkg}")
        return False

# Run checks
REQUIRED = [
    ("seaborn", "seaborn", False),      # optional (plots)
    ("matplotlib", "matplotlib", True), # critical
    ("pandas", "pandas", True),         # critical
    ("networkx", "networkx", True),     # critical
    ("reportlab", "reportlab", False),  # optional (PDF reports)
    ("PIL", "pillow", False),           # optional (images in reports)
]

for mod, pkg, critical in REQUIRED:
    check_dependency(mod, pkg, critical)

# Local modules (root level)
from data_io import load_data, load_from_url
from cleaning import Cleaner
from viz import hist_plot, scatter_plot, route_map, kpi_bar_chart
from evaluation import evaluate_kpis
from reports import make_report
from pipeline import run_full_pipeline, run_step

# Models (inside models/ folder)
from models.baseline_a_star import run_a_star
# RL parts (env + dqn_agent) are imported lazily later in the Model tab

# ---------------------------- UX Helpers ----------------------------
def _coerce_metrics(metrics: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Accepts either flat metrics or nested-by-model metrics and returns a normalized dict:
      {"Baseline": {..}, "RL": {...}}   (missing parts allowed)
    """
    if not metrics:
        return{}
    # already nested by model?
    if all(isinstance(v, dict) for v in metrics.values()):
        return metrics
    # flat -> tuck under "Baseline"
    return {"Baseline": metrics}

def compute_high_level_summary(ctx: dict) -> Dict[str, Any]:
    """Derive executive-level KPIs from current context (safe defaults)."""
    edges = ctx.get("edges_df")
    clean = ctx.get("edges_clean")
    baseline = ctx.get("baseline") or {}
    rl = ctx.get("rl_results") or {}

    rows = int(len(edges)) if edges is not None else 0
    cols = int(edges.shape[1]) if edges is not None else 0
    rows_clean = int(len(clean)) if clean is not None else 0

    base_len = baseline.get("weighted_length")
    rl_len = rl.get("weighted_length")

    improvement = None
    if base_len is not None and rl_len is not None and base_len > 0:
        improvement = round(100.0 * (base_len - rl_len) / base_len, 2)

    last_report = None
    rep_dir = os.path.join("artifacts", "reports")
    if os.path.isdir(rep_dir):
        files = [os.path.join(rep_dir, f) for f in os.listdir(rep_dir)]
        files = [f for f in files if os.path.isfile(f)]
        if files:
            last_report = max(files, key=lambda p: os.path.getmtime(p))

    return {
        "rows": rows,
        "cols": cols,
        "rows_clean": rows_clean,
        "baseline_weighted_length": base_len,
        "rl_weighted_length": rl_len,
        "improvement_pct": improvement,
        "last_report": last_report
    }

def executive_summary_panel(ctx: dict):
    """Render top cards & a KPI chart for non-technical viewers."""
    summary = compute_high_level_summary(ctx)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows (raw)", summary["rows"])
    with c2:
        st.metric("Rows (cleaned)", summary["rows_clean"])
    with c3:
        st.metric("Columns", summary["cols"])
    with c4:
        delta = f'{summary["improvement_pct"]}%' if summary["improvement_pct"] is not None else "-"
        st.metric("Efficiency Gain vs. Baseline", value="-" if delta == "-" else delta)

   
    metrics = _coerce_metrics(ctx.get("metrics", {}))
    if metrics:
        recs = []
        for model, kpis in metrics.items():
            for k, v in kpis.items():
                if isinstance(v, (int, float)):
                    recs.append({"Model": model, "KPI": k, "Value": v})

        if recs:
            fig = px.bar(recs, x="KPI", y="Value", color="Model", barmode="group",
                         title="Key Performance Indicators", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Report status"):
        if summary["last_report"]:
            ts = datetime.fromtimestamp(os.path.getmtime(summary["last_report"])).strftime("%Y-%m-%d %H:%M:%S")
            st.write(f"**Latest report:** {os.path.basename(summary['last_report'])} (generated {ts})")
            with open(summary["last_report"], "rb") as fh:
                st.download_button("Download latest report", data=fh.read(),
                                   file_name=os.path.basename(summary["last_report"]))
        else:
            st.info("No reports generated yet. Use the **Reports** or **Pipeline** tab to create one.")
                  
    
# ---------------------------- App Setup ----------------------------
ART_DIR = "artifacts"
LOG_DIR = os.path.join(ART_DIR, "logs")
DATA_DIR = os.path.join(ART_DIR, "datasets")
REP_DIR = os.path.join(ART_DIR, "reports")
for d in (ART_DIR, LOG_DIR, DATA_DIR, REP_DIR):
    os.makedirs(d, exist_ok=True)

st.set_page_config(page_title="RL Supply-Chain Simulator", layout="wide")
st.title("RL Supply-Chain Routing Simulator — Milestone 3/4")

if "ctx" not in st.session_state:
    st.session_state.ctx = {"seed": 42}
ctx = st.session_state.ctx

# ---------------------------- Utilities ----------------------------


def log_run(event: str, payload: dict):
    """Append a JSON line to artifacts/logs/runs.jsonl"""
    rec = {"ts": datetime.utcnow().isoformat() + "Z",
           "event": event, **payload}
    with open(os.path.join(LOG_DIR, "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _safe_read(path: str, default_text: str = "File not found."):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default_text


# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.header("Configuration")

    cfg_file = st.file_uploader(
        "Load config.yaml (optional)", type=["yaml", "yml"])
    if cfg_file:
        try:
            cfg = yaml.safe_load(cfg_file.read())
            if isinstance(cfg, dict):
                ctx.update(cfg)
                st.success("Configuration loaded.")
        except Exception as e:
            st.warning(f"Failed to parse YAML: {e}")

    ctx["seed"] = st.number_input(
        "Random seed", value=int(ctx.get("seed", 42)), step=1)

    st.divider()
    st.subheader("Synthetic Data Params")
    synth_defaults = ctx.get("synth", {})
    n_nodes = st.slider("n_nodes", 10, 200, int(
        synth_defaults.get("n_nodes", 40)))
    edge_prob = st.slider("edge_prob", 0.05, 0.9, float(
        synth_defaults.get("edge_prob", 0.25)))
    speed_mph = st.slider("speed_mph", 20, 75, int(
        synth_defaults.get("speed_mph", 45)))
    delay_prob = st.slider("delay_prob", 0.0, 0.5, float(
        synth_defaults.get("delay_prob", 0.15)))
    if st.button("Generate Synthetic Data"):
        with st.spinner("Generating synthetic graph..."):
            ctx["synth_params"] = {
                "n_nodes": n_nodes,
                "edge_prob": edge_prob,
                "speed_mph": speed_mph,
                "delay_prob": delay_prob,
            }
            ctx.update(
                load_data(path=None, synth_params=ctx["synth_params"], seed=ctx["seed"]))
            log_run("generate_synth", {"params": ctx["synth_params"]})
        st.success("Synthetic data generated.")

    st.divider()
    st.subheader("Upload or URL Load")
    up = st.file_uploader(
        "Edges CSV (origin_id,dest_id,distance_km,travel_time_est,fuel_rate)",
        type=["csv"],
    )
    url = st.text_input("...or CSV URL")

    c1, c2 = st.columns(2)
    with c1:
        if up and st.button("Load CSV File"):
            try:
                tmp = pd.read_csv(up)
                tmp_path = os.path.join(
                    DATA_DIR, f"upload_{int(datetime.utcnow().timestamp())}.csv")
                tmp.to_csv(tmp_path, index=False)
                ctx["path"] = tmp_path
                with st.spinner("Loading file..."):
                    ctx.update(load_data(path=ctx["path"]))
                log_run("load_file", {"path": tmp_path, "rows": int(len(tmp))})
                st.success("CSV loaded.")
            except Exception as e:
                st.error(f"File load failed: {e}")

    with c2:
        if url and st.button("Load from URL"):
            try:
                with st.spinner("Fetching URL..."):
                    df = load_from_url(url)
                tmp_path = os.path.join(
                    DATA_DIR, f"url_{int(datetime.utcnow().timestamp())}.csv")
                df.to_csv(tmp_path, index=False)
                ctx["path"] = tmp_path
                ctx.update(load_data(path=ctx["path"]))
                log_run("load_url", {"url": url, "rows": int(len(df))})
                st.success("URL CSV loaded.")
            except Exception as e:
                st.error(f"Failed to load URL: {e}")
                log_run("load_url_error", {"url": url, "error": str(e)})

    st.divider()
    st.subheader("Charts")
    ctx["use_mpl"] = st.checkbox(
        "Use Matplotlib for plots (optional)", value=False)

# ---------------------------- Tabs ----------------------------
tabs = ["Overview", "Data", "Explore", "Clean", "Model",
        "Results", "Reports", "Pipeline", "Help"]
T0, T1, T2, T3, T4, T5, T6, T7, TH = st.tabs(tabs)

# ---------------------------- Overview ----------------------------
with T0:
    st.subheader("Executive Summary")
    st.write(
        "This dashboard demonstrates how AI can optimize supply-chain routes. "
        "It compares a traditional pathfinding algorithm (A*) with an AI agent (Reinforcement Learning) "
        "to reduce travel time and fuel use. Use the tabs to load data, explore, clean, model, and generate reports."
    )
    executive_summary_panel(ctx)

    st.divider()
    st.subheader("What to do next (Quick Start)")
    cA, CB, cC = st.columns(3)
    with cA:
        st.markdown("**1) Load Data** \nUpload a CSV or generate synthetic data from the sidebar.")
    with cB:
        st.markdown("**2) Run Pipeline** \nUse the Pipeline tab for one_click end-to-end execution.")
    with cC:
        st.markdown("**3) Download Report** \nGo to Reports to export a PDF/TXXT summary.")

# ---------------------------- Data ----------------------------
with T1:
    st.subheader("Data Preview & Quality")
    if ctx.get("edges_df") is None:
        st.info("Load a CSV or generate synthetic data from the **sidebar** to begin.")
    else:
        df = ctx["edges_df"]
        missing_total = int(df.isna().sum().sum())
        numeric_cnt = int(df.select_dtypes(include="numer").shape[1])
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Rows", len(df))
        with c2: st.metric("Columns", df.shape[1])
        with c3: st.metriC("Numeric Columns", numeric_cnt)
        with c4: st.metric("Missing Values", missing_total)
    
        sch = ctx.get("schema", {"ok": True, "missing": [], "non_numeric": []})
        if not sch.get("ok", True):
            st.warning("Schema checks found potential issues:")
            if sch.get("missing"): st.write("• Missing columns:", sch["missing"])
            if sch.get("non_numeric"): st.write("•Non-numeric columns:", sch["non_numeric"])
        else:
            st.success("Schema check: OK")
    
        st.markdown("**Sample (first 50 rows)**")
        st.dataframe(df.head(50), use_container_width=True)
    
        if ctx.get("nodes_df") is not None:
                try:
                    st.plotly_chart(route_map(ctx["nodes_df"]), use_container_width=True)
                except Exception:
                    st.info("Map preview uses Plotly. If it fails, continue with other tabs.")

# ---------------------------- Explore ----------------------------
with T2:
    st.subheader("Exploration")
    if ctx.get("edges_df") is None:
        st.info("Load or generate data to explore.")
    else:
        df = ctx["edges_df"]
        num_cols = [c for c in df.columns if pd.adi.types.is_numeric_dtypw(df[c])]
        use_mpl = bool(ctx.get("use_mpl", False))

        with st.expander("Quick Insights (auto-generated summary)"):
            if num-cols:
              corr_text = "N/A"
                try:
                    corr = df[num_cols].corr(numeric_only=True).stack().reset_index()
                    corr.columns = ["X", "Y", "r"]
                    corr = corr[corr["X"] != corr["Y"]].sort_values('r', ascending=False)
                    if not corr.empty:
                        top = corr.iloc[0]
                        corr_text = f"Strongest correlation: **{top['X']} vs {top['Y']}** (r={top['r']:.2f})."
                except Exception:
                    pass
                skew_text = []
                for c in num_cols[:3]:
                    try:
                        sk = df[c].skew()
                        if abs(sk) > 1.0:
                            skew_text.append(f"**{c}** appears highly skewed (skew=(sk:2f}).")
                    except Exception:
                        pass
                st.markdown(
                    f"- Numeric features detected: **{len(num_cols)}**  \n"
                    f"- {corr_text}  \n"
                    + ("- " + "  \n- "n- ".join(skew_text) if skew_text else "- No extreme skew detected in the first few numeric features.")
                )
            else:
                st.write("No numeric columns detected.")

        if num_cols:
            col = st.selectbox("Histogram column", num_cols)
            fig = hist_plot(df, col, use_matplotlib=use_mpl)
            if fig is not None:
                st.pyplot(fig, use_container_width=True) if use_mpl else st.plotly_chart(fig, use_container_width=True)

            if len(num_cols) >= 2:
                cA, cB = st.columns(2)
                with cA: x = st.selectbox("X", num_cols, index=0)
                with cB: y = st.selectbox("Y", num_cols, index=1)
                fig2 = scatter_plot(df, x, y, use_matplotlib=use_mpl)
                if fig2 is not None:
                    st.pyplot(fig2, use_container_width=True) if use_mpl else st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Summary statistics**")
        st.dataframe(df.describe().T, use_container_width=True)
        

# ---------------------------- Clean ----------------------------
with T3:
    st.subheader("Cleaning")
    st.caption("These steps improve data quality and comprability before modeling.")
    if ctx.get("edges_df") is None:
        st.info("Load or generate data first")
    else:
        df = ctx["edges_df"]
        normalize = st.checkbox("Normalize numeric columns", value=true,
                                help="Scales numeric values so features are comparable.")
        iqr_mult = st.slider("IQR multiplier (cap outliers)", 0.5, 3.0, 1.5,
                             help="Caps extreme values using the interquartile range rule.")
        drop_na = st.checbox("Drop rows with missing values", value=False,
                             help="Removes rows containing missing values.")

    if st.button("Apply Cleaning"):
        with st.spinner("Cleaning data..."):
            raw_rows = len(df)
            if drop_na:
                df = df.drop_na()
            cleaner = Cleaner(normalize=normalize, iqr_mult=iqr_mult)
            df_clean = cleaner.fit_transform(df)
            ctx["edges_clean"] = df_clean
            ctx["cleaner"] = cleaner
            clean_path = os.path.join(DATA_DIR, f"cleaned_{int(datetime.utcnow().timestamp())}.csv")
            df_clean.to_csv(clean_path, index=False)

            st.success("Cleaning complete.")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Rows (before)", raw_rows)
            with c2: st.metric("Rows (after)", len(df_clean))
            with c3: st.metric("Rows dropped, raw_rows - len(df_clean))

            st.markdown("**Preview (first 50 rows)**")
            st.dataframe(df_clean.head(50), use_container_width=True)

# ---------------------------- Model ----------------------------
with T4:
    st.subheader("Modeling")
    st.info("Compare a traditional pathfinding method (A*) with an AI agent that learns (Reinforcement Learning).")
    st.markdown("**Model Cards**")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**⏱️ A* Pathfinding** \nFinds shortest paths using a proven heuristic. Transparent and fast.")
    with col_m2:
        st.markdown("**🧠 Deep Q-Network (RL)** \nLearns routing decisions from experience. Adaptive and data-driven.")

    if ctx.get("G") is not None:
        weight = st.selectbox("Edge weight (A*)",
                              ["distance_km", "travel_time_est", "fuel_rate"])
        if st.button("Run A* Baseline"):
            with st.spinner("Running baseline A*..."):
                ctx["baseline"] = run_a_star(ctx["G"], weight=weight)
                log_run("run_astar", {
                        "weight": weight, "length": ctx["baseline"]["weighted_length"]})
            st.write(ctx["baseline"])

        # RL controls (lazy import)
        tt = st.number_input("DQN total_timesteps (demo)",
                             value=200)  # keep small for cloud
        colA, colB = st.columns(2)

        with colA:
            if st.button("Train DQN (toy env)"):
                try:
                    from models.env import RoutingEnv
                    from models.dqn_agent import train_dqn

                    env = RoutingEnv(ctx["G"])
                    with st.spinner("Training RL model (small demo)..."):
                        ctx["rl_model_path"] = train_dqn(
                            env, total_timesteps=int(tt))
                        log_run("train_dqn", {"timesteps": int(
                            tt), "model_path": ctx["rl_model_path"]})
                    st.success(
                        f"Trained RL model saved: {ctx['rl_model_path']}")
                except Exception as e:
                    st.error(f"RL training unavailable: {e}")

        with colB:
            if st.button("Infer DQN"):
                try:
                    from models.env import RoutingEnv
                    from models.dqn_agent import infer_dqn

                    if not ctx.get("rl_model_path"):
                        st.warning("Train a model first.")
                    else:
                        env = RoutingEnv(ctx["G"])
                        with st.spinner("Running inference..."):
                            ctx["rl_results"] = infer_dqn(
                                ctx["rl_model_path"], env)
                            log_run("infer_dqn", ctx["rl_results"])
                        st.write(ctx["rl_results"])
                except Exception as e:
                    st.error(f"RL inference unavailable: {e}")

# ---------------------------- Results ----------------------------
with T5:
    metrics = _coerce_metrics(ctx.get("metrics", {}))
    if not metrics:
        st.info("Run a model (A* and/or RL) to view results.")
    else:
        # headline KPIs
        base = metrics.get("Baseline", {})
        rl = metrics.get("RL", {})

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Baseline weighted Length", f"{base.get('weighted_length', '-')}")
        with c2:
            st.metric("RL Weighted Length", f"{rl.get('weighted_length', '-')}")
        with c3:
            b = base.get("weighted_length"); r = rl.get("weighted_length")
            if isinstance(b, (int, float)) and isinstance(r, (int,float)) and b > 0:
                st.metric("Efficiency Gain", f"{round(100*(b-r)/b,2)}%")
            else:
                st.metric("Efficiency Gain", "-")

        # KPI comparison bar chart
        try:
            # prefer your viz.kpi_bar_chart if it returns a Matplotlib fig
            fig = kpi_bar_chart(metrics)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
        except Exception:
            # fallback to Plotly if needed
            recs = []
            for model, kpis in metrics.items():
                for k, v in kpis.items():
                    if isinstance(v, (int, float)):
                        recs.append({"Model": model, "KPI": k, "Value": v})
            if recs:
                fig = px.bar(recs, x="KPI", y="Value", color="Model", barmode="group", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw results (JSON)"):
            st.json(metrics)

# ---------------------------- Reports ----------------------------
with T6:
    st.subheader("Report Generation")
    st.caption("Create a timestamped PDF/TXT report with dataset details, cleaning steps, model KPIs, and reproduciblity metadata.")
    if st.button("Generate Report"):
        try:
            with st.spinner("Rendering report..."):
                path = make_report(ctx, plots=[])
                log_run("report", {"path": path})
            st.success(f"Report created: {path}")

            if os.path.exists(path):
                with open(path, "rb") as fh:
                    st.download_button("Download Report", data=fh.read(), file_name=os.path.basemname(path))
        except Exception as e:
            st.error(f"Report generation failed: {e}")

    if ctx.get("edges_clean") is not None:
        st.download_button(
            "Download Cleaned CSV",
            data=ctx["edges_clean"].to_csv(index=False),
            file_name="cleaned.csv",
        )

# ---------------------------- Pipeline ----------------------------
with T7:
    st.subheader("Pipeline - One Click Run (Baseline)")
    st.caption("Generates synthetic data → cleans → runs A* → evaluates KPIs → produces a report.")
    if st.button("Run Full Pipeline (Synthetic → Report)"):
        try:
            step = st.empty()
            with st.spinner("Executing pipeline..."):
                step.markdown("**Step 1/5:** Generating synthetic data...")
                ctx["synth_params"] = {"n_nodes": 30, "edge_prob": 0.3, "speed_mph": 40, "delay_prob": 0.1}
                ctx.update(load_data(path=None, synth_params=ctx["synth_params"], seed=ctx["seed"]))

                step.markdown("**Step 2/5:** Cleaning data...")
                cleaner = Cleaner(normalize=True, iqr_mult=1.5)
                ctx["edges_clean"] = cleaner.fit_transform(ctx["edges_df"])

                step.markdown("**Step 3/5:** Running baseline model (A*)...")
                ctx["baseline"] = run_a_star(ctx["G"], weight="distance_km")

                step.markdown("**Step 4/5:** Evaluating KPIs...")
                ctx["metrics"] = evaluate_kpis(ctx.get("baseline"), ctx.get("rl_results"))

                step.markdown("**Step 5/5:** Generating report...")
                path = make_report(ctx, plots=[])
                log_run("pipeline_full", {"report": path})

            step.markdown("✅ ** Pipeline finished.**")
            st.json(ctx["metrics'])

            if os.path.exists(path):
                    with open(path, "rb") as fh:
                        st.download_button("Download Report", data=fh.read(), file_name=os.path.basename(path))
        except Exception as e:
                    st.error(f"Pipeline failed: {e}")
    
# ---------------------------- Help ----------------------------
with TH:
    st.subheader("Help")
    st.markdown("**Quick Start (non-technical)**")
    st.markdown("1) Use the **Overview** to understand what the app does.  \n
                "2) Load data from the **sidebar**.  \n
                "3) Click **Pipeline** to run everything.  \n
                "4) Get your **Report** in PDF/TXT.")

    with st.expander("FAQ"):
            st.markdown("**Can I use my own data?** Yes - upload a CSV with the required columns shown on the Data tab.")
            st.markdown("**What is 'weighted length'?** A composite cost for routing that balancs distance/time/fuel.")
            st.markdown("**Is it reproducible?** Yes - synthetic generation logs parameters and seeds for replication.")

    st.markdown("---")
    st.markdown("**Full Guide (from HELP.md)**")
    help_text = _safe_read("HELP.md", default_text="HELP.md not found. Please include HELP.d]md next to app.py.")
    st.markdown(help_text)
