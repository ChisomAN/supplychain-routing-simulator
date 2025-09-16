# app.py — RL Supply-Chain Routing Simulator (Milestone 3/4)
# Streamlit UI: Data | Explore | Clean | Model | Results | Reports | Pipeline | Help

# ---------------------------- Imports ----------------------------
import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st
import yaml

# Local modules (root level)
from data_io import load_data, load_from_url
from cleaning import Cleaner
from viz import hist_plot, scatter_plot, route_map
from evaluation import evaluate_kpis
from reports import make_report
from pipeline import run_full_pipeline, run_step

# Models (inside models/ folder)
from models.baseline_a_star import run_a_star
# RL parts (env + dqn_agent) are imported lazily later in the Model tab


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
tabs = ["Data", "Explore", "Clean", "Model",
        "Results", "Reports", "Pipeline", "Help"]
T1, T2, T3, T4, T5, T6, T7, TH = st.tabs(tabs)

# ---------------------------- Data ----------------------------
with T1:
    st.subheader("Data Preview")
    if ctx.get("edges_df") is not None:
        sch = ctx.get("schema", {"ok": True, "missing": [], "non_numeric": []})
        if not sch.get("ok", True):
            st.warning("Schema issues detected:")
            if sch.get("missing"):
                st.write("Missing columns:", sch["missing"])
            if sch.get("non_numeric"):
                st.write("Non-numeric columns:", sch["non_numeric"])
        st.dataframe(ctx["edges_df"].head(50), use_container_width=True)

    if ctx.get("nodes_df") is not None:
        try:
            st.plotly_chart(
                route_map(ctx["nodes_df"]), use_container_width=True)
        except Exception:
            st.info("Map preview uses Plotly. If it fails, continue with other tabs.")

# ---------------------------- Explore ----------------------------
with T2:
    st.subheader("Exploration")
    if ctx.get("edges_df") is not None:
        num_cols = [c for c in ctx["edges_df"].columns if pd.api.types.is_numeric_dtype(
            ctx["edges_df"][c])]
        use_mpl = bool(ctx.get("use_mpl", False))

        if num_cols:
            col = st.selectbox("Histogram column", num_cols)
            fig = hist_plot(ctx["edges_df"], col, use_matplotlib=use_mpl)
            if fig is not None:
                if use_mpl:
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.plotly_chart(fig, use_container_width=True)

            if len(num_cols) >= 2:
                cA, cB = st.columns(2)
                with cA:
                    x = st.selectbox("X", num_cols, index=0)
                with cB:
                    y = st.selectbox("Y", num_cols, index=1)

                fig2 = scatter_plot(ctx["edges_df"], x,
                                    y, use_matplotlib=use_mpl)
                if fig2 is not None:
                    if use_mpl:
                        st.pyplot(fig2, use_container_width=True)
                    else:
                        st.plotly_chart(fig2, use_container_width=True)

        st.write("Summary stats:")
        st.dataframe(ctx["edges_df"].describe().T, use_container_width=True)

# ---------------------------- Clean ----------------------------
with T3:
    st.subheader("Cleaning")
    if ctx.get("edges_df") is not None:
        normalize = st.checkbox("Normalize numeric columns", value=True)
        iqr_mult = st.slider("IQR multiplier (cap outliers)", 0.5, 3.0, 1.5)
        drop_na = st.checkbox("Drop rows with missing values", value=False)

        if st.button("Apply Cleaning"):
            with st.spinner("Cleaning data..."):
                df = ctx["edges_df"].copy()
                if drop_na:
                    df = df.dropna()
                cleaner = Cleaner(normalize=normalize, iqr_mult=iqr_mult)
                df_clean = cleaner.fit_transform(df)
                ctx["edges_clean"] = df_clean
                ctx["cleaner"] = cleaner
                clean_path = os.path.join(
                    DATA_DIR, f"cleaned_{int(datetime.utcnow().timestamp())}.csv")
                df_clean.to_csv(clean_path, index=False)
                log_run(
                    "clean",
                    {
                        "normalize": normalize,
                        "iqr_mult": iqr_mult,
                        "drop_na": drop_na,
                        "out": clean_path,
                        "rows_in": int(len(df)),
                        "rows_out": int(len(df_clean)),
                    },
                )
            st.success("Cleaning complete.")
            st.dataframe(df_clean.head(50), use_container_width=True)

# ---------------------------- Model ----------------------------
with T4:
    st.subheader("Modeling")
    st.info(
        "Baseline A* is available. RL (DQN) is optional and imported only when used.")

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
    st.subheader("Results & Comparison")
    if ctx.get("baseline") or ctx.get("rl_results"):
        ctx["metrics"] = evaluate_kpis(
            ctx.get("baseline"), ctx.get("rl_results"))
        st.json(ctx["metrics"])

# ---------------------------- Reports ----------------------------
with T6:
    st.subheader("Report Generation")
    if st.button("Generate Report"):
        try:
            with st.spinner("Rendering report..."):
                path = make_report(ctx)  # PDF if ReportLab, else TXT
                log_run("report", {"path": path})
            st.success(f"Report created: {path}")

            if os.path.exists(path):
                with open(path, "rb") as fh:
                    st.download_button(
                        "Download Report", data=fh.read(), file_name=os.path.basename(path))
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
    st.subheader("Pipeline — One Click Run (Baseline)")
    if st.button("Run Full Pipeline (Synthetic → Report)"):
        try:
            with st.spinner("Executing pipeline..."):
                ctx["synth_params"] = {
                    "n_nodes": 30, "edge_prob": 0.3, "speed_mph": 40, "delay_prob": 0.1}
                ctx.update(
                    load_data(path=None, synth_params=ctx["synth_params"], seed=ctx["seed"]))
                cleaner = Cleaner(normalize=True, iqr_mult=1.5)
                ctx["edges_clean"] = cleaner.fit_transform(ctx["edges_df"])
                ctx["baseline"] = run_a_star(ctx["G"], weight="distance_km")
                ctx["metrics"] = evaluate_kpis(
                    ctx.get("baseline"), ctx.get("rl_results"))
                path = make_report(ctx)
                log_run("pipeline_full", {"report": path})
            st.success("Pipeline finished.")
            st.json(ctx["metrics"])
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    st.download_button(
                        "Download Report", data=fh.read(), file_name=os.path.basename(path))
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

# ---------------------------- Help ----------------------------
with TH:
    st.subheader("Help")
    st.markdown(
        """
        This section renders **HELP.md** so end users can read guidance inside the app.
        If the file isn't found, a short notice is displayed.
        """,
        help="Place HELP.md in the same folder as app.py."
    )
    help_text = _safe_read(
        "HELP.md",
        default_text="HELP.md not found. Please include HELP.md next to app.py."
    )
    st.markdown(help_text)
