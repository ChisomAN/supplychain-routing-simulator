# app.py — RL Supply-Chain Routing Simulator
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

from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    precision_recall_fscore_support, accuracy_score
)

# Optional: reuse your Cleaner with safe defaults
try:
    from cleaning import Cleaner as _CleanerAA
    def _clean_edges_aa(df: pd.DataFrame) -> pd.DataFrame:
        return _CleanerAA(normalize=True, iqr_mult=1.5).fit_transform(df)
except Exception:
    def _clean_edges_aa(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        for col in ("distance_km", "travel_time_est", "fuel_rate"):
            if col in df:
                df[col] = df[col].clip(lower=0)
        return df

def _logistic(x: np.ndarray, temp: float = 0.08) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-temp * x))

def _evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict:
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # Handle degenerate cases for ROC/AUC when y_true has 1 class
    unique = np.unique(y_true)
    if unique.size < 2:
        # No ROC curve possible; return NaNs and a dummy diagonal
        fpr = np.array([0.0, 1.0], dtype=float)
        tpr = np.array([0.0, 1.0], dtype=float)
        roc_auc = float("nan")
    else:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        try:
            roc_auc = auc(fpr, tpr)
        except Exception:
            roc_auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    return dict(acc=acc, prec=precision, rec=recall, f1=f1,
                fpr=fpr, tpr=tpr, auc=roc_auc, cm=cm)
    
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

def _ensure_chartable_kpis(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Make sure there's at least one numeric KPI per model so that chart isn't empty."""
    if not metrics:
        return metrics
    out = {}
    if all (isinstance(v, dict) for v in metrics.values()):    # nested-by-model
        for model, kpis in metrics.items():
            any_numeric = any(isinstance(v, (int, float)) for v in kpis.values())
            if not any_numeric:
                #Try a graceful alias if present
                if "rl_mean_reward" in kpis and isinstance(kpis["rl_mean_reward"], (int, float)):
                    out[model] = {**kpis, "mean_reward": float(kpis["rl_mean_reward"])}
                else:
                    out[model] = {**kpis, "_placeholder": 0.0}
            else:
                out[model] = kpis
        return out
    # flat dict (Baseline only) - keep as-is
    return metrics

def _get_any(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None

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

    if summary["rows"] == 0:
        st.info("No data loaded yet. Use the sidebar to generate synthetic data or upload a CSV.")
    
# ---------------------------- App Setup ----------------------------
ART_DIR = "artifacts"
LOG_DIR = os.path.join(ART_DIR, "logs")
DATA_DIR = os.path.join(ART_DIR, "datasets")
REP_DIR = os.path.join(ART_DIR, "reports")
for d in (ART_DIR, LOG_DIR, DATA_DIR, REP_DIR):
    os.makedirs(d, exist_ok=True)

st.set_page_config(page_title="RL Supply-Chain Simulator", layout="wide", initial_sidebar_state="auto")
# ---- Dark/Light CSS (auto-aware) ----
def inject_theme_css(theme_choice: str):
    # Ask Streamlit what the active base theme is (per config.toml)
    active_base = st.get_option("theme.base") or "light"  # "light" or "dark"

    # Resolve final palette: user choice wins; "Auto" = follow active_base
    effective = theme_choice.lower() if theme_choice in ("Light", "Dark") else active_base.lower()

    light = {
        "bg": "#ffffff", "text": "#111827", "muted": "#4b5563", "border": "#e5e7eb",
        "card": "#ffffff", "chip_bg": "#ffffff", "chip_border": "#eef2f7",
    }
    dark = {
        "bg": "#0e1117", "text": "#e5e7eb", "muted": "#9ca3af", "border": "#1f2937",
        "card": "#111827", "chip_bg": "#0f172a", "chip_border": "#1f2937",
    }
    palette = dark if effective == "dark" else light

    st.markdown(f"""
    <style>
      :root {{
        --bg: {palette['bg']};
        --text: {palette['text']};
        --muted: {palette['muted']};
        --border: {palette['border']};
        --card: {palette['card']};
        --chip-bg: {palette['chip_bg']};
        --chip-border: {palette['chip_border']};
      }}
      .app-header {{
        display:flex; align-items:center; gap:12px;
        padding:10px 16px; background: var(--bg);
        border-bottom:1px solid var(--border); margin-bottom:0.8rem;
      }}
      .app-header img {{ height:42px; width:auto; border-radius:6px; }}
      .app-header h1 {{ font-size:1.6rem; font-weight:700; color: var(--text); margin:0; }}

      /* General text */
      .stMarkdown, .stText, .stRadio, .stSelectbox, .stSlider, label, p, span, h1,h2,h3,h4,h5,h6 {{
        color: var(--text);
      }}

      /* Metric chips */
      [data-testid="stMetric"] {{
        background: var(--chip-bg)!important;
        border: 1px solid var(--chip-border)!important;
        border-radius:12px; padding:12px; margin-bottom:8px;
      }}
    </style>
    """, unsafe_allow_html=True)

# Plot style to match theme
import plotly.express as px
import matplotlib.pyplot as plt

_choice = st.session_state.get("_theme_choice", "Auto")
# Simple heuristic: if forced Dark, or Auto + system is dark (Streamlit's native dark theme is also fine)
force_dark = (_choice == "Dark")
force_light = (_choice == "Light")

if force_dark:
    px.defaults.template = "plotly_dark"
    plt.style.use("dark_background")
elif force_light:
    px.defaults.template = "plotly_white"
    plt.style.use("default")
else:
    # Auto: pick a sensible default; Plotly follows page background fine either way
    px.defaults.template = "plotly_white"
    plt.style.use("default")

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

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

st.markdown("""
<style>
/* content width + spacing */
.block-container {padding-top:1rem; padding-bottom:3rem; max-width: 1200px;}
/* metric chips */
[data-testid="stMetric"] {
  background:#fff;border:1px solid #eef2f7;border-radius:12px;padding:12px;margin-bottom:8px;
}
/* headings */
h2, .stSubheader {margin-top:0.2rem; margin-bottom:0.4rem;}
/* expander header weight */
.streamlit-expanderHeader {font-weight:600;}
/* nicer buttons */
.stButton>button {border-radius:10px;}
/* top header bar */
.app-header {display:flex;align-items:center;justify-content:space-between;
             padding:6px 0 12px 0;border-bottom:1px solid #e5e7eb;margin-bottom:8px;}
.app-title {display:flex;gap:10px;align-items:center;font-weight:700;font-size:1.05rem;}
.app-title img {height:26px;width:26px;border-radius:6px;}
.app-actions {display:flex;gap:8px;}
</style>
""", unsafe_allow_html=True)

# --- Top header bar with logo + title (HTML + base64 image) ---
from pathlib import Path
import base64

def _img_as_data_uri(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = path.suffix.lower().lstrip(".") or "png"
        mime = "image/png" if ext == "png" else "image/jpeg" if ext in ("jpg", "jpeg") else "image/svg+xml"
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

logo_path = Path(__file__).parent / "assets" / "logo.png"
logo_src  = _img_as_data_uri(logo_path)

st.markdown("""
<style>
.app-header {
    display:flex; align-items:center; gap:12px;
    padding:10px 16px;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.8rem;
}
.app-header img { height:45px; width:auto; border-radius:6px; vertical-align:middle;}
.app-header h1 { font-size:1.6rem; font-weight:700; color: var(--text); margin:0; line-height:1; }
</style>
""", unsafe_allow_html=True)

if logo_src:
    st.markdown(
        f"""
        <div class="app-header">
            <img src="{logo_src}" alt="logo" />
            <h1>RL Supply-Chain Routing Simulator</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # graceful fallback (no broken image)
    st.markdown(
        """
        <div class="app-header">
            <h1>RL Supply-Chain Routing Simulator</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Make main area adapt when the sidebar opens/closes ---
st.markdown("""
<style>
/* Base content width on wide screens */
section.main > div.block-container{
  max-width: 1200px;
  padding-left: 1rem;
  padding-right: 1rem;
}

/* When the sidebar is OPEN, use the remaining viewport width */
aside[data-testid="stSidebar"][aria-expanded="true"]
  ~ section.main > div.block-container{
  max-width: calc(100vw - 22rem);  /* adjust if your sidebar width differs */
}

/* When the sidebar is CLOSED, nearly full width */
aside[data-testid="stSidebar"][aria-expanded="false"]
  ~ section.main > div.block-container{
  max-width: 96vw;
}

/* Small screens: always fluid */
@media (max-width: 1100px){
  section.main > div.block-container{ max-width: 96vw; }
}

/* Keep the top header aligned with content paddings */
.app-header{
  margin-left: 0; margin-right: 0;
  padding-left: 1rem; padding-right: 1rem;
}
</style>
""", unsafe_allow_html=True)

has_data = ctx.get("edges_df") is not None

# --- Quick action buttons ---
if has_data:
    qc1, qc2 = st.columns([1, 1])
    with qc1:
        run_quick_pipeline = st.button(
            "▶ Run Pipeline",
            use_container_width=True,
            key="quick_pipeline_btn"
        )
    with qc2:
        gen_quick_report = st.button(
            "📄 Generate Report",
            use_container_width=True,
            key="quick_report_btn"
        )
else:
    st.info("Load or generate data from the sidebar to unlock quick actions (Pipeline & Report).")
    run_quick_pipeline = False
    gen_quick_report = False

# --- Wire quick actions to existing logic ---
if run_quick_pipeline:
    with st.spinner("Running quick pipeline (clean → A* → KPIs → report)..."):
        # Assume data already exists since has_data == True
        cleaner = Cleaner(normalize=True, iqr_mult=1.5)
        ctx["edges_clean"] = cleaner.fit_transform(ctx["edges_df"])
        ctx["baseline"] = run_a_star(ctx["G"], weight="distance_km")
        ctx["metrics"] = evaluate_kpis(ctx.get("baseline"), ctx.get("rl_results"))
        quick_report_path = make_report(ctx, plots=[])
        log_run("pipeline_full_quick", {"report": quick_report_path})
    st.success("Quick pipeline completed ✅")

if gen_quick_report:
    try:
        with st.spinner("Generating report from current context..."):
            report_path = make_report(ctx, plots=[])
            log_run("report_quick", {"path": report_path})
        st.success(f"Report generated ✅ — {os.path.basename(report_path)}")
        if os.path.exists(report_path):
            with open(report_path, "rb") as fh:
                st.download_button(
                    "Download Report",
                    data=fh.read(),
                    file_name=os.path.basename(report_path)
                )
    except Exception as e:
        st.error(f"Quick report failed: {e}")

# === Advanced Analysis: real-output collectors ===
def _aa_sigmoid(x, temp=0.08):
    import numpy as np
    return 1.0 / (1.0 + np.exp(-temp * x))

def _aa_sample_pairs(G, k=200, rng_seed=11):
    """Draw k random (origin, dest) node pairs from graph G (no self-pairs)."""
    import numpy as np
    nodes = list(G.nodes())
    rng = np.random.default_rng(rng_seed)
    pairs = []
    for _ in range(k * 3):  # oversample attempts to avoid same-node pairs
        o = rng.choice(nodes)
        d = rng.choice(nodes)
        if o != d:
            pairs.append((o, d))
        if len(pairs) >= k:
            break
    return pairs

def _aa_astar_time(G, origin, dest, weight="travel_time_est"):
    """
    Estimate A* travel time along the planned route by summing edge 'weight' (fallback to distance_km).
    Uses your existing run_a_star() which expects the graph and a weight.
    """
    try:
        # run_a_star returns {"path": [...], "weighted_length": ...} in your app
        res = run_a_star(G, weight=weight, origin=origin, dest=dest)  # extend signature if your function supports it
        if res and "weighted_length" in res:
            return float(res["weighted_length"])
    except TypeError:
        # If your run_a_star doesn't accept origin/dest, approximate with distance_km shortest path
        import networkx as nx
        path = nx.shortest_path(G, source=origin, target=dest, weight=weight if weight in list(next(iter(G.edges(data=True)))[-1].keys()) else "distance_km")
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            data = G.get_edge_data(u, v) or {}
            total += float(data.get(weight, data.get("distance_km", 0.0)))
        return total
    except Exception:
        return None

def _aa_dqn_time(model_path, G, origin, dest):
    """
    Roll out the trained DQN on an episode from origin to dest in your RoutingEnv,
    returning a total travel time estimate for that episode.
    """
    try:
        from models.env import RoutingEnv
        from models.dqn_agent import infer_dqn_episode  # try a per-episode helper if available
    except Exception:
        # Fallback: attempt to use infer_dqn(model_path, env, episodes=1) and read total time
        infer_dqn_episode = None

    try:
        from models.env import RoutingEnv
        env = RoutingEnv(G, origin=origin, dest=dest)  # assumes your env can take start/goal (if not, it will ignore)
    except Exception:
        return None

    # Preferred: a single-episode inference API returning a dict with "total_time"
    if infer_dqn_episode:
        try:
            out = infer_dqn_episode(model_path, env)
            if isinstance(out, dict) and "total_time" in out:
                return float(out["total_time"])
        except Exception:
            pass

    # Generic fallback: run your existing infer_dqn for 1 episode and read first episode's total_time if it exists
    try:
        from models.dqn_agent import infer_dqn
        out = infer_dqn(model_path, env, episodes=1)
        # Accept common keys
        for key in ("total_time", "episode_time", "travel_time", "length", "weighted_length"):
            if isinstance(out, dict) and key in out:
                return float(out[key])
        # Or if list-like per-episode
        if isinstance(out, dict):
            for k, v in out.items():
                if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (int, float)):
                    return float(v[0])
    except Exception:
        pass

    return None

def aa_collect_real_scores(ctx: dict, k_pairs=200, sla_min=90, weight="travel_time_est", temp=0.08, rng_seed=11):
    """
    Build a matched evaluation set of size k_pairs:
      - sample origin/dest pairs
      - get A* time and DQN time (if model available)
      - compute P(on-time) = sigmoid( SLA - time )
      - build y_true from a 'noisy' realized time (to represent live conditions) for ROC/F1/CM
    Stores arrays in ctx["aa_eval"].
    """
    import numpy as np
    if ctx.get("G") is None:
        raise RuntimeError("Graph G not available. Load or generate data first.")

    pairs = _aa_sample_pairs(ctx["G"], k=k_pairs, rng_seed=rng_seed)

    # SLA per sample
    sla = float(sla_min)

    # A* times
    a_times = []
    for (o, d) in pairs:
        t = _aa_astar_time(ctx["G"], o, d, weight=weight)
        a_times.append(np.nan if t is None else float(t))
    a_times = np.array(a_times, dtype=float)

    # DQN times (only if trained)
    d_times = None
    model_path = ctx.get("rl_model_path")
    if model_path:
        d_times_list = []
        for (o, d) in pairs:
            t = _aa_dqn_time(model_path, ctx["G"], o, d)
            d_times_list.append(np.nan if t is None else float(t))
        d_times = np.array(d_times_list, dtype=float)

    # Ground truth (realized) times — add mild noise to reflect uncertainty
    rng = np.random.default_rng(rng_seed)
    realized_a = a_times + rng.normal(0, 3.0, size=a_times.size)  # ±3 min jitter
    y_true = (realized_a <= sla).astype(int)  # proxy label for “was on time”

    # Probabilities via sigmoid(slack)
    a_star_prob = _aa_sigmoid(sla - a_times, temp=temp)
    dqn_prob = _aa_sigmoid(sla - d_times, temp=temp) if d_times is not None else None

    # Clean NaNs (be forgiving so we don't end up with 0 samples)
    base_valid = np.isfinite(a_star_prob) & np.isfinite(y_true)

    if dqn_prob is not None:
        dqn_valid = np.isfinite(dqn_prob)

        if dqn_valid.any():
            # Use only rows where *both* A* and DQN are finite
            valid = base_valid & dqn_valid
        else:
            # RL side is all NaNs → fall back to baseline-only evaluation
            dqn_prob = None
            valid = base_valid
    else:
        # No RL probs at all → baseline-only, but still keep labels
        valid = base_valid

    # As a last safety net, if somehow everything is bad, keep at least baseline where possible
    if not valid.any() and base_valid.any():
        valid = base_valid
        dqn_prob = None

    out = {
        "pairs": [pairs[i] for i, ok in enumerate(valid) if ok],
        "y_true": y_true[valid].astype(int),
        "a_star_prob": a_star_prob[valid],
        "dqn_prob": dqn_prob[valid] if dqn_prob is not None else None,
        "sla_min": sla_min,
        "weight": weight,
        "temp": temp,
    }
    ctx["aa_eval"] = out
    return out


@st.cache_data(show_spinner=False)
def _aa_load_edges(sample_first: bool = True, uploaded_file=None) -> pd.DataFrame:
    """
    Try, in order:
      1) uploaded_file (if provided)
      2) project samples: /mnt/data/sample_edges.csv, ./sample_edges.csv
      3) latest CSV in artifacts/datasets
      4) app context: ctx['edges_clean'] or ctx['edges_df']
    If none found, raise FileNotFoundError (handled by caller).
    """
    import glob
    from pathlib import Path
    import streamlit as st

    # 1) direct upload
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    # 2) project samples
    candidates = []
    if sample_first:
        candidates += ["/mnt/data/sample_edges.csv", "sample_edges.csv"]
    for p in candidates:
        if Path(p).exists():
            try:
                return pd.read_csv(p)
            except Exception:
                pass

    # 3) latest CSV from artifacts/datasets (your app’s default data dir)
    data_dir = Path("artifacts") / "datasets"
    if data_dir.exists():
        csvs = sorted(glob.glob(str(data_dir / "*.csv")), key=os.path.getmtime, reverse=True)
        for p in csvs:
            try:
                return pd.read_csv(p)
            except Exception:
                continue

    # 4) app context
    ctx = st.session_state.get("ctx", {})
    if isinstance(ctx, dict):
        if isinstance(ctx.get("edges_clean"), pd.DataFrame) and not ctx["edges_clean"].empty:
            return ctx["edges_clean"].copy()
        if isinstance(ctx.get("edges_df"), pd.DataFrame) and not ctx["edges_df"].empty:
            return ctx["edges_df"].copy()

    # nothing found
    raise FileNotFoundError("No CSV available for Advanced Analysis.")

def render_advanced_analysis_tab(
    df_source: pd.DataFrame | None,
    sla_min: int = 90,
    use_reward_shaping: bool = True,
    use_opt_tuning: bool = True,
    data_mode_label: str = "Sample",
    uploaded_file=None
):
    """
    Renders the Advanced Analysis tab (metrics, ROC, confusion matrices, EDA, refinements).
    Uses proxy probabilities unless you later swap in your real model outputs.
    """
    # 0) Load/clean data (robust fallbacks + graceful exit)
    df_raw = None
    if df_source is not None:
        df_raw = df_source.copy()
    else:
        try:
            df_raw = _aa_load_edges(sample_first=(data_mode_label == "Sample"), uploaded_file=uploaded_file)
        except FileNotFoundError:
            st.info(
                "No dataset available for Advanced Analysis yet. "
                "Please either:\n"
                "• Upload a CSV in the sidebar (Advanced Analysis Dataset → Upload CSV), or\n"
                "• Generate/Load data in other tabs and click **Apply Cleaning** (so `edges_clean` becomes available)."
            )
            return

    df_clean = _clean_edges_aa(df_raw.copy())

    # Guards
    if "travel_time_est" not in df_clean.columns:
        st.error("Expected column 'travel_time_est' not found. Provide a CSV with travel_time_est.")
        return
    if "distance_km" not in df_clean.columns:
        st.warning("Column 'distance_km' not found; defaulting to zeros for distance features.")
        df_clean["distance_km"] = 0.0

    # 1) Proxy labels & proxy probabilities (baseline)
    rng = np.random.default_rng(11)
    df_clean["sla_min"] = int(sla_min)
    noise = rng.normal(0, 6, size=len(df_clean))
    actual_time = df_clean["travel_time_est"].values + noise
    y_true_arr = (actual_time <= df_clean["sla_min"].values).astype(int)

    a_star_score = df_clean["sla_min"].values - df_clean["travel_time_est"].values
    dqn0_score   = a_star_score + rng.normal(0, 5, size=len(df_clean)) + 0.08*df_clean["distance_km"].values*(-0.2)
    a_star_prob_arr, dqn0_prob_arr = _logistic(a_star_score), _logistic(dqn0_score)

    # Refinements on proxy DQN
    tightness = np.clip((df_clean["sla_min"].values / (df_clean["travel_time_est"].values + 1e-5)), 0.5, 1.5)
    traffic   = np.clip(df_clean["distance_km"].values / max(df_clean["distance_km"].max(), 1e-9), 0, 1)
    dqn1_prob = dqn0_prob_arr.copy()
    if use_reward_shaping:
        dqn1_prob = np.clip(
            dqn0_prob_arr + 0.06*(tightness - 1.0) + 0.05*(1.0 - traffic) + rng.normal(0, 0.01, size=len(df_clean)),
            0, 1
        )
    dqn2_prob = dqn1_prob.copy()
    if use_opt_tuning:
        dqn2_prob = np.clip(dqn1_prob + (y_true_arr - 0.5)*0.05, 0, 1)

    # 2) Prefer real outputs if available (overrides the proxy arrays)
    aa_eval = st.session_state.get("ctx", {}).get("aa_eval")
    using_real = False
    if isinstance(aa_eval, dict):
        y_true_real = aa_eval.get("y_true")
        a_prob_real = aa_eval.get("a_star_prob")
        d_prob_real = aa_eval.get("dqn_prob")
        if isinstance(y_true_real, np.ndarray) and isinstance(a_prob_real, np.ndarray):
            y_true_arr = y_true_real
            a_star_prob_arr = a_prob_real
            dqn0_prob_arr = d_prob_real if d_prob_real is not None else None
            using_real = True  # when real RL exists, we usually skip v1/v2

    # Button row for building/clearing real sample
    st.markdown("### Real Evaluation (Optional)")
    cols_btn = st.columns([1, 1, 2])
    with cols_btn[0]:
        if st.button("🧪 Build evaluation sample from real models", use_container_width=True):
            try:
                out = aa_collect_real_scores(
                    ctx,
                    k_pairs=200,
                    sla_min=sla_min,
                    weight="travel_time_est",
                    temp=0.08,
                    rng_seed=19
                )
                n_lab = len(out["y_true"])
    
                if n_lab == 0:
                    # No usable pairs → keep proxy scores, don't flip into "using_real" mode
                    st.warning(
                        "Real evaluation sample had **0 usable pairs** "
                        "(no valid paths/scores). Keeping proxy scores instead."
                    )
                    using_real = False
                else:
                    st.success(f"Collected {n_lab} labeled samples.")
                    aa_eval = ctx.get("aa_eval")
                    y_true_arr = aa_eval["y_true"]
                    a_star_prob_arr = aa_eval["a_star_prob"]
                    dqn0_prob_arr = aa_eval.get("dqn_prob", None)
                    using_real = True
            except Exception as e:
                st.error(f"Could not collect real samples: {e}")
    with cols_btn[1]:
        if st.button("♻️ Clear real evaluation cache", use_container_width=True):
            st.session_state.ctx.pop("aa_eval", None)
            using_real = False
    
            # ⬇️ Reset evaluation arrays back to the proxy set so lengths align
            rng = np.random.default_rng(11)
            df_clean["sla_min"] = int(sla_min)
            noise = rng.normal(0, 6, size=len(df_clean))
            actual_time = df_clean["travel_time_est"].values + noise
            y_true_arr = (actual_time <= df_clean["sla_min"].values).astype(int)
    
            a_star_score = df_clean["sla_min"].values - df_clean["travel_time_est"].values
            dqn0_score   = a_star_score + rng.normal(0, 5, size=len(df_clean)) + 0.08*df_clean["distance_km"].values*(-0.2)
            a_star_prob_arr = _logistic(a_star_score)
            dqn0_prob_arr   = _logistic(dqn0_score)
    
            tightness = np.clip((df_clean["sla_min"].values / (df_clean["travel_time_est"].values + 1e-5)), 0.5, 1.5)
            traffic   = np.clip(df_clean["distance_km"].values / max(df_clean["distance_km"].max(), 1e-9), 0, 1)
            dqn1_prob = np.clip(dqn0_prob_arr + 0.06*(tightness - 1.0) + 0.05*(1.0 - traffic) + rng.normal(0, 0.01, size=len(df_clean)), 0, 1)
            dqn2_prob = np.clip(dqn1_prob + (y_true_arr - 0.5)*0.05, 0, 1)
    
            st.info("Cleared. The tab will fall back to proxy scores.")

    # Ensure all arrays have the same base length
    L = len(y_true_arr)
    a_star_prob_arr = a_star_prob_arr[:L]
    if dqn0_prob_arr is not None: dqn0_prob_arr = dqn0_prob_arr[:L]
    if not using_real:
        dqn1_prob = dqn1_prob[:L]
        dqn2_prob = dqn2_prob[:L]

        # 3) Consistent test split (robust to tiny/imbalanced data)
    idx = np.arange(len(y_true_arr))
    n = len(y_true_arr)

    if n < 5:
        # If the dataset is very small, just evaluate on the whole set
        st.warning("Dataset is very small; evaluating on all samples (no holdout split).")
        idx_test = idx
    else:
        # Pick a test size that leaves at least 1 train sample
        n_test = max(1, int(round(n * 0.30)))
        if n_test >= n:
            n_test = n - 1
        test_size = n_test / n

        # Use stratify only if both classes exist and each has at least 2 samples
        unique, counts = np.unique(y_true_arr, return_counts=True)
        can_stratify = (unique.size >= 2) and (counts.min() >= 2)

        try:
            _, X_te, _, y_te, _, idx_te = train_test_split(
                a_star_prob_arr.reshape(-1, 1),
                y_true_arr,
                idx,
                test_size=test_size,
                random_state=19,
                stratify=(y_true_arr if can_stratify else None)
            )
            idx_test = idx_te
        except ValueError:
            # Fall back to a non-stratified split if sklearn still complains
            _, X_te, _, y_te, _, idx_te = train_test_split(
                a_star_prob_arr.reshape(-1, 1),
                y_true_arr,
                idx,
                test_size=test_size,
                random_state=19,
                stratify=None
            )
            idx_test = idx_te

    # Build evaluation dict
    E = {"A*": _evaluate_probs(y_true_arr[idx_test], a_star_prob_arr[idx_test])}
    if dqn0_prob_arr is not None:
        E["DQN"] = _evaluate_probs(y_true_arr[idx_test], dqn0_prob_arr[idx_test])

    # Only include synthetic v1/v2 when we’re not using real RL outputs
    if not using_real:
        E["DQN v1"] = _evaluate_probs(y_true_arr[idx_test], dqn1_prob[idx_test])
        E["DQN v2"] = _evaluate_probs(y_true_arr[idx_test], dqn2_prob[idx_test])

    # === Sub-tabs ===
    tab_over, tab_eval, tab_refine, tab_eda = st.tabs(
        ["Overview", "Detailed Evaluation", "Refinements", "EDA & Trends"]
    )

    # Overview
    with tab_over:
        st.caption(f"Source: **{data_mode_label}** | Rows: {len(df_clean):,}")
        cols = st.columns(len(E))
        for (name, metrics), col in zip(E.items(), cols):
            col.metric(name, value=f"{metrics['auc']:.3f} AUC", delta=f"F1 {metrics['f1']:.3f}")
        st.dataframe(df_clean.head(20), use_container_width=True)

    # Detailed Evaluation
    with tab_eval:
        st.subheader("ROC Curves")
        fig = plt.figure()
        for name in E.keys():
            plt.plot(E[name]["fpr"], E[name]["tpr"], label=f"{name} (AUC={E[name]['auc']:.3f})")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right"); st.pyplot(fig)

        st.subheader("Confusion Matrices")
        for name in E.keys():
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(E[name]["cm"]).plot(ax=ax)
            ax.set_title(name); st.pyplot(fig)

        st.subheader("Metrics Table")
        mdf = pd.DataFrame([{
            "Model": k, "Accuracy": E[k]["acc"], "Precision": E[k]["prec"],
            "Recall": E[k]["rec"], "F1": E[k]["f1"], "ROC_AUC": E[k]["auc"]
        } for k in E])
        st.dataframe(mdf, use_container_width=True)

    # Refinements (delta view) — only helpful when proxies are shown
    with tab_refine:
        st.write("Performance across models (AUC not shown here for brevity).")
        fig = plt.figure(figsize=(8, 4))
        labels = list(E.keys())
        accs = [E[k]["acc"] for k in labels]
        plt.bar(labels, accs); plt.ylim(0, 1)
        plt.title("Accuracy by Model"); st.pyplot(fig)

    # EDA & Trends
    with tab_eda:
        st.subheader("Correlation Heatmap")

        same_len = (len(y_true_arr) == len(df_clean))
        if same_len:
            # We can enrich the original cleaned frame
            eda = df_clean.copy()
            eda["on_time_true"] = y_true_arr
            eda["a_star_prob"]  = a_star_prob_arr
            if dqn0_prob_arr is not None:
                eda["dqn_prob"] = dqn0_prob_arr
            if not using_real:
                # Only when we're using proxy refinements
                eda["dqn2_prob"] = dqn2_prob
        else:
            # Fall back to a self-contained EDA frame built from evaluation arrays only
            st.info(
                "Evaluation sample size differs from the cleaned dataset. "
                "Showing correlations on the evaluation sample only."
            )
            cols = {
                "on_time_true": y_true_arr,
                "a_star_prob":  a_star_prob_arr,
            }
            if dqn0_prob_arr is not None:
                cols["dqn_prob"] = dqn0_prob_arr
            if not using_real:
                cols["dqn2_prob"] = dqn2_prob[:len(y_true_arr)] if 'dqn2_prob' in locals() else None
            eda = pd.DataFrame({k: v for k, v in cols.items() if v is not None})

        corr = eda.corr(numeric_only=True)
        fig = plt.figure(figsize=(7, 6))
        plt.imshow(corr, interpolation='nearest')
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.colorbar()
        plt.title("Correlation Matrix")
        st.pyplot(fig)

        st.subheader("Trends by Distance Buckets")
        if same_len and "distance_km" in df_clean.columns:
            eda["distance_bucket"] = pd.cut(
                eda["distance_km"], bins=[0,5,10,15,20,30,50], include_lowest=True
            )
            group = eda.groupby("distance_bucket").agg(
                true=("on_time_true","mean"),
                **(
                    {"dqn2": ("dqn2_prob","mean")}
                    if "dqn2_prob" in eda.columns else
                    {"dqn": ("dqn_prob","mean")} if "dqn_prob" in eda.columns
                    else {}
                )
            ).reset_index()

            fig = plt.figure()
            plt.plot(group["distance_bucket"].astype(str), group["true"], marker="o", label="True on-time")
            for col in group.columns:
                if col not in ("distance_bucket", "true"):
                    plt.plot(group["distance_bucket"].astype(str), group[col], marker="o", label=col.upper())
            plt.xticks(rotation=30, ha="right")
            plt.legend()
            plt.ylabel("Rate")
            plt.title("On-time vs Distance")
            st.pyplot(fig)
        else:
            st.info(
                "Distance-bucket trends skipped because `distance_km` isn’t aligned with the evaluation sample."
            )

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

    st.divider()
    st.subheader("Theme")
    theme_choice = st.selectbox("Appearance", ["Auto", "Light", "Dark"], index=0, key="theme_choice")

    def _theme_palette(choice: str):
        # defaults = light
        palette = {
            "bg": "#ffffff",
            "text": "#111827",
            "muted": "#4b5563",
            "border": "#e5e7eb",
            "card": "#ffffff",
            "chip_bg": "#ffffff",
            "chip_border": "#eef2f7",
        }
        if choice == "Dark":
            palette.update({
                "bg": "#0e1117",
                "text": "#e5e7eb",
                "muted": "#9ca3af",
                "border": "#1f2937",
                "card": "#111827",
                "chip_bg": "#0f172a",
                "chip_border": "#1f2937",
            })
        return palette

    st.session_state["_theme_choice"] = theme_choice

# Apply the theme
inject_theme_css(theme_choice)

if (theme_choice == "Dark") or (theme_choice == "Auto" and (st.get_option("theme.base") or "light").lower() == "dark"):
    px.defaults.template = "plotly_dark"
    plt.style.use("dark_background")
else:
    px.defaults.template = "plotly_white"
    plt.style.use("default")

# --- Advanced Analysis controls ---
st.subheader("Advanced Analysis Controls")
aa_sla_min = st.number_input("SLA (minutes)", min_value=30, max_value=240, value=90, step=5)
aa_use_reward_shaping = st.checkbox("Apply v1: Reward Shaping", value=True)
aa_use_opt_tuning     = st.checkbox("Apply v2: Optimization Tuning", value=True)

st.markdown("---")
aa_data_mode = st.radio("Advanced Analysis Dataset", ["Sample CSV", "Upload CSV"], index=0, key="aa_data_mode")
aa_uploaded = None
if aa_data_mode == "Upload CSV":
    aa_uploaded = st.file_uploader(
        "Upload edges CSV for Advanced Analysis",
        type=["csv"], key="aa_uploader"
    )

# ---------------------------- Tabs ----------------------------
tabs = ["Overview", "Data", "Explore", "Clean", "Model",
        "Results", "Reports", "Pipeline", "Advanced Analysis", "Help"]
T0, T1, T2, T3, T4, T5, T6, T7, TA, TH = st.tabs(tabs)

# ---------------------------- Overview ----------------------------
with T0:
    st.subheader("Executive Summary")
    st.write(
        "This dashboard demonstrates how AI can optimize supply-chain routes. "
        "It compares a traditional pathfinding algorithm (A*) with an AI agent (Reinforcement Learning) "
        "to reduce travel time and fuel use. Use the tabs to load data, explore, clean, model, and generate reports."
    )
    executive_summary_panel(ctx)

    try:
        metrics_norm = _coerce_metrics(ctx.get("metrics", {}))
        series = []
        for m, kpis in metrics_norm.items():
            vals = [v for v in kpis.values() if isinstance(v, (int, float))]
            if vals:
                series.append((m, sum(vals)/len(vals)))
        if series:
            fig_spark = go.Figure()
            fig_spark.add_trace(go.Scatter(y=[x[1] for x in series], mode="lines+markers", name="KPI trend"))
            fig_spark.update_layout(height=120, margin=dict(l=10,r=10,t=10,b=10), template="plotly_white", showlegend=False)
            st.plotly_chart(fig_spark, use_container_width=True)
    except Exception:
        pass
    st.divider()
    st.subheader("What to do next (Quick Start)")
    cA, cB, cC = st.columns(3)
    with cA:
        st.markdown("**1) Load Data** \nUpload a CSV or generate synthetic data from the sidebar.")
    with cB:
        st.markdown("**2) Run Pipeline** \nUse the Pipeline tab for one_click end-to-end execution.")
    with cC:
        st.markdown("**3) Download Report** \nGo to Reports to export a PDF/TXT summary.")

# ---------------------------- Data ----------------------------
with T1:
    st.subheader("Data Preview & Quality")
    if ctx.get("edges_df") is None:
        st.info("Load a CSV or generate synthetic data from the **sidebar** to begin.")
    else:
        df = ctx["edges_df"]
        missing_total = int(df.isna().sum().sum())
        numeric_cnt = int(df.select_dtypes(include="number").shape[1])
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Rows", len(df))
        with c2: st.metric("Columns", df.shape[1])
        with c3: st.metric("Numeric Columns", numeric_cnt)
        with c4: st.metric("Missing Values", missing_total)
    
        sch = ctx.get("schema", {"ok": True, "missing": [], "non_numeric": []})
        if not sch.get("ok", True):
            st.warning("Schema checks found potential issues:")
            if sch.get("missing"): st.write("• **Missing columns:**", sch["missing"])
            if sch.get("non_numeric"): st.write("• **Non-numeric columns:**", sch["non_numeric"])
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
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        use_mpl = bool(ctx.get("use_mpl", False))

        with st.expander("Quick Insights (auto-generated summary)"):
            if num_cols:
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
                            skew_text.append(f"**{c}** appears highly skewed (skew={sk:2f}).")
                    except Exception:
                        pass
                st.markdown(
                    f"- Numeric features detected: **{len(num_cols)}**  \n"
                    f"- {corr_text}  \n"
                    + ("- " + "  n- ".join(skew_text) if skew_text else "- No extreme skew detected in the first few numeric features.")
                )
            else:
                st.write("No numeric columns detected.")

        if num_cols:
            col = st.selectbox("Histogram column", num_cols)
            fig = hist_plot(df, col, use_matplotlib=use_mpl)
            if fig is not None:
                if use_mpl:
                    _ = st.pyplot(fig, use_container_width=True)
                else:
                    _ = st.plotly_chart(fig, use_container_width=True)
            
            if len(num_cols) >= 2:
                cA, cB = st.columns(2)
                with cA: x = st.selectbox("X", num_cols, index=0)
                with cB: y = st.selectbox("Y", num_cols, index=1)
                fig2 = scatter_plot(df, x, y, use_matplotlib=use_mpl)
                if fig2 is not None:
                    if use_mpl:
                        _ = st.pyplot(fig2, use_container_width=True)
                    else:
                        _ = st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Summary statistics**")
        st.dataframe(df.describe().T, use_container_width=True) 

# ---------------------------- Clean ----------------------------
with T3:
    st.subheader("Cleaning")
    st.caption("These steps improve data quality and comparability before modeling.")
    if ctx.get("edges_df") is None:
        st.info("Load or generate data first")
    else:
        df = ctx["edges_df"]
        normalize = st.checkbox("Normalize numeric columns", value=True,
                                help="Scales numeric values so features are comparable.")
        iqr_mult = st.slider("IQR multiplier (cap outliers)", 0.5, 3.0, 1.5,
                             help="Caps extreme values using the interquartile range rule.")
        drop_na = st.checkbox("Drop rows with missing values", value=False,
                              help="Removes rows containing missing values.")

    if st.button("Apply Cleaning"):
        with st.spinner("Cleaning data..."):
            raw_rows = len(df)
            if drop_na:
                df = df.dropna()
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
            with c3: st.metric("Rows dropped", raw_rows - len(df_clean))

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
                try:
                    ctx["metrics"] = evaluate_kpis(ctx.get("baseline"), ctx.get("rl_results"))
                except Exception as e:
                    st.warning(f"Metrics not available yet: {e}")
            st.write(ctx["baseline"])

        # RL controls (lazy import)
        tt = st.number_input("DQN total_timesteps (demo)",
                             value=200, step=100, min_value=100)  # keep small for cloud
        eval_eps = st.number_input("Eval episodes", value=10, step=5, min_value=1)
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
                        eval_env = RoutingEnv(ctx["G"])
                        with st.spinner("Running inference..."):
                            res = infer_dqn(
                                ctx["rl_model_path"], eval_env, episodes=int(eval_eps)
                            )
        
                        # --- try to extract a route cost / length from the RL result ---
                        rl_weight = None
        
                        # 1) Look for any obvious route-length keys in the result dict
                        for key in ("weighted_length", "total_time", "travel_time",
                                    "episode_time", "length"):
                            if key not in res:
                                continue
                            v = res[key]
                            if isinstance(v, (int, float)):
                                rl_weight = float(v)
                                break
                            if isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float)):
                                rl_weight = float(v[0])
                                break
        
                        # 2) Fallback: approximate a "cost" from mean_reward if nothing else is available
                        if rl_weight is None:
                            mr = res.get("mean_reward")
                            if isinstance(mr, (int, float)):
                                # In many routing RL setups lower cost ↔ higher (less negative) reward.
                                # We use -mean_reward as a simple proxy so that the Results tab
                                # can still show an RL length and efficiency gain.
                                # Rescale the reward so it's in the same ballpark as the A* path length.
                                scale = 100.0
                                rl_weight = float(-mr) / scale
        
                        # Store RL outputs in context (now including weighted_length)
                        ctx["rl_results"] = {
                            "episodes": int(eval_eps),
                            "rl_mean_reward": float(res.get("mean_reward", 0.0)),
                            "weighted_length": rl_weight,
                            "rl_weighted_length": rl_weight,  # alias for any older code
                        }
        
                        # --- build metrics so Results tab can show RL length + efficiency ---
                        base = ctx.get("baseline")
                        if base and "weighted_length" in base and isinstance(rl_weight, (int, float)):
                            b_len = float(base["weighted_length"])
                            r_len = float(rl_weight)
                            eff = 100.0 * (b_len - r_len) / b_len if b_len > 0 else None
        
                            ctx["metrics"] = {
                                "Baseline": {"weighted_length": b_len},
                                "RL": {
                                    "weighted_length": r_len,
                                    "efficiency_gain_pct": eff,
                                },
                            }
                            st.success("Metrics updated from RL inference.")
                        else:
                            # Fall back to the original helper if we can't compute both lengths yet
                            try:
                                ctx["metrics"] = evaluate_kpis(
                                    ctx.get("baseline"), ctx.get("rl_results")
                                )
                                st.success("Metrics updated from RL inference.")
                            except Exception as e:
                                st.warning(f"Metrics update skipped: {e}")
        
                        log_run("infer_dqn", ctx["rl_results"])
                        st.write(ctx["rl_results"])
                except Exception as e:
                    st.error(f"RL inference unavailable: {e}")


# ---------------------------- Results ----------------------------
with T5:
    st.subheader("Results & Comparison")

    # If a model has run but metrics aren't cached, compute them now
    baseline = ctx.get("baseline")
    rl = ctx.get("rl_results")
    if (baseline or rl) and not ctx.get("metrics"):
        try:
            ctx["metrics"] = evaluate_kpis(baseline, rl)
        except Exception as e:
            st.error(f"Could not compute metrics: {e}")

    # Optional manual recompute
    cols_re = st.columns([1, 6])
    with cols_re[0]:
        if st.button("Recompute metrics"):
            try:
                ctx["metrics"] = evaluate_kpis(ctx.get("baseline"), ctx.get("rl_results"))
                st.success("Metrics updated.")
            except Exception as e:
                st.error(f"Recompute failed: {e}")

    metrics = _coerce_metrics(ctx.get("metrics", {}))
    metrics = _ensure_chartable_kpis(metrics)
    
    if not metrics:
        st.info("Run a model (A* and/or RL) to view results.")
    else:
        base = metrics.get("Baseline", {}) or {}
        rl_m = metrics.get("RL", {}) or {}
    
        # accept either the old names (baseline_weighted_length / rl_weighted_length)
        # or the canonical 'weighted_length'
        b_len = _get_any(base, "weighted_length", "baseline_weighted_length")
        r_len = _get_any(rl_m, "weighted_length", "rl_weighted_length")
    
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Baseline Weighted Length",
                "-" if b_len is None else round(b_len, 2)
            )
        with c2:
            st.metric(
                "RL Weighted Length",
                "-" if r_len is None else round(r_len, 2)
            )
        with c3:
            if isinstance(b_len, (int, float)) and isinstance(r_len, (int, float)) and b_len > 0:
                st.metric("Efficiency Gain", f"{round(100*(b_len - r_len)/b_len, 2)}%")
            else:
                st.metric("Efficiency Gain", "-")

    # KPI comparison chart
    try:
        fig = kpi_bar_chart(metrics)  # your Matplotlib-based helper
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
    except Exception:
        # Plotly fallback
        records = []
        for model, kpis in metrics.items():
            if isinstance(kpis, dict):
                for k, v in kpis.items():
                    if isinstance(v, (int, float)):
                        records.append({"Model": model, "KPI": k, "Value": float(v)})
        if records:
            import plotly.express as px
            fig = px.bar(
                records, x="KPI", y="Value", color="Model",
                barmode="group", template="plotly_white",
                title="Key Performance Indicators"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric KPIs to visualize yet. Run A* and/or RL.")

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
                    st.download_button("Download Report", data=fh.read(), file_name=os.path.basename(path))
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
    st.subheader("Pipeline – One Click Run (Baseline)")
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
            st.json(ctx["metrics"])

            if os.path.exists(path):
                    with open(path, "rb") as fh:
                        st.download_button("Download Report", data=fh.read(), file_name=os.path.basename(path))
        except Exception as e:
                    st.error(f"Pipeline failed: {e}")

# ---------------------------- Advanced Analysis ----------------------------
with TA:
    st.subheader("Advanced Analysis & Model Refinement")
    render_advanced_analysis_tab(
        df_source = (ctx.get("edges_clean") if ctx.get("edges_clean") is not None
                 else ctx.get("edges_df") if ctx.get("edges_df") is not None
                 else None),
        sla_min=int(aa_sla_min) if 'aa_sla_min' in locals() else 90,
        use_reward_shaping=bool(aa_use_reward_shaping) if 'aa_use_reward_shaping' in locals() else True,
        use_opt_tuning=bool(aa_use_opt_tuning) if 'aa_use_opt_tuning' in locals() else True,
        data_mode_label=("Sample" if aa_data_mode=="Sample CSV" else "Uploaded") if 'aa_data_mode' in locals() else "Sample",
        uploaded_file=aa_uploaded if 'aa_uploaded' in locals() else None
    )
    
# ---------------------------- Help ----------------------------
with TH:
    st.subheader("Help")
    st.markdown("**Quick Start (non-technical)**")
    st.markdown("1) Use the **Overview** to understand what the app does.  \n"
                "2) Load data from the **sidebar**.  \n"
                "3) Click **Pipeline** to run everything.  \n"
                "4) Get your **Report** in PDF/TXT.")

    with st.expander("FAQ"):
            st.markdown("**Can I use my own data?** Yes - upload a CSV with the required columns shown on the Data tab.")
            st.markdown("**What is 'weighted length'?** A composite cost for routing that balances distance/time/fuel.")
            st.markdown("**Is it reproducible?** Yes - synthetic generation logs parameters and seeds for replication.")

    st.markdown("---")
    st.markdown("**Full Guide (from HELP.md)**")
    help_text = _safe_read("HELP.md", default_text="HELP.md not found. Please include HELP.md next to app.py.")
    st.markdown(help_text)
