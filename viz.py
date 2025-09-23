# viz.py
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import plotly.express as px
# --------------------------
# Histogram Plot
# --------------------------
def hist_plot(df, column, use_matplotlib=False):
    if use_matplotlib:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[column], kde=True, bins=20, ax=ax)
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        return fig
    else:
        return px.histogram(df, x=column, nbins=20, marginal="box", title=f"Histogram of {column}")

# --------------------------
# Scatter Plot
# --------------------------
def scatter_plot(df, x, y, use_matplotlib=False):
    if use_matplotlib:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x=x, y=y, ax=ax)
        ax.set_title(f"Scatter Plot: {x} vs {y}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        return fig
    else:
        return px.scatter(df, x=x, y=y, trendline="ols", title=f"{x} vs {y}")

# --------------------------
# Route Map (Graph Visual)
# --------------------------
def route_map(nodes_df):
    """Draws route map from a nodes/edges dataframe (for Streamlit)."""
    G = nx.from_pandas_edgelist(nodes_df, "origin_id", "dest_id", ["distance_km"])
    pos = nx.spring_layout(G, seed=42)
    weights = nx.get_edge_attributes(G, "distance_km")

    fig, ax = plt.subplots(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, ax=ax)
    ax.set_title("Route Map")
    return fig

# --------------------------
# KPI Bar Chart (Baseline vs RL)
# --------------------------
def kpi_bar_chart(metrics_dict, save_path=None):
    """
    Plot KPI comparison as a bar chart.
    Handles both nested and flat metric dictionaries.
    """
    records = []

    for model, kpis in metrics_dict.items():
        if isinstance(kpis, dict):
            # Expected nested dict of KPIs
            for k, v in kpis.items():
                records.append({"Model": model, "KPI": k, "Value": v})
        else:
            # Flat float/int — wrap as single KPI
            records.append({"Model": model, "KPI": "score", "Value": kpis})

    df = pd.DataFrame(records)

    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="KPI", y="Value", hue="Model")
    plt.title("KPI Comparison: Baseline vs RL")
    plt.xlabel("Key Performance Indicator")
    plt.ylabel("Value")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        return None
    else:
        return plt.gcf()
