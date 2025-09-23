# viz.py
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd

# --------------------------
# Histogram Plot
# --------------------------
def hist_plot(df, column, save_path=None):
    plt.figure(figsize=(6, 4))
    sns.histplot(df[column], kde=True, bins=20)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# --------------------------
# Scatter Plot
# --------------------------
def scatter_plot(df, x, y, save_path=None):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f"Scatter Plot: {x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# --------------------------
# Route Map (Graph Visual)
# --------------------------
def route_map(edges, save_path=None):
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)
    weights = nx.get_edge_attributes(G, "weight")

    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.title("Route Map")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# --------------------------
# KPI Bar Chart (Baseline vs RL)
# --------------------------
def kpi_bar_chart(metrics_dict, save_path=None):
    """
    Plot KPI comparison as a bar chart.

    Args:
        metrics_dict (dict): Dictionary like:
            {"Baseline": {"weighted_length": 12.9, "time": 4.2},
             "RL": {"weighted_length": 10.5, "time": 3.7}}
        save_path (str): If provided, saves plot to file.
    """
    records = []
    for model, kpis in metrics_dict.items():
        for k, v in kpis.items():
            records.append({"Model": model, "KPI": k, "Value": v})
    df = pd.DataFrame(records)

    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="KPI", y="Value", hue="Model")
    plt.title("KPI Comparison: Baseline vs RL")
    plt.xlabel("Key Performance Indicator")
    plt.ylabel("Value")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
