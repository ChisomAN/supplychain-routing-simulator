import pandas as pd
import plotly.express as px

# Optional Matplotlib; if unavailable, stay on Plotly
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def hist_plot(df: pd.DataFrame, col: str, use_matplotlib: bool = False):
    if col not in df.columns:
        return None
    if use_matplotlib and _HAS_MPL:
        fig, ax = plt.subplots()
        df[col].dropna().plot(kind="hist", bins=30, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        fig.tight_layout()
        return fig
    return px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")


def scatter_plot(df: pd.DataFrame, x: str, y: str, use_matplotlib: bool = False):
    if x not in df.columns or y not in df.columns:
        return None
    if use_matplotlib and _HAS_MPL:
        fig, ax = plt.subplots()
        ax.scatter(df[x], df[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{x} vs {y}")
        fig.tight_layout()
        return fig
    return px.scatter(df, x=x, y=y, title=f"{x} vs {y}")


def route_map(nodes_df: pd.DataFrame, routes: list[tuple[int, int]] | None = None):
    fig = px.scatter(nodes_df, x="x", y="y", text="node_id", title="Nodes Map")
    return fig
