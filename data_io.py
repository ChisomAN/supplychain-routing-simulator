import io
import numpy as np
import pandas as pd
import networkx as nx
import requests
from typing import Optional, Dict, Tuple, List

REQUIRED_COLUMNS: List[str] = [
    "origin_id",
    "dest_id",
    "distance_km",
    "travel_time_est",
    "fuel_rate",
]


def validate_schema(df: pd.DataFrame) -> Dict:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    numeric_cols = ["distance_km", "travel_time_est", "fuel_rate"]
    non_numeric = [
        c for c in numeric_cols if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    return {"ok": len(missing) == 0 and len(non_numeric) == 0, "missing": missing, "non_numeric": non_numeric}


def _graph_to_edges_df(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({
            "origin_id": u,
            "dest_id": v,
            "distance_km": float(d.get("distance_km", 1.0)),
            "travel_time_est": float(d.get("travel_time_est", 1.0)),
            "fuel_rate": float(d.get("fuel_rate", 0.1)),
        })
    return pd.DataFrame(rows)


def generate_synthetic_graph(n_nodes: int = 40, edge_prob: float = 0.25,
                             speed_mph: float = 45.0, delay_prob: float = 0.15,
                             seed: int = 42) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
    pos = {i: (rng.uniform(-1, 1), rng.uniform(-1, 1)) for i in G.nodes}
    nx.set_node_attributes(G, pos, "pos")
    for u, v in G.edges():
        dxy = np.array(pos[u]) - np.array(pos[v])
        distance_km = float(np.linalg.norm(dxy) * 10.0)
        base_time_min = distance_km / (speed_mph * 1.60934 / 60.0)
        shock = rng.binomial(1, delay_prob)
        travel_time_est = base_time_min * (1.0 + 0.5 * shock)
        fuel_rate = 0.08 + 0.04 * rng.random()
        G[u][v]["distance_km"] = distance_km
        G[u][v]["travel_time_est"] = float(travel_time_est)
        G[u][v]["fuel_rate"] = float(fuel_rate)
    edges_df = _graph_to_edges_df(G)
    nodes_df = pd.DataFrame({
        "node_id": list(G.nodes()),
        "x": [pos[i][0] for i in G.nodes()],
        "y": [pos[i][1] for i in G.nodes()],
    })
    return G, nodes_df, edges_df


def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def load_from_url(url: str, timeout: int = 20) -> pd.DataFrame:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))


def load_data(path: Optional[str] = None, synth_params: Optional[Dict] = None,
              seed: int = 42) -> Dict:
    """Load CSV or synthesize a graph-backed edge table. Return a context dict."""
    if path:
        df = pd.read_csv(path)
        sch = validate_schema(df)
        if set(["origin_id", "dest_id"]).issubset(df.columns):
            G = nx.from_pandas_edgelist(df, source="origin_id", target="dest_id",
                                        edge_attr=True, create_using=nx.Graph())
        else:
            G = nx.Graph()
        nodes = pd.DataFrame({"node_id": list(G.nodes())})
        return {"G": G, "nodes_df": nodes, "edges_df": df, "schema": sch}
    sp = synth_params or {}
    G, nodes_df, edges_df = generate_synthetic_graph(
        n_nodes=sp.get("n_nodes", 40),
        edge_prob=sp.get("edge_prob", 0.25),
        speed_mph=sp.get("speed_mph", 45.0),
        delay_prob=sp.get("delay_prob", 0.15),
        seed=seed,
    )
    sch = validate_schema(edges_df)
    return {"G": G, "nodes_df": nodes_df, "edges_df": edges_df, "schema": sch}
