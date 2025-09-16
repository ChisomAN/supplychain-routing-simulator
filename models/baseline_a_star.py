import networkx as nx
from typing import Dict


def run_a_star(G: nx.Graph, source: int | None = None, target: int | None = None,
               weight: str = "distance_km") -> Dict:
    if source is None:
        source = list(G.nodes())[0]
    if target is None:
        target = list(G.nodes())[-1]
    try:
        path = nx.astar_path(G, source, target, weight=weight)
    except Exception:
        path = []
    length = 0.0
    for u, v in zip(path, path[1:]):
        length += G[u][v].get(weight, 1.0)
    return {"source": source, "target": target, "path": path, "weighted_length": length}
