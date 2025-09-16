import math
from typing import Dict


def evaluate_kpis(baseline: Dict | None = None, rl: Dict | None = None) -> Dict:
    out = {}
    if baseline:
        out["baseline_weighted_length"] = float(
            baseline.get("weighted_length", math.nan))
    if rl:
        out["rl_mean_reward"] = float(rl.get("mean_reward", math.nan))
    return out
