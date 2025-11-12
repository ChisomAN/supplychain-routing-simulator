import math
from typing import Dict


def evaluate_kpis(baseline: Dict | None = None, rl: Dict | None = None) -> Dict:
    """
    Small, robust KPI helper used by the Pipeline and quick actions.
    - Always returns a plain dict of simple numbers.
    - Skips RL metrics when no RL run has happened or values are NaN.
    """
    out: Dict[str, float] = {}

    # ---- Baseline KPIs ----
    if baseline:
        b_len = baseline.get("weighted_length")
        if isinstance(b_len, (int, float)) and math.isfinite(b_len):
            out["baseline_weighted_length"] = float(b_len)

    # ---- RL KPIs (optional) ----
    if rl:
        # mean reward (only if it is a real finite number)
        mr = rl.get("mean_reward")
        if isinstance(mr, (int, float)) and math.isfinite(mr):
            out["rl_mean_reward"] = float(mr)

        # RL weighted length, if available
        rl_len = rl.get("weighted_length") or rl.get("rl_weighted_length")
        if isinstance(rl_len, (int, float)) and math.isfinite(rl_len):
            out["rl_weighted_length"] = float(rl_len)

        # If we have both lengths, compute efficiency gain (for Pipeline JSON)
        b_len = out.get("baseline_weighted_length")
        if b_len is not None and "rl_weighted_length" in out and b_len > 0:
            out["efficiency_gain_pct"] = 100.0 * (b_len - out["rl_weighted_length"]) / b_len

    return out
