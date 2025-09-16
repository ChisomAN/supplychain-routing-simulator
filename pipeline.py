from typing import Dict

from data_io import load_data
from cleaning import Cleaner
from models.baseline_a_star import run_a_star
from evaluation import evaluate_kpis
from reports import make_report


def run_step(step: str, ctx: Dict) -> Dict:
    ctx = dict(ctx)
    if step == "load":
        ctx.update(load_data(path=ctx.get("path"), synth_params=ctx.get(
            "synth_params"), seed=ctx.get("seed", 42)))
    elif step == "clean":
        cleaner = Cleaner(normalize=ctx.get("normalize", True),
                          iqr_mult=ctx.get("iqr_mult", 1.5))
        ctx["edges_clean"] = cleaner.fit_transform(
            ctx["edges_df"]) if "edges_df" in ctx else None
        ctx["cleaner"] = cleaner
    elif step == "baseline":
        if ctx.get("G"):
            ctx["baseline"] = run_a_star(
                ctx["G"], weight=ctx.get("baseline_weight", "distance_km"))
    elif step == "evaluate":
        ctx["metrics"] = evaluate_kpis(
            ctx.get("baseline"), ctx.get("rl_results"))
    elif step == "report":
        ctx["report_path"] = make_report(ctx)
    return ctx


def run_full_pipeline(ctx: Dict) -> Dict:
    for step in ["load", "clean", "baseline", "evaluate", "report"]:
        ctx = run_step(step, ctx)
    return ctx
