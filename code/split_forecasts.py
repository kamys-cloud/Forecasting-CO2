"""Rebuild the per-state working files (WORK/pred_test, WORK/exp/<variant>, WORK/exp_<panel>/base, WORK/sb, WORK/pr,
WORK/opsv_ridge/<variant>, WORK/opsv_nl/<variant>) and the selected
settings from the consolidated day-by-day forecasts in forecasts/, so that the analysis and figure scripts can be run without refitting
the models. Usage: python split_forecasts.py"""
import os, sys, json
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import FORECASTS, WORK, ensure

EXP = {"fourier_ridge": "fourier", "fourier_nonlinear": "fourier_nl", "comp_annual": "comp_annual", "comp_doy": "comp_doy", "comp_month": "comp_month",
       "comp_year": "comp_year", "comp_cal": "comp_cal", "target_cal": "target_cal", "realtime": "realtime", "resid": "resid",
       "factorial_annual_fourier": "fa", "factorial_weekly_fourier": "fw", "factorial_cyclic_spline": "cs", "fourier_hgb_expanded": "faw",
       "cumulative_7d": "cum7", "cumulative_14d": "cum14", "degree_day_climatology": "clim", "degree_day_oracle": "oracle",
       "operational_lagged": "ops", "operational_fourier": "ops_f", "operational_fourier_climatology": "ops_fc", "operational_fourier_gefs": "ops_fw",
       "operational_fourier_observed_weather": "ops_fo"}


def split(name, outdir, by=("state", "horizon")):
    fn = os.path.join(FORECASTS, f"{name}.csv.gz")
    if not os.path.exists(fn): return 0
    d = pd.read_csv(fn); ensure(outdir); n = 0
    for key, g in d.groupby(list(by)):
        g = g.dropna(axis=1, how="all")
        f = f"{key[0]}_h{key[1]}.csv" if len(by) == 2 else f"{key}.csv" if not isinstance(key, tuple) else f"{key[0]}.csv"
        g.to_csv(os.path.join(outdir, f), index=False); n += 1
    return n


print("pred_test", split("baseline", os.path.join(WORK, "pred_test")))
for name, var in EXP.items(): print(var, split(name, os.path.join(WORK, "exp", var)))
for p in ("balanced", "highcov", "filled"): print(p, split(f"panel_{p}", os.path.join(WORK, f"exp_{p}", "base")))
print("sb", split("classical", os.path.join(WORK, "sb"), by=("state",)))
print("pr", split("prophet", os.path.join(WORK, "pr"), by=("state",)))
# perfect-prognosis / forecast-trained variants of the operational models (ridge and GAM/HGB were fitted as separate runs)
fn = os.path.join(FORECASTS, "operational_training_variants.csv.gz")
if os.path.exists(fn):
    V = pd.read_csv(fn); n = 0
    for (v, s, h), g in V.groupby(["variant", "state", "horizon"]):
        for sub, ms in (("opsv_ridge", ["ridge"]), ("opsv_nl", ["gam", "hgb"])):
            x = g.dropna(subset=ms)
            if x.empty: continue
            x[["state", "horizon", "date", "target_date", "target"] + ms].to_csv(os.path.join(ensure(WORK, sub, v), f"{s}_h{h}.csv"), index=False); n += 1
    print("opsv", n)
S = pd.read_csv(os.path.join(FORECASTS, "selected_settings.csv"))
DES = {v: v for v in EXP.values()}
for (s, h, des), g in S.groupby(["state", "horizon", "design"]):
    best = {r.model: json.loads(r.setting) for r in g.itertuples()}
    if des == "baseline": json.dump(best, open(os.path.join(WORK, "pred_test", f"{s}_h{h}_settings.json"), "w"))
    elif des in DES: json.dump({"best": best}, open(os.path.join(ensure(WORK, "exp", des), f"{s}_h{h}.json"), "w"))
    elif des.startswith("exp_"): json.dump({"best": best}, open(os.path.join(ensure(WORK, des, "base"), f"{s}_h{h}.json"), "w"))
    elif des.startswith("opsv_"):
        v = des[5:]
        for sub, ms in (("opsv_ridge", ["ridge"]), ("opsv_nl", ["gam", "hgb"])):
            b_ = {m: best[m] for m in ms if m in best}
            if b_: json.dump({"best": b_}, open(os.path.join(ensure(WORK, sub, v), f"{s}_h{h}.json"), "w"))
print("settings written")
