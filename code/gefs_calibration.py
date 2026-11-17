"""Calibrate GEFS state degree-day forecasts to NOAA CPC observed degree days.
For each state, lead h and quantity (HDD, CDD): observed = a + b * forecast, fitted by least squares on target dates in the 12 months before
the first forecast origin (17 July 2021 - 16 July 2022); calibrated forecasts are clipped at zero.
Writes data/weather/gefs_dd_calibrated.csv (state, date = issue day, h, target_date, hdd_raw, cdd_raw, hdd_fc, cdd_fc) and data/weather/gefs_skill.csv; reads the per-issue files of gefs_degree_days.py (WORK/gefs) or data/weather/gefs_dd_raw_by_issue.csv.gz."""
import os, glob, json
import numpy as np, pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WEATHER as WX, EXTERNAL, WORK
FS = sorted(glob.glob(f"{WORK}/gefs/*.csv"))
G = (pd.concat([pd.read_csv(f).assign(date=os.path.basename(f)[:8]) for f in FS]) if FS else
     pd.read_csv(f"{WX}/gefs_dd_raw_by_issue.csv.gz").rename(columns={"issue_date": "date"}).astype({"date": str}))
G["date"] = pd.to_datetime(G.date); G["target_date"] = pd.to_datetime(G.target_date)
O = pd.read_csv(f"{EXTERNAL}/cpc_degree_days.csv", parse_dates=["date"]).rename(columns={"date": "target_date", "hdd": "hdd_obs", "cdd": "cdd_obs"})
M = G.merge(O, on=["state", "target_date"], how="left")
CAL0, CAL1, ORIG = pd.Timestamp("2021-07-17"), pd.Timestamp("2022-07-16"), pd.Timestamp("2022-07-17")
out, coef = [], []
for (s, h), g in M.groupby(["state", "h"]):
    g = g.copy(); c = g[(g.target_date >= CAL0) & (g.target_date <= CAL1)].dropna(subset=["hdd_obs", "cdd_obs"])
    for q in ("hdd", "cdd"):
        x, y = c[f"{q}_f"].values, c[f"{q}_obs"].values
        if len(c) > 60 and np.var(x) > 1e-9: b, a = np.polyfit(x, y, 1)
        else: a, b = 0.0, 1.0
        g[f"{q}_fc"] = np.clip(a + b * g[f"{q}_f"].values, 0, None); coef.append(dict(state=s, h=h, q=q, a=a, b=b, n=len(c)))
    out.append(g)
C = pd.concat(out).rename(columns={"hdd_f": "hdd_raw", "cdd_f": "cdd_raw"})
C[["state", "date", "h", "target_date", "hdd_raw", "cdd_raw", "hdd_fc", "cdd_fc"]].sort_values(["state", "h", "date"]).to_csv(
    f"{WX}/gefs_dd_calibrated.csv", index=False, date_format="%Y-%m-%d")
pd.DataFrame(coef).to_csv(f"{WX}/gefs_calibration_coefficients.csv", index=False)
# skill on the evaluation period (issue dates from the first origin): correlation and RMSE against CPC, and against climatology
E = C[C.date >= ORIG].dropna(subset=["hdd_obs", "cdd_obs"]).copy()
E["dd_fc"] = E.hdd_fc + E.cdd_fc; E["dd_obs"] = E.hdd_obs + E.cdd_obs
hist = O[O.target_date < ORIG].copy(); hist["doy"] = hist.target_date.dt.dayofyear
cl = hist.groupby(["state", "doy"])[["hdd_obs", "cdd_obs"]].mean().reset_index()
E["doy"] = E.target_date.dt.dayofyear; E = E.merge(cl.rename(columns={"hdd_obs": "hdd_cl", "cdd_obs": "cdd_cl"}), on=["state", "doy"], how="left")
E["dd_cl"] = E.hdd_cl + E.cdd_cl
rows = []
for h, g in E.groupby("h"):
    a = g.groupby("state").apply(lambda x: pd.Series(dict(r=np.corrcoef(x.dd_fc, x.dd_obs)[0, 1], rmse=np.sqrt(np.mean((x.dd_fc - x.dd_obs) ** 2)),
                                                           rmse_cl=np.sqrt(np.mean((x.dd_cl - x.dd_obs) ** 2)))), include_groups=False)
    rows.append(dict(h=int(h), median_r=float(a.r.median()), median_rmse=float(a.rmse.median()), median_rmse_clim=float(a.rmse_cl.median()),
                     median_skill=float((1 - a.rmse ** 2 / a.rmse_cl ** 2).median()), n_states=len(a)))
pd.DataFrame(rows).to_csv(f"{WX}/gefs_skill.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())
miss = int(pd.date_range(ORIG, "2025-12-30").difference(G.date.unique()).size)
json.dump(dict(missing_issue_days=miss, skill=rows, calibration_window=["2021-07-17", "2022-07-16"]), open(f"{WX}/gefs_summary.json", "w"), indent=1)
print("missing issue days in evaluation:", miss)
