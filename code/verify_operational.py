"""Independent recomputation of the operational numbers (operational forecasts, GEFS skill) from the raw forecast files, with no code
shared with the analysis scripts. Usage: python verify_operational.py"""
import glob, json, os, pickle, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK as V, WEATHER as WX, RESULTS, EXTERNAL, STATE_DAILY
R = json.load(open(f"{RESULTS}/operational/results.json")); H = [1, 2, 3, 5, 7, 10, 14]
DATA = STATE_DAILY
bad, n = [], [0]


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def chk(name, a, b, tol):
    n[0] += 1
    if b is None or not abs(a - b) <= tol: bad.append((name, round(a, 5), b))


# ---- 1. operational national mean R2 (dates on which every variant and the benchmark forecasts exist)
states = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{V}/pred_test/*_h1.csv")); assert len(states) == 45
for h in H:
    vs = [v for v in ("ops", "ops_f", "ops_fc", "ops_fw", "ops_fo") if all(os.path.exists(f"{V}/exp/{v}/{s}_h{h}.csv") for s in states)]
    acc = {f"{v}:{m}": [] for v in vs for m in ("ridge", "gam", "hgb")}
    for s in states:
        b = pd.read_csv(f"{V}/pred_test/{s}_h{h}.csv")[["date", "target", "ridge", "gam", "hgb"]].rename(columns={"ridge": "b_r", "gam": "b_g", "hgb": "b_h"})
        b = b.merge(pd.read_csv(f"{V}/exp/fourier/{s}_h{h}.csv")[["date", "fourier_ridge"]], on="date")
        for v in vs:
            x = pd.read_csv(f"{V}/exp/{v}/{s}_h{h}.csv")[["date", "ridge", "gam", "hgb"]].rename(columns=lambda c: c if c == "date" else f"{v}:{c}")
            b = b.merge(x, on="date")
        b = b.dropna()
        for k in acc: acc[k].append(r2(b.target, b[k]))
    for k, x in acc.items(): chk(f"ops mean {k} h{h}", float(np.mean(x)), R["ops_mean"][str(h)].get(k), tol=1.5e-3)

# ---- 2. GEFS skill: median over states of the correlation between calibrated forecast and CPC degree days (HDD + CDD)
G = pd.read_csv(f"{WX}/gefs_dd_calibrated.csv", parse_dates=["date", "target_date"])
C = pd.read_csv(f"{EXTERNAL}/cpc_degree_days.csv", parse_dates=["date"]).rename(columns={"date": "target_date"})
G = G[G.date >= "2022-07-17"].merge(C, on=["state", "target_date"])
SK = {r["h"]: r for r in R["gefs_skill"]}
for h, g in G.groupby("h"):
    if h not in SK: continue
    med = g.groupby("state").apply(lambda x: np.corrcoef(x.hdd_fc + x.cdd_fc, x.hdd + x.cdd)[0, 1]).median()
    chk(f"GEFS median r h{h}", float(med), SK[h]["median_r"], tol=2e-3)

print(f"Operational verification: {n[0] - len(bad)}/{n[0]} recomputed values match")
for b in bad: print("  MISMATCH", b)
