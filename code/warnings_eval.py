"""P90 exceedance warnings scored on identical held-out origins.

Event: emissions on the target day exceed the state's 90th percentile of daily emissions before the first origin (2019-01-01 to
2022-07-16). Raw rule: warn when the point forecast exceeds that threshold. Calibrated rule: warn when the forecast exceeds a
model- and state-specific threshold chosen on the first evaluation year (origins 2022-07-17 to 2023-07-16) so that warnings are as
frequent as events there. BOTH rules are scored on the same held-out origins (2023-07-17 to 2025-12-31).
Scores (pooled over 45 states): event frequency, warning frequency, probability of detection (recall), precision (1 - false-alarm
ratio), false-alarm rate (false alarms / non-events), Peirce skill score and relative economic value (Richardson, 2000) at cost-loss
ratios 0.05, 0.1, 0.2, 0.3, 0.5 and at the base rate; 95% intervals from the joint 28-day date-block bootstrap (2,000 replicates).
Usage: python warnings_eval.py OUT_DIR"""
import os, sys
import numpy as np, pandas as pd
from jboot import block_weights
import panels as pn

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
ROOT = pn.ROOT; NB = 2000
CAL_END = pd.Timestamp("2023-07-17"); FIRST = pd.Timestamp("2022-07-17")
CL = [0.05, 0.1, 0.2, 0.3, 0.5]
STATES = sorted(f[:-4] for f in os.listdir(os.path.join(ROOT, "data", "state_daily")))
THR = {}
for s in STATES:
    y = pd.read_csv(os.path.join(ROOT, "data", "state_daily", f"{s}.csv"), parse_dates=["date"]).set_index("date").co2_mass.asfreq("D")
    THR[s] = float(y[y.index < FIRST].quantile(0.9))


def value(h, m, f, c, cl):
    n = h + m + f + c; s = (h + m) / n
    Ef = (f + h) / n * cl + m / n; Ec = min(cl, s); Ep = s * cl
    return (Ec - Ef) / (Ec - Ep) if Ec > Ep else np.nan


def scores(t):
    h, m, f, c = t; n = t.sum()
    out = dict(event_freq=(h + m) / n, warn_freq=(h + f) / n, pod=h / (h + m), precision=h / (h + f) if h + f else np.nan,
               far_ratio=f / (h + f) if h + f else np.nan, false_alarm_rate=f / (f + c), pss=h / (h + m) - f / (f + c))
    out["value_base_rate"] = value(h, m, f, c, (h + m) / n)
    for cl in CL: out[f"value_{cl}"] = value(h, m, f, c, cl)
    return out


def contingency(df, models):
    """per-model daily contingency counts on the held-out origins (rows = dates) for raw and calibrated rules"""
    res = {}
    dates = pd.DatetimeIndex(sorted(df.date[df.date >= CAL_END].unique()))
    di = {d: i for i, d in enumerate(dates)}
    for mdl in models:
        raw = np.zeros((len(dates), 4)); cal = np.zeros((len(dates), 4))
        for s, g in df.groupby("state"):
            g = g[g[mdl].notna()]; t = THR[s]
            A = g[g.date < CAL_END]; B = g[g.date >= CAL_END]
            r = float(np.mean(A.target > t)); tau = float(np.quantile(A[mdl], 1 - r)) if 0 < r < 1 else t
            o = (B.target > t).values; idx = B.date.map(di).values
            for arr, thr_ in ((raw, t), (cal, tau)):
                p_ = (B[mdl] > thr_).values
                for k, v in enumerate([o & p_, o & ~p_, ~o & p_, ~o & ~p_]): np.add.at(arr[:, k], idx, v.astype(float))
        res[mdl] = dict(raw=raw, calibrated=cal)
    return res, len(dates)


rows = []
SETS = [("benchmark", lambda h: pn.benchmark(h), ["persistence", "ridge", "fourier_ridge", "gam", "hgb", "sarimax", "prophet"]),
        ("operational", lambda h: pn.operational(h), ["ops_ridge", "opsf_ridge", "opsfw_ridge", "opsfw_gam", "opsfw_hgb"])]
for setname, loader, models in SETS:
    for h in (1, 7, 14):
        df = loader(h); C, n = contingency(df, models)
        rng = np.random.default_rng(42); W = [block_weights(n, rng, 28) for _ in range(NB)]
        for mdl in models:
            for rule in ("raw", "calibrated"):
                arr = C[mdl][rule]; pt = scores(arr.sum(0))
                bs = pd.DataFrame([scores((w[:, None] * arr).sum(0)) for w in W])
                row = dict(set=setname, horizon=h, model=mdl, rule=rule, n_days=n, **pt)
                for k in pt: row[k + "_lo"] = float(np.nanpercentile(bs[k], 2.5)); row[k + "_hi"] = float(np.nanpercentile(bs[k], 97.5))
                # paired difference calibrated - raw of PSS
                rows.append(row)
            d = np.array([scores((w[:, None] * C[mdl]["calibrated"]).sum(0))["pss"] - scores((w[:, None] * C[mdl]["raw"]).sum(0))["pss"] for w in W])
            rows[-1]["pss_gain_lo"] = float(np.percentile(d, 2.5)); rows[-1]["pss_gain_hi"] = float(np.percentile(d, 97.5))
        print(setname, h, "done", flush=True)
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "warnings_identical_origins.csv"), index=False)
show = ["set", "horizon", "model", "rule", "event_freq", "warn_freq", "pod", "precision", "false_alarm_rate", "pss", "pss_lo", "pss_hi", "value_0.1", "value_0.3"]
print(T[show].round(3).to_string())
