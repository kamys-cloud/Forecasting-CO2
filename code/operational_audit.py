"""Audit of the operational GEFS experiment.
(1) Calibration stability: MSE skill (vs the 2019 to mid-2022 day-of-year climatology) and bias of raw, fixed-calibrated (12 months
    before the first origin) and rolling-recalibrated GEFS degree-day forecasts by evaluation year and meteorological season. The
    rolling calibration refits a + b*forecast monthly on the 365 days of target dates ending the day before the issue day.
(2) Value of GEFS forecasts for CO2 by lead, season and state: Fourier ridge, GAM and HGB with and without GEFS degree days
    (operational information set): national mean state R2, and MSE skill 1 - MSE(with)/MSE(without) by season and by state.
(3) Ensemble spread: relation of squared CO2 forecast error to the GEFS 2-m temperature spread (rank correlation by state and lead,
    controlling for month), and a Gaussian predictive distribution centred on the Fourier ridge + GEFS forecast with sigma either
    constant or linear in spread (both fitted on the first evaluation year, 2022-07-17 to 2023-07-16, by state and lead), scored on
    later origins by CRPS, 90% interval coverage and the Brier score of P90 exceedance.
Usage: python operational_audit.py OUT_DIR"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
import panels as pn

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
ROOT = pn.ROOT; ORIG = pd.Timestamp("2022-07-17"); CAL_END = pd.Timestamp("2023-07-17")
SEAS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}

# ---------------------------------------------------------------- (1) calibration stability
G = pd.read_csv(os.path.join(ROOT, "data", "weather", "gefs_dd_calibrated.csv"), parse_dates=["date", "target_date"])
O = pd.read_csv(os.path.join(ROOT, "data", "external", "cpc_degree_days.csv"), parse_dates=["date"]).rename(
    columns={"date": "target_date", "hdd": "hdd_obs", "cdd": "cdd_obs"})
M = G.merge(O, on=["state", "target_date"], how="left").dropna(subset=["hdd_obs", "cdd_obs"])
hist = O[O.target_date < ORIG].copy(); hist["doy"] = hist.target_date.dt.dayofyear
cl = hist.groupby(["state", "doy"])[["hdd_obs", "cdd_obs"]].mean().rename(columns=lambda c: c.replace("obs", "cl")).reset_index()
M["doy"] = M.target_date.dt.dayofyear; M = M.merge(cl, on=["state", "doy"], how="left")
# rolling recalibration (monthly refits, trailing 365 days of verified targets before the issue month)
roll = []
for (s, h), g in M.groupby(["state", "h"]):
    g = g.sort_values("date").copy()
    for q in ("hdd", "cdd"):
        g[f"{q}_roll"] = np.nan
        for m0 in pd.date_range(ORIG.replace(day=1), "2025-12-01", freq="MS"):
            tr = g[(g.target_date < m0) & (g.target_date >= m0 - pd.Timedelta(days=365))]
            sel = (g.date >= m0) & (g.date < m0 + pd.offsets.MonthBegin(1))
            if len(tr) > 60 and tr[f"{q}_raw"].var() > 1e-9: b, a = np.polyfit(tr[f"{q}_raw"], tr[f"{q}_obs"], 1)
            else: a, b = 0.0, 1.0
            g.loc[sel, f"{q}_roll"] = np.clip(a + b * g.loc[sel, f"{q}_raw"], 0, None)
    roll.append(g)
M = pd.concat(roll); E = M[M.date >= ORIG].copy()
E["obs"] = E.hdd_obs + E.cdd_obs; E["clim"] = E.hdd_cl + E.cdd_cl
for k in ("raw", "fc", "roll"): E[k] = E[f"hdd_{k}"] + E[f"cdd_{k}"]
E["year"] = E.target_date.dt.year; E["season"] = E.target_date.dt.month.map(SEAS)
rows = []
for by in ("year", "season"):
    for (key, h), g in E.groupby([by, "h"]):
        per = g.groupby("state").apply(lambda x: pd.Series({**{f"skill_{k}": 1 - np.mean((x[k] - x.obs) ** 2) / np.mean((x.clim - x.obs) ** 2) for k in ("raw", "fc", "roll")},
                                                            **{f"bias_{k}": np.mean(x[k] - x.obs) for k in ("raw", "fc", "roll")}}), include_groups=False)
        rows.append(dict(by=by, key=key, h=int(h), **per.median().to_dict()))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "gefs_calibration_stability.csv"), index=False)
print(pd.DataFrame(rows)[lambda d: d.h.isin([1, 7, 14])].round(3).to_string())

# ---------------------------------------------------------------- (2) value of GEFS forecasts by lead, season, state
rs, rst = [], []
for h in pn.H:
    d = pn.operational(h); d["season"] = d.target_date.dt.month.map(SEAS)
    for m in ("ridge", "gam", "hgb"):
        a, b = f"opsf_{m}", f"opsfw_{m}"
        for seas, g in d.groupby("season"):
            per = g.groupby("state").apply(lambda x: 1 - np.mean((x.target - x[b]) ** 2) / np.mean((x.target - x[a]) ** 2), include_groups=False)
            rs.append(dict(h=h, model=m, season=seas, median_mse_skill=float(per.median()), n_states_improved=int((per > 0).sum())))
        per = d.groupby("state").apply(lambda x: pd.Series(dict(r2_without=1 - np.sum((x.target - x[a]) ** 2) / np.sum((x.target - x.target.mean()) ** 2),
                                                               r2_with=1 - np.sum((x.target - x[b]) ** 2) / np.sum((x.target - x.target.mean()) ** 2))), include_groups=False)
        per["h"] = h; per["model"] = m; rst.append(per.reset_index())
S = pd.DataFrame(rs); S.to_csv(os.path.join(OUT, "gefs_value_by_season.csv"), index=False)
ST = pd.concat(rst); ST["gain"] = ST.r2_with - ST.r2_without; ST.to_csv(os.path.join(OUT, "gefs_value_by_state.csv"), index=False)
print(S[S.model == "ridge"].pivot(index="season", columns="h", values="median_mse_skill").round(3))
print(ST[ST.model == "ridge"].groupby("h").gain.describe()[["min", "25%", "50%", "75%", "max"]].round(3), ST[ST.model == "ridge"].groupby("h").apply(lambda x: (x.gain > 0).sum()))

# ---------------------------------------------------------------- (3) ensemble spread
SP = pd.read_csv(os.path.join(ROOT, "data", "weather", "gefs_spread.csv.gz"), parse_dates=["date"]).rename(columns={"h": "horizon"})
THR = {}
for s in sorted(f[:-4] for f in os.listdir(os.path.join(ROOT, "data", "state_daily"))):
    y = pd.read_csv(os.path.join(ROOT, "data", "state_daily", f"{s}.csv"), parse_dates=["date"]).set_index("date").co2_mass
    THR[s] = float(y[y.index < ORIG].quantile(0.9))


def crps_norm(y, mu, sd):
    z = (y - mu) / sd; return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


rows = []
for h in pn.H:
    d = pn.operational(h)[["state", "date", "target_date", "target", "opsfw_ridge"]].merge(SP[SP.horizon == h][["state", "date", "tsd_f"]], on=["state", "date"], how="left")
    d = d.dropna(); d["e"] = d.target - d.opsfw_ridge; d["month"] = d.target_date.dt.month
    rho = []
    for s, g in d.groupby("state"):
        g = g.copy(); g["e2r"] = g.e ** 2 / g.groupby("month").e.transform(lambda x: np.mean(x ** 2))
        g["spr"] = g.tsd_f / g.groupby("month").tsd_f.transform("mean")
        rho.append(stats.spearmanr(g.spr, g.e2r).statistic)
    sc = {"const": [], "spread": []}; cov = {"const": [], "spread": []}; bs = {"const": [], "spread": []}
    for s, g in d.groupby("state"):
        A = g[g.date < CAL_END]; B = g[g.date >= CAL_END]
        # sigma models fitted on the calibration year: |e| * sqrt(pi/2) estimates sigma for a Gaussian
        s0 = np.sqrt(np.mean(A.e ** 2))
        X = np.column_stack([np.ones(len(A)), A.tsd_f]); b = np.linalg.lstsq(X, np.abs(A.e) * np.sqrt(np.pi / 2), rcond=None)[0]
        s1 = np.clip(b[0] + b[1] * B.tsd_f.values, 0.25 * s0, None)
        for k, sd in (("const", np.full(len(B), s0)), ("spread", s1)):
            sc[k].append(np.mean(crps_norm(B.target.values, B.opsfw_ridge.values, sd)) / np.mean(np.abs(B.target - B.target.mean())))
            cov[k].append(np.mean(np.abs(B.e.values) <= 1.645 * sd))
            p = 1 - stats.norm.cdf((THR[s] - B.opsfw_ridge.values) / sd); o = (B.target.values > THR[s]).astype(float)
            bs[k].append(np.mean((p - o) ** 2))
    rows.append(dict(h=h, median_spearman_spread_vs_sq_error=float(np.median(rho)), n_states_positive=int(np.sum(np.array(rho) > 0)),
                     crps_rel_const=float(np.median(sc["const"])), crps_rel_spread=float(np.median(sc["spread"])),
                     n_states_crps_better=int(np.sum(np.array(sc["spread"]) < np.array(sc["const"]))),
                     coverage90_const=float(np.median(cov["const"])), coverage90_spread=float(np.median(cov["spread"])),
                     brier_p90_const=float(np.mean(bs["const"])), brier_p90_spread=float(np.mean(bs["spread"])),
                     n_states_brier_better=int(np.sum(np.array(bs["spread"]) < np.array(bs["const"])))))
SPR = pd.DataFrame(rows); SPR.to_csv(os.path.join(OUT, "gefs_spread_reliability.csv"), index=False); print(SPR.round(4).to_string())
