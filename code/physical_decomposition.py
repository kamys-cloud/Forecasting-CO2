"""Physically informed decomposition of state CO2:  E(t) = sum_i H_i(t) EF_i  ->  activity (heat input) x intensity (EF),
fuel-class contributions (facility-level emission factors), degree-day (weather-driven demand) and generation-mix (EIA-923) links.
Outputs data/derived/*.csv"""
import glob, json, os
import numpy as np, pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import EXTERNAL, DERIVED as P, CAMPD, STATE_DAILY as SUP
STATES = sorted(f[:-4] for f in os.listdir(SUP))
cols = ["state", "name", "fid", "date", "load", "steam", "so2", "co2", "nox", "heat"]
F = pd.concat([pd.read_csv(f).set_axis(cols, axis=1) for f in sorted(glob.glob(os.path.join(CAMPD, "daily-emissions-facility-aggregation-*.csv")))])
F["date"] = pd.to_datetime(F.date); F = F[(F.date >= "2019-01-01") & (F.date <= "2025-12-31") & F.state.isin(STATES)]
F = F.drop_duplicates(["state", "fid", "date"])
# ---------------------------------------------------------------- facility emission factors and fuel classes
rep = F[F.co2.notna() & F.heat.notna() & (F.heat > 0)]
fac = rep.groupby(["state", "fid"]).agg(co2=("co2", "sum"), heat=("heat", "sum"))
fac["ef"] = fac.co2 / fac.heat                        # short tons CO2 per MMBtu
fac["cls"] = np.where(fac.ef >= 0.090, "coal", np.where(fac.ef < 0.066, "gas", "other"))
fac.reset_index().to_csv(f"{P}/facility_ef.csv", index=False)
print(fac.ef.describe(), fac.cls.value_counts(), (fac.groupby("cls").co2.sum() / fac.co2.sum()).round(3))
F = F.merge(fac.cls.reset_index(), on=["state", "fid"], how="left")
# ---------------------------------------------------------------- state-day activity / intensity / fuel components
g = F[F.co2.notna()].groupby(["state", "date"])
S = pd.DataFrame({"E": g.co2.sum(), "H": g.heat.sum(min_count=1), "L": g.load.sum(min_count=1)})
for c in ("coal", "gas", "other"):
    gc = F[F.co2.notna() & (F.cls == c)].groupby(["state", "date"])
    S[f"E_{c}"] = gc.co2.sum(); S[f"H_{c}"] = gc.heat.sum(min_count=1)
S = S.fillna({f"{p}_{c}": 0.0 for p in ("E", "H") for c in ("coal", "gas", "other")}).reset_index()
S.to_csv(f"{P}/state_daily_components.csv", index=False)
DD = pd.read_csv(f"{EXTERNAL}/cpc_degree_days.csv", parse_dates=["date"])


def clim(x, dates):
    """day-of-year climatology of a series expressed relative to its own annual mean (removes interannual level change), 15-day smoothed."""
    s = pd.Series(x.values, index=pd.DatetimeIndex(dates)).asfreq("D")
    rel = s / s.groupby(s.index.year).transform("mean")
    c = rel.groupby(rel.index.dayofyear).mean().reindex(range(1, 367)).interpolate()
    c = pd.concat([c.iloc[-15:], c, c.iloc[:15]]).rolling(15, center=True, min_periods=1).mean().iloc[15:-15]
    c.index = range(1, 367); return c


rows, prof = [], []
for s in STATES:
    x = S[S.state == s].set_index("date").sort_index()
    cE, cH = clim(x.E, x.index), clim(x.H, x.index)
    lE, lH = np.log(cE), np.log(cH); lEF = lE - lH
    vE = lE.var()
    # fuel-class contributions to the seasonal anomaly of E (relative to annual mean): anomaly_E = sum_k anomaly_Ek
    an = {}
    for c in ("coal", "gas", "other"):
        sk = pd.Series(x[f"E_{c}"].values, index=x.index).asfreq("D")
        rel = sk / pd.Series(x.E.values, index=x.index).asfreq("D").groupby(lambda t: t.year).transform("mean")
        ck = rel.groupby(rel.index.dayofyear).mean().reindex(range(1, 367)).interpolate()
        ck = pd.concat([ck.iloc[-15:], ck, ck.iloc[:15]]).rolling(15, center=True, min_periods=1).mean().iloc[15:-15]; ck.index = range(1, 367)
        an[c] = ck - ck.mean()
    aE = sum(an.values())
    share = {c: float(np.cov(an[c], aE)[0, 1] / aE.var()) for c in an}
    # degree days: climatology of HDD, CDD; linear fit of E climatology on HDD/CDD climatology
    d = DD[DD.state == s].set_index("date")
    ch = d.hdd.groupby(d.index.dayofyear).mean().reindex(range(1, 367)).interpolate()
    cc = d.cdd.groupby(d.index.dayofyear).mean().reindex(range(1, 367)).interpolate()
    smooth = lambda c: pd.Series(np.convolve(np.r_[c.values[-7:], c.values, c.values[:7]], np.ones(15) / 15, "valid"), index=c.index)
    ch, cc = smooth(ch), smooth(cc)
    X = np.column_stack([np.ones(366), ch, cc]); b = np.linalg.lstsq(X, cE.values, rcond=None)[0]; fitDD = X @ b
    r2_dd_clim = 1 - np.sum((cE - fitDD) ** 2) / np.sum((cE - cE.mean()) ** 2)
    doy = np.arange(1, 367)
    def four(K): return np.column_stack([np.ones(366)] + [f(2 * np.pi * k * doy / 365.25) for k in range(1, K + 1) for f in (np.sin, np.cos)])
    r2_f1 = 1 - np.sum((cE - four(1) @ np.linalg.lstsq(four(1), cE.values, rcond=None)[0]) ** 2) / np.sum((cE - cE.mean()) ** 2)
    r2_f3 = 1 - np.sum((cE - four(3) @ np.linalg.lstsq(four(3), cE.values, rcond=None)[0]) ** 2) / np.sum((cE - cE.mean()) ** 2)
    # daily (not climatological) degree-day regression with year and weekday effects, versus calendar Fourier K=3
    j = x[["E"]].join(d[["hdd", "cdd"]], how="inner").dropna()
    j["E"] = j.E / j.E.groupby(j.index.year).transform("mean")
    Y = j.E.values; dw = pd.get_dummies(j.index.dayofweek, drop_first=True).values.astype(float)
    yr = pd.get_dummies(j.index.year, drop_first=True).values.astype(float)
    def r2(Xm):
        bb = np.linalg.lstsq(Xm, Y, rcond=None)[0]; return 1 - np.sum((Y - Xm @ bb) ** 2) / np.sum((Y - Y.mean()) ** 2)
    jd = j.index.dayofyear.values
    fd = lambda K: np.column_stack([f(2 * np.pi * k * jd / 365.25) for k in range(1, K + 1) for f in (np.sin, np.cos)])
    base = np.column_stack([np.ones(len(j)), dw, yr])
    r2_daily_dd = r2(np.column_stack([base, j.hdd, j.cdd]))
    r2_daily_f1 = r2(np.column_stack([base, fd(1)])); r2_daily_f3 = r2(np.column_stack([base, fd(3)]))
    r2_daily_dd_f3 = r2(np.column_stack([base, j.hdd, j.cdd, fd(3)]))
    rows.append(dict(state=s, var_share_activity=float(lH.var() / vE), var_share_intensity=float(lEF.var() / vE),
                     var_share_cov=float(2 * np.cov(lH, lEF)[0, 1] / vE), amp_E=float(cE.max() - cE.min()), amp_H=float(cH.max() - cH.min()),
                     amp_EF=float((cE / cH).max() - (cE / cH).min()), share_coal=share["coal"], share_gas=share["gas"], share_other=share["other"],
                     coal_frac=float(x.E_coal.sum() / x.E.sum()), r2_clim_degree_days=float(r2_dd_clim), r2_clim_fourier1=float(r2_f1),
                     r2_clim_fourier3=float(r2_f3), corr_clim_E_hdd=float(np.corrcoef(cE, ch)[0, 1]), corr_clim_E_cdd=float(np.corrcoef(cE, cc)[0, 1]),
                     r2_daily_degree_days=float(r2_daily_dd), r2_daily_fourier1=float(r2_daily_f1), r2_daily_fourier3=float(r2_daily_f3),
                     r2_daily_dd_plus_fourier3=float(r2_daily_dd_f3), b_hdd=float(b[1]), b_cdd=float(b[2])))
    for dd_, vals in [("E", cE), ("H", cH), ("EF", cE / cH), ("coal", an["coal"]), ("gas", an["gas"]), ("other", an["other"]),
                      ("fit_degree_days", pd.Series(fitDD, index=range(1, 367))), ("hdd", ch), ("cdd", cc)]:
        for k, v in zip(range(1, 367), np.asarray(vals)): prof.append(dict(state=s, series=dd_, doy=k, value=float(v)))
T = pd.DataFrame(rows); T.to_csv(f"{P}/seasonal_decomposition.csv", index=False)
pd.DataFrame(prof).to_csv(f"{P}/seasonal_profiles_phys.csv", index=False)
# ---------------------------------------------------------------- generation mix (EIA-923 monthly, state level)
NAME = json.load(open(f"{EXTERNAL}/state_names.json"))
gm = []
for y in range(2019, 2026):
    g_ = pd.read_excel(f"{EXTERNAL}/raw/generation_monthly.xlsx", sheet_name=f"{y}_Final", header=4)
    g_.columns = ["year", "month", "state", "producer", "source", "gen"]; gm.append(g_)
G = pd.concat(gm); G = G[(G.producer == "Total Electric Power Industry") & G.state.isin(STATES)]
G["gen"] = pd.to_numeric(G.gen, errors="coerce")
Gp = G.pivot_table(index=["state", "year", "month"], columns="source", values="gen", aggfunc="sum").fillna(0).reset_index()
fos = [c for c in ["Coal", "Natural Gas", "Petroleum", "Other Gases"] if c in Gp.columns]
ren = [c for c in ["Hydroelectric Conventional", "Wind", "Solar Thermal and Photovoltaic", "Geothermal", "Wood and Wood Derived Fuels", "Other Biomass"] if c in Gp.columns]
Gp["fossil"] = Gp[fos].sum(1); Gp["nonfossil"] = Gp["Total"] - Gp["fossil"]; Gp["renew"] = Gp[ren].sum(1)
Gp["nuclear"] = Gp.get("Nuclear", 0.0)
Sm = S.assign(year=S.date.dt.year, month=S.date.dt.month).groupby(["state", "year", "month"]).agg(E=("E", "sum"), L=("L", "sum"), ndays=("E", "count")).reset_index()
M = Sm.merge(Gp[["state", "year", "month", "Total", "fossil", "nonfossil", "renew", "nuclear"]], on=["state", "year", "month"])
M.to_csv(f"{P}/monthly_generation_mix.csv", index=False)
mix = []
for s, m in M.groupby("state"):
    m = m.copy(); m["ann"] = m.groupby("year").E.transform("mean")
    for c in ("E", "L", "Total", "fossil", "nonfossil", "renew", "nuclear"):
        m[c + "_a"] = (m[c] - m.groupby("year")[c].transform("mean")) / m.groupby("year")["Total"].transform("mean")
    # fossil anomaly = demand (total) anomaly - non-fossil anomaly ; covariance shares
    vf = m.fossil_a.var()
    mix.append(dict(state=s, share_total_gen=float(np.cov(m.Total_a, m.fossil_a)[0, 1] / vf),
                    share_nonfossil=float(-np.cov(m.nonfossil_a, m.fossil_a)[0, 1] / vf),
                    corr_campd_load_eia_fossil=float(np.corrcoef(m.L, m.fossil)[0, 1]),
                    corr_campd_load_eia_fossil_changes=float(np.corrcoef(np.diff(m.L), np.diff(m.fossil))[0, 1]),
                    corr_campd_E_eia_fossil=float(np.corrcoef(m.E, m.fossil)[0, 1]),
                    ratio_campd_load_eia_fossil=float(m.L.sum() / m.fossil.sum()), months=int(len(m))))
pd.DataFrame(mix).to_csv(f"{P}/generation_mix_shares.csv", index=False)
print(T.describe().T[["mean", "50%", "min", "max"]].round(3)); print(pd.DataFrame(mix).describe().T[["mean", "50%", "min", "max"]].round(3))
