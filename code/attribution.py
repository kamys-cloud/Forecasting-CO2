"""Physical attribution beyond climatological correlation.

Part A - reconstruction residual. State emissions are rebuilt from facility heat input with a time-invariant facility emission factor,
E^(t) = sum_i H_i(t) EF_i with EF_i = sum_t CO2_i(t) / sum_t H_i(t) over 2019-2025 (and, as a check, with year-specific EF_i,y). The
residual E - E^ measures within-facility changes in emission factor (fuel switching, unit mix inside a plant) that the decomposition
treats as constant; it is reported for daily levels, for the day-of-year climatology and for daily anomalies.

Part B - conditional, out-of-sample attribution. Same-day regressions of y(t) = log E(t) - log(mean E over the 365 days ending t-1)
(a de-levelled series that removes interannual shifts without using future data) are fitted on 2019-01-01 to 2022-07-16 and evaluated
on 2022-07-17 to 2025-12-31. Predictor sets: W weekday; F annual Fourier harmonics 1-3; D observed population-weighted HDD and CDD
(linear and squared); C controls for generation mix and persistent conditions (monthly non-fossil share of state generation, EIA-923;
y(t-1), y(t-7) and the mean of y over t-7..t-1). The incremental out-of-sample R2 of D is compared without and with F and C, and
tested with a Diebold-Mariano statistic on the squared-error differences (Newey-West, 7 lags; Holm across states). A BA-level
version adds same-day wind-and-solar, hydro-and-nuclear generation and net interchange (EIA-930) for ERCOT, CAISO, NYISO, ISO-NE.
Usage: python attribution.py CAMPD_DIR OUT_DIR"""
import os, sys, glob, json
import numpy as np, pandas as pd
from scipy import stats
import ba_data as bd

CAMPD, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
ROOT = bd.ROOT
STATES = sorted(f[:-4] for f in os.listdir(os.path.join(ROOT, "data", "state_daily")))
TRAIN_END, TEST_START = pd.Timestamp("2022-07-16"), pd.Timestamp("2022-07-17")


# =============================================================== Part A
def part_a():
    fr = []
    for fn in sorted(glob.glob(os.path.join(CAMPD, "daily-emissions-facility-aggregation-*.csv"))):
        fr.append(pd.read_csv(fn, usecols=["State", "Facility ID", "Date", "CO2 Mass (short tons)", "Heat Input (mmBtu)"]))
    F = pd.concat(fr).rename(columns={"State": "state", "Facility ID": "fac", "Date": "date", "CO2 Mass (short tons)": "co2",
                                      "Heat Input (mmBtu)": "heat"})
    F["date"] = pd.to_datetime(F.date)
    F = F[(F.date >= "2019-01-01") & (F.date <= "2025-12-31") & F.state.isin(STATES)].drop_duplicates(["state", "fac", "date"])
    F = F[F.co2.notna()]
    ok = F.heat.notna() & (F.heat > 0)
    ef = F[ok].groupby(["state", "fac"]).apply(lambda g: g.co2.sum() / g.heat.sum()).rename("ef")
    F["year"] = F.date.dt.year
    efy = F[ok].groupby(["state", "fac", "year"]).apply(lambda g: g.co2.sum() / g.heat.sum()).rename("efy")
    F = F.join(ef, on=["state", "fac"]).join(efy, on=["state", "fac", "year"])
    F["rec"] = np.where(ok, F.heat * F.ef, F.co2)            # facility-days without heat input keep their reported CO2
    F["recy"] = np.where(ok, F.heat * F.efy, F.co2)
    S = F.groupby(["state", "date"]).agg(E=("co2", "sum"), Ec=("rec", "sum"), Ey=("recy", "sum"), share_noheat=("heat", lambda h: 0))
    nh = F[~ok].groupby(["state", "date"]).co2.sum()
    S["noheat"] = nh.reindex(S.index).fillna(0) / S.E
    rows = []
    for s in STATES:
        x = S.loc[s]
        def r2(a, b): return 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)
        clim = lambda v: (v / v.groupby(v.index.year).transform("mean")).groupby(v.index.dayofyear).mean()
        cE, cC = clim(x.E), clim(x.Ec)
        anE = x.E / x.E.groupby(x.index.year).transform("mean") - cE.reindex(x.index.dayofyear).values
        anC = x.Ec / x.Ec.groupby(x.index.year).transform("mean") - cC.reindex(x.index.dayofyear).values
        rows.append(dict(state=s, resid_share_daily=1 - r2(x.E.values, x.Ec.values), resid_share_daily_yearEF=1 - r2(x.E.values, x.Ey.values),
                         mape_daily=float(100 * np.mean(np.abs(x.E - x.Ec) / x.E)), mape_daily_yearEF=float(100 * np.mean(np.abs(x.E - x.Ey) / x.E)),
                         resid_share_climatology=1 - r2(cE.values, cC.values), resid_share_anomaly=1 - r2(anE.values, anC.values),
                         amp_E=float(cE.max() - cE.min()), amp_rec=float(cC.max() - cC.min()), share_co2_without_heat=float(x.noheat.mean())))
    A = pd.DataFrame(rows); A.to_csv(os.path.join(OUT, "reconstruction_residual.csv"), index=False)
    return A


# =============================================================== Part B helpers
def ols_oos(Xtr, ytr, Xte, yte):
    Xtr1 = np.column_stack([np.ones(len(Xtr)), Xtr]); Xte1 = np.column_stack([np.ones(len(Xte)), Xte])
    b = np.linalg.lstsq(Xtr1, ytr, rcond=None)[0]; e = yte - Xte1 @ b
    return 1 - np.sum(e ** 2) / np.sum((yte - yte.mean()) ** 2), e, b


def dm(e1, e2, lags=7):
    """one-sided DM (H1: model 2 smaller squared error), Newey-West variance"""
    d = e1 ** 2 - e2 ** 2; n = len(d); dc = d - d.mean()
    v = np.sum(dc ** 2) / n
    for k in range(1, lags + 1): v += 2 * (1 - k / (lags + 1)) * np.sum(dc[k:] * dc[:-k]) / n
    t = d.mean() / np.sqrt(v / n); return float(t), float(1 - stats.norm.cdf(t))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


def design(y, idx, dd, extra=None):
    t = idx; doy = t.dayofyear.values
    W = pd.get_dummies(t.dayofweek).values[:, 1:].astype(float)
    Fo = np.column_stack([f(2 * np.pi * k * doy / 365.25) for k in (1, 2, 3) for f in (np.sin, np.cos)])
    h = dd.hdd.reindex(t).values; c = dd.cdd.reindex(t).values
    D = np.column_stack([h, c, h ** 2, c ** 2])
    yl = y.reindex(t - pd.Timedelta(days=1)).values; y7 = y.reindex(t - pd.Timedelta(days=7)).values
    ym = y.shift(1).rolling(7).mean().reindex(t).values
    C = np.column_stack([yl, y7, ym] + ([extra] if extra is not None else []))
    return dict(W=W, F=Fo, D=D, C=C)


SETS = {"W": ["W"], "W+D": ["W", "D"], "W+F": ["W", "F"], "W+F+D": ["W", "F", "D"], "W+F+C": ["W", "F", "C"], "W+F+C+D": ["W", "F", "C", "D"]}


def fit_sets(y, blocks, idx):
    ok = np.all(np.isfinite(np.column_stack([y.reindex(idx).values] + [blocks[k] for k in blocks])), axis=1)
    tr = ok & (idx <= TRAIN_END); te = ok & (idx >= TEST_START)
    yv = y.reindex(idx).values; res = {}; errs = {}
    for name, parts in SETS.items():
        X = np.column_stack([blocks[p] for p in parts])
        r2, e, _ = ols_oos(X[tr], yv[tr], X[te], yv[te]); res[name] = r2; errs[name] = e
    return res, errs, int(te.sum())


def part_b():
    DD = pd.read_csv(os.path.join(ROOT, "data", "external", "cpc_degree_days.csv"), parse_dates=["date"])
    M = pd.read_csv(os.path.join(ROOT, "data", "derived", "monthly_generation_mix.csv"))
    rows = []
    for s in STATES:
        x = pd.read_csv(os.path.join(ROOT, "data", "state_daily", f"{s}.csv"), parse_dates=["date"]).set_index("date").co2_mass.asfreq("D")
        lvl = x.shift(1).rolling(365, min_periods=300).mean()
        y = np.log(x) - np.log(lvl)
        dd = DD[DD.state == s].set_index("date")[["hdd", "cdd"]].asfreq("D")
        m = M[M.state == s].set_index(["year", "month"])
        nf = (m.nonfossil / m.Total)
        idx = y.dropna().index
        nfs = nf.reindex(list(zip(idx.year, idx.month))).values
        bl = design(y, idx, dd, nfs)
        res, errs, n = fit_sets(y, bl, idx)
        t1, p1 = dm(errs["W"], errs["W+D"]); t2, p2 = dm(errs["W+F"], errs["W+F+D"]); t3, p3 = dm(errs["W+F+C"], errs["W+F+C+D"])
        rows.append(dict(state=s, n_test=n, **{f"r2_{k}": v for k, v in res.items()},
                         inc_D_alone=res["W+D"] - res["W"], inc_D_given_F=res["W+F+D"] - res["W+F"], inc_D_given_FC=res["W+F+C+D"] - res["W+F+C"],
                         inc_F_given_D=res["W+F+D"] - res["W+D"], p_D_alone=p1, p_D_given_F=p2, p_D_given_FC=p3))
    B = pd.DataFrame(rows)
    for c in ("p_D_alone", "p_D_given_F", "p_D_given_FC"): B[c + "_holm"] = holm(B[c].values)
    B.to_csv(os.path.join(OUT, "conditional_attribution_states.csv"), index=False)
    # ---------------- anomaly regression: CO2 anomaly vs degree-day anomaly (climatologies from the training period only)
    an = []
    for s in STATES:
        x = pd.read_csv(os.path.join(ROOT, "data", "state_daily", f"{s}.csv"), parse_dates=["date"]).set_index("date").co2_mass.asfreq("D")
        y = (np.log(x) - np.log(x.shift(1).rolling(365, min_periods=300).mean())).dropna()
        dd = DD[DD.state == s].set_index("date")[["hdd", "cdd"]].asfreq("D")
        tr = y.index <= TRAIN_END
        cy = y[tr].groupby(y[tr].index.dayofyear).mean(); cy = smooth(cy)
        ya = y - cy.reindex(y.index.dayofyear).values
        cdd_ = dd[dd.index <= TRAIN_END]; ch = smooth(cdd_.hdd.groupby(cdd_.index.dayofyear).mean()); cc = smooth(cdd_.cdd.groupby(cdd_.index.dayofyear).mean())
        ha = dd.hdd.reindex(y.index).values - ch.reindex(y.index.dayofyear).values
        ca = dd.cdd.reindex(y.index).values - cc.reindex(y.index.dayofyear).values
        X = np.column_stack([ha, ca]); ok = np.isfinite(X).all(1) & np.isfinite(ya.values)
        trm = ok & tr; tem = ok & (y.index >= TEST_START)
        r2, e, b = ols_oos(X[trm], ya.values[trm], X[tem], ya.values[tem])
        yl = ya.shift(1).values; X2 = np.column_stack([yl]); ok2 = tem & np.isfinite(yl)
        r2p, _, _ = ols_oos(X2[trm & np.isfinite(yl)], ya.values[trm & np.isfinite(yl)], X2[ok2], ya.values[ok2])
        X3 = np.column_stack([yl, ha, ca]); r2pd, _, _ = ols_oos(X3[trm & np.isfinite(yl)], ya.values[trm & np.isfinite(yl)], X3[ok2], ya.values[ok2])
        an.append(dict(state=s, r2_anom_dd=r2, b_hdd=b[1], b_cdd=b[2], r2_anom_persistence=r2p, r2_anom_persistence_dd=r2pd))
    pd.DataFrame(an).to_csv(os.path.join(OUT, "anomaly_attribution_states.csv"), index=False)
    return B, pd.DataFrame(an)


def smooth(c):
    c = c.reindex(range(1, 367)).interpolate().bfill().ffill()
    v = np.convolve(np.r_[c.values[-7:], c.values, c.values[:7]], np.ones(15) / 15, "valid")
    return pd.Series(v, index=range(1, 367))


def part_c():
    """BA level: CAMPD crosswalk CO2 with same-day renewable, hydro-nuclear generation and net interchange."""
    rows = []
    for ba in bd.BAS:
        x = bd.ba_co2(ba, "crosswalk"); x = x.where(x > 0)
        y = np.log(x) - np.log(x.shift(1).rolling(365, min_periods=300).mean())
        b = bd.eia930(ba); dd = bd.degree_days(ba)
        lvl = b.netgen.shift(1).rolling(365, min_periods=300).mean()
        vre = (b.wind + b.solar) / lvl; hn = (b.hydro + b.nuclear) / lvl; imp = (b.demand - b.netgen) / lvl
        idx = y.dropna().index
        extra = np.column_stack([vre.reindex(idx).values, hn.reindex(idx).values, imp.reindex(idx).values,
                                 (lambda m: m)(pd.Series(0.0, index=idx).values)])[:, :3]
        bl = design(y, idx, dd, None); bl["C"] = np.column_stack([bl["C"], extra])
        res, errs, n = fit_sets(y, bl, idx)
        _, p2 = dm(errs["W+F"], errs["W+F+D"]); _, p3 = dm(errs["W+F+C"], errs["W+F+C+D"])
        rows.append(dict(ba=ba, n_test=n, **{f"r2_{k}": v for k, v in res.items()}, inc_D_alone=res["W+D"] - res["W"],
                         inc_D_given_F=res["W+F+D"] - res["W+F"], inc_D_given_FC=res["W+F+C+D"] - res["W+F+C"], p_D_given_F=p2, p_D_given_FC=p3))
    C = pd.DataFrame(rows); C.to_csv(os.path.join(OUT, "conditional_attribution_ba.csv"), index=False)
    return C


if __name__ == "__main__":
    B, AN = part_b(); C = part_c()
    print(B.describe().T[["50%", "25%", "75%", "min", "max"]].round(4).to_string())
    print((B[["p_D_alone_holm", "p_D_given_F_holm", "p_D_given_FC_holm"]] < 0.05).sum())
    print(AN.describe().T[["50%", "25%", "75%"]].round(4)); print(C.round(4).to_string())
    A = part_a(); print(A.describe().T[["50%", "25%", "75%", "min", "max"]].round(4).to_string())
