"""Benchmark analyses. Usage: python analysis_benchmark.py SECTION [SECTION ...]   (sections: core, mechanism, ablation, variants,
classical, fnl, panels, cost). Results accumulate in results/benchmark/results.json; tables in results/benchmark/*.csv."""
import os
for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[_k] = "1"
import sys, glob, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
sys.path.insert(0, os.path.dirname(__file__))
from jboot import Panel, boot, ci

from paths import WORK as W, WORK as V, STATE_DAILY as DATA, DERIVED, EXTERNAL, ensure
OUT = ensure(os.path.join(os.path.dirname(DATA), "..", "results", "benchmark"))
H = [1, 2, 3, 5, 7, 10, 14]
STATES = sorted(f[:-4] for f in os.listdir(DATA))
RJ = f"{OUT}/results.json"
R = json.load(open(RJ)) if os.path.exists(RJ) else {}
NB = 2000


def save():
    json.dump(R, open(RJ, "w"), indent=1, default=float)


def nw_var(d, L):
    d = np.asarray(d, float); n = len(d); c = d - d.mean(); v = np.sum(c * c) / n
    for l in range(1, L + 1): v += 2 * (1 - l / (L + 1)) * np.sum(c[l:] * c[:-l]) / n
    return v / n


def dm(ea, eb, h):
    d = ea ** 2 - eb ** 2; n = len(d); L = max(h - 1, max(1, int(np.floor(4 * (n / 100) ** (2 / 9)))))
    v = nw_var(d, L)
    if not np.isfinite(v) or v <= 0: return 0.0, 1.0
    s = d.mean() / np.sqrt(v) * np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 0))
    return s, 2 * (1 - stats.t.cdf(abs(s), df=n - 1))


def holm(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p); a = np.empty(m); run = 0
    for i, k in enumerate(o): run = max(run, min((m - i) * p[k], 1)); a[k] = run
    return a


def load_base(h):
    """baseline predictions (benchmark engine) + Fourier ridge, merged on date."""
    out = []
    for s in STATES:
        b = pd.read_csv(f"{W}/pred_test/{s}_h{h}.csv")
        f = pd.read_csv(f"{V}/exp/fourier/{s}_h{h}.csv")[["date", "fourier_ridge"]]
        m = b.merge(f, on="date", how="left")
        fn = f"{V}/exp/fourier_nl/{s}_h{h}.csv"
        if os.path.exists(fn): m = m.merge(pd.read_csv(fn)[["date", "fourier_gam", "fourier_hgb"]], on="date", how="left")
        sb = f"{V}/sb/{s}.csv"
        if os.path.exists(sb):
            c = pd.read_csv(sb); c = c[c.horizon == h][["date", "snaive7", "snaive364", "fets", "sarimax"]]
            m = m.merge(c, on="date", how="left")
        out.append(m)
    return out


def emissions_weights():
    w = {}
    for s in STATES:
        b = pd.read_csv(f"{W}/pred_test/{s}_h1.csv"); w[s] = b.target.mean()
    return pd.Series(w)


# =============================================================== CORE
def core():
    EW = emissions_weights(); R["emission_weights_share_top5"] = float(EW.sort_values().iloc[-5:].sum() / EW.sum())
    models = ["persistence", "ridge", "fourier_ridge", "gam", "hgb"]
    tab, pair_rows, dmrows, yr_rows, nat_rows = {}, [], [], [], []
    PAIRS = [("hgb", "ridge"), ("gam", "ridge"), ("gam", "hgb"), ("fourier_ridge", "ridge"), ("fourier_ridge", "gam"), ("fourier_ridge", "hgb")]
    panels = {}
    for h in H:
        fr = load_base(h); P = Panel(fr, models); panels[h] = P
        r2 = {m: P.r2(m) for m in models}
        tab[h] = {m: float(np.nanmean(r2[m])) for m in models}
        tab[h].update({f"ew_{m}": float(np.average(r2[m], weights=EW[P.states].values)) for m in models})
        # national aggregate forecast: dates on which every state has a forecast
        full = P.mask.all(0)
        Y = np.nansum(np.where(P.mask, P.y, 0), 0)[full]
        for m in models:
            F = np.nansum(np.where(P.mask, P.p[m], 0), 0)[full]
            nat_rows.append(dict(horizon=h, model=m, n_days=int(full.sum()), R2=float(r2_score(Y, F)),
                                 RMSE_kt=float(np.sqrt(np.mean((Y - F) ** 2)) / 1e3), MAE_kt=float(np.mean(np.abs(Y - F)) / 1e3),
                                 MAPE=float(100 * np.mean(np.abs(Y - F) / Y)), mean_kt=float(Y.mean() / 1e3)))
        # state-level RMSE in tons and as % of mean
        for m in models:
            rm = np.sqrt(np.nanmean(np.where(P.mask, (P.y - P.p[m]) ** 2, np.nan), 1)); mu = np.nanmean(np.where(P.mask, P.y, np.nan), 1)
            tab[h][f"rmse_pct_median_{m}"] = float(np.median(100 * rm / mu)); tab[h][f"rmse_t_sum_{m}"] = float(rm.sum())
        # joint-bootstrap CIs for pairwise national differences + state-bootstrap for comparison
        def stat(rr, a, b): return float(np.nanmean(rr["x"](a) - rr["x"](b)))
        bs = boot({"x": P}, lambda rr: np.array([stat(rr, a, b) for a, b in PAIRS]), n_boot=NB)
        for k, (a, b) in enumerate(PAIRS):
            d = r2[a] - r2[b]; lo, hi = ci(bs[:, k]); lo90, hi90 = ci(bs[:, k], .90)
            pair_rows.append(dict(horizon=h, a=a, b=b, mean=float(np.nanmean(d)), ci_lo=lo, ci_hi=hi, ci90_lo=lo90, ci90_hi=hi90,
                                  n_a_better=int((d > 0).sum()),
                                  ew_mean=float(np.average(d, weights=EW[P.states].values))))
        # per-state DM tests for GAM vs HGB and Fourier ridge vs GAM/HGB
        for i, s in enumerate(P.states):
            mk = P.mask[i]; y = P.y[i, mk]
            for a, b in [("gam", "hgb"), ("fourier_ridge", "gam"), ("fourier_ridge", "hgb")]:
                st, p = dm(y - P.p[b][i, mk], y - P.p[a][i, mk], h)   # positive: a has lower loss
                dmrows.append(dict(state=s, horizon=h, a=a, b=b, stat=st, p=p))
        # evaluation-year breakdown
        years = P.dates.year
        for yv in sorted(set(years)):
            w = (years == yv).astype(float)
            rr = {m: P.r2(m, w) for m in models}
            yr_rows.append(dict(horizon=h, year=int(yv), n_days=int(w.sum()), **{m: float(np.nanmean(rr[m])) for m in models},
                                hgb_gt_ridge=int((rr["hgb"] > rr["ridge"]).sum()), gam_gt_ridge=int((rr["gam"] > rr["ridge"]).sum()),
                                fr_ge_gam=int((rr["fourier_ridge"] >= rr["gam"]).sum())))
    PR = pd.DataFrame(pair_rows); PR.to_csv(f"{OUT}/pairs_joint_bootstrap.csv", index=False)
    DMr = pd.DataFrame(dmrows)
    for (a, b), g in DMr.groupby(["a", "b"]):
        DMr.loc[g.index, "holm"] = holm(g.p.values)
    DMr.to_csv(f"{OUT}/dm_extra.csv", index=False)
    pd.DataFrame(yr_rows).to_csv(f"{OUT}/by_year.csv", index=False)
    pd.DataFrame(nat_rows).to_csv(f"{OUT}/national_aggregate.csv", index=False)
    R["mean_r2"] = {str(h): v for h, v in tab.items()}
    dsum = {}
    for (a, b), g in DMr.groupby(["a", "b"]):
        dsum[f"{a}_vs_{b}"] = {str(h): dict(nominal=int((x.p < .05).sum()), holm_a=int(((x.holm < .05) & (x.stat > 0)).sum()),
                                            holm_b=int(((x.holm < .05) & (x.stat < 0)).sum())) for h, x in g.groupby("horizon")}
        dsum[f"{a}_vs_{b}"]["total_holm"] = int((g.holm < .05).sum())
    R["dm_extra"] = dsum
    # TOST equivalence of national GAM-HGB difference, margin 0.01 R2 (90% joint-bootstrap CI within +-margin)
    R["tost_gam_hgb"] = {str(r.horizon): dict(ci90=[r.ci90_lo, r.ci90_hi], equivalent=bool(r.ci90_lo > -0.01 and r.ci90_hi < 0.01))
                         for r in PR[(PR.a == "gam") & (PR.b == "hgb")].itertuples()}
    R["tost_fr_gam"] = {str(r.horizon): dict(ci90=[r.ci90_lo, r.ci90_hi], equivalent=bool(r.ci90_lo > -0.01 and r.ci90_hi < 0.01))
                        for r in PR[(PR.a == "fourier_ridge") & (PR.b == "gam")].itertuples()}
    # horizon-slope with joint bootstrap (common shocks across states and horizons)
    U = sorted(set().union(*[set(p.dates) for p in panels.values()])); U = pd.DatetimeIndex(U)
    maps = {h: U.get_indexer(panels[h].dates) for h in H}
    rng = np.random.default_rng(42)
    from jboot import block_weights

    def slope(dd):
        hh = np.repeat(H, len(STATES)); yy = np.concatenate([dd[h] for h in H]); ok = ~np.isnan(yy)
        return np.polyfit(hh[ok], yy[ok], 1)[0]
    pt = {h: panels[h].r2("hgb") - panels[h].r2("ridge") for h in H}; ptg = {h: panels[h].r2("gam") - panels[h].r2("ridge") for h in H}
    sl, slg = [], []
    for _ in range(NB):
        w = block_weights(len(U), rng)
        sl.append(slope({h: panels[h].r2("hgb", w[maps[h]]) - panels[h].r2("ridge", w[maps[h]]) for h in H}))
        slg.append(slope({h: panels[h].r2("gam", w[maps[h]]) - panels[h].r2("ridge", w[maps[h]]) for h in H}))
    R["slope_joint"] = dict(hgb_ridge=[slope(pt), *ci(np.array(sl))], gam_ridge=[slope(ptg), *ci(np.array(slg))],
                            hgb_ridge_p_le0=float(np.mean(np.array(sl) <= 0)))
    save()


# =============================================================== MECHANISM: seasonal predictability + characteristics
def seasonal_design(td):
    dt = pd.DatetimeIndex(td); doy, dow = dt.dayofyear.values, dt.dayofweek.values
    X = [np.ones(len(dt))]
    for k in (1, 2, 3): X += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    X += [(dow == j).astype(float) for j in range(1, 7)]
    return np.column_stack(X)


def mechanism():
    P0 = pd.read_csv(f"{W}/core/panel.csv")
    rows = []
    for s in STATES:
        y = pd.read_csv(f"{DATA}/{s}.csv", parse_dates=["date"]).set_index("date").co2_mass
        for h in H:
            b = pd.read_csv(f"{W}/pred_test/{s}_h{h}.csv", parse_dates=["date", "target_date"])
            fr = pd.read_csv(f"{V}/exp/fourier/{s}_h{h}.csv")
            # training-only seasonal climatology refitted on the same 14-day schedule (target dates <= refit date)
            preds = np.empty(len(b)); start = b.date.min()
            for T in pd.date_range(start, b.date.max(), freq="14D"):
                te = (b.date >= T) & (b.date < T + pd.Timedelta(days=14))
                if not te.any(): continue
                hist = y.loc[:T].dropna(); Xs = seasonal_design(hist.index)
                beta = np.linalg.solve(Xs.T @ Xs, Xs.T @ hist.values)
                preds[te.values] = seasonal_design(b.target_date[te]) @ beta
            rows.append(dict(state=s, horizon=h, seasonal_r2=r2_score(b.target, preds), fr=r2_score(fr.target, fr.fourier_ridge)))
    M = pd.DataFrame(rows).merge(P0[["state", "horizon", "persist", "ridge", "gam", "hgb", "region"]], on=["state", "horizon"])
    M["d_hr"] = M.hgb - M.ridge; M["d_gr"] = M.gam - M.ridge; M["d_fr"] = M.fr - M.ridge; M["d_gf"] = M.gam - M.fr
    # state characteristics
    eia = pd.read_csv(f"{EXTERNAL}/eia_seds_2024_electric_power_co2.csv")
    NAME = json.load(open(f"{EXTERNAL}/state_names.json"))
    cov = pd.read_csv(f"{DERIVED}/coverage_by_state.csv").set_index("state")
    ch = []
    for s in STATES:
        d = pd.read_csv(f"{DATA}/{s}.csv", parse_dates=["date"]).set_index("date")
        e = eia[eia.State == NAME[s]].iloc[0]
        m30 = d.co2_mass.rolling(30, center=True, min_periods=20).mean()
        dow = d.co2_mass.groupby(d.index.dayofweek).mean()
        ch.append(dict(state=s, coal_share=e.Coal / e.Total, log_mean_co2=np.log(d.co2_mass.mean()),
                       heat_rate=(d.heat_input.sum() / d.gross_load_mwh.sum()), load_cv=d.gross_load_mwh.std() / d.gross_load_mwh.mean(),
                       seasonal_amp=m30.std() / d.co2_mass.mean(), weekly_amp=(dow.max() - dow.min()) / dow.mean(),
                       facilities=cov.loc[s, "facilities"], balanced_share=cov.loc[s, "bal_share"]))
    CH = pd.DataFrame(ch); M = M.merge(CH, on="state"); M.to_csv(f"{OUT}/mechanism_panel.csv", index=False)
    import statsmodels.formula.api as smf
    res = {}
    for dep in ["d_hr", "d_gr", "d_fr", "d_gf"]:
        for name, form in [("h_only", f"{dep} ~ horizon"), ("pers", f"{dep} ~ persist"),
                           ("pers_seas", f"{dep} ~ persist + seasonal_r2"),
                           ("pers_seas_hfe", f"{dep} ~ persist + seasonal_r2 + C(horizon)")]:
            f = smf.ols(form, data=M).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(M.state)[0]})
            res[f"{dep}|{name}"] = dict(params=f.params.to_dict(), pvalues=f.pvalues.to_dict(), r2=f.rsquared, n=int(f.nobs))
    # standardized cross-sectional regression at 14 days on state characteristics
    X14 = M[M.horizon == 14].copy()
    zcols = ["coal_share", "log_mean_co2", "heat_rate", "load_cv", "seasonal_amp", "weekly_amp", "balanced_share"]
    for c in zcols + ["persist", "seasonal_r2"]: X14[c + "_z"] = (X14[c] - X14[c].mean()) / X14[c].std()
    cs = {}
    for dep in ["d_gr", "d_hr", "d_fr", "seasonal_r2", "persist"]:
        rhs = " + ".join(c + "_z" for c in (zcols if dep in ("seasonal_r2", "persist") else zcols))
        f = smf.ols(f"{dep} ~ {rhs}", data=X14).fit(cov_type="HC3")
        cs[dep] = dict(params=f.params.to_dict(), pvalues=f.pvalues.to_dict(), r2=f.rsquared, n=int(f.nobs))
    cs["corr"] = {c: {dep: float(np.corrcoef(X14[c], X14[dep])[0, 1]) for dep in ["d_gr", "d_hr", "d_fr", "seasonal_r2", "persist"]} for c in zcols}
    R["mechanism"] = dict(panel_models=res, cross_section_14=cs,
                          seasonal_r2_by_h={str(h): float(M[M.horizon == h].seasonal_r2.mean()) for h in H},
                          corr_seas_dgr=float(np.corrcoef(M.seasonal_r2, M.d_gr)[0, 1]), corr_pers_seas=float(np.corrcoef(M.seasonal_r2, M.persist)[0, 1]))
    save()


# =============================================================== ABLATION across horizons (+ uncertainty, by-year, by-state)
def ablation():
    variants = ["comp_annual", "comp_doy", "comp_month", "comp_year", "comp_cal"]
    rows, ybs, dmrows = [], [], []
    for h in H:
        base = {s: pd.read_csv(f"{W}/pred_test/{s}_h{h}.csv") for s in STATES}
        for v in variants:
            fns = [f"{V}/exp/{v}/{s}_h{h}.csv" for s in STATES]
            if not all(os.path.exists(f) and os.path.getsize(f) > 0 for f in fns): continue
            frames = []
            for s, fn in zip(STATES, fns):
                a = pd.read_csv(fn).rename(columns={"ridge": "ridge_v", "gam": "gam_v", "hgb": "hgb_v"})
                frames.append(base[s].merge(a[["date", "ridge_v", "gam_v", "hgb_v"]], on="date"))
            ms = ["ridge", "gam", "hgb", "ridge_v", "gam_v", "hgb_v"]; P = Panel(frames, ms)
            r2 = {m: P.r2(m) for m in ms}
            bs = boot({"x": P}, lambda rr: np.array([np.nanmean(rr["x"](m) - rr["x"](m + "_v")) for m in ("ridge", "gam", "hgb")]
                                                      + [np.nanmean(rr["x"]("gam_v") - rr["x"]("ridge_v")), np.nanmean(rr["x"]("hgb_v") - rr["x"]("ridge_v"))]),
                      n_boot=NB)
            row = dict(horizon=h, variant=v)
            for k, m in enumerate(("ridge", "gam", "hgb")):
                loss = r2[m] - r2[m + "_v"]
                row.update({f"{m}_full": float(np.mean(r2[m])), f"{m}_var": float(np.mean(r2[m + "_v"])), f"{m}_loss": float(loss.mean()),
                            f"{m}_loss_ci": ci(bs[:, k]), f"{m}_worse": int((loss > 0).sum()), f"{m}_loss_median": float(np.median(loss))})
            row.update(gam_minus_ridge_var=float(np.mean(r2["gam_v"] - r2["ridge_v"])), gam_minus_ridge_var_ci=ci(bs[:, 3]),
                       hgb_minus_ridge_var=float(np.mean(r2["hgb_v"] - r2["ridge_v"])), hgb_minus_ridge_var_ci=ci(bs[:, 4]),
                       best_var=max(("ridge", "gam", "hgb"), key=lambda m: np.mean(r2[m + "_v"])))
            rows.append(row)
            # per-state DM (full vs variant) and per-state-year losses (annual pair only)
            for i, s in enumerate(P.states):
                mk = P.mask[i]; y = P.y[i, mk]
                for m in ("ridge", "gam", "hgb"):
                    st, p = dm(y - P.p[m + "_v"][i, mk], y - P.p[m][i, mk], h)   # positive: full model has lower loss
                    dmrows.append(dict(state=s, horizon=h, variant=v, model=m, stat=st, p=p, loss=float(r2[m][i] - r2[m + "_v"][i]),
                                       r2_full=float(r2[m][i]), r2_var=float(r2[m + "_v"][i])))
                if v == "comp_annual":
                    years = P.dates.year
                    for yv in sorted(set(years)):
                        w = (years == yv).astype(float)
                        for m in ("ridge", "gam", "hgb"):
                            ybs.append(dict(state=s, horizon=h, year=int(yv), model=m,
                                            loss=float(P.r2(m, w)[i] - P.r2(m + "_v", w)[i])))
    A = pd.DataFrame(rows); A.to_csv(f"{OUT}/ablation_components.csv", index=False)
    DMa = pd.DataFrame(dmrows)
    if len(DMa):
        for key, g in DMa.groupby(["variant", "model"]): DMa.loc[g.index, "holm"] = holm(g.p.values)
        DMa.to_csv(f"{OUT}/ablation_dm.csv", index=False)
    YB = pd.DataFrame(ybs)
    if len(YB): YB.to_csv(f"{OUT}/ablation_annual_by_state_year.csv", index=False)
    R["ablation_components"] = A.to_dict(orient="records")
    if len(DMa):
        R["ablation_dm"] = {f"{v}|{m}|{h}": dict(holm_full_better=int(((g.holm < .05) & (g.stat > 0)).sum()),
                                                 holm_var_better=int(((g.holm < .05) & (g.stat < 0)).sum()))
                            for (v, m, h), g in DMa.groupby(["variant", "model", "horizon"])}
    if len(YB):
        R["ablation_by_year"] = {f"{m}|{h}|{y}": dict(mean=float(g.loss.mean()), worse=int((g.loss > 0).sum()), n=int(len(g)))
                                 for (m, h, y), g in YB.groupby(["model", "horizon", "year"])}
    save()


# =============================================================== VARIANTS: target-date calendar, real-time inputs, seasonal residuals
def variants():
    out = {}
    for v in ["target_cal", "realtime", "resid"]:
        rows = []
        for h in H:
            fns = [f"{V}/exp/{v}/{s}_h{h}.csv" for s in STATES]
            if not all(os.path.exists(f) and os.path.getsize(f) > 0 for f in fns): continue
            frames = []
            for s, fn in zip(STATES, fns):
                b = pd.read_csv(f"{W}/pred_test/{s}_h{h}.csv")
                a = pd.read_csv(fn).rename(columns={"ridge": "ridge_v", "gam": "gam_v", "hgb": "hgb_v"})
                keep = ["date", "ridge_v", "gam_v", "hgb_v"] + (["seasonal"] if v == "resid" else [])
                frames.append(b.merge(a[keep], on="date"))
            ms = ["ridge", "gam", "hgb", "ridge_v", "gam_v", "hgb_v"] + (["seasonal"] if v == "resid" else [])
            P = Panel(frames, ms); r2 = {m: P.r2(m) for m in ms}
            row = dict(horizon=h, n=len(frames))
            for m in ("ridge", "gam", "hgb"):
                row[f"{m}_full"] = float(np.mean(r2[m])); row[f"{m}_var"] = float(np.mean(r2[m + "_v"]))
            if v == "resid":
                # skill beyond the training-estimated seasonal component: 1 - SSE(model)/SSE(seasonal)
                for m in ("ridge_v", "gam_v", "hgb_v"):
                    sse_m = np.nansum(np.where(P.mask, (P.y - P.p[m]) ** 2, np.nan), 1)
                    sse_s = np.nansum(np.where(P.mask, (P.y - P.p["seasonal"]) ** 2, np.nan), 1)
                    row[f"{m[:-2]}_resid_skill"] = float(np.mean(1 - sse_m / sse_s))
                row["seasonal_r2"] = float(np.mean(r2["seasonal"]))
                bs = boot({"x": P}, lambda rr: np.array([np.nanmean(rr["x"]("gam_v") - rr["x"]("ridge_v")),
                                                         np.nanmean(rr["x"]("hgb_v") - rr["x"]("ridge_v"))]), n_boot=NB)
                row["gam_minus_ridge_resid"] = float(np.mean(r2["gam_v"] - r2["ridge_v"])); row["gam_minus_ridge_resid_ci"] = ci(bs[:, 0])
                row["hgb_minus_ridge_resid"] = float(np.mean(r2["hgb_v"] - r2["ridge_v"])); row["hgb_minus_ridge_resid_ci"] = ci(bs[:, 1])
            else:
                bs = boot({"x": P}, lambda rr: np.array([np.nanmean(rr["x"](m + "_v") - rr["x"](m)) for m in ("ridge", "gam", "hgb")]), n_boot=NB)
                for k, m in enumerate(("ridge", "gam", "hgb")): row[f"{m}_change_ci"] = ci(bs[:, k])
            row["best_var"] = max(("ridge", "gam", "hgb"), key=lambda m: np.mean(r2[m + "_v"]))
            rows.append(row)
        out[v] = rows
        pd.DataFrame(rows).to_csv(f"{OUT}/variant_{v}.csv", index=False)
    R["variants"] = out; save()


# =============================================================== CLASSICAL seasonal baselines
def classical():
    rows = []
    for h in H:
        fr = load_base(h)
        if not all("sarimax" in f for f in fr): return
        ms = ["persistence", "snaive7", "snaive364", "fets", "sarimax", "ridge", "fourier_ridge", "gam", "hgb"]
        P = Panel(fr, ms); r2 = {m: P.r2(m) for m in ms}
        bs = boot({"x": P}, lambda rr: np.array([np.nanmean(rr["x"]("fourier_ridge") - rr["x"]("sarimax")),
                                                 np.nanmean(rr["x"]("gam") - rr["x"]("sarimax"))]), n_boot=NB)
        row = dict(horizon=h, **{m: float(np.mean(r2[m])) for m in ms})
        row["fr_minus_sarimax_ci"] = ci(bs[:, 0]); row["gam_minus_sarimax_ci"] = ci(bs[:, 1])
        row["sarimax_gt_ridge"] = int((r2["sarimax"] > r2["ridge"]).sum()); row["sarimax_gt_gam"] = int((r2["sarimax"] > r2["gam"]).sum())
        row["fr_gt_sarimax"] = int((r2["fourier_ridge"] > r2["sarimax"]).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{OUT}/classical.csv", index=False); R["classical"] = rows
    if all(os.path.exists(f"{V}/sb/{s}.json") for s in STATES):   # fit timings exist only after refitting (stage A)
        t = [json.load(open(f"{V}/sb/{s}.json")) for s in STATES]
        R["classical_timing"] = {m: float(np.mean([x[m]["fit_s_per_refit"] for x in t])) for m in ("fets", "sarimax")}
    save()


# =============================================================== FOURIER for GAM/HGB
def fnl():
    rows = []
    for h in H:
        fr = load_base(h)
        if not all("fourier_hgb" in f for f in fr): return
        ms = ["ridge", "fourier_ridge", "gam", "hgb", "fourier_gam", "fourier_hgb"]
        P = Panel(fr, ms); r2 = {m: P.r2(m) for m in ms}
        bs = boot({"x": P}, lambda rr: np.array([np.nanmean(rr["x"]("fourier_gam") - rr["x"]("fourier_ridge")),
                                                 np.nanmean(rr["x"]("fourier_hgb") - rr["x"]("fourier_ridge"))]), n_boot=NB)
        row = dict(horizon=h, **{m: float(np.mean(r2[m])) for m in ms})
        row["fgam_minus_fr_ci"] = ci(bs[:, 0]); row["fhgb_minus_fr_ci"] = ci(bs[:, 1])
        row["fgam_gt_fr"] = int((r2["fourier_gam"] > r2["fourier_ridge"]).sum()); row["fhgb_gt_fr"] = int((r2["fourier_hgb"] > r2["fourier_ridge"]).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{OUT}/fourier_nonlinear.csv", index=False); R["fourier_nonlinear"] = rows; save()


# =============================================================== PANELS
def panels():
    out = {}
    for pnl in ["balanced", "highcov"]:
        rows = []
        for h in (1, 7, 14):
            fns = sorted(glob.glob(f"{V}/exp_{pnl}/base/*_h{h}.csv"))
            if not fns: continue
            fr = [pd.read_csv(f) for f in fns]
            P = Panel(fr, ["ridge", "gam", "hgb"]); r2 = {m: P.r2(m) for m in ("ridge", "gam", "hgb")}
            full = pd.read_csv(f"{W}/core/panel.csv"); full = full[(full.horizon == h) & full.state.isin(P.states)]
            rows.append(dict(horizon=h, n_states=len(fr), **{m: float(np.mean(r2[m])) for m in ("ridge", "gam", "hgb")},
                             hgb_minus_ridge=float(np.mean(r2["hgb"] - r2["ridge"])), gam_minus_ridge=float(np.mean(r2["gam"] - r2["ridge"])),
                             full_hgb_minus_ridge=float((full.hgb - full.ridge).mean()), full_gam_minus_ridge=float((full.gam - full.ridge).mean()),
                             hgb_gt_ridge=int((r2["hgb"] > r2["ridge"]).sum()), gam_gt_ridge=int((r2["gam"] > r2["ridge"]).sum())))
        out[pnl] = rows
    R["panels"] = out; save()


# =============================================================== COMPUTE COST & COMPLEXITY
def cost():
    """Single-thread fit time per refit (ms), tuning candidates, and model size (median over state-horizons)."""
    if not glob.glob(f"{V}/sb/*.json"):
        print("cost: fit timings are recorded only when the models are refitted (stage A); released values kept"); return
    def tim(var, m):
        js = glob.glob(f"{V}/exp/{var}/*.json")
        v = [json.load(open(j))["timing"][m]["fit_s_per_refit"] for j in js if os.path.getsize(j) > 0 and m in json.load(open(j)).get("timing", {})]
        return 1000 * float(np.median(v)) if v else float("nan")
    sets = {}
    for f in glob.glob(f"{W}/pred_test/*_settings.json"):
        k = os.path.basename(f).replace("_settings.json", ""); sets[k] = json.load(open(f))
    def Kof(var, m):
        return [json.load(open(j))["best"][m][0] for j in glob.glob(f"{V}/exp/{var}/*.json") if os.path.getsize(j) > 0]
    Kfr = Kof("fourier", "fourier_ridge"); Kfg = Kof("fourier_nl", "fourier_gam") or [np.nan]; Kfh = Kof("fourier_nl", "fourier_hgb") or [np.nan]
    nf = lambda K: 27 + 2 * (K - 1) + 4
    hgb_leaves = [s["hgb"]["max_iter"] * min(31, 2 ** s["hgb"]["max_depth"]) for s in sets.values()]
    sb = [json.load(open(f)) for f in glob.glob(f"{V}/sb/*.json")]
    rows = [
        dict(model="Persistence", fit_ms=0.0, candidates=0, size="0 parameters"),
        dict(model="Seasonal naive", fit_ms=0.0, candidates=0, size="0 parameters"),
        dict(model="Ridge", fit_ms=tim("target_cal", "ridge"), candidates=12, size="28 coefficients"),
        dict(model="Fourier ridge", fit_ms=tim("fourier", "fourier_ridge"), candidates=72,
             size=f"{int(np.median([nf(k) for k in Kfr])) + 1} coefficients (median; {min(nf(k) for k in Kfr) + 1}-{max(nf(k) for k in Kfr) + 1})"),
        dict(model="GAM", fit_ms=tim("target_cal", "gam"), candidates=12, size="163 coefficients (6 spline bases × 27 + 1)"),
        dict(model="HGB", fit_ms=tim("target_cal", "hgb"), candidates=12,
             size=f"<= {int(np.median(hgb_leaves))} leaves (median; up to {max(hgb_leaves)})"),
        dict(model="Fourier GAM", fit_ms=tim("fourier_nl", "fourier_gam"), candidates=72,
             size=f"{int(np.nanmedian([6 * nf(k) + 1 for k in Kfg]))} coefficients (median)" if Kfg[0] == Kfg[0] else ""),
        dict(model="Fourier HGB", fit_ms=tim("fourier_nl", "fourier_hgb"), candidates=72, size="as HGB (median bound 400 leaves)"),
        dict(model="Fourier-ETS", fit_ms=1000 * float(np.median([x["fets"]["fit_s_per_refit"] for x in sb])) if sb else np.nan, candidates=1,
             size="11 parameters (7 regression, 4 smoothing) and initial states"),
        dict(model="SARIMAX", fit_ms=1000 * float(np.median([x["sarimax"]["fit_s_per_refit"] for x in sb])) if sb else np.nan, candidates=1,
             size="13 parameters (7 regression, 5 ARMA, 1 variance)"),
    ]
    R["cost_table"] = rows; R["fourier_K_counts"] = {str(k): int(v) for k, v in pd.Series(Kfr).value_counts().sort_index().items()}
    R["fourier_K_by_h"] = {str(h): float(np.median([json.load(open(j))["best"]["fourier_ridge"][0] for j in glob.glob(f"{V}/exp/fourier/*_h{h}.json")])) for h in H}
    save()


if __name__ == "__main__":
    for sec in sys.argv[1:]: globals()[sec]()
    print("ok", sys.argv[1:])
