"""Carbon-aware scheduling in ERCOT, CAISO, NYISO and ISO-NE: three separate outcomes.

(1) Forecast accuracy of daily average carbon intensity (R2 by lead).
(2) Ranking of low-intensity days: a deferrable one-day block is placed, at the end of each issue day t, on the day of t+1..t+7 with
    the lowest forecast average intensity; scored by the share of the perfect-foresight reduction in average intensity captured
    (relative to the mean of the seven candidate days), by the within-week Spearman rank correlation between forecast and realized
    intensity, and by the share of weeks in which the truly lowest day is chosen.
(3) Modelled avoided emissions: the same decisions scored with marginal emission factors (MEF) estimated from day-to-day changes in
    CAMPD CO2 of the BA's facilities and in EIA-930 demand, by season x net-load tercile (Siler-Evans-type regression).
Intensity definitions: crosswalk (primary: CAMPD CO2 of facilities assigned to the BA in EIA-860 / EIA-930 net generation), proxy
(CAMPD CO2 of the mapped states, as in the state-proxy design) and eia (EIA estimate of CO2 of generated electricity). Decisions made
with crosswalk forecasts are also scored on the EIA intensity.
(4) A deferrable electrolytic hydrogen process (parameters in PROCESS) scheduled once per week under minimum-run, campaign and
    deadline constraints; realized average- and marginal-basis emissions and start-up costs relative to a fixed schedule.
Usage: python scheduling_eval.py OUT_DIR"""
import os, sys, json, itertools
import numpy as np, pandas as pd
import ba_data as bd

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
ROOT = bd.ROOT; HS = list(range(1, 8)); ORIG = pd.Timestamp("2022-07-17"); LAST = pd.Timestamp("2025-12-31")
NB = 2000; rng = np.random.default_rng(42)
LAB = {("orig", "ridge"): "Ridge", ("orig", "gam"): "GAM", ("orig", "hgb"): "HGB", ("four", "ridge"): "Fourier ridge",
       ("four", "gam"): "Fourier GAM", ("four", "hgb"): "Fourier HGB", ("four_gefs", "ridge"): "Fourier ridge + GEFS",
       ("four_gefs", "gam"): "Fourier GAM + GEFS", ("four_gefs", "hgb"): "Fourier HGB + GEFS"}
PROCESS = dict(power_mw=20.0, kwh_per_kg_h2=55.0, run_days_per_week=4, min_campaign_days=2, max_campaigns=2,
               startup_energy_mwh=20.0, startup_cost_usd=3000.0)


def load_fc(kind):
    f = pd.read_csv(os.path.join(ROOT, "forecasts", f"scheduling_intensity_{kind}.csv.gz"), parse_dates=["date", "target_date"])
    return {(ba, LAB[(v, m)]): {h: g[g.horizon == h].set_index("date").forecast for h in HS}
            for (ba, v, m), g in f.groupby(["balancing_authority", "variant", "model"])}


def proxy_fc():
    """forecasts trained on the state-proxy intensity, for decision transfer to the facility-mapped intensity"""
    return load_fc("proxy")


def decide(F, origins):
    """index of chosen lead (0..6) per origin; NaN if a forecast is missing"""
    M = np.column_stack([F[h].reindex(origins).values for h in HS]); ok = ~np.isnan(M).any(1)
    ch = np.full(len(origins), -1); ch[ok] = np.argmin(M[ok], 1); return ch, M


def window(I, origins):
    return np.column_stack([I.reindex(origins + pd.Timedelta(days=h)).values for h in HS])


def blocks_ci(num, den, block=28):
    n = len(num); nb = int(np.ceil(n / block)); res = []
    for _ in range(NB):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, n - block + 1, nb)])[:n]
        res.append(num[idx].sum() / den[idx].sum())
    return [float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))]


# ---------------------------------------------------------------- marginal emission factors
def season(t): return np.select([t.month.isin([12, 1, 2]), t.month.isin([3, 4, 5]), t.month.isin([6, 7, 8])], ["DJF", "MAM", "JJA"], "SON")


def mef(ba):
    E = bd.ba_co2(ba, "crosswalk"); b = bd.eia930(ba)
    nl = (b.demand - b.wind.fillna(0) - b.solar.fillna(0))
    d = pd.DataFrame(dict(E=E, D=b.demand, NL=nl)).loc["2019-01-01":"2025-12-31"].dropna()
    q = d.NL.quantile([1 / 3, 2 / 3]).values
    d["bin"] = pd.Series(season(d.index), index=d.index) + "-" + pd.cut(d.NL, [-np.inf, q[0], q[1], np.inf], labels=["L", "M", "H"]).astype(str)
    dd = d[["E", "D", "NL"]].diff().dropna(); dd["bin"] = d.bin.reindex(dd.index)
    dd = dd[(d.index.to_series().diff().dt.days.reindex(dd.index) == 1)]
    bins = sorted(dd.bin.unique()); X = np.column_stack([np.where(dd.bin == k, dd.D, 0.0) for k in bins]
                                                       + [pd.get_dummies(dd.index.dayofweek).values[:, 1:].astype(float), np.ones(len(dd))])
    beta = np.linalg.lstsq(X, dd.E.values, rcond=None)[0][:len(bins)]
    bs = []
    n = len(dd); L = 28
    for _ in range(500):
        idx = np.concatenate([np.arange(s, s + L) for s in rng.integers(0, n - L + 1, int(np.ceil(n / L)))])[:n]
        bs.append(np.linalg.lstsq(X[idx], dd.E.values[idx], rcond=None)[0][:len(bins)])
    bs = np.array(bs)
    tab = pd.DataFrame(dict(ba=ba, bin=bins, mef_kg_per_mwh=beta, lo=np.percentile(bs, 2.5, 0), hi=np.percentile(bs, 97.5, 0),
                            n=[int((dd.bin == k).sum()) for k in bins]))
    daily = d.bin.map(dict(zip(bins, beta))).asfreq("D")
    return tab, daily, float(np.linalg.lstsq(np.column_stack([dd.D, np.ones(n)]), dd.E.values, rcond=None)[0][0])


# ---------------------------------------------------------------- weekly process scheduling
def patterns(n=7, k=4, mincamp=2, maxcamp=2):
    out = []
    for days in itertools.combinations(range(n), k):
        x = np.zeros(n, int); x[list(days)] = 1
        runs = [len(list(g)) for v, g in itertools.groupby(x) if v == 1]
        if len(runs) <= maxcamp and min(runs) >= mincamp: out.append(x)
    return np.array(out)


PAT = patterns(7, PROCESS["run_days_per_week"], PROCESS["min_campaign_days"], PROCESS["max_campaigns"])
STARTS = np.array([[1 if (x[i] == 1 and (i == 0 or x[i - 1] == 0)) else 0 for i in range(7)] for x in PAT])
E_DAY = PROCESS["power_mw"] * 24.0


def energy(pidx):
    return PAT[pidx] * E_DAY + STARTS[pidx] * PROCESS["startup_energy_mwh"]


def process_week(Ihat, Iact, Mact):
    """choose the pattern minimizing forecast average-basis emissions; return realized (avg kg, marginal kg, starts)"""
    en = PAT * E_DAY + STARTS * PROCESS["startup_energy_mwh"]
    j = int(np.argmin(en @ Ihat)); e = en[j]
    return float(e @ Iact), float(e @ Mact), int(STARTS[j].sum()), j


if __name__ == "__main__":
    FX, FE, FP = load_fc("crosswalk"), load_fc("eia"), proxy_fc()
    agree, cap_rows, acc_rows, mef_tabs, proc_rows, mef_summ, ORIGINS = [], [], [], [], [], [], []
    origins = pd.date_range(ORIG, LAST - pd.Timedelta(days=7))
    for ba in bd.BAS:
        I = {k: bd.intensity(ba, k) for k in ("crosswalk", "proxy", "eia")}
        tab, MEF, mef_all = mef(ba); mef_tabs.append(tab)
        # ---------------- (A) agreement of intensity definitions
        ev = slice(ORIG, LAST)
        A = pd.DataFrame({k: v.loc[ev] for k, v in I.items()}).dropna()
        clim = lambda s: s - s.groupby(s.index.dayofyear).transform("mean")
        Wx = window(I["crosswalk"], origins); Wp = window(I["proxy"], origins); We = window(I["eia"], origins)
        okw = ~np.isnan(Wx).any(1) & ~np.isnan(Wp).any(1) & ~np.isnan(We).any(1)
        ora_x = (Wx.mean(1) - Wx.min(1))
        for other, Wo in (("proxy", Wp), ("eia", We)):
            j = np.argmin(np.where(np.isnan(Wo), np.inf, Wo), 1); r = np.arange(len(origins))
            sav = Wx.mean(1) - Wx[r, j]
            sp = [pd.Series(Wx[i]).corr(pd.Series(Wo[i]), method="spearman") for i in np.where(okw)[0]]
            agree.append(dict(ba=ba, other=other, corr_level=float(A.crosswalk.corr(A[other])), corr_anomaly=float(clim(A.crosswalk).corr(clim(A[other]))),
                              mean_ratio_other_to_crosswalk=float((A[other] / A.crosswalk).mean()),
                              same_lowest_day=float(np.mean(np.argmin(Wx[okw], 1) == j[okw])), median_within_week_spearman=float(np.median(sp)),
                              transfer_capture=float(sav[okw].sum() / ora_x[okw].sum()), transfer_ci=blocks_ci(sav[okw], ora_x[okw]),
                              mean_crosswalk=float(A.crosswalk.mean()), mean_other=float(A[other].mean())))
        # ---------------- (B,C) decisions: accuracy, ranking, average and marginal savings
        Wm = window(MEF, origins)
        rules = {"Perfect foresight": {h: I["crosswalk"].shift(-h) for h in HS},
                 "Same weekday last week": {h: I["crosswalk"].shift(7 * int(np.ceil((h + 1) / 7)) - h) for h in HS},
                 "Next day": {h: pd.Series(float(h), index=I["crosswalk"].index) for h in HS}}
        for (b_, lab), F in FX.items():
            if b_ == ba: rules[lab] = F
        rules["Fourier ridge + GEFS (proxy-trained)"] = FP[(ba, "Fourier ridge + GEFS")]
        rules["Perfect foresight (marginal)"] = {h: MEF.shift(-h) for h in HS}
        ch = {k: decide(v, origins)[0] for k, v in rules.items()}
        valid = okw & ~np.isnan(Wm).any(1) & np.all(np.stack([c >= 0 for c in ch.values()]), 0)
        r = np.arange(len(origins))
        oraA = (Wx.mean(1) - Wx.min(1))[valid]; oraM = (Wm.mean(1) - Wm.min(1))[valid]; oraE = (We.mean(1) - We.min(1))[valid]
        ORIGINS.append(pd.DataFrame(dict(ba=ba, origin=origins[valid])))
        for lab, c in ch.items():
            savA = (Wx.mean(1) - Wx[r, c])[valid]; savM = (Wm.mean(1) - Wm[r, c])[valid]; savE = (We.mean(1) - We[r, c])[valid]
            F = rules[lab]
            M = np.column_stack([F[h].reindex(origins).values for h in HS])[valid]
            sp = np.nanmedian([pd.Series(M[i]).corr(pd.Series(Wx[valid][i]), method="spearman") for i in range(len(M))]) \
                if lab not in ("Next day",) else np.nan
            cap_rows.append(dict(ba=ba, rule=lab, n=int(valid.sum()), capture_avg=float(savA.sum() / oraA.sum()), capture_avg_ci=blocks_ci(savA, oraA),
                                 saving_avg_kg_mwh=float(savA.mean()), capture_eia=float(savE.sum() / oraE.sum()), capture_eia_ci=blocks_ci(savE, oraE),
                                 saving_marginal_kg_mwh=float(savM.mean()), saving_marginal_ci=[float(x) for x in np.percentile(
                                     [savM[np.concatenate([np.arange(s, s + 28) for s in rng.integers(0, len(savM) - 27, int(np.ceil(len(savM) / 28)))])[:len(savM)]].mean()
                                      for _ in range(NB)], [2.5, 97.5])],
                                 capture_marginal=float(savM.sum() / oraM.sum()), within_week_spearman=float(sp),
                                 hit_lowest=float(np.mean(c[valid] == np.argmin(Wx[valid], 1))),
                                 perfect_avg_kg_mwh=float(oraA.mean()), perfect_marginal_kg_mwh=float(oraM.mean()),
                                 mean_intensity=float(Wx[valid].mean()), mean_mef=float(Wm[valid].mean())))
        for (b_, lab), F in FX.items():
            if b_ != ba: continue
            for h in HS:
                y = I["crosswalk"].reindex(F[h].index + pd.Timedelta(days=h)).values; p = F[h].values; ok = ~np.isnan(y)
                acc_rows.append(dict(ba=ba, model=lab, horizon=h, r2=float(1 - np.sum((y[ok] - p[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))))
        for (b_, lab), F in FE.items():
            if b_ != ba or lab != "Fourier ridge + GEFS": continue
            c = decide(F, origins)[0]; ok = valid & (c >= 0)
            savE = (We.mean(1) - We[r, c])[ok]; oE = (We.mean(1) - We.min(1))[ok]
            cap_rows.append(dict(ba=ba, rule="Fourier ridge + GEFS (EIA-trained)", n=int(ok.sum()), capture_eia=float(savE.sum() / oE.sum()),
                                 capture_eia_ci=blocks_ci(savE, oE)))
        mef_summ.append(dict(ba=ba, mef_pooled=mef_all, mef_mean_eval=float(MEF.loc[ev].mean()), mean_avg_intensity=float(I["crosswalk"].loc[ev].mean()),
                             corr_daily_avg_vs_mef=float(pd.concat([I["crosswalk"], MEF], axis=1).loc[ev].dropna().corr().iloc[0, 1])))
        # ---------------- (D) weekly process scheduling
        weeks = pd.date_range(ORIG, LAST - pd.Timedelta(days=7), freq="7D")
        Wxk = window(I["crosswalk"], weeks); Wmk = window(MEF, weeks)
        prules = {"Fixed (days 1-4)": None, "Perfect foresight": "pf", "Same weekday last week": rules["Same weekday last week"]}
        for lab in ("Ridge", "Fourier ridge", "Fourier ridge + GEFS", "Fourier GAM + GEFS", "Fourier HGB + GEFS"): prules[lab] = rules[lab]
        res = {}
        okk = ~np.isnan(Wxk).any(1) & ~np.isnan(Wmk).any(1)
        for lab, F in prules.items():
            vals = []
            for i, t in enumerate(weeks):
                if not okk[i]: vals.append((np.nan,) * 3); continue
                if F is None: j = int(np.where((PAT == np.array([1, 1, 1, 1, 0, 0, 0])).all(1))[0][0]); e = energy(j); vals.append((e @ Wxk[i], e @ Wmk[i], STARTS[j].sum())); continue
                Ih = Wxk[i] if isinstance(F, str) else np.array([F[h].get(t, np.nan) for h in HS])
                if np.isnan(Ih).any(): vals.append((np.nan,) * 3); continue
                a, m_, s_, _ = process_week(Ih, Wxk[i], Wmk[i]); vals.append((a, m_, s_))
            res[lab] = np.array(vals, float)
        okp = np.all(np.stack([~np.isnan(v).any(1) for v in res.values()]), 0)
        base = res["Fixed (days 1-4)"][okp]; pf = res["Perfect foresight"][okp]
        h2 = PROCESS["run_days_per_week"] * E_DAY * 1000 / PROCESS["kwh_per_kg_h2"] / 1000     # t H2 per week
        for lab, v in res.items():
            v = v[okp]; dA = (base[:, 0] - v[:, 0]) / 1000; dM = (base[:, 1] - v[:, 1]) / 1000; dS = v[:, 2] - base[:, 2]
            cost = dS * PROCESS["startup_cost_usd"]
            def bci(x):
                n = len(x); L = 4; bb = [x[np.concatenate([np.arange(s, s + L) for s in rng.integers(0, n - L + 1, int(np.ceil(n / L)))])[:n]].mean() for _ in range(NB)]
                return [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]
            proc_rows.append(dict(ba=ba, rule=lab, weeks=int(okp.sum()), avg_emissions_t_week=float(v[:, 0].mean() / 1000),
                                  avg_reduction_t_week=float(dA.mean()), avg_reduction_ci=bci(dA), avg_reduction_pct=float(100 * dA.sum() / (base[:, 0].sum() / 1000)),
                                  capture_of_perfect=float(dA.sum() / ((base[:, 0] - pf[:, 0]).sum() / 1000)),
                                  marginal_reduction_t_week=float(dM.mean()), marginal_reduction_ci=bci(dM),
                                  marginal_reduction_pct=float(100 * dM.sum() / (base[:, 1].sum() / 1000)),
                                  extra_starts_week=float(dS.mean()), extra_cost_usd_week=float(cost.mean()),
                                  abatement_cost_usd_per_t_avg=float(cost.sum() / dA.sum()) if dA.sum() > 0 else np.nan,
                                  kg_co2_per_kg_h2_avg=float(v[:, 0].mean() / (h2 * 1000)), h2_t_week=h2))
        print(ba, "done", flush=True)
    pd.concat(ORIGINS).to_csv(os.path.join(OUT, "evaluation_origins.csv"), index=False)
    pd.DataFrame(agree).to_csv(os.path.join(OUT, "intensity_definition_agreement.csv"), index=False)
    pd.DataFrame(cap_rows).to_csv(os.path.join(OUT, "scheduling_capture.csv"), index=False)
    pd.DataFrame(acc_rows).to_csv(os.path.join(OUT, "intensity_forecast_accuracy.csv"), index=False)
    pd.concat(mef_tabs).to_csv(os.path.join(OUT, "marginal_emission_factors.csv"), index=False)
    pd.DataFrame(mef_summ).to_csv(os.path.join(OUT, "marginal_summary.csv"), index=False)
    pd.DataFrame(proc_rows).to_csv(os.path.join(OUT, "process_scheduling.csv"), index=False)
    json.dump(dict(process=PROCESS, feasible_patterns=PAT.tolist()), open(os.path.join(OUT, "process_definition.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(agree).drop(columns=["transfer_ci"]).round(3).to_string())
    C = pd.DataFrame(cap_rows)
    print(C[["ba", "rule", "capture_avg", "capture_eia", "capture_marginal", "saving_avg_kg_mwh", "saving_marginal_kg_mwh", "within_week_spearman", "hit_lowest"]].round(3).to_string())
    print(pd.DataFrame(mef_summ).round(3)); print(pd.concat(mef_tabs).round(1).to_string())
    print(pd.DataFrame(proc_rows)[["ba", "rule", "avg_emissions_t_week", "avg_reduction_t_week", "avg_reduction_pct", "capture_of_perfect", "marginal_reduction_t_week", "marginal_reduction_pct", "extra_starts_week", "abatement_cost_usd_per_t_avg"]].round(2).to_string())
