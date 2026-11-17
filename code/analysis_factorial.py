"""Factorial, equivalence and decision analyses. Usage: python analysis_factorial.py SECTION [...]   sections: factorial equiv yearblock application weather prophet cost oos
Results accumulate in results/factorial/results.json; tables in results/factorial/*.csv"""
import os
for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[_k] = "1"
import sys, glob, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jboot import Panel, boot, ci, block_weights

from paths import WORK as W, WORK as V, STATE_DAILY as DATA, RESULTS, ensure
OUT = ensure(RESULTS, "factorial")
H = [1, 2, 3, 5, 7, 10, 14]
STATES = sorted(f[:-4] for f in os.listdir(DATA))
RJ = f"{OUT}/results.json"
R = json.load(open(RJ)) if os.path.exists(RJ) else {}
NB = 2000
MARGIN = 0.01


def save(): json.dump(R, open(RJ, "w"), indent=1, default=float)


def frames(h, extra=True):
    """per-state frames with every model on the identical rows: base, Fourier, factorial variants, classical, Prophet."""
    out = []
    for s in STATES:
        b = pd.read_csv(f"{W}/pred_test/{s}_h{h}.csv")[["state", "date", "target_date", "target", "persistence", "ridge", "gam", "hgb"]]
        f = pd.read_csv(f"{V}/exp/fourier/{s}_h{h}.csv")[["date", "fourier_ridge"]]
        n = pd.read_csv(f"{V}/exp/fourier_nl/{s}_h{h}.csv")[["date", "fourier_gam", "fourier_hgb"]]
        m = b.merge(f, on="date").merge(n, on="date")
        if extra:
            for v in ("fa", "fw", "cs"):
                fn = f"{V}/exp/{v}/{s}_h{h}.csv"
                if os.path.exists(fn):
                    x = pd.read_csv(fn)[["date", "ridge", "gam", "hgb"]].rename(columns={k: f"{v}_{k}" for k in ("ridge", "gam", "hgb")})
                    m = m.merge(x, on="date", how="left")
            c = pd.read_csv(f"{V}/sb/{s}.csv"); c = c[c.horizon == h][["date", "snaive7", "snaive364", "fets", "sarimax"]]
            m = m.merge(c, on="date", how="left")
            fn = f"{V}/pr/{s}.csv"
            if os.path.exists(fn):
                pz = pd.read_csv(fn); m = m.merge(pz[pz.horizon == h][["date", "prophet"]], on="date", how="left")
        assert len(m) == len(b)
        out.append(m)
    return out


def verdict(lo90, hi90, lo95, hi95, margin=MARGIN):
    """S: superior (95% CI > 0); I: inferior (95% CI < 0); E: equivalent (90% CI within +-margin); N: non-inferior (90% lower > -margin); U: undetermined"""
    e = lo90 > -margin and hi90 < margin; ni = lo90 > -margin
    if lo95 > 0: return "S" + ("/N" if ni else "")
    if hi95 < 0: return "I" + ("/E" if e else "")
    if e: return "E"
    if ni: return "N"
    return "U"


# ================================================================== factorial (model family x seasonal basis)
FAMS = ["ridge", "gam", "hgb"]
BASES = {"orig": lambda m: m, "fw": lambda m: f"fw_{m}", "fa": lambda m: f"fa_{m}", "faw": lambda m: f"fourier_{m}", "cs": lambda m: f"cs_{m}"}


def factorial():
    rows, crow = [], []
    CONTR = []
    for m in FAMS:
        CONTR += [(f"fa_{m}", BASES["orig"](m), f"{m}: +annual Fourier vs original"), (f"fw_{m}", m, f"{m}: +weekly Fourier vs original"),
                  (f"fourier_{m}", f"fa_{m}", f"{m}: +weekly given annual"), (f"cs_{m}", f"fa_{m}", f"{m}: cyclic spline vs annual Fourier")]
    CONTR += [("gam", "ridge", "GAM vs ridge, original basis"), ("hgb", "ridge", "HGB vs ridge, original basis"),
              ("fa_gam", "fa_ridge", "GAM vs ridge, annual Fourier basis"), ("fa_hgb", "fa_ridge", "HGB vs ridge, annual Fourier basis"),
              ("cs_gam", "cs_ridge", "GAM vs ridge, cyclic-spline basis"), ("cs_hgb", "cs_ridge", "HGB vs ridge, cyclic-spline basis"),
              ("fourier_gam", "fourier_ridge", "GAM vs ridge, annual+weekly Fourier basis"), ("fourier_hgb", "fourier_ridge", "HGB vs ridge, annual+weekly Fourier basis")]
    for h in H:
        fr = frames(h)
        models = [BASES[b](m) for b in BASES for m in FAMS]
        P = Panel(fr, models); r2 = {k: P.r2(k) for k in models}
        row = dict(horizon=h, **{k: float(np.nanmean(v)) for k, v in r2.items()})
        rows.append(row)
        bs = boot({"x": P}, lambda rr: np.array([np.nanmean(rr["x"](a) - rr["x"](b)) for a, b, _ in CONTR]), n_boot=NB)
        for k, (a, b, lab) in enumerate(CONTR):
            lo, hi = ci(bs[:, k]); lo9, hi9 = ci(bs[:, k], .90); d = r2[a] - r2[b]
            crow.append(dict(horizon=h, a=a, b=b, label=lab, mean=float(np.nanmean(d)), ci_lo=lo, ci_hi=hi, ci90_lo=lo9, ci90_hi=hi9,
                             n_a_better=int(np.sum(d > 0)), verdict=verdict(lo9, hi9, lo, hi)))
        # two-way decomposition (family x basis) of mean R2 at this horizon over the 5 bases
        Mx = np.array([[row[BASES[b](m)] for b in BASES] for m in FAMS])
        g = Mx.mean(); fam = Mx.mean(1) - g; bas = Mx.mean(0) - g; inter = Mx - g - fam[:, None] - bas[None, :]
        row.update(sd_family=float(fam.std()), sd_basis=float(bas.std()), sd_interaction=float(inter.std()))
    pd.DataFrame(rows).to_csv(f"{OUT}/factorial_mean_r2.csv", index=False); pd.DataFrame(crow).to_csv(f"{OUT}/factorial_contrasts.csv", index=False)
    R["factorial"] = rows; R["factorial_contrasts"] = crow
    sel = {}
    for v in ("fa", "cs"):
        for m in FAMS:
            ks = [json.load(open(j))["best"][m][0] for j in glob.glob(f"{V}/exp/{v}/*.json")]
            sel[f"{v}_{m}"] = {str(k): int(n) for k, n in pd.Series(ks).value_counts().sort_index().items()}
    R["factorial_selection"] = sel; save()


# ================================================================== equivalence / non-inferiority and absolute error
def equiv():
    PAIRS = [("fourier_ridge", "ridge"), ("fourier_ridge", "gam"), ("fourier_ridge", "hgb"), ("fourier_ridge", "fourier_gam"),
             ("fourier_ridge", "fourier_hgb"), ("fourier_ridge", "sarimax"), ("fourier_ridge", "prophet"), ("gam", "hgb"), ("sarimax", "gam")]
    rows, agg = [], []
    for h in H:
        fr = frames(h)
        models = sorted(set(sum(PAIRS, ())))
        P = Panel(fr, models); r2 = {k: P.r2(k) for k in models}
        bs = boot({"x": P}, lambda rr: np.array([np.nanmean(rr["x"](a) - rr["x"](b)) for a, b in PAIRS]), n_boot=NB)
        for k, (a, b) in enumerate(PAIRS):
            lo, hi = ci(bs[:, k]); lo9, hi9 = ci(bs[:, k], .90); d = r2[a] - r2[b]
            rows.append(dict(horizon=h, a=a, b=b, mean=float(np.nanmean(d)), ci_lo=lo, ci_hi=hi, ci90_lo=lo9, ci90_hi=hi9,
                             n_a_better=int(np.sum(d > 0)), verdict=verdict(lo9, hi9, lo, hi)))
        # absolute error: national aggregate (sum of 45 states) RMSE and MAE; bootstrap over the same date blocks
        full = P.mask.all(0); Y = np.where(P.mask, P.y, 0).sum(0)[full]
        F = {k: np.where(P.mask, P.p[k], 0).sum(0)[full] for k in models}
        e2 = {k: (Y - F[k]) ** 2 for k in models}; mean_y = Y.mean()
        rng = np.random.default_rng(42); n = len(Y)
        rmse = {k: np.sqrt(e2[k].mean()) for k in models}
        bsr = []
        for _ in range(NB):
            wgt = block_weights(n, rng)
            bsr.append([np.sqrt(np.average(e2[a], weights=wgt)) - np.sqrt(np.average(e2[b], weights=wgt)) for a, b in PAIRS])
        bsr = np.array(bsr)
        for k, (a, b) in enumerate(PAIRS):
            lo, hi = ci(bsr[:, k])
            agg.append(dict(horizon=h, a=a, b=b, rmse_a_kt=rmse[a] / 1e3, rmse_b_kt=rmse[b] / 1e3, diff_kt=(rmse[a] - rmse[b]) / 1e3,
                            diff_ci_kt=[lo / 1e3, hi / 1e3], diff_pct_mean=100 * (rmse[a] - rmse[b]) / mean_y,
                            diff_ci_pct_mean=[100 * lo / mean_y, 100 * hi / mean_y], mean_kt=mean_y / 1e3,
                            mae_a_kt=float(np.abs(Y - F[a]).mean() / 1e3), mae_b_kt=float(np.abs(Y - F[b]).mean() / 1e3)))
    pd.DataFrame(rows).to_csv(f"{OUT}/equivalence.csv", index=False); pd.DataFrame(agg).to_csv(f"{OUT}/aggregate_error.csv", index=False)
    R["equiv"] = rows; R["aggregate_error"] = agg
    # margin calibration: relative RMSE change implied by dR2 = 0.01 at the mean Fourier-ridge R2 of each horizon
    R["margin_rmse_pct"] = {str(h): float(100 * (1 - np.sqrt((1 - (r2m + MARGIN)) / (1 - r2m))))
                            for h, r2m in [(h, json.load(open(f"{RESULTS}/benchmark/results.json"))["mean_r2"][str(h)]["fourier_ridge"]) for h in H]}
    save()


# ================================================================== year-block inference (exact enumeration of the 4^4 year resamples)
def yearblock():
    CON = [("hgb", "ridge"), ("gam", "ridge"), ("fourier_ridge", "gam"), ("fourier_ridge", "hgb"), ("fourier_ridge", "ridge")]
    out, per_year = {}, []
    slopes = {}
    for h in H:
        fr = frames(h, extra=False)
        models = ["ridge", "gam", "hgb", "fourier_ridge"]
        P = Panel(fr, models); yrs = P.dates.year.values; Y = sorted(set(yrs))
        for y in Y:
            w = (yrs == y).astype(float); rr = {m: P.r2(m, w) for m in models}
            per_year.append(dict(horizon=h, year=int(y), **{f"{a}-{b}": float(np.nanmean(rr[a] - rr[b])) for a, b in CON}))
        stats = []
        for combo in itertools.product(Y, repeat=len(Y)):
            w = np.zeros(len(yrs))
            for y in combo: w += (yrs == y)
            rr = {m: P.r2(m, w) for m in models}
            stats.append([np.nanmean(rr[a] - rr[b]) for a, b in CON])
        stats = np.array(stats)
        out[str(h)] = {f"{a}-{b}": dict(ci=[float(np.percentile(stats[:, k], 2.5)), float(np.percentile(stats[:, k], 97.5))],
                                       share_positive=float(np.mean(stats[:, k] > 0))) for k, (a, b) in enumerate(CON)}
    PY = pd.DataFrame(per_year); PY.to_csv(f"{OUT}/yearblock_per_year.csv", index=False)
    # slope of HGB-ridge on horizon, per evaluation year (leave-years-in) and leave-one-year-out
    sl = {}
    for y in sorted(PY.year.unique()):
        d = PY[PY.year == y]; sl[int(y)] = float(np.polyfit(d.horizon, d["hgb-ridge"], 1)[0])
    R["yearblock"] = out; R["yearblock_slope_by_year"] = sl
    R["yearblock_sign_consistency"] = {k: {str(h): int((PY[PY.horizon == h][k] > 0).sum()) for h in H} for k in [f"{a}-{b}" for a, b in CON]}
    save()


# ================================================================== application: cumulative totals and threshold warnings
def relvalue(hit, miss, fa, cn, cl):
    """Richardson (2000) relative economic value for a deterministic warning at cost/loss ratio cl."""
    n = hit + miss + fa + cn; s = (hit + miss) / n
    if s == 0: return np.nan
    E_f = (fa + hit) / n * cl + miss / n; E_c = min(cl, s); E_p = s * cl
    return float((E_c - E_f) / (E_c - E_p)) if E_c > E_p else np.nan


def application():
    res = {}
    # thresholds: 90th percentile of daily (or 7-day total) emissions before the first forecast origin (2022-07-17)
    thr = {}
    for s in STATES:
        y = pd.read_csv(f"{DATA}/{s}.csv", parse_dates=["date"]).set_index("date").co2_mass.asfreq("D")
        pre = y[y.index < "2022-07-17"]
        thr[s] = dict(d90=float(pre.quantile(0.9)), w90=float(pre.rolling(7).sum().quantile(0.9)), w14=float(pre.rolling(14).sum().quantile(0.9)))
    warn = []
    for h in (1, 7, 14):
        fr = frames(h)
        for m in ["persistence", "ridge", "fourier_ridge", "gam", "hgb", "sarimax", "prophet"]:
            tot = np.zeros(4)
            for f in fr:
                t = thr[f.state.iloc[0]]["d90"]; ok = f[m].notna()
                o = f.target[ok] > t; p_ = f[m][ok] > t
                tot += [np.sum(o & p_), np.sum(o & ~p_), np.sum(~o & p_), np.sum(~o & ~p_)]
            hit, miss, fa, cn = tot; base = (hit + miss) / tot.sum()
            warn.append(dict(target="daily", horizon=h, model=m, events=int(hit + miss), base_rate=float(base), pod=float(hit / (hit + miss)),
                             far=float(fa / (hit + fa)) if hit + fa else np.nan, csi=float(hit / (hit + miss + fa)),
                             pss=float(hit / (hit + miss) - fa / (fa + cn)), value=relvalue(hit, miss, fa, cn, base)))
    cum = []
    for H_, key in [(7, "w90"), (14, "w14")]:
        fr = []
        for s in STATES:
            a = pd.read_csv(f"{V}/exp/cum{H_}/{s}_h{H_}.csv")
            fr.append(a)
        for m in ["persistence", "ridge", "fourier_ridge", "gam", "hgb"]:
            tot = np.zeros(4); r2s, mape = [], []
            for a in fr:
                ok = a[m].notna(); y, p_ = a.target[ok].values, a[m][ok].values.astype(float)
                r2s.append(1 - np.sum((y - p_) ** 2) / np.sum((y - y.mean()) ** 2)); mape.append(100 * np.mean(np.abs(y - p_) / y))
                t = thr[a.state.iloc[0]][key]; o = y > t; q = p_ > t
                tot += [np.sum(o & q), np.sum(o & ~q), np.sum(~o & q), np.sum(~o & ~q)]
            hit, miss, fa, cn = tot; base = (hit + miss) / tot.sum()
            cum.append(dict(window=H_, model=m, mean_r2=float(np.mean(r2s)), median_mape=float(np.median(mape)), events=int(hit + miss),
                            base_rate=float(base), pod=float(hit / (hit + miss)), far=float(fa / (hit + fa)) if hit + fa else np.nan,
                            csi=float(hit / (hit + miss + fa)), pss=float(hit / (hit + miss) - fa / (fa + cn)), value=relvalue(hit, miss, fa, cn, base)))
        # national total of the next H_ days
        Y = None
        for m in ["ridge", "fourier_ridge", "gam", "hgb", "persistence"]:
            S_ = pd.concat([a.set_index("date")[["target", m]].astype(float) for a in fr], axis=1, keys=range(len(fr)))
            yy = S_.xs("target", axis=1, level=1).dropna().sum(1); pp = S_.xs(m, axis=1, level=1).loc[yy.index].sum(1)
            res[f"national_cum{H_}_{m}"] = dict(rmse_kt=float(np.sqrt(np.mean((yy - pp) ** 2)) / 1e3), mape=float(100 * np.mean(np.abs(yy - pp) / yy)),
                                                 mean_kt=float(yy.mean() / 1e3))
    # calibrated warnings: each model's forecast threshold is set on the first evaluation year (origins before 2023-07-17) so that its
    # warning frequency equals the observed exceedance frequency there; scored on origins from 2023-07-17 (raw thresholds scored on the same days)
    cal = []
    def score(o, p_):
        o, p_ = np.asarray(o), np.asarray(p_); return [np.sum(o & p_), np.sum(o & ~p_), np.sum(~o & p_), np.sum(~o & ~p_)]
    def calib(frs, key_fn, models, label, hz):
        for m in models:
            raw, cl = np.zeros(4), np.zeros(4)
            for f in frs:
                f = f[f[m].notna()]; t = key_fn(f)
                d = pd.to_datetime(f.date); A = f[d < "2023-07-17"]; B = f[d >= "2023-07-17"]
                r = float(np.mean(A.target > t)); tau = float(np.quantile(A[m], 1 - r)) if 0 < r < 1 else t
                raw += score(B.target > t, B[m] > t); cl += score(B.target > t, B[m] > tau)
            for nm, tot in (("raw", raw), ("calibrated", cl)):
                hit, miss, fa, cn = tot; base = (hit + miss) / tot.sum()
                cal.append(dict(target=label, horizon=hz, model=m, rule=nm, events=int(hit + miss), pod=float(hit / (hit + miss)),
                                far=float(fa / (hit + fa)) if hit + fa else np.nan, csi=float(hit / (hit + miss + fa)),
                                pss=float(hit / (hit + miss) - fa / (fa + cn)), value=relvalue(hit, miss, fa, cn, base)))
    for h in (1, 7, 14):
        fr = frames(h)
        calib(fr, lambda f: thr[f.state.iloc[0]]["d90"], ["persistence", "ridge", "fourier_ridge", "gam", "hgb", "sarimax", "prophet"], "daily", h)
    for H_, key in [(7, "w90"), (14, "w14")]:
        fr = [pd.read_csv(f"{V}/exp/cum{H_}/{s}_h{H_}.csv") for s in STATES]
        calib(fr, lambda f, key=key: thr[f.state.iloc[0]][key], ["persistence", "ridge", "fourier_ridge", "gam", "hgb"], f"cum{H_}", H_)
    R["warnings_calibrated"] = cal; pd.DataFrame(cal).to_csv(f"{OUT}/warnings_calibrated.csv", index=False)
    pd.DataFrame(warn).to_csv(f"{OUT}/warnings_daily.csv", index=False); pd.DataFrame(cum).to_csv(f"{OUT}/cumulative.csv", index=False)
    R["warnings"] = warn; R["cumulative"] = cum; R["cumulative_national"] = res; save()


# ================================================================== weather-informed representation
def weather():
    rows = []
    for h in H:
        fr = frames(h, extra=False); out = []
        for f in fr:
            s = f.state.iloc[0]; m = f.copy()
            for v in ("clim", "oracle"):
                fn = f"{V}/exp/{v}/{s}_h{h}.csv"
                if os.path.exists(fn):
                    x = pd.read_csv(fn)[["date", "ridge", "gam", "hgb"]].rename(columns={k: f"{v}_{k}" for k in ("ridge", "gam", "hgb")})
                    m = m.merge(x, on="date", how="left")
            out.append(m)
        cols = [c for c in out[0].columns if c.startswith(("clim_", "oracle_"))]
        if not cols: continue
        P = Panel(out, ["ridge", "gam", "hgb", "fourier_ridge"] + cols); r2 = {k: P.r2(k) for k in ["ridge", "gam", "hgb", "fourier_ridge"] + cols}
        pairs = [("clim_ridge", "ridge"), ("clim_ridge", "fourier_ridge"), ("oracle_ridge", "fourier_ridge"), ("clim_gam", "gam"), ("clim_gam", "clim_ridge")]
        pairs = [p for p in pairs if p[0] in cols and (p[1] in cols or p[1] in ("ridge", "gam", "fourier_ridge"))]
        bs = boot({"x": P}, lambda rr: np.array([np.nanmean(rr["x"](a) - rr["x"](b)) for a, b in pairs]), n_boot=NB)
        row = dict(horizon=h, **{k: float(np.nanmean(v)) for k, v in r2.items()})
        for k, (a, b) in enumerate(pairs): row[f"{a}-{b}"] = float(np.nanmean(r2[a] - r2[b])); row[f"{a}-{b}_ci"] = ci(bs[:, k])
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{OUT}/weather.csv", index=False); R["weather"] = rows; save()


# ================================================================== classical incl. Prophet
def prophet():
    rows = []
    for h in H:
        fr = frames(h); ms = ["prophet", "sarimax", "fourier_ridge", "ridge", "gam"]
        P = Panel(fr, ms); r2 = {m: P.r2(m) for m in ms}
        rows.append(dict(horizon=h, **{m: float(np.nanmean(v)) for m, v in r2.items()}, prophet_gt_ridge=int(np.sum(r2["prophet"] > r2["ridge"]))))
    R["prophet"] = rows
    t = [json.load(open(f)) for f in glob.glob(f"{V}/pr/*.json")]
    t = [x for x in t if np.isfinite(x.get("fit_s_per_refit", np.nan))]   # timings exist only after refitting (stage A)
    if t: R["prophet_timing"] = dict(fit_s_per_refit=float(np.median([x["fit_s_per_refit"] for x in t])), predict_s_per_refit=float(np.median([x["predict_s_per_refit"] for x in t])))
    save()


# ================================================================== out-of-sample seasonal aggregates (for the fitted-response figure)
def oos():
    rows = []
    for s in ["TX", "PA", "CA", "ND", "GA", "NM"]:
        b = pd.read_csv(f"{W}/pred_test/{s}_h14.csv", parse_dates=["date", "target_date"])
        f = pd.read_csv(f"{V}/exp/fourier/{s}_h14.csv")[["date", "fourier_ridge"]]; f["date"] = pd.to_datetime(f.date)
        m = b.merge(f, on="date")
        ann = m.groupby(m.target_date.dt.year).target.transform("mean")
        m["wk"] = ((m.target_date.dt.dayofyear - 1) // 7).clip(upper=51)
        for c in ("target", "ridge", "fourier_ridge", "gam", "hgb"):
            v = (m[c] / ann - 1) * 100
            g = v.groupby(m.wk).mean().reindex(range(52))
            g = pd.Series(np.convolve(np.r_[g.values[-1:], g.values, g.values[:1]], np.ones(3) / 3, "valid"), index=range(52))
            for k, val in g.items(): rows.append(dict(state=s, series=c, week=int(k), value=float(val)))
        # seasonal bias of each model by month (mean forecast error as % of annual mean)
        for c in ("ridge", "fourier_ridge", "gam", "hgb"):
            e = ((m[c] - m.target) / ann * 100).groupby(m.target_date.dt.month).mean()
            for mo, val in e.items(): rows.append(dict(state=s, series=f"bias_{c}", week=int(mo), value=float(val)))
    pd.DataFrame(rows).to_csv(f"{OUT}/oos_seasonal.csv", index=False)
    D = pd.DataFrame(rows); summ = {}
    for s in D.state.unique():
        t = D[(D.state == s) & (D.series == "target")].set_index("week").value
        summ[s] = {c: float(np.corrcoef(t, D[(D.state == s) & (D.series == c)].set_index("week").value)[0, 1]) for c in ("ridge", "fourier_ridge", "gam", "hgb")}
        summ[s].update({f"bias_amp_{c}": float(D[(D.state == s) & (D.series == f"bias_{c}")].value.abs().mean()) for c in ("ridge", "fourier_ridge", "gam", "hgb")})
    R["oos_seasonal"] = summ; save()


# ================================================================== cost: feature construction, tuning, refits, updating
def cost():
    rows = []
    def collect(var, m, lab):
        base = f"{V}/exp_timing" if var.startswith("timing") else f"{V}/exp"
        var = {"timing_faw": "faw"}.get(var, var)
        js = [json.load(open(j)) for j in glob.glob(f"{base}/{var}/*.json") if os.path.getsize(j) > 0]
        js = [j for j in js if "tune_s" in j and m in j["tune_s"]]
        if not js: return
        rows.append(dict(model=lab, feature_s=float(np.median([j["feature_s"] for j in js])), tune_s=float(np.median([j["tune_s"][m] for j in js])),
                         candidates=int(js[0][f"{m}_candidates"]), fit_ms_per_refit=1000 * float(np.median([j["timing"][m]["fit_s_per_refit"] for j in js])),
                         refits=int(np.median([j["timing"][m]["n_fits"] for j in js])),
                         predict_ms_per_forecast=1000 * float(np.median([j["timing"][m]["predict_s_total"] / 1259 for j in js]))))
    for var, m, lab in [("fw", "ridge", "Ridge (weekly-Fourier basis, 12 candidates)"), ("fa", "ridge", "Ridge with annual Fourier terms"),
                        ("fa", "gam", "GAM with annual Fourier terms"), ("fa", "hgb", "HGB with annual Fourier terms"),
                        ("timing_base", "ridge", "Ridge"), ("timing_base", "gam", "GAM"), ("timing_base", "hgb", "HGB"),
                        ("timing_faw", "ridge", "Fourier ridge"), ("timing_faw", "gam", "Fourier GAM"), ("timing_faw", "hgb", "Fourier HGB")]:
        collect(var, m, lab)
    if not rows:
        print("cost: fit timings are recorded only when the models are refitted (stage A); released values kept"); return
    R["cost"] = rows; pd.DataFrame(rows).to_csv(f"{OUT}/cost.csv", index=False)
    T = [json.load(open(f)) for f in glob.glob(f"{V}/sbt/*.json")]
    if T:
        R["cost_classical"] = {m: {k: float(np.median([t[m][k] for t in T])) for k in T[0][m]} for m in T[0]}
    save()


# ================================================================== expanded HGB search with the annual+weekly basis; gap-filled series
def hgbx():
    rows = []
    for h in (1, 7, 14):
        fr = frames(h, extra=False); out = []
        for f in fr:
            s = f.state.iloc[0]; x = pd.read_csv(f"{V}/exp/faw/{s}_h{h}.csv")[["date", "hgbx"]]
            out.append(f.merge(x, on="date", how="left"))
        P = Panel(out, ["fourier_hgb", "hgbx", "fourier_ridge", "fourier_gam"]); r2 = {m: P.r2(m) for m in ["fourier_hgb", "hgbx", "fourier_ridge", "fourier_gam"]}
        bs = boot({"x": P}, lambda rr: np.array([np.nanmean(rr["x"]("hgbx") - rr["x"]("fourier_hgb")), np.nanmean(rr["x"]("fourier_ridge") - rr["x"]("hgbx"))]), n_boot=NB)
        lo9, hi9 = ci(bs[:, 1], .90); lo, hi = ci(bs[:, 1])
        rows.append(dict(horizon=h, **{m: float(np.nanmean(v)) for m, v in r2.items()}, hgbx_minus_fhgb_ci=ci(bs[:, 0]),
                         fr_minus_hgbx=float(np.nanmean(r2["fourier_ridge"] - r2["hgbx"])), fr_minus_hgbx_ci=[lo, hi], fr_minus_hgbx_ci90=[lo9, hi9],
                         verdict=verdict(lo9, hi9, lo, hi)))
    R["hgbx"] = rows; save()


def filled():
    rows = []
    P0 = pd.read_csv(f"{W}/core/panel.csv")
    for h in (1, 7, 14):
        fs = sorted(glob.glob(f"{V}/exp_filled/base/*_h{h}.csv"))
        fr = [pd.read_csv(f) for f in fs]
        P = Panel(fr, ["ridge", "gam", "hgb"]); r2 = {m: P.r2(m) for m in ("ridge", "gam", "hgb")}
        full = P0[(P0.horizon == h)]
        rows.append(dict(horizon=h, n_states=len(fr), **{m: float(np.nanmean(r2[m])) for m in ("ridge", "gam", "hgb")},
                         gam_minus_ridge=float(np.nanmean(r2["gam"] - r2["ridge"])), hgb_minus_ridge=float(np.nanmean(r2["hgb"] - r2["ridge"])),
                         full_gam_minus_ridge=float((full.gam - full.ridge).mean()), full_hgb_minus_ridge=float((full.hgb - full.ridge).mean())))
    R["filled"] = rows; save()


if __name__ == "__main__":
    for sec in sys.argv[1:]: globals()[sec]()
    print("ok", sys.argv[1:])
