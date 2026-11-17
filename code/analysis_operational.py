"""Operational analyses (information set at issue, weather forecasts). Usage: python analysis_operational.py SECTION [...]
sections: gefs ops opsdec.  Results accumulate in results/operational/results.json"""
import os
for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[_k] = "1"
import sys, json, glob, subprocess
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jboot import Panel, boot, ci, block_weights

from paths import WORK as W, WORK as V, STATE_DAILY as DATA, WEATHER, RESULTS, ensure
OUT = ensure(RESULTS, "operational")
H = [1, 2, 3, 5, 7, 10, 14]; STATES = sorted(f[:-4] for f in os.listdir(DATA))
RJ = f"{OUT}/results.json"; R = json.load(open(RJ)) if os.path.exists(RJ) else {}
VARS = ["ops", "ops_f", "ops_fc", "ops_fw", "ops_fo"]; FAMS = ["ridge", "gam", "hgb"]; MARGIN = 0.01


def save(): json.dump(R, open(RJ, "w"), indent=1, default=float)


def verdict(lo90, hi90, lo95, hi95, margin=MARGIN):
    e = lo90 > -margin and hi90 < margin; ni = lo90 > -margin
    if lo95 > 0: return "S" + ("/N" if ni else "")
    if hi95 < 0: return "I" + ("/E" if e else "")
    return "E" if e else ("N" if ni else "U")


def gefs():
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gefs_calibration.py")], check=True)
    g = json.load(open(f"{WEATHER}/gefs_summary.json")); R["gefs_skill"] = g["skill"]; R["gefs_missing_issue_days"] = g["missing_issue_days"]
    C = pd.read_csv(f"{WEATHER}/gefs_calibration_coefficients.csv")
    R["gefs_calibration"] = {q: dict(median_a=float(C[C.q == q].a.median()), median_b=float(C[C.q == q].b.median())) for q in ("hdd", "cdd")}
    save()


def frames(h, variants=VARS, base=True):
    out = []
    for s in STATES:
        m = None
        for v in variants:
            fn = f"{V}/exp/{v}/{s}_h{h}.csv"
            if not os.path.exists(fn): continue
            x = pd.read_csv(fn)[["date", "target"] + FAMS].rename(columns={k: f"{v}:{k}" for k in FAMS})
            m = x if m is None else m.merge(x.drop(columns="target"), on="date", how="inner")
        if base:   # benchmark models and lagged persistence on the same dates
            b = pd.read_csv(f"{W}/pred_test/{s}_h{h}.csv")[["date", "ridge", "gam", "hgb"]].rename(columns=lambda c: c if c == "date" else f"bench:{c}")
            f_ = pd.read_csv(f"{V}/exp/fourier/{s}_h{h}.csv")[["date", "fourier_ridge"]].rename(columns={"fourier_ridge": "bench:fourier_ridge"})
            m = m.merge(b, on="date", how="left").merge(f_, on="date", how="left")
            y = pd.read_csv(f"{DATA}/{s}.csv", parse_dates=["date"]).set_index("date").co2_mass.asfreq("D")
            m["ops:persistence"] = y.shift(1).reindex(pd.to_datetime(m.date)).values      # last finalized observation
        m.insert(0, "state", s); out.append(m)
    return out


CON = [("ops_f:ridge", "ops:ridge", "Ridge: + Fourier − lagged inputs only"),
       ("ops:gam", "ops:ridge", "GAM − ridge, lagged inputs only"), ("ops:hgb", "ops:ridge", "HGB − ridge, lagged inputs only"),
       ("ops_f:gam", "ops_f:ridge", "GAM − ridge, + Fourier"), ("ops_f:hgb", "ops_f:ridge", "HGB − ridge, + Fourier"),
       ("ops_fc:ridge", "ops_f:ridge", "Ridge: + degree-day climatology"), ("ops_fw:ridge", "ops_f:ridge", "Ridge: + GEFS forecast"),
       ("ops_fw:gam", "ops_f:gam", "GAM: + GEFS forecast"), ("ops_fw:hgb", "ops_f:hgb", "HGB: + GEFS forecast"),
       ("ops_fw:ridge", "ops_fc:ridge", "Ridge: GEFS forecast − climatology"),
       ("ops_fw:gam", "ops_fw:ridge", "GAM − ridge, + Fourier + GEFS"), ("ops_fw:hgb", "ops_fw:ridge", "HGB − ridge, + Fourier + GEFS"),
       ("ops_fo:gam", "ops_fo:ridge", "GAM − ridge, + Fourier + observed weather"), ("ops_fo:hgb", "ops_fo:ridge", "HGB − ridge, + Fourier + observed weather"),
       ("ops_fo:ridge", "ops_fw:ridge", "Ridge: observed − forecast weather"), ("ops_fo:hgb", "ops_fw:hgb", "HGB: observed − forecast weather"),
       ("ops_f:ridge", "bench:fourier_ridge", "Fourier ridge: operational − benchmark information")]


def ops():
    mean, crow, agg = {}, [], []
    for h in H:
        fr = frames(h)
        avail = [v for v in VARS if all(f"{v}:ridge" in f.columns for f in fr)]
        models = [f"{v}:{m}" for v in avail for m in FAMS] + ["bench:ridge", "bench:gam", "bench:hgb", "bench:fourier_ridge", "ops:persistence"]
        fr = [f.dropna(subset=models) for f in fr]
        P = Panel(fr, models); r2 = {k: P.r2(k) for k in models}
        CON_h = [c for c in CON if c[0] in models and c[1] in models]
        mean[str(h)] = {k: float(np.nanmean(v)) for k, v in r2.items()}
        mean[str(h)]["n_dates_median"] = float(np.median(P.mask.sum(1)))
        bs = boot({"x": P}, lambda rr: np.array([np.nanmean(rr["x"](a) - rr["x"](b)) for a, b, _ in CON_h]))
        for k, (a, b, lab) in enumerate(CON_h):
            lo, hi = ci(bs[:, k]); lo9, hi9 = ci(bs[:, k], .90); d = r2[a] - r2[b]
            crow.append(dict(horizon=h, a=a, b=b, label=lab, mean=float(np.nanmean(d)), ci_lo=lo, ci_hi=hi, ci90_lo=lo9, ci90_hi=hi9,
                             n_a_better=int(np.sum(d > 0)), verdict=verdict(lo9, hi9, lo, hi)))
        # national daily total (days with all 45 states)
        full = P.mask.all(0); Y = np.where(P.mask, P.y, 0).sum(0)[full]
        AM = [k for k in ["ops:persistence", "ops:ridge", "ops_f:ridge", "ops_f:gam", "ops_f:hgb", "ops_fw:ridge", "ops_fw:gam", "ops_fw:hgb", "ops_fo:ridge"] if k in models]
        Fm = {k: np.where(P.mask, P.p[k], 0).sum(0)[full] for k in AM}; e2 = {k: (Y - Fm[k]) ** 2 for k in AM}
        rng = np.random.default_rng(42); n = len(Y); PA = [(k, "ops_fw:ridge") for k in AM if k != "ops_fw:ridge"]
        bsr = []
        for _ in range(2000):
            w = block_weights(n, rng); bsr.append([np.sqrt(np.average(e2[a], weights=w)) - np.sqrt(np.average(e2[b], weights=w)) for a, b in PA])
        bsr = np.array(bsr)
        row = dict(horizon=h, mean_kt=float(Y.mean() / 1e3), **{f"rmse_kt:{k}": float(np.sqrt(e2[k].mean()) / 1e3) for k in AM})
        for j, (a, b) in enumerate(PA):
            lo, hi = ci(bsr[:, j]); row[f"diff_kt:{a}"] = float((np.sqrt(e2[a].mean()) - np.sqrt(e2[b].mean())) / 1e3); row[f"diff_ci_kt:{a}"] = [lo / 1e3, hi / 1e3]
        agg.append(row)
    R["ops_mean"] = mean; R["ops_contrasts"] = crow; R["ops_aggregate"] = agg
    pd.DataFrame(crow).to_csv(f"{OUT}/ops_contrasts.csv", index=False); pd.DataFrame(agg).to_csv(f"{OUT}/ops_aggregate.csv", index=False)
    pd.DataFrame(mean).T.to_csv(f"{OUT}/ops_mean_r2.csv"); save()


def opsdec():
    """calibrated P90 warnings under the operational information set (thresholds before the first origin; calibration on the first year)"""
    thr = {}
    for s in STATES:
        y = pd.read_csv(f"{DATA}/{s}.csv", parse_dates=["date"]).set_index("date").co2_mass.asfreq("D")
        thr[s] = float(y[y.index < "2022-07-17"].quantile(0.9))
    rows = []
    for h in (1, 7, 14):
        fr = frames(h)
        for m in ["ops:persistence", "ops:ridge", "ops_f:ridge", "ops_f:gam", "ops_f:hgb", "ops_fw:ridge", "ops_fw:gam", "ops_fw:hgb"]:
            tot = {"raw": np.zeros(4), "calibrated": np.zeros(4)}
            for f in fr:
                f = f[f[m].notna()]; t = thr[f.state.iloc[0]]; d = pd.to_datetime(f.date)
                A, B_ = f[d < "2023-07-17"], f[d >= "2023-07-17"]
                r = float(np.mean(A.target > t)); tau = float(np.quantile(A[m], 1 - r)) if 0 < r < 1 else t
                for rule, th in (("raw", t), ("calibrated", tau)):
                    o = (B_.target > t).values; p_ = (B_[m] > th).values
                    tot[rule] += [np.sum(o & p_), np.sum(o & ~p_), np.sum(~o & p_), np.sum(~o & ~p_)]
            for rule, (hit, miss, fa, cn) in tot.items():
                rows.append(dict(horizon=h, model=m, rule=rule, pod=float(hit / (hit + miss)), far=float(fa / (hit + fa)) if hit + fa else np.nan,
                                 pss=float(hit / (hit + miss) - fa / (fa + cn))))
    R["ops_warnings"] = rows; pd.DataFrame(rows).to_csv(f"{OUT}/ops_warnings.csv", index=False); save()


if __name__ == "__main__":
    for sec in sys.argv[1:]: globals()[sec]()
    print("ok", sys.argv[1:])
