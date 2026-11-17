"""Independent recomputation of the benchmark and factorial numbers from the raw forecast and validation files (no code shared with the
analysis scripts)."""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK as V, RESULTS, DERIVED, STATE_DAILY
R = json.load(open(f"{RESULTS}/factorial/results.json")); H = [1, 2, 3, 5, 7, 10, 14]
bad, n = [], [0]
DATA = STATE_DAILY


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def chk(name, a, b, tol=6e-5):
    n[0] += 1
    if b is None or not abs(a - b) <= tol: bad.append((name, a, b))


states = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{V}/pred_test/*_h1.csv")); assert len(states) == 45


def load(s, h):
    b = pd.read_csv(f"{V}/pred_test/{s}_h{h}.csv")
    for v in ("fa", "fw", "cs"):
        fn = f"{V}/exp/{v}/{s}_h{h}.csv"
        if os.path.exists(fn):
            x = pd.read_csv(fn)[["date", "ridge", "gam", "hgb"]].rename(columns=lambda c: c if c == "date" else f"{v}_{c}"); b = b.merge(x, on="date", how="left")
    b = b.merge(pd.read_csv(f"{V}/exp/fourier/{s}_h{h}.csv")[["date", "fourier_ridge"]], on="date")
    b = b.merge(pd.read_csv(f"{V}/exp/fourier_nl/{s}_h{h}.csv")[["date", "fourier_gam", "fourier_hgb"]], on="date")
    fn = f"{V}/pr/{s}.csv"
    if os.path.exists(fn): p = pd.read_csv(fn); b = b.merge(p[p.horizon == h][["date", "prophet"]], on="date", how="left")
    c = pd.read_csv(f"{V}/sb/{s}.csv"); b = b.merge(c[c.horizon == h][["date", "sarimax"]], on="date", how="left")
    return b


FAC = {r["horizon"]: r for r in R.get("factorial", [])}; PZ = {r["horizon"]: r for r in R.get("prophet", [])}
AGG = pd.DataFrame(R.get("aggregate_error", []))
for h in H:
    fr = [load(s, h) for s in states]
    for k in ["fa_ridge", "fa_gam", "fa_hgb", "fw_ridge", "fw_gam", "fw_hgb", "cs_ridge", "cs_gam", "cs_hgb"]:
        if h in FAC and k in FAC[h] and all(k in f for f in fr): chk(f"factorial {k} h{h}", float(np.mean([r2(f.target, f[k]) for f in fr])), FAC[h][k])
    if h in PZ and all("prophet" in f for f in fr):
        chk(f"prophet h{h}", float(np.mean([r2(f.target[f.prophet.notna()], f.prophet.dropna()) for f in fr])), PZ[h]["prophet"], tol=2e-3)
    # national daily total RMSE (days with all 45 states)
    if len(AGG):
        tot = pd.concat([f.set_index("target_date")[["target", "ridge", "gam", "fourier_ridge"]] for f in fr], axis=1, keys=range(45))
        tot = tot.dropna(); S = {c: tot.xs(c, axis=1, level=1).sum(1) for c in ["target", "ridge", "gam", "fourier_ridge"]}
        for b_ in ("ridge", "gam"):
            ref = AGG[(AGG.horizon == h) & (AGG.a == "fourier_ridge") & (AGG.b == b_)]
            if len(ref):
                chk(f"national RMSE fourier_ridge h{h}", float(np.sqrt(np.mean((S["target"] - S["fourier_ridge"]) ** 2)) / 1e3), float(ref.rmse_a_kt.iloc[0]), tol=0.05)
                chk(f"national RMSE {b_} h{h}", float(np.sqrt(np.mean((S["target"] - S[b_]) ** 2)) / 1e3), float(ref.rmse_b_kt.iloc[0]), tol=0.05)
    # per-year differences
    PY = pd.DataFrame([]) if not os.path.exists(f"{RESULTS}/factorial/yearblock_per_year.csv") else pd.read_csv(f"{RESULTS}/factorial/yearblock_per_year.csv")
    for y in (2022, 2023, 2024, 2025):
        d = []
        for f in fr:
            g = f[pd.to_datetime(f.date).dt.year == y]; d.append(r2(g.target, g.fourier_ridge) - r2(g.target, g.gam))
        ref = PY[(PY.horizon == h) & (PY.year == y)]
        if len(ref): chk(f"year {y} FR-GAM h{h}", float(np.mean(d)), float(ref["fourier_ridge-gam"].iloc[0]), tol=2e-3)
# weather
for r in R.get("weather", []):
    h = r["horizon"]
    for v in ("clim", "oracle"):
        fs = [f"{V}/exp/{v}/{s}_h{h}.csv" for s in states]
        if all(os.path.exists(f) for f in fs):
            for m in ("ridge", "gam", "hgb"):
                vals = []
                for s, f in zip(states, fs):
                    b = pd.read_csv(f"{V}/pred_test/{s}_h{h}.csv")[["date", "target"]].merge(pd.read_csv(f)[["date", m]], on="date", how="left")
                    vals.append(r2(b.target[b[m].notna()], b[m].dropna()))
                chk(f"weather {v}_{m} h{h}", float(np.mean(vals)), r.get(f"{v}_{m}"), tol=2e-3)
# cumulative MAPE and warnings
CU = {(r["window"], r["model"]): r for r in R.get("cumulative", [])}
for H_ in (7, 14):
    fs = [f"{V}/exp/cum{H_}/{s}_h{H_}.csv" for s in states]
    if not all(os.path.exists(f) for f in fs): continue
    for m in ("persistence", "ridge", "fourier_ridge", "gam", "hgb"):
        mape = []
        for f in fs:
            a = pd.read_csv(f); ok = a[m].notna(); mape.append(100 * np.mean(np.abs(a.target[ok] - a[m][ok]) / a.target[ok]))
        if (H_, m) in CU: chk(f"cum{H_} {m} MAPE", float(np.median(mape)), CU[(H_, m)]["median_mape"], tol=1e-6)
WR = {(r["horizon"], r["model"]): r for r in R.get("warnings", [])}
for h in (1, 14):
    for m in ("ridge", "fourier_ridge", "gam"):
        if (h, m) not in WR: continue
        tp = fn_ = fp = tn = 0
        for s in states:
            y = pd.read_csv(f"{DATA}/{s}.csv", parse_dates=["date"]).set_index("date").co2_mass.asfreq("D")
            t = y[y.index < "2022-07-17"].quantile(0.9)
            b = load(s, h) if m != "fourier_ridge" else load(s, h)
            ok = b[m].notna(); o = b.target[ok] > t; p = b[m][ok] > t
            tp += int((o & p).sum()); fn_ += int((o & ~p).sum()); fp += int((~o & p).sum()); tn += int((~o & ~p).sum())
        chk(f"warning PSS {m} h{h}", tp / (tp + fn_) - fp / (fp + tn), WR[(h, m)]["pss"], tol=1e-9)
# validation numbers
EM = pd.read_csv(f"{DERIVED}/eia_multiyear.csv"); VS = json.load(open(f"{DERIVED}/summary.json"))["eia_multiyear"]
chk("EIA median |diff| util+IPP", float(np.median(np.abs(EM.ratio_util_ipp - 1)) * 100), VS["median_abs_pct_diff_util_ipp"], tol=1e-6)
chk("EIA share within 10% util+IPP", float(np.mean(np.abs(EM.ratio_util_ipp - 1) <= 0.10)), VS["share_within_10pct_util_ipp"], tol=1e-6)
chk("EIA state-years", float(len(EM)), float(VS["n_state_years"]), tol=0)
PH = pd.read_csv(f"{DERIVED}/seasonal_decomposition.csv")
C = pd.read_csv(f"{DERIVED}/state_daily_components.csv", parse_dates=["date"])
for s in ("TX", "PA"):
    c = C[C.state == s].set_index("date"); E = c.E if "E" in c else c.co2
    rel = E / E.groupby(E.index.year).transform("mean"); cl = rel.groupby(rel.index.dayofyear).mean()
    cl = pd.concat([cl.iloc[-15:], cl, cl.iloc[:15]]).rolling(15, center=True).mean().iloc[15:-15]
    chk(f"amp_E {s}", float(cl.max() - cl.min()), float(PH.set_index("state").loc[s, "amp_E"]), tol=0.03)
print(f"checked {n[0]} numbers; mismatches: {len(bad)}")
for b in bad: print("  MISMATCH", b)
