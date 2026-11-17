"""Fitted annual-calendar responses. For each representative state at h=14, the final-refit
models (last 14-day block, selected settings) are evaluated along valid calendar dates of 2025 (month and
day-of-year encodings varied jointly), holding all other predictors at their training values:
GAM/ridge: other inputs at training medians (exact partial effect for additive models);
HGB: partial dependence = mean prediction over 400 training rows with the calendar path substituted.
Year-end continuity: response on 31 Dec vs 1 Jan."""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[k] = '1'
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK, RESULTS, STATE_DAILY, ensure
from models import model_for, RIDGE_ALPHA_GRID, HGB_GRID
from engine import eligible_training, prepare, FULL_FEATURES, metrics
STATES = sys.argv[1:] or ["TX", "PA", "CA", "ND", "NM", "GA"]
H = 14
OUT = ensure(RESULTS, "benchmark")
FDATA = STATE_DAILY
ANN = ["month_sin", "month_cos", "doy_sin", "doy_cos"]


def fourier_cols(dates, K):
    dt = pd.DatetimeIndex(dates); doy, dow = dt.dayofyear.values, dt.dayofweek.values
    c = {}
    for k in range(2, K + 1):
        c[f"fa_s{k}"] = np.sin(2 * np.pi * k * doy / 365.25); c[f"fa_c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    for k in (2, 3):
        c[f"fw_s{k}"] = np.sin(2 * np.pi * k * dow / 7); c[f"fw_c{k}"] = np.cos(2 * np.pi * k * dow / 7)
    return pd.DataFrame(c)


rows, summ = [], []
path = pd.date_range("2025-01-01", "2025-12-31")
tail = pd.date_range("2026-01-01", "2026-01-31")
allp = path.append(tail)
for s in STATES:
    sd = pd.read_csv(f"{FDATA}/{s}.csv"); d = prepare(sd, H)
    sets = json.load(open(f"{WORK}/pred_test/{s}_h{H}_settings.json"))
    fsel = json.load(open(f"{WORK}/exp/fourier/{s}_h{H}.json"))["best"]["fourier_ridge"]
    last_block = pd.date_range(pd.to_datetime(sd.date).min() + pd.Timedelta(days=30), pd.to_datetime(sd.date).max())
    start = last_block[len(last_block) // 2]
    blocks = pd.date_range(start, d.date.max(), freq="14D"); T = blocks[-1]
    tr = eligible_training(d, T)
    X, y = tr[FULL_FEATURES], tr.target.to_numpy()
    med = X.median()
    rng = np.random.default_rng(0); samp = X.iloc[rng.choice(len(X), min(400, len(X)), replace=False)]
    mon, doy = allp.month.values, allp.dayofyear.values
    cal = pd.DataFrame({"month_sin": np.sin(2 * np.pi * mon / 12), "month_cos": np.cos(2 * np.pi * mon / 12),
                        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25)})
    prof = {}
    for m in ("ridge", "gam", "hgb"):
        mod = model_for(m, sets["settings"][m] if "settings" in sets else sets[m]); mod.fit(X.to_numpy(), y)
        if m in ("ridge", "gam"):
            Z = pd.DataFrame([med.values] * len(allp), columns=FULL_FEATURES)
            for c in ANN: Z[c] = cal[c].values
            prof[m] = mod.predict(Z.to_numpy())
        else:
            vals = []
            base = samp.to_numpy()
            ci = [FULL_FEATURES.index(c) for c in ANN]
            for i in range(len(allp)):
                Z = base.copy(); Z[:, ci] = cal.iloc[i].values; vals.append(mod.predict(Z).mean())
            prof[m] = np.array(vals)
    # Fourier ridge (selected K): annual harmonics on the same path, other inputs at medians
    K, a = fsel
    Xf = pd.concat([X.reset_index(drop=True), fourier_cols(tr.date, K)], axis=1)
    modf = model_for("ridge", a); modf.fit(Xf.to_numpy(), y)
    Zf = pd.DataFrame([Xf.median().values] * len(allp), columns=Xf.columns)
    for c in ANN: Zf[c] = cal[c].values
    fa = fourier_cols(allp, K)
    for c in fa.columns:
        if c.startswith("fa_"): Zf[c] = fa[c].values
    prof["fourier_ridge"] = modf.predict(Zf.to_numpy())
    # empirical profile: mean 14-day-ahead target by origin day of year (training period), smoothed 15-day
    emp = pd.Series(tr.target.values, index=pd.DatetimeIndex(tr.date)).groupby(lambda t: t.dayofyear).mean()   # aligned to the origin date
    emp = emp.reindex(range(1, 367)).interpolate().rolling(15, center=True, min_periods=1).mean()
    mean_y = float(np.mean(y))
    for m, v in prof.items():
        v = np.asarray(v); vy = v[:len(path)]
        jump = float(v[len(path)] - v[len(path) - 1])            # 1 Jan 2026 minus 31 Dec 2025
        daily = np.abs(np.diff(vy))
        summ.append(dict(state=s, model=m, amplitude_pct=float(100 * (vy.max() - vy.min()) / mean_y),
                         yearend_jump_pct=float(100 * abs(jump) / mean_y), median_daily_step_pct=float(100 * np.median(daily) / mean_y),
                         max_daily_step_pct=float(100 * daily.max() / mean_y),
                         corr_with_empirical=float(np.corrcoef(vy, emp.reindex(path.dayofyear).values)[0, 1])))
        for dte, val in zip(allp, v):
            rows.append(dict(state=s, model=m, date=dte, value_pct=100 * (val - vy.mean()) / mean_y))
    for dte in path:
        rows.append(dict(state=s, model="empirical", date=dte, value_pct=100 * (emp.loc[dte.dayofyear] - emp.mean()) / mean_y))
    print(s, "done", flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/fitted_responses.csv", index=False)
pd.DataFrame(summ).to_csv(f"{OUT}/fitted_responses_summary.csv", index=False)
print(pd.DataFrame(summ).round(3).to_string())
