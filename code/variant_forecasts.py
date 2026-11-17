"""Design variants on the benchmark protocol (same prepare(), validation block, 14-day refits, outcome cut-offs): Fourier models,
calendar-component removals, target-date calendar, real-time (lagged) inputs, seasonal-residual target, facility panels.
Usage: python variant_forecasts.py OUTDIR DATA_DIR VARIANT HORIZONS(comma) MODELS(comma) STATE [STATE ...]
VARIANT: comp_annual|comp_doy|comp_month|comp_year|comp_weekly|target_cal|realtime|resid|fourier|base
"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[k] = '1'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
OUT, DATA, VARIANT, HS, MODELS = sys.argv[1:6]; STATES = sys.argv[6:]
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from models import model_for, RIDGE_ALPHA_GRID, HGB_GRID
from engine import eligible_training, prepare, FULL_FEATURES, metrics
from engine import engineer_features, CALENDAR_FEATURES, LAG_FEATURES, OPERATIONAL_FEATURES as OPS

HS = [int(x) for x in HS.split(",")]; MODELS = MODELS.split(",")
OUT = Path(OUT) / VARIANT; OUT.mkdir(parents=True, exist_ok=True)
CAL = CALENDAR_FEATURES + ["year"]
DROP = {"comp_annual": ["month_sin", "month_cos", "doy_sin", "doy_cos"], "comp_doy": ["doy_sin", "doy_cos"],
        "comp_month": ["month_sin", "month_cos"], "comp_year": ["year"], "comp_cal": CALENDAR_FEATURES + ["year"], "comp_weekly": ["dow_sin", "dow_cos", "is_weekend"]}
KMAX = 6


def calendar_cols(dates):
    dt = pd.to_datetime(pd.Series(dates).values)
    dow, mon, doy = dt.dayofweek.values, dt.month.values, dt.dayofyear.values
    return pd.DataFrame({"month_sin": np.sin(2 * np.pi * mon / 12), "month_cos": np.cos(2 * np.pi * mon / 12),
                         "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
                         "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
                         "is_weekend": (dow >= 5).astype(int), "year": dt.year.values})


def fourier_cols(dates, K):
    dt = pd.to_datetime(pd.Series(dates).values); doy, dow = dt.dayofyear.values, dt.dayofweek.values
    cols = {}
    for k in range(2, K + 1):
        cols[f"fa_s{k}"] = np.sin(2 * np.pi * k * doy / 365.25); cols[f"fa_c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    for k in (2, 3):
        cols[f"fw_s{k}"] = np.sin(2 * np.pi * k * dow / 7); cols[f"fw_c{k}"] = np.cos(2 * np.pi * k * dow / 7)
    return pd.DataFrame(cols)


def seasonal_design(target_dates):
    dt = pd.to_datetime(pd.Series(target_dates).values); doy, dow = dt.dayofyear.values, dt.dayofweek.values
    X = [np.ones(len(dt))]
    for k in (1, 2, 3):
        X += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    X += [(dow == j).astype(float) for j in range(1, 7)]
    return np.column_stack(X)


def build(state, h):
    sd = pd.read_csv(f"{DATA}/{state}.csv")
    d = prepare(sd, h)                                        # calendar-aligned rows, targets built before complete-case filtering
    dates = pd.date_range(pd.to_datetime(sd.date).min() + pd.Timedelta(days=30), pd.to_datetime(sd.date).max())
    start = dates[len(dates) // 2]
    feats = list(FULL_FEATURES)
    if VARIANT in DROP:
        feats = [f for f in FULL_FEATURES if f not in DROP[VARIANT]]
    elif VARIANT == "target_cal":
        tc = calendar_cols(d.target_date); d = d.copy()
        for c in CAL: d[c] = tc[c].values
    elif VARIANT == "realtime":
        s2 = sd.copy(); s2["date"] = pd.to_datetime(s2.date)
        s2 = s2.set_index("date").sort_index().reindex(pd.date_range(s2.date.min(), s2.date.max(), freq="D"))
        s2.index.name = "date"; ef = engineer_features(s2.reset_index()).set_index("date")
        lagged = ef[LAG_FEATURES + OPS].shift(1)              # inputs observed through t-1 only
        d = d.copy()
        for c in LAG_FEATURES + OPS: d[c] = lagged[c].reindex(d.date).values
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=FULL_FEATURES).reset_index(drop=True)
    elif VARIANT.startswith("fourier"):
        fc = fourier_cols(d.date, KMAX); d = pd.concat([d.reset_index(drop=True), fc], axis=1)
    return d, feats, start


def feats_fourier(K):
    return list(FULL_FEATURES) + [f"fa_{t}{k}" for k in range(2, K + 1) for t in ("s", "c")] + [f"fw_{t}{k}" for k in (2, 3) for t in ("s", "c")]


def fitpred(model, setting, Xa, ya, Xb):
    m = model_for(model, setting); t0 = time.perf_counter(); m.fit(Xa, ya); t1 = time.perf_counter()
    p = m.predict(Xb); t2 = time.perf_counter()
    return p, t1 - t0, t2 - t1


def run(state, h):
    d, feats, start = build(state, h)
    dev = eligible_training(d, start); val = dev.iloc[-max(int(len(dev) * .15), 20):]
    tr = eligible_training(dev, val.date.iloc[0])
    resid = VARIANT == "resid"

    def target(rows, fit_rows):
        if not resid: return rows.target.to_numpy(), None
        S_fit = seasonal_design(fit_rows.target_date); beta = np.linalg.lstsq(S_fit, fit_rows.target.to_numpy(), rcond=None)[0]
        return rows.target.to_numpy() - seasonal_design(rows.target_date) @ beta, beta

    best, meta = {}, {}
    base_of = lambda m: m.split("_", 1)[1] if m.startswith("fourier_") else m
    plans = {}
    for m in MODELS:
        g = HGB_GRID if base_of(m) == "hgb" else RIDGE_ALPHA_GRID
        plans[m] = [(K, s) for K in range(1, KMAX + 1) for s in g] if m.startswith("fourier_") else [(None, s) for s in g]
    yt, _ = target(tr, tr); yv, _ = target(val, tr)
    for m, grid in plans.items():
        sc = []
        for K, s in grid:
            cols = feats_fourier(K) if m.startswith("fourier_") else feats
            p, _, _ = fitpred(base_of(m), s, tr[cols].to_numpy(), yt, val[cols].to_numpy()); sc.append(metrics(yv, p)["R2"])
        best[m] = grid[int(np.argmax(sc))]
        if m.startswith("fourier_"): meta[f"{m}_val_by_K"] = {K: float(max(sc[i] for i, (kk, _) in enumerate(grid) if kk == K)) for K in range(1, KMAX + 1)}
    frames, timing = [], {m: [0.0, 0, 0.0] for m in plans}
    for block in pd.date_range(start, d.date.max(), freq="14D"):
        test = d[(d.date >= block) & (d.date < block + pd.Timedelta(days=14))]
        if test.empty: continue
        train = eligible_training(d, block); assert train.target_date.max() <= block
        ya, beta = target(train, train)
        f = test[["date", "target_date", "target"]].copy()
        if resid:
            f["seasonal"] = seasonal_design(test.target_date) @ beta
        for m, (K, s) in best.items():
            cols = feats_fourier(K) if m.startswith("fourier_") else feats
            p, ft, pt = fitpred(base_of(m), s, train[cols].to_numpy(), ya, test[cols].to_numpy())
            f[m] = p + (f["seasonal"].to_numpy() if resid else 0)
            timing[m][0] += ft; timing[m][1] += 1; timing[m][2] += pt
        frames.append(f)
    p = pd.concat(frames, ignore_index=True); p.insert(0, "horizon", h); p.insert(0, "state", state)
    meta.update(best={k: [v[0], v[1]] for k, v in best.items()},
                timing={m: dict(fit_s_per_refit=v[0] / max(v[1], 1), n_fits=v[1], predict_s_total=v[2]) for m, v in timing.items()})
    return p, meta


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for s in STATES:
            for h in HS:
                fn = OUT / f"{s}_h{h}.csv"
                if fn.exists(): continue
                t = time.time()
                try: p, meta = run(s, h)
                except Exception as e:
                    print(f"SKIP {VARIANT} {s} h{h}: {e!r}", flush=True); continue
                p.to_csv(fn, index=False); (OUT / f"{s}_h{h}.json").write_text(json.dumps(meta, default=str))
                print(f"{VARIANT} {s} h{h} {time.time() - t:.1f}s", flush=True)
    print("DONE", flush=True)
