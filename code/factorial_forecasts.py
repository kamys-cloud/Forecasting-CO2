"""Seasonal-representation factorial and related designs on the benchmark protocol (prepare(), validation block, 14-day refits, cut-offs).
Usage: python factorial_forecasts.py OUTDIR DATA_DIR VARIANT HORIZONS MODELS STATE [STATE ...]
VARIANT
  fa      original 27 + annual Fourier harmonics k=2..K of the origin day of year (K in 1..6, tuned jointly)
  fw      original 27 + weekly Fourier harmonics k=2,3 (fixed)
  cs      original 27 + periodic cubic B-spline basis of the origin day of year (n_knots in {0,4,6,8,10,12}, tuned jointly; 0 = none)
  faw     original 27 + annual (K tuned) + weekly harmonics  (the Fourier models; used with the expanded HGB search)
  cum7 / cum14   target = sum of y(t+1..t+H) (next-week / next-two-weeks total), outcome cutoff t+H; models ridge,gam,hgb,
                 or fourier_ridge (annual K tuned + weekly)
  clim    original 27 + degree-day climatology for the target day of year (expected HDD, CDD from 2019 to the first origin)
          + observed HDD, CDD on the origin day (both known when the forecast is issued)
  oracle  original 27 + observed HDD, CDD on the target day (NOT available in advance; upper bound for perfect weather forecasts)
MODELS: ridge,gam,hgb (fourier_ridge for cum*). With 'hgbx' the expanded 60-setting random HGB search is used.
"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[k] = '1'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
OUT, DATA, VARIANT, HS, MODELS = sys.argv[1:6]; STATES = sys.argv[6:]
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.preprocessing import SplineTransformer
from models import model_for, RIDGE_ALPHA_GRID, HGB_GRID
from engine import eligible_training, prepare, FULL_FEATURES, metrics

HS = [int(x) for x in HS.split(",")]; MODELS = MODELS.split(",")
OUT = Path(OUT) / VARIANT; OUT.mkdir(parents=True, exist_ok=True)
V = os.path.dirname(os.path.abspath(__file__))
from paths import EXTERNAL
KMAX = 6
CS_KNOTS = [0, 4, 6, 8, 10, 12]
rng = np.random.default_rng(42)          # identical construction to the expanded search of regen.py
EXP_HGB = [{"learning_rate": float(rng.uniform(0.01, 0.3)), "max_depth": int(rng.integers(2, 7)),
            "max_iter": int(rng.integers(40, 300)), "min_samples_leaf": int(rng.integers(5, 60))} for _ in range(60)]
DD = None
if VARIANT in ("clim", "oracle"):
    DD = pd.read_csv(f"{EXTERNAL}/cpc_degree_days.csv", parse_dates=["date"])    # state, date, hdd, cdd (population-weighted)


def fa_cols(dates, K):
    doy = pd.to_datetime(pd.Series(dates).values).dayofyear.values
    return {f"fa_{t}{k}": (np.sin if t == "s" else np.cos)(2 * np.pi * k * doy / 365.25) for k in range(2, K + 1) for t in ("s", "c")}


def fw_cols(dates):
    dow = pd.to_datetime(pd.Series(dates).values).dayofweek.values
    return {f"fw_{t}{k}": (np.sin if t == "s" else np.cos)(2 * np.pi * k * dow / 7) for k in (2, 3) for t in ("s", "c")}


def cs_cols(dates, n):
    if n == 0: return {}
    doy = pd.to_datetime(pd.Series(dates).values).dayofyear.values.astype(float).reshape(-1, 1)
    st = SplineTransformer(n_knots=n + 1, degree=3, knots=np.linspace(0.5, 366.5, n + 1).reshape(-1, 1), extrapolation="periodic")
    B = st.fit_transform(doy)
    return {f"cs{n}_{j}": B[:, j] for j in range(B.shape[1])}


def build(state, h):
    sd = pd.read_csv(f"{DATA}/{state}.csv")
    hh = {"cum7": 7, "cum14": 14}.get(VARIANT, h)
    d = prepare(sd, hh).reset_index(drop=True)
    dates = pd.date_range(pd.to_datetime(sd.date).min() + pd.Timedelta(days=30), pd.to_datetime(sd.date).max())
    start = dates[len(dates) // 2]
    extra = {}
    if VARIANT in ("cum7", "cum14"):
        y = pd.to_datetime(sd.date); s_ = pd.Series(sd.co2_mass.values, index=y).reindex(pd.date_range(y.min(), y.max()))
        fw = sum(s_.shift(-j) for j in range(1, hh + 1))                   # NaN if any day missing
        d["target"] = fw.reindex(pd.to_datetime(d.date)).values
        d = d.dropna(subset=["target"]).reset_index(drop=True)
    if VARIANT in ("fa", "faw", "cum7", "cum14"):
        for K in range(2, KMAX + 1): pass
        extra.update(fa_cols(d.date, KMAX))
    if VARIANT in ("fw", "faw", "cum7", "cum14"): extra.update(fw_cols(d.date))
    if VARIANT == "cs":
        for n in CS_KNOTS: extra.update(cs_cols(d.date, n))
    if VARIANT in ("clim", "oracle"):
        dd = DD[DD.state == state].set_index("date")[["hdd", "cdd"]]
        if VARIANT == "clim":
            hist = dd[dd.index < start]
            clim = hist.groupby(hist.index.dayofyear).mean().reindex(range(1, 367)).interpolate()
            clim = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]]).rolling(15, center=True).mean().iloc[15:-15]
            clim.index = range(1, 367)
            tdoy = pd.to_datetime(d.target_date).dt.dayofyear.values
            extra["hdd_clim_target"] = clim.hdd.reindex(tdoy).values; extra["cdd_clim_target"] = clim.cdd.reindex(tdoy).values
            extra["hdd_origin"] = dd.hdd.reindex(pd.to_datetime(d.date)).values; extra["cdd_origin"] = dd.cdd.reindex(pd.to_datetime(d.date)).values
        else:
            extra["hdd_target"] = dd.hdd.reindex(pd.to_datetime(d.target_date)).values; extra["cdd_target"] = dd.cdd.reindex(pd.to_datetime(d.target_date)).values
    if extra:
        d = pd.concat([d, pd.DataFrame(extra)], axis=1)
        d = d.dropna(subset=list(extra)).reset_index(drop=True)
    return d, start


def cols_for(m, K):
    base = list(FULL_FEATURES)
    if VARIANT == "fa": return base + [f"fa_{t}{k}" for k in range(2, K + 1) for t in ("s", "c")]
    if VARIANT == "fw": return base + [f"fw_{t}{k}" for k in (2, 3) for t in ("s", "c")]
    if VARIANT == "faw" or (VARIANT.startswith("cum") and m.startswith("fourier_")):
        return base + [f"fa_{t}{k}" for k in range(2, K + 1) for t in ("s", "c")] + [f"fw_{t}{k}" for k in (2, 3) for t in ("s", "c")]
    if VARIANT == "cs": return base + ([f"cs{K}_{j}" for j in range(K)] if K else [])
    if VARIANT == "clim": return base + ["hdd_clim_target", "cdd_clim_target", "hdd_origin", "cdd_origin"]
    if VARIANT == "oracle": return base + ["hdd_target", "cdd_target"]
    return base


def fitpred(model, setting, Xa, ya, Xb):
    m = model_for(model, setting); t0 = time.perf_counter(); m.fit(Xa, ya); t1 = time.perf_counter()
    p = m.predict(Xb); t2 = time.perf_counter()
    return p, t1 - t0, t2 - t1


def base_of(m): return {"fourier_ridge": "ridge", "hgbx": "hgb"}.get(m, m)


def run(state, h):
    t_feat = time.perf_counter(); d, start = build(state, h); t_feat = time.perf_counter() - t_feat
    dev = eligible_training(d, start); val = dev.iloc[-max(int(len(dev) * .15), 20):]
    tr = eligible_training(dev, val.date.iloc[0])
    tuneK = VARIANT in ("fa", "faw") or (VARIANT.startswith("cum"))
    Ks = CS_KNOTS if VARIANT == "cs" else (list(range(1, KMAX + 1)) if VARIANT in ("fa", "faw") else [None])
    best, meta, t_tune = {}, {}, {}
    for m in MODELS:
        g = EXP_HGB if m == "hgbx" else (HGB_GRID if base_of(m) == "hgb" else RIDGE_ALPHA_GRID)
        KK = list(range(1, KMAX + 1)) if (VARIANT.startswith("cum") and m.startswith("fourier_")) else ([None] if VARIANT.startswith("cum") else Ks)
        grid = [(K, s) for K in KK for s in g]
        t0 = time.perf_counter(); sc = []
        for K, s in grid:
            c = cols_for(m, K)
            p, _, _ = fitpred(base_of(m), s, tr[c].to_numpy(), tr.target.to_numpy(), val[c].to_numpy()); sc.append(metrics(val.target.to_numpy(), p)["R2"])
        t_tune[m] = time.perf_counter() - t0
        best[m] = grid[int(np.argmax(sc))]
        if len(KK) > 1: meta[f"{m}_val_by_K"] = {str(K): float(max(sc[i] for i, (kk, _) in enumerate(grid) if kk == K)) for K in KK}
        meta[f"{m}_candidates"] = len(grid)
    frames, timing = [], {m: [0.0, 0, 0.0] for m in MODELS}
    for block in pd.date_range(start, d.date.max(), freq="14D"):
        test = d[(d.date >= block) & (d.date < block + pd.Timedelta(days=14))]
        if test.empty: continue
        train = eligible_training(d, block); assert train.target_date.max() <= block
        f = test[["date", "target_date", "target"]].copy()
        for m, (K, s) in best.items():
            c = cols_for(m, K)
            p, ft, pt = fitpred(base_of(m), s, train[c].to_numpy(), train.target.to_numpy(), test[c].to_numpy())
            f[m] = p; timing[m][0] += ft; timing[m][1] += 1; timing[m][2] += pt
        frames.append(f)
    p = pd.concat(frames, ignore_index=True); p.insert(0, "horizon", h); p.insert(0, "state", state)
    if VARIANT.startswith("cum"):
        hh = int(VARIANT[3:]); p["persistence"] = None
        x = d.set_index("date").co2_mass if "co2_mass" in d.columns else None
        if x is not None: p["persistence"] = hh * x.reindex(p.date).values
    meta.update(best={k: [v[0], v[1]] for k, v in best.items()}, feature_s=t_feat, tune_s=t_tune,
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
                    import traceback; traceback.print_exc()
                    print(f"SKIP {VARIANT} {s} h{h}: {e!r}", flush=True); continue
                p.to_csv(fn, index=False); (OUT / f"{s}_h{h}.json").write_text(json.dumps(meta, default=str))
                print(f"{VARIANT} {s} h{h} {time.time() - t:.1f}s", flush=True)
    print("DONE", flush=True)
