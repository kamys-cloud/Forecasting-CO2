"""Operational experiments: the information available at issue on the benchmark protocol (prepare(), validation block, 14-day refits,
outcome cut-offs).
  All variants: CO2-history and operational inputs lagged one day (only data finalized by day t-1 when the forecast is issued at the
  end of day t), as the 'realtime' variant of run_exp2.
  ops     lagged 27 inputs
  ops_f   + annual and weekly Fourier terms (annual K as selected for the benchmark Fourier model of the same state, horizon, family)
  ops_fc  + Fourier + degree-day climatology of the target day of year (2019 to the first origin) + observed HDD/CDD of day t-1
  ops_fw  + Fourier + degree days of the target day: observed in training and validation ('perfect prog'), the calibrated GEFS
          ensemble-mean forecast issued on day t at prediction + observed HDD/CDD of day t-1
  ops_fo  + Fourier + observed target-day degree days at prediction as well (oracle; not available in advance) + day t-1 HDD/CDD
Usage: python operational_forecasts.py OUTDIR DATA_DIR VARIANT HORIZONS MODELS STATE [STATE ...]"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[k] = '1'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
OUT, DATA, VARIANT, HS, MODELS = sys.argv[1:6]; STATES = sys.argv[6:]
from paths import EXTERNAL, WEATHER, WORK
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from models import model_for, RIDGE_ALPHA_GRID, HGB_GRID
from engine import eligible_training, prepare, FULL_FEATURES, metrics
from engine import engineer_features, LAG_FEATURES, OPERATIONAL_FEATURES as OPS

HS = [int(x) for x in HS.split(",")]; MODELS = MODELS.split(",")
OUT = Path(OUT) / VARIANT; OUT.mkdir(parents=True, exist_ok=True)
V = os.path.dirname(os.path.abspath(__file__))
KMAX = 6
FOUR = VARIANT != "ops"
WX = VARIANT in ("ops_fc", "ops_fw", "ops_fo")
DD = pd.read_csv(f"{EXTERNAL}/cpc_degree_days.csv", parse_dates=["date"]) if WX else None
GF = pd.read_csv(f"{WEATHER}/gefs_dd_calibrated.csv", parse_dates=["date"]) if VARIANT == "ops_fw" else None
WXC = ["hdd_t", "cdd_t", "hdd_l1", "cdd_l1"]


def fa_cols(dates):
    doy = pd.to_datetime(pd.Series(dates).values).dayofyear.values
    return {f"fa_{t}{k}": (np.sin if t == "s" else np.cos)(2 * np.pi * k * doy / 365.25) for k in range(2, KMAX + 1) for t in ("s", "c")}


def fw_cols(dates):
    dow = pd.to_datetime(pd.Series(dates).values).dayofweek.values
    return {f"fw_{t}{k}": (np.sin if t == "s" else np.cos)(2 * np.pi * k * dow / 7) for k in (2, 3) for t in ("s", "c")}


def build(state, h):
    sd = pd.read_csv(f"{DATA}/{state}.csv")
    d = prepare(sd, h).reset_index(drop=True)
    dates = pd.date_range(pd.to_datetime(sd.date).min() + pd.Timedelta(days=30), pd.to_datetime(sd.date).max())
    start = dates[len(dates) // 2]
    s2 = sd.copy(); s2["date"] = pd.to_datetime(s2.date)
    s2 = s2.set_index("date").sort_index().reindex(pd.date_range(s2.date.min(), s2.date.max(), freq="D"))
    s2.index.name = "date"; ef = engineer_features(s2.reset_index()).set_index("date")
    lagged = ef[LAG_FEATURES + OPS].shift(1)                 # inputs observed through t-1 only
    for c in LAG_FEATURES + OPS: d[c] = lagged[c].reindex(pd.to_datetime(d.date)).values
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=FULL_FEATURES).reset_index(drop=True)
    extra = {}
    if FOUR: extra.update(fa_cols(d.date)); extra.update(fw_cols(d.date))
    nfill = 0
    if WX:
        dd = DD[DD.state == state].set_index("date")[["hdd", "cdd"]]
        od, td = pd.to_datetime(d.date), pd.to_datetime(d.target_date)
        extra["hdd_l1"] = dd.hdd.reindex(od - pd.Timedelta(days=1)).values; extra["cdd_l1"] = dd.cdd.reindex(od - pd.Timedelta(days=1)).values
        hist = dd[dd.index < start]
        clim = hist.groupby(hist.index.dayofyear).mean().reindex(range(1, 367)).interpolate()
        clim = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]]).rolling(15, center=True).mean().iloc[15:-15]; clim.index = range(1, 367)
        if VARIANT == "ops_fc":
            extra["hdd_t"] = clim.hdd.reindex(td.dt.dayofyear).values; extra["cdd_t"] = clim.cdd.reindex(td.dt.dayofyear).values
        else:
            extra["hdd_t"] = dd.hdd.reindex(td).values; extra["cdd_t"] = dd.cdd.reindex(td).values
            if VARIANT == "ops_fw":
                g = GF[(GF.state == state) & (GF.h == h)].set_index("date")
                fh = g.hdd_fc.reindex(od).values; fc_ = g.cdd_fc.reindex(od).values
                miss = np.isnan(fh) | np.isnan(fc_); nfill = int(np.sum(miss & (od >= start)))
                fh = np.where(miss, clim.hdd.reindex(td.dt.dayofyear).values, fh); fc_ = np.where(miss, clim.cdd.reindex(td.dt.dayofyear).values, fc_)
                extra["hdd_fc"] = fh; extra["cdd_fc"] = fc_
    if extra:
        d = pd.concat([d, pd.DataFrame(extra)], axis=1)
        need = [c for c in extra if c not in ("hdd_fc", "cdd_fc")]
        d = d.dropna(subset=need).reset_index(drop=True)
    return d, start, nfill


def KBENCH(state, h, m):
    h = {4: 5, 6: 7}.get(h, h)      # horizons outside the benchmark set use the order selected at the next benchmark horizon
    if m == "ridge": return int(json.load(open(f"{WORK}/exp/fourier/{state}_h{h}.json"))["best"]["fourier_ridge"][0])
    return int(json.load(open(f"{WORK}/exp/fourier_nl/{state}_h{h}.json"))["best"][f"fourier_{m}"][0])


def cols_for(K):
    c = list(FULL_FEATURES)
    if FOUR: c += [f"fa_{t}{k}" for k in range(2, K + 1) for t in ("s", "c")] + [f"fw_{t}{k}" for k in (2, 3) for t in ("s", "c")]
    if WX: c += WXC
    return c


def fitpred(model, setting, Xa, ya, Xb):
    m = model_for(model, setting); t0 = time.perf_counter(); m.fit(Xa, ya); t1 = time.perf_counter()
    p = m.predict(Xb); t2 = time.perf_counter()
    return p, t1 - t0, t2 - t1


def run(state, h):
    t_feat = time.perf_counter(); d, start, nfill = build(state, h); t_feat = time.perf_counter() - t_feat
    dev = eligible_training(d, start); val = dev.iloc[-max(int(len(dev) * .15), 20):]
    tr = eligible_training(dev, val.date.iloc[0])
    best, meta, t_tune = {}, {}, {}
    for m in MODELS:
        g = HGB_GRID if m == "hgb" else RIDGE_ALPHA_GRID
        # annual complexity K is taken from the benchmark Fourier models of the same state, horizon, and family (selected on the same
        # pre-origin validation block); the model's own hyperparameters are re-selected for this information set
        Ks = [KBENCH(state, h, m)] if FOUR else [None]
        grid = [(K, s) for K in Ks for s in g]
        t0 = time.perf_counter(); sc = []
        for K, s in grid:
            c = cols_for(K or 1)
            p, _, _ = fitpred(m, s, tr[c].to_numpy(), tr.target.to_numpy(), val[c].to_numpy()); sc.append(metrics(val.target.to_numpy(), p)["R2"])
        t_tune[m] = time.perf_counter() - t0; best[m] = grid[int(np.argmax(sc))]; meta[f"{m}_candidates"] = len(grid)
    frames, timing = [], {m: [0.0, 0, 0.0] for m in MODELS}
    for block in pd.date_range(start, d.date.max(), freq="14D"):
        test = d[(d.date >= block) & (d.date < block + pd.Timedelta(days=14))].copy()
        if test.empty: continue
        train = eligible_training(d, block); assert train.target_date.max() <= block
        if VARIANT == "ops_fw": test["hdd_t"] = test.hdd_fc.values; test["cdd_t"] = test.cdd_fc.values
        f = test[["date", "target_date", "target"]].copy()
        for m, (K, s) in best.items():
            c = cols_for(K or 1)
            p, ft, pt = fitpred(m, s, train[c].to_numpy(), train.target.to_numpy(), test[c].to_numpy())
            f[m] = p; timing[m][0] += ft; timing[m][1] += 1; timing[m][2] += pt
        frames.append(f)
    p = pd.concat(frames, ignore_index=True); p.insert(0, "horizon", h); p.insert(0, "state", state)
    meta.update(best={k: [v[0], v[1]] for k, v in best.items()}, feature_s=t_feat, tune_s=t_tune, forecast_filled=nfill,
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
