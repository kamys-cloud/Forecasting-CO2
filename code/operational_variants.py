"""Operational forecasts with GEFS degree-day forecasts: perfect-prognosis versus forecast-trained models, ensemble spread, and a
two-day lag of observed degree days.

All variants use the operational information set (CO2-history and operating inputs observed through t-1), the annual and weekly
Fourier terms (annual order K as selected for the benchmark Fourier model of the same state, horizon and family), and degree days
of day t-1 (observed) and of the target day t+h. They differ in the target-day degree days used for training/validation and in the
training window:
  pp      perfect prognosis: observed target-day degree days in training and validation, GEFS forecast at prediction (full window)
  pp_r    as pp, training restricted to origins from 2021-07-10 (first archived GEFS issue used here)
  ft_r    forecast-trained: GEFS forecast issued at each training origin in training, validation and prediction (origins >= 2021-07-10)
  ft_rs   as ft_r plus the GEFS ensemble spread of 2-m temperature for the target day
  pp_l2   as pp, with observed degree days of day t-2 instead of t-1 (robustness to CPC publication delay)
Hyperparameters are re-selected on the last 15% of eligible rows before the first origin (2022-07-17); 14-day refits.
Usage: python operational_variants.py OUT_DIR VARIANT HORIZONS MODELS STATE [STATE ...]"""
import os
for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[_k] = "1"
import sys, json, time, glob
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from engine import prepare, eligible_training, engineer_features, first_origin, r2, FULL_FEATURES, LAG_FEATURES, OPERATIONAL_FEATURES
from models import model_for, RIDGE_ALPHA_GRID, HGB_GRID

from paths import ROOT, WORK
OUT, VARIANT, HS, MODELS = sys.argv[1:5]; STATES = sys.argv[5:]
HS = [int(x) for x in HS.split(",")]; MODELS = MODELS.split(",")
OUTV = os.path.join(OUT, VARIANT); os.makedirs(OUTV, exist_ok=True)
GEFS_START = pd.Timestamp("2021-07-10")
DD = pd.read_csv(os.path.join(ROOT, "data", "external", "cpc_degree_days.csv"), parse_dates=["date"])
GF = pd.read_csv(os.path.join(ROOT, "data", "weather", "gefs_dd_calibrated.csv"), parse_dates=["date"])
SP = pd.read_csv(os.path.join(ROOT, "data", "weather", "gefs_spread.csv.gz"), parse_dates=["date"]) if VARIANT == "ft_rs" else None
SELF = os.path.join(ROOT, "forecasts", "selected_settings.csv")
SEL = pd.read_csv(SELF) if os.path.exists(SELF) else None


def K_bench(state, h, m):
    """Annual Fourier order selected for the benchmark Fourier model of the same state, horizon and family: from the benchmark fits in
    WORK/exp (as in operational_forecasts.py), or from forecasts/selected_settings.csv when those fits are not present."""
    key = ("fourier", "fourier_ridge") if m == "ridge" else ("fourier_nl", f"fourier_{m}")
    fn = os.path.join(WORK, "exp", key[0], f"{state}_h{h}.json")
    if os.path.exists(fn): return int(json.load(open(fn))["best"][key[1]][0])
    r = SEL[(SEL.state == state) & (SEL.horizon == h) & (SEL.design == key[0]) & (SEL.model == key[1])]
    return int(json.loads(r.setting.iloc[0])[0])


def build(state, h):
    sd = pd.read_csv(os.path.join(ROOT, "data", "state_daily", f"{state}.csv"))
    d = prepare(sd, h).reset_index(drop=True); start = first_origin(sd)
    s2 = sd.copy(); s2["date"] = pd.to_datetime(s2.date)
    s2 = s2.set_index("date").sort_index().reindex(pd.date_range(s2.date.min(), s2.date.max(), freq="D")); s2.index.name = "date"
    ef = engineer_features(s2.reset_index()).set_index("date")
    lagged = ef[LAG_FEATURES + OPERATIONAL_FEATURES].shift(1)
    for c in LAG_FEATURES + OPERATIONAL_FEATURES: d[c] = lagged[c].reindex(pd.to_datetime(d.date)).values
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=FULL_FEATURES).reset_index(drop=True)
    od, td = pd.to_datetime(d.date), pd.to_datetime(d.target_date)
    doy, dow = td.dt.dayofyear.values, td.dt.dayofweek.values
    od_doy, od_dow = od.dt.dayofyear.values, od.dt.dayofweek.values
    ex = {}
    for k in range(2, 7):
        ex[f"fa_s{k}"] = np.sin(2 * np.pi * k * od_doy / 365.25); ex[f"fa_c{k}"] = np.cos(2 * np.pi * k * od_doy / 365.25)
    for k in (2, 3):
        ex[f"fw_s{k}"] = np.sin(2 * np.pi * k * od_dow / 7); ex[f"fw_c{k}"] = np.cos(2 * np.pi * k * od_dow / 7)
    dd = DD[DD.state == state].set_index("date")[["hdd", "cdd"]]
    lag = 2 if VARIANT == "pp_l2" else 1
    ex["hdd_l1"] = dd.hdd.reindex(od - pd.Timedelta(days=lag)).values; ex["cdd_l1"] = dd.cdd.reindex(od - pd.Timedelta(days=lag)).values
    ex["hdd_obs"] = dd.hdd.reindex(td).values; ex["cdd_obs"] = dd.cdd.reindex(td).values
    g = GF[(GF.state == state) & (GF.h == h)].set_index("date")
    ex["hdd_fc"] = g.hdd_fc.reindex(od).values; ex["cdd_fc"] = g.cdd_fc.reindex(od).values
    if SP is not None:
        sp = SP[(SP.state == state) & (SP.h == h)].set_index("date"); ex["tsd"] = sp.tsd_f.reindex(od).values
    d = pd.concat([d, pd.DataFrame(ex)], axis=1)
    need = ["hdd_l1", "cdd_l1", "hdd_obs", "cdd_obs"]
    if VARIANT in ("pp_r", "ft_r", "ft_rs"):
        d = d[od >= GEFS_START]; need += ["hdd_fc", "cdd_fc"] + (["tsd"] if VARIANT == "ft_rs" else [])
    d = d.dropna(subset=need).reset_index(drop=True)
    if VARIANT.startswith("ft"):
        d["hdd_t"] = d.hdd_fc; d["cdd_t"] = d.cdd_fc
    else:
        d["hdd_t"] = d.hdd_obs; d["cdd_t"] = d.cdd_obs
    return d, start


def cols_for(K):
    c = list(FULL_FEATURES) + [f"fa_{t}{k}" for k in range(2, K + 1) for t in ("s", "c")] + [f"fw_{t}{k}" for k in (2, 3) for t in ("s", "c")]
    c += ["hdd_t", "cdd_t", "hdd_l1", "cdd_l1"]
    if VARIANT == "ft_rs": c += ["tsd"]
    return c


def run(state, h):
    d, start = build(state, h)
    dev = eligible_training(d, start); val = dev.iloc[-max(int(len(dev) * .15), 20):]; tr = eligible_training(dev, val.date.iloc[0])
    best = {}
    for m in MODELS:
        K = K_bench(state, h, m); c = cols_for(K); sc = []
        grid = HGB_GRID if m == "hgb" else RIDGE_ALPHA_GRID
        for s in grid:
            mod = model_for(m, s).fit(tr[c].to_numpy(), tr.target.to_numpy()); sc.append(r2(val.target, mod.predict(val[c].to_numpy())))
        best[m] = (K, grid[int(np.argmax(sc))])
    frames = []
    for block in pd.date_range(start, d.date.max(), freq="14D"):
        test = d[(d.date >= block) & (d.date < block + pd.Timedelta(days=14))].copy()
        if test.empty: continue
        train = eligible_training(d, block)
        test["hdd_t"] = test.hdd_fc.values; test["cdd_t"] = test.cdd_fc.values
        f = test[["date", "target_date", "target"]].copy()
        for m, (K, s) in best.items():
            c = cols_for(K); f[m] = model_for(m, s).fit(train[c].to_numpy(), train.target.to_numpy()).predict(test[c].to_numpy())
        frames.append(f)
    p = pd.concat(frames, ignore_index=True); p.insert(0, "horizon", h); p.insert(0, "state", state)
    return p, dict(best={k: [v[0], v[1]] for k, v in best.items()}, n_train_first=int(len(eligible_training(d, start))))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        for s in STATES:
            for h in HS:
                fn = os.path.join(OUTV, f"{s}_h{h}.csv")
                if os.path.exists(fn): continue
                t = time.time(); p, meta = run(s, h)
                p.to_csv(fn, index=False); json.dump(meta, open(fn.replace(".csv", ".json"), "w"), default=str)
                print(VARIANT, s, h, f"{time.time() - t:.1f}s", flush=True)
