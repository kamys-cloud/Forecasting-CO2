"""Prophet baseline (Taylor & Letham 2018): additive piecewise-linear trend + annual (10 Fourier pairs) and weekly (3 pairs) seasonality
+ U.S. holidays, re-estimated at each 14-day refit date with data observed by that date. Prophet has no autoregressive state, so forecasts
for origins inside a block come from the fit at the block start (information through the block start only).
Usage: python prophet_baseline.py PRED_DIR OUT_DIR STATE [STATE ...]"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[k] = '1'
import sys, json, time, logging, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR); logging.getLogger("prophet").setLevel(logging.ERROR)
from prophet import Prophet
PRED, OUT = sys.argv[1:3]; STATES = sys.argv[3:]
from paths import STATE_DAILY as DATA
H = [1, 2, 3, 5, 7, 10, 14]
os.makedirs(OUT, exist_ok=True)
for s in STATES:
    if os.path.exists(f"{OUT}/{s}.csv"): continue
    y = pd.read_csv(f"{DATA}/{s}.csv", parse_dates=["date"])[["date", "co2_mass"]].rename(columns={"date": "ds", "co2_mass": "y"})
    ev = {h: pd.read_csv(f"{PRED}/{s}_h{h}.csv", parse_dates=["date", "target_date"])[["date", "target_date", "target"]] for h in H}
    origins = sorted(set().union(*[set(e.date) for e in ev.values()]))
    blocks = list(pd.date_range(min(origins), max(origins), freq="14D"))
    fc, tfit, tpred = {}, 0.0, 0.0
    for T in blocks:
        hist = y[y.ds <= T].dropna()
        t0 = time.perf_counter()
        m = Prophet(yearly_seasonality=10, weekly_seasonality=3, daily_seasonality=False)
        m.add_country_holidays(country_name="US"); m.fit(hist)
        t1 = time.perf_counter()
        fut = pd.DataFrame({"ds": pd.date_range(T + pd.Timedelta(days=1), T + pd.Timedelta(days=28))})
        p = m.predict(fut); tpred += time.perf_counter() - t1; tfit += t1 - t0
        for d, v in zip(p.ds, p.yhat): fc[(T, d)] = v
    rows = []
    for h in H:
        e = ev[h].copy()
        blk = [max(b for b in blocks if b <= t) for t in e.date]
        e["prophet"] = [fc.get((b, td), np.nan) for b, td in zip(blk, e.target_date)]
        e.insert(0, "horizon", h); e.insert(0, "state", s); rows.append(e)
    pd.concat(rows).to_csv(f"{OUT}/{s}.csv", index=False)
    json.dump(dict(fit_s_per_refit=tfit / len(blocks), predict_s_per_refit=tpred / len(blocks), n_fits=len(blocks)), open(f"{OUT}/{s}.json", "w"))
    print(s, f"{tfit:.0f}s", flush=True)
