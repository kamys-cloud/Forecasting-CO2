"""Classical seasonal baselines on the identical forecast origins/targets of the main benchmark.
Usage: python classical_baselines.py PRED_DIR OUT_DIR STATE [STATE ...]
  snaive7  : y(t+h-7*ceil(h/7))            (weekly seasonal naive; uses only data observed by t)
  snaive364: y(t+h-364)                     (annual seasonal naive, same weekday one year earlier)
  fets     : annual Fourier regression (K=3) + ETS(A,Ad,A) with weekly seasonality on the remainder;
             parameters re-estimated every 14 calendar days (same refit dates), states updated daily.
  sarimax  : regression with SARIMA(2,0,1)(1,0,1)_7 errors and annual Fourier terms (K=3);
             parameters re-estimated every 112 days (8 refit blocks), state updated daily.
"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[k] = '1'
import sys, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

PRED, OUT = sys.argv[1:3]; STATES = sys.argv[3:]
from paths import STATE_DAILY as DATA
H = [1, 2, 3, 5, 7, 10, 14]
os.makedirs(OUT, exist_ok=True)


def fourier(idx, K=3):
    doy = idx.dayofyear.values
    return np.column_stack([f(2 * np.pi * k * doy / 365.25) for k in range(1, K + 1) for f in (np.sin, np.cos)])


def run(state):
    sd = pd.read_csv(f"{DATA}/{state}.csv", parse_dates=["date"]).set_index("date").co2_mass
    cal = pd.date_range(sd.index.min(), sd.index.max() + pd.Timedelta(days=14), freq="D")
    y = sd.reindex(cal); scale = float(np.nanmean(y)); ys = y / scale
    evals = {h: pd.read_csv(f"{PRED}/{state}_h{h}.csv", parse_dates=["date", "target_date"])[["date", "target_date", "target"]] for h in H}
    origins = sorted(set().union(*[set(e.date) for e in evals.values()]))
    start = min(origins)
    blocks = list(pd.date_range(start, max(origins), freq="14D"))
    FX = pd.DataFrame(fourier(cal), index=cal)
    fc = {m: {} for m in ("fets", "sarimax")}
    timing = {"fets": [0.0, 0], "sarimax": [0.0, 0]}
    yff = ys.ffill()
    sres = None
    for bi, T in enumerate(blocks):
        orig_blk = [t for t in origins if T <= t < T + pd.Timedelta(days=14)]
        if not orig_blk: continue
        hist = ys.loc[:T]
        # ---------------- Fourier + ETS
        t0 = time.perf_counter()
        ok = hist.notna().values
        Xh = np.column_stack([np.ones(len(hist)), FX.loc[hist.index].values])
        beta = np.linalg.lstsq(Xh[ok], hist.values[ok], rcond=None)[0]
        ann = pd.Series(np.column_stack([np.ones(len(cal)), FX.values]) @ beta, index=cal)
        rem = (yff - ann)
        e = ExponentialSmoothing(rem.loc[:T].values, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7).fit()
        prm = dict(smoothing_level=e.params["smoothing_level"], smoothing_trend=e.params["smoothing_trend"],
                   smoothing_seasonal=e.params["smoothing_seasonal"], damping_trend=e.params["damping_trend"])
        timing["fets"][0] += time.perf_counter() - t0; timing["fets"][1] += 1
        for t in orig_blk:
            if t == T:
                f14 = e.forecast(14)
            else:
                e2 = ExponentialSmoothing(rem.loc[:t].values, trend="add", damped_trend=True, seasonal="add",
                                          seasonal_periods=7, initialization_method="known",
                                          initial_level=e.params["initial_level"], initial_trend=e.params["initial_trend"],
                                          initial_seasonal=e.params["initial_seasons"]).fit(**prm, optimized=False)
                f14 = e2.forecast(14)
            for h in H:
                tgt = t + pd.Timedelta(days=h)
                fc["fets"][(t, h)] = (f14[h - 1] + ann.loc[tgt]) * scale
        # ---------------- SARIMAX (re-estimate every 8 blocks; daily state update)
        if bi % 8 == 0 or sres is None:
            t0 = time.perf_counter()
            mod = SARIMAX(hist.values, exog=FX.loc[hist.index].values, order=(2, 0, 1), seasonal_order=(1, 0, 1, 7), trend="c")
            sres = mod.fit(disp=False, maxiter=200, start_params=None if bi == 0 else sparams)
            sparams = sres.params
            timing["sarimax"][0] += time.perf_counter() - t0; timing["sarimax"][1] += 1
            cur_end = T
        else:
            # bring the state up to T using the frozen parameters
            new = ys.loc[cur_end + pd.Timedelta(days=1):T]
            if len(new): sres = sres.extend(new.values, exog=FX.loc[new.index].values); cur_end = T
        r = sres
        for t in orig_blk:
            if t > cur_end:
                new = ys.loc[cur_end + pd.Timedelta(days=1):t]
                r = r.extend(new.values, exog=FX.loc[new.index].values) if len(new) else r
                sres, cur_end = r, t
            fut = pd.date_range(t + pd.Timedelta(days=1), periods=14)
            f14 = r.forecast(14, exog=FX.loc[fut].values)
            for h in H:
                fc["sarimax"][(t, h)] = f14[h - 1] * scale
    rows = []
    for h, e in evals.items():
        e = e.copy(); e.insert(0, "horizon", h); e.insert(0, "state", state)
        e["snaive7"] = [y.get(tt - pd.Timedelta(days=7 * int(np.ceil(h / 7))), np.nan) for tt in e.target_date]
        e["snaive364"] = [y.get(tt - pd.Timedelta(days=364), np.nan) for tt in e.target_date]
        # fall back to the last observed value when the seasonal lag falls on a missing day
        yff_raw = sd.reindex(cal).ffill()
        e["snaive7"] = e.snaive7.fillna(pd.Series([yff_raw.get(tt - pd.Timedelta(days=7 * int(np.ceil(h / 7)))) for tt in e.target_date]))
        e["snaive364"] = e.snaive364.fillna(pd.Series([yff_raw.get(tt - pd.Timedelta(days=364)) for tt in e.target_date]))
        e["fets"] = [fc["fets"][(t, h)] for t in e.date]; e["sarimax"] = [fc["sarimax"][(t, h)] for t in e.date]
        rows.append(e)
    out = pd.concat(rows, ignore_index=True)
    meta = {m: dict(fit_s_per_refit=v[0] / max(v[1], 1), n_fits=v[1]) for m, v in timing.items()}
    return out, meta


if __name__ == "__main__":
    for s in STATES:
        fn = f"{OUT}/{s}.csv"
        if os.path.exists(fn): continue
        t = time.time(); out, meta = run(s)
        out.to_csv(fn, index=False); json.dump(meta, open(f"{OUT}/{s}.json", "w"))
        print(f"{s} {time.time() - t:.0f}s", flush=True)
    print("DONE", flush=True)
