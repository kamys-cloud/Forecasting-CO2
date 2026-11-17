"""Direct forecasts of daily average carbon intensity for the scheduling analyses.

For issue day t (end of day; data finalized through t-1) and h = 1..7, intensity I(t+h) is forecast by ridge, GAM and HGB under the
operational protocol of the state experiments: hyperparameters (and, for ridge, the annual Fourier order K) are selected once on the
last 15% of eligible rows before the first origin, and models are refitted every 14 days on rows whose target date is on or before the
refit date. Inputs mirror the 27 state predictors: the eight calendar encodings of the origin; intensity history (lags 1, 2, 3, 7, 14,
30 days from t-1; trailing 7-, 14-, 30-day means and standard deviations); operational inputs of day t-1 (net generation, demand, and
fossil, wind-and-solar, hydro-and-nuclear shares of generation). Variants: orig; four (+ annual harmonics 2..K and weekly harmonics 2-3;
K tuned for ridge and reused by GAM and HGB); four_gefs (+ population-weighted degree days of t-1 and of the target day: observed CPC
values in training and validation, calibrated GEFS forecasts issued on day t at prediction).
Usage: python scheduling_forecasts.py KIND OUT_CSV_GZ [BA,...]     KIND = crosswalk | eia | proxy"""
import os
for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[_k] = "1"
import sys, json, time
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from models import model_for, RIDGE_ALPHA_GRID, HGB_GRID
import ba_data as bd

KIND, OUT = sys.argv[1], sys.argv[2]
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else bd.BAS
HS = list(range(1, 8)); ORIG = pd.Timestamp("2022-07-17")


def features(ba, I):
    b = bd.eia930(ba); g = bd.netgen(ba); dd = bd.degree_days(ba); gf = bd.gefs_degree_days(ba)
    Il = I.asfreq("D").shift(1)
    base = pd.DataFrame(index=I.asfreq("D").index)
    for L in (0, 1, 2, 6, 13, 29): base[f"i_l{L + 1}"] = Il.shift(L)
    for k in (7, 14, 30): base[f"i_m{k}"] = Il.rolling(k).mean(); base[f"i_s{k}"] = Il.rolling(k).std()
    base["g_l1"] = g.shift(1); base["d_l1"] = b.demand.shift(1)
    base["fos_l1"] = ((b.coal + b.gas + b.oil) / g).shift(1); base["vre_l1"] = ((b.wind + b.solar) / g).shift(1)
    base["hn_l1"] = ((b.hydro + b.nuclear) / g).shift(1)
    t = base.index
    base["month_sin"] = np.sin(2 * np.pi * t.month / 12); base["month_cos"] = np.cos(2 * np.pi * t.month / 12)
    base["dow_sin"] = np.sin(2 * np.pi * t.dayofweek / 7); base["dow_cos"] = np.cos(2 * np.pi * t.dayofweek / 7)
    base["doy_sin"] = np.sin(2 * np.pi * t.dayofyear / 365.25); base["doy_cos"] = np.cos(2 * np.pi * t.dayofyear / 365.25)
    base["is_weekend"] = (t.dayofweek >= 5).astype(float); base["year"] = t.year.astype(float)
    feats = list(base.columns)
    for k in range(2, 7):
        base[f"fa_s{k}"] = np.sin(2 * np.pi * k * t.dayofyear / 365.25); base[f"fa_c{k}"] = np.cos(2 * np.pi * k * t.dayofyear / 365.25)
    for k in (2, 3):
        base[f"fw_s{k}"] = np.sin(2 * np.pi * k * t.dayofweek / 7); base[f"fw_c{k}"] = np.cos(2 * np.pi * k * t.dayofweek / 7)
    base["hdd_l1"] = dd.hdd.shift(1).reindex(t).values; base["cdd_l1"] = dd.cdd.shift(1).reindex(t).values
    return base, feats, dd, gf


def forecast(ba, I, variant, fam, Kcache):
    base, feats, dd, gf = features(ba, I); t = base.index; out = {}; meta = {}
    for h in HS:
        d = base.copy(); td = t + pd.Timedelta(days=h); d["td"] = td; d["y"] = I.reindex(td).values
        d["hdd_t"] = dd.hdd.reindex(td).values; d["cdd_t"] = dd.cdd.reindex(td).values
        fh = gf.xs(h, level="h").reindex(t); d["hdd_fc"] = fh.hdd_fc.values; d["cdd_fc"] = fh.cdd_fc.values
        need = feats + ["y", "hdd_l1", "cdd_l1", "hdd_t", "cdd_t"]
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=need)
        d = d[d.index >= t[0] + pd.Timedelta(days=45)]

        def cols(K):
            c = list(feats)
            if variant != "orig":
                c += [f"fa_{q}{k}" for k in range(2, K + 1) for q in ("s", "c")] + [f"fw_{q}{k}" for k in (2, 3) for q in ("s", "c")]
            if variant == "four_gefs": c += ["hdd_l1", "cdd_l1", "hdd_t", "cdd_t"]
            return c
        dev = d[d.td <= ORIG]; nv = max(int(len(dev) * .15), 20); va = dev.iloc[-nv:]; tr = dev[dev.td <= va.index[0]]
        Ks = ([Kcache[(variant, h)]] if fam != "ridge" else list(range(1, 7))) if variant != "orig" else [1]
        grid = [(K, s_) for K in Ks for s_ in (HGB_GRID if fam == "hgb" else RIDGE_ALPHA_GRID)]
        sc = []
        for K, s_ in grid:
            c = cols(K); m = model_for(fam, s_).fit(tr[c].to_numpy(), tr.y.to_numpy())
            sc.append(-np.mean((m.predict(va[c].to_numpy()) - va.y.to_numpy()) ** 2))
        K, s_ = grid[int(np.argmax(sc))]; c = cols(K)
        if fam == "ridge": Kcache[(variant, h)] = K
        meta[h] = dict(K=K, setting=s_)
        preds = []
        for blk in pd.date_range(ORIG, d.index.max(), freq="14D"):
            te = d[(d.index >= blk) & (d.index < blk + pd.Timedelta(days=14))].copy()
            if te.empty: continue
            trn = d[d.td <= blk]
            if variant == "four_gefs": te["hdd_t"] = te.hdd_fc.fillna(te.hdd_t); te["cdd_t"] = te.cdd_fc.fillna(te.cdd_t)
            m = model_for(fam, s_).fit(trn[c].to_numpy(), trn.y.to_numpy())
            preds.append(pd.DataFrame(dict(date=te.index, target_date=te.td.values, forecast=m.predict(te[c].to_numpy()))))
        f = pd.concat(preds); f["horizon"] = h; out[h] = f
    return pd.concat(out.values()), meta


if __name__ == "__main__":
    rows, metas = [], {}
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT, parse_dates=["date", "target_date"]); rows.append(prev)
        done = set(map(tuple, prev[["balancing_authority", "variant", "model"]].drop_duplicates().values))
    else:
        done = set()
    mfile = OUT.replace(".csv.gz", "_settings.json")
    if os.path.exists(mfile): metas = json.load(open(mfile))
    with threadpool_limits(limits=1):
        for ba in ONLY:
            I = bd.intensity(ba, KIND); Kc = {}
            for k_, K in metas.get(f"{ba}|K", {}).items():
                v, h = k_.split(":"); Kc[(v, int(h))] = K
            for variant in ("orig", "four", "four_gefs"):
                for fam in ("ridge", "gam", "hgb"):
                    if (ba, variant, fam) in done: continue
                    t0 = time.time(); f, meta = forecast(ba, I, variant, fam, Kc)
                    f.insert(0, "model", fam); f.insert(0, "variant", variant); f.insert(0, "balancing_authority", ba)
                    rows.append(f); metas[f"{ba}|{variant}|{fam}"] = {str(k): v for k, v in meta.items()}
                    metas[f"{ba}|K"] = {f"{v}:{h}": K for (v, h), K in Kc.items()}
                    pd.concat(rows).to_csv(OUT, index=False); json.dump(metas, open(mfile, "w"), indent=1, default=str)
                    print(KIND, ba, variant, fam, f"{time.time() - t0:.0f}s", flush=True)
