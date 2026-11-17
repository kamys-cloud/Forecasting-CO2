"""Load day-by-day forecasts (forecasts/*.csv.gz) into aligned per-horizon panels."""
import os
import numpy as np, pandas as pd

from paths import ROOT
H = [1, 2, 3, 5, 7, 10, 14]
_CACHE = {}


def load(name, rename=None):
    if name not in _CACHE:
        _CACHE[name] = pd.read_csv(os.path.join(ROOT, "forecasts", f"{name}.csv.gz"), parse_dates=["date", "target_date"])
    d = _CACHE[name]
    return d.rename(columns=rename) if rename else d


BENCH = {"baseline": None, "fourier_ridge": None, "fourier_nonlinear": None, "classical": None, "prophet": None}


def benchmark(h):
    """identical rows of persistence, ridge, GAM, HGB, Fourier ridge/GAM/HGB, classical baselines, Prophet."""
    b = load("baseline"); b = b[b.horizon == h][["state", "date", "target_date", "target", "persistence", "ridge", "gam", "hgb"]]
    for nm, cols in (("fourier_ridge", ["fourier_ridge"]), ("fourier_nonlinear", ["fourier_gam", "fourier_hgb"]),
                     ("classical", ["snaive7", "snaive364", "fets", "sarimax"]), ("prophet", ["prophet"])):
        x = load(nm); x = x[x.horizon == h][["state", "date"] + cols]
        b = b.merge(x, on=["state", "date"], how="left")
    return b


def operational(h):
    out = None
    for nm, pre in (("operational_lagged", "ops_"), ("operational_fourier", "opsf_"), ("operational_fourier_gefs", "opsfw_"),
                    ("operational_fourier_climatology", "opsfc_"), ("operational_fourier_observed_weather", "opsfo_")):
        x = load(nm); x = x[x.horizon == h]
        if x.empty: continue
        x = x[["state", "date", "target_date", "target", "ridge", "gam", "hgb"]].rename(columns={m: pre + m for m in ("ridge", "gam", "hgb")})
        out = x if out is None else out.merge(x.drop(columns=["target_date", "target"]), on=["state", "date"], how="left")
    return out


def frames(df):
    return [g.reset_index(drop=True) for _, g in df.groupby("state", sort=True)]
