"""Daily balancing-authority series used by the scheduling analyses: net generation and demand (EIA-930), CAMPD CO2 under the
facility crosswalk and under the state proxy, EIA-estimated CO2 of generated electricity, population-weighted degree days (observed
and GEFS forecasts) over the states of the BA footprint."""
import os
import numpy as np, pandas as pd

from paths import ROOT
BAS = ["ERCO", "CISO", "NYIS", "ISNE"]
NAME = {"ERCO": "ERCOT", "CISO": "CAISO", "NYIS": "NYISO", "ISNE": "ISO-NE"}
PROXY_STATES = {"ERCO": ["TX"], "CISO": ["CA"], "NYIS": ["NY"], "ISNE": ["CT", "MA", "NH", "RI"]}
WEATHER_STATES = {"ERCO": ["TX"], "CISO": ["CA"], "NYIS": ["NY"], "ISNE": ["CT", "MA", "NH", "RI", "ME", "VT"]}
ST2KG = 907.18474                      # short ton -> kg


def path(*p): return os.path.join(ROOT, *p)


def state_population():
    C = pd.read_csv(path("data", "weather", "cenpop.txt"), encoding="utf-8-sig")
    from gefs_names import ABBR
    C = C[C.STNAME.isin(ABBR)]; C["st"] = C.STNAME.map(ABBR)
    return C.groupby("st").POPULATION.sum()


def netgen(ba):
    """EIA-930 daily net generation (MWh); non-positive days or days below 60% of the 29-day rolling median set missing."""
    B = pd.read_csv(path("data", "external", "eia930_daily.csv"), parse_dates=["date"])
    g = B[B.ba == ba].set_index("date").netgen.asfreq("D").astype(float)
    med = g.rolling(29, center=True, min_periods=5).median()
    return g.where((g > 0) & (g >= 0.6 * med))


def eia930(ba):
    B = pd.read_csv(path("data", "external", "eia930_daily.csv"), parse_dates=["date"])
    return B[B.ba == ba].set_index("date").asfreq("D")


def intensity(ba, kind="crosswalk"):
    """Daily average carbon intensity of generation, kg CO2/MWh.
    crosswalk: CAMPD CO2 of facilities assigned to the BA (EIA-860) / EIA-930 net generation
    proxy:     CAMPD CO2 of the mapped states / EIA-930 net generation
    eia:       EIA estimate of CO2 from generated electricity (Hourly Electric Grid Monitor) / EIA-930 net generation"""
    g = netgen(ba)
    if kind == "eia":
        G = pd.read_csv(path("data", "external", "gridmonitor_ba_daily.csv"), parse_dates=["date"])
        e = G[G.ba == ba].set_index("date").co2_generated_t.asfreq("D") * 1000.0
        return (e / g).replace([np.inf, -np.inf], np.nan)
    X = pd.read_csv(path("data", "derived", "ba_daily_co2.csv"), parse_dates=["date"])
    x = X[X.ba == ba].set_index("date")
    c = (x.co2_xw if kind == "crosswalk" else x.co2_proxy).asfreq("D") * ST2KG
    return (c / g).replace([np.inf, -np.inf], np.nan)


def ba_co2(ba, kind="crosswalk"):
    X = pd.read_csv(path("data", "derived", "ba_daily_co2.csv"), parse_dates=["date"])
    x = X[X.ba == ba].set_index("date")
    return (x.co2_xw if kind == "crosswalk" else x.co2_proxy).asfreq("D") * ST2KG


def degree_days(ba):
    """Population-weighted observed (CPC) degree days over the BA weather states."""
    pop = state_population(); sts = WEATHER_STATES[ba]; w = pop.reindex(sts) / pop.reindex(sts).sum()
    D = pd.read_csv(path("data", "external", "cpc_degree_days.csv"), parse_dates=["date"])
    D = D[D.state.isin(sts)].copy(); D["w"] = D.state.map(w)
    return D.assign(hdd=D.hdd * D.w, cdd=D.cdd * D.w).groupby("date")[["hdd", "cdd"]].sum(min_count=len(sts)).asfreq("D")


def gefs_degree_days(ba):
    """Population-weighted calibrated GEFS degree-day forecasts (issue date, lead h)."""
    pop = state_population(); sts = WEATHER_STATES[ba]; w = pop.reindex(sts) / pop.reindex(sts).sum()
    G = pd.read_csv(path("data", "weather", "gefs_dd_calibrated.csv"), parse_dates=["date", "target_date"])
    G = G[G.state.isin(sts)].copy(); G["w"] = G.state.map(w)
    G["hdd_fc"] *= G.w; G["cdd_fc"] *= G.w
    return G.groupby(["date", "h"])[["hdd_fc", "cdd_fc"]].sum(min_count=len(sts))
