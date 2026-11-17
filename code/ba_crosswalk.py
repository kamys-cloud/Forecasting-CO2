"""Facility-to-balancing-authority crosswalk for the CAMPD facility-day records and the resulting daily BA emissions.

Each CAMPD facility (Facility ID = EIA plant code, ORIS) is assigned the balancing-authority code reported for that plant in
EIA-860 Schedule 2 (Plant) of the same calendar year; if the plant is absent from that year's file the nearest available year is
used, and the assignment method is recorded. Daily BA emissions are the sums over facilities assigned to the BA.

Outputs (OUT_DIR):
  facility_ba_crosswalk.csv  facility x year: state, BA code, source year, match method, CO2 (short tons) in that year
  ba_daily_co2.csv           daily CO2 (short tons) by BA for the crosswalk definition and for the state-proxy definition
  boundary_summary.csv       boundary error of the state proxy by BA and year
Usage: python ba_crosswalk.py CAMPD_DIR EIA860_DIR OUT_DIR"""
import sys, glob, os
import numpy as np, pandas as pd

campd, e860, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
PROXY = {"ERCO": ["TX"], "CISO": ["CA"], "NYIS": ["NY"], "ISNE": ["CT", "MA", "NH", "RI"]}
YEARS = range(2019, 2026)

# ---------------------------------------------------------------- CAMPD facility-days (2019-2025, duplicates removed)
fr = []
for fn in sorted(glob.glob(os.path.join(campd, "daily-emissions-facility-aggregation-*.csv"))):
    d = pd.read_csv(fn, usecols=["State", "Facility ID", "Date", "CO2 Mass (short tons)", "Heat Input (mmBtu)", "Gross Load (MWh)"])
    fr.append(d)
F = pd.concat(fr, ignore_index=True).rename(columns={"State": "state", "Facility ID": "fac", "Date": "date", "Gross Load (MWh)": "gload",
                                                      "CO2 Mass (short tons)": "co2", "Heat Input (mmBtu)": "heat"})
F["date"] = pd.to_datetime(F.date)
F = F[(F.date >= "2019-01-01") & (F.date <= "2025-12-31")]
F = F.drop_duplicates(["state", "fac", "date"]).copy()
F["year"] = F.date.dt.year

# ---------------------------------------------------------------- EIA-860 plant -> BA by year
P = []
for y in YEARS:
    p = pd.read_excel(os.path.join(e860, f"2___Plant_Y{y}.xlsx"), skiprows=1,
                      usecols=["Plant Code", "State", "Balancing Authority Code", "Latitude", "Longitude"])
    p = p.rename(columns={"Plant Code": "fac", "State": "state860", "Balancing Authority Code": "ba", "Latitude": "lat", "Longitude": "lon"})
    p["year"] = y
    P.append(p.dropna(subset=["fac"]))
P = pd.concat(P, ignore_index=True); P["fac"] = P.fac.astype(int)
fy = F.groupby(["fac", "year", "state"], as_index=False).co2.sum()
rows = []
by_fac = {k: g.set_index("year") for k, g in P.groupby("fac")}
for r in fy.itertuples(index=False):
    g = by_fac.get(int(r.fac))
    if g is None:
        rows.append((r.fac, r.year, r.state, np.nan, np.nan, "unmatched", r.co2)); continue
    g = g[g.ba.notna()]
    if g.empty:
        rows.append((r.fac, r.year, r.state, np.nan, np.nan, "no BA reported", r.co2)); continue
    if r.year in g.index:
        rows.append((r.fac, r.year, r.state, g.loc[r.year, "ba"], r.year, "same year", r.co2)); continue
    yy = min(g.index, key=lambda q: (abs(q - r.year), -q))
    rows.append((r.fac, r.year, r.state, g.loc[yy, "ba"], yy, "nearest year", r.co2))
X = pd.DataFrame(rows, columns=["facility_id", "year", "state", "ba", "ba_source_year", "method", "co2_short_tons"])
X.to_csv(os.path.join(out, "facility_ba_crosswalk.csv"), index=False)

# ---------------------------------------------------------------- daily BA emissions under both definitions
F = F.merge(X[["facility_id", "year", "ba"]].rename(columns={"facility_id": "fac"}), on=["fac", "year"], how="left")
recs = []
for ba, sts in PROXY.items():
    xw = F[F.ba == ba].groupby("date").agg(co2_xw=("co2", "sum"), heat_xw=("heat", "sum"), n_xw=("fac", "nunique"))
    px = F[F.state.isin(sts)].groupby("date").agg(co2_proxy=("co2", "sum"), n_proxy=("fac", "nunique"))
    both = F[(F.ba == ba) & F.state.isin(sts)].groupby("date").co2.sum().rename("co2_both")
    d = pd.concat([xw, px, both], axis=1).reindex(pd.date_range("2019-01-01", "2025-12-31")).fillna(0.0)
    d["ba"] = ba; d.index.name = "date"; recs.append(d.reset_index())
B = pd.concat(recs, ignore_index=True)
B.to_csv(os.path.join(out, "ba_daily_co2.csv"), index=False)

# ---------------------------------------------------------------- boundary error of the state proxy
S = []
for ba, sts in PROXY.items():
    for y in list(YEARS) + ["all"]:
        g = X if y == "all" else X[X.year == y]
        prox = g[g.state.isin(sts)]; xw = g[g.ba == ba]
        out_of_ba = prox[prox.ba != ba]; out_of_states = xw[~xw.state.isin(sts)]
        S.append(dict(ba=ba, year=y, co2_proxy=prox.co2_short_tons.sum(), co2_crosswalk=xw.co2_short_tons.sum(),
                      proxy_share_outside_ba=out_of_ba.co2_short_tons.sum() / prox.co2_short_tons.sum(),
                      crosswalk_share_outside_states=out_of_states.co2_short_tons.sum() / max(xw.co2_short_tons.sum(), 1e-9),
                      outside_ba_codes=";".join(f"{k}:{v / prox.co2_short_tons.sum():.4f}" for k, v in
                                                out_of_ba.groupby(out_of_ba.ba.fillna("none")).co2_short_tons.sum().sort_values(ascending=False).items()),
                      outside_state_codes=";".join(f"{k}:{v / xw.co2_short_tons.sum():.4f}" for k, v in
                                                   out_of_states.groupby("state").co2_short_tons.sum().sort_values(ascending=False).items()
                                                   if xw.co2_short_tons.sum() > 0),
                      n_fac_proxy=prox.facility_id.nunique(), n_fac_crosswalk=xw.facility_id.nunique()))
pd.DataFrame(S).to_csv(os.path.join(out, "boundary_summary.csv"), index=False)
print(X.method.value_counts()); print((X.groupby("method").co2_short_tons.sum() / X.co2_short_tons.sum()).round(5))
print(pd.DataFrame(S)[lambda s: s.year == "all"].to_string())
