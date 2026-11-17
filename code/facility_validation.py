"""Facility-level validation of the state-day series (2019-2025, 45 states) and alternative facility panels.
Outputs: data/derived summary CSVs and the balanced and high-coverage panel series in data/panels/{balanced,highcov}."""
import glob, json, os
import numpy as np, pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DERIVED as OUT, STATE_DAILY as SUP, CAMPD, PANELS, EXTERNAL, WORK as W
STATES = sorted(f[:-4] for f in os.listdir(SUP))
cols = ["state", "name", "fid", "date", "load", "steam", "so2", "co2", "nox", "heat"]
fr = []
for f in sorted(glob.glob(os.path.join(CAMPD, "daily-emissions-facility-aggregation-*.csv"))):
    d = pd.read_csv(f); d.columns = cols; fr.append(d)
F = pd.concat(fr, ignore_index=True)
F["date"] = pd.to_datetime(F.date)
F = F[(F.date >= "2019-01-01") & (F.date <= "2025-12-31")]
n0 = len(F); F = F.drop_duplicates(["state", "fid", "date"]); ndup = n0 - len(F)
F = F[F.state.isin(STATES)]
res = dict(facility_day_records=int(len(F)), duplicates_removed=int(ndup), facilities=int(F.fid.nunique()))

# ---- absent vs zero at facility-day level
rep = F.co2.notna()
res.update(records_without_co2=int((~rep).sum()), share_records_without_co2=float((~rep).mean()),
           share_without_co2_also_without_load_or_heat=float((F.loc[~rep, "load"].isna() & F.loc[~rep, "heat"].isna()).mean()),
           reported_zero_co2_records=int((F.co2 == 0).sum()), share_reported_zero=float((F.co2 == 0).sum() / rep.sum()))


def agg(G):
    a = G.groupby(["state", "date"]).agg(co2_mass=("co2", lambda s: s.sum(min_count=1)), so2_mass=("so2", lambda s: s.sum(min_count=1)),
                                          nox_mass=("nox", lambda s: s.sum(min_count=1)), gross_load_mwh=("load", lambda s: s.sum(min_count=1)),
                                          heat_input=("heat", lambda s: s.sum(min_count=1)), n_facilities=("fid", "count"), n_operating=("co2", "count"))
    return a.reset_index()


A = agg(F)
# ---- reconstruction vs supplied aggregates
rec = []
for s in STATES:
    sup = pd.read_csv(f"{SUP}/{s}.csv", parse_dates=["date"]).set_index("date")
    mine = A[A.state == s].set_index("date")
    j = sup[["co2_mass", "n_facilities"]].join(mine[["co2_mass", "n_facilities"]], rsuffix="_rec", how="outer")
    both = j.dropna(subset=["co2_mass", "co2_mass_rec"])
    rel = (both.co2_mass_rec - both.co2_mass).abs() / both.co2_mass.abs().clip(lower=1)
    rec.append(dict(state=s, days_supplied=int(sup.co2_mass.notna().sum()), days_rebuilt=int(mine.co2_mass.notna().sum()),
                    max_rel_diff=float(rel.max()), share_days_exact=float((rel < 1e-9).mean()),
                    nfac_match=float((both.n_facilities == both.n_facilities_rec).mean())))
REC = pd.DataFrame(rec); REC.to_csv(f"{OUT}/reconstruction_check.csv", index=False)
res["reconstruction_max_rel_diff"] = float(REC.max_rel_diff.max()); res["reconstruction_min_share_exact"] = float(REC.share_days_exact.min())

# ---- coverage and panel composition per state
days = pd.date_range("2019-01-01", "2025-12-31"); ND = len(days)
Fr = F[rep].copy(); Fr["year"] = Fr.date.dt.year
cov = []
series_bal, series_hc = {}, {}
for s in STATES:
    G = Fr[Fr.state == s]
    tot = G.co2.sum()
    per_fac = G.groupby("fid").agg(ndays=("date", "nunique"), co2=("co2", "sum"), years=("year", "nunique"),
                                   first=("date", "min"), last=("date", "max"))
    hc = per_fac.index[per_fac.ndays >= 0.95 * ND]; bal = per_fac.index[per_fac.years == 7]
    entries = int((per_fac["first"] > pd.Timestamp("2019-12-31")).sum()); exits = int((per_fac["last"] < pd.Timestamp("2025-01-01")).sum())
    daily = A[A.state == s].set_index("date").reindex(days)
    nf = daily.n_operating
    st = G[G.fid.isin(hc)].groupby("date").co2.sum().reindex(days); sb = G[G.fid.isin(bal)].groupby("date").co2.sum().reindex(days)
    full = daily.co2_mass
    yearly_n = G.groupby("year").fid.nunique()
    share_bal_by_year = (G[G.fid.isin(bal)].groupby("year").co2.sum() / G.groupby("year").co2.sum())
    cov.append(dict(state=s, facilities=int(len(per_fac)), facilities_2019=int(yearly_n.get(2019, 0)), facilities_2025=int(yearly_n.get(2025, 0)),
                    entries=entries, exits=exits, hc_facilities=int(len(hc)), hc_share=float(per_fac.loc[hc, "co2"].sum() / tot),
                    hc_corr=float(pd.concat([full, st], axis=1).dropna().corr().iloc[0, 1]) if len(hc) else np.nan,
                    bal_facilities=int(len(bal)), bal_share=float(per_fac.loc[bal, "co2"].sum() / tot),
                    bal_corr=float(pd.concat([full, sb], axis=1).dropna().corr().iloc[0, 1]),
                    bal_share_min_year=float(share_bal_by_year.min()), bal_share_max_year=float(share_bal_by_year.max()),
                    reporting_fac_mean=float(nf.mean()), reporting_fac_cv=float(nf.std() / nf.mean()),
                    reporting_fac_min=float(nf.min()), reporting_fac_max=float(nf.max()),
                    missing_state_days=int(full.isna().sum()), zero_state_days=int((full == 0).sum()),
                    share_fac_days_without_co2=float(F[(F.state == s)].co2.isna().mean())))
    for panel, ids, store in [("balanced", bal, series_bal), ("highcov", hc, series_hc)]:
        if len(ids) == 0: continue
        sub = agg(F[(F.state == s) & F.fid.isin(ids)])
        sub = sub.drop(columns="state").assign(state=s)
        store[s] = sub
COV = pd.DataFrame(cov); COV.to_csv(f"{OUT}/coverage_by_state.csv", index=False)
for panel, store in [("balanced", series_bal), ("highcov", series_hc)]:
    os.makedirs(f"{PANELS}/{panel}", exist_ok=True)
    for s, sub in store.items():
        sub[["date", "co2_mass", "so2_mass", "nox_mass", "gross_load_mwh", "heat_input", "n_facilities", "state"]].to_csv(
            f"{PANELS}/{panel}/{s}.csv", index=False)

# ---- independent inventory: EIA SEDS 2024 electric power sector CO2 (thousand metric tons)
eia = pd.read_csv(f"{EXTERNAL}/eia_seds_2024_electric_power_co2.csv")
NAME = {"AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona", "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
        "FL": "Florida", "GA": "Georgia", "IA": "Iowa", "IL": "Illinois", "IN": "Indiana", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
        "MA": "Massachusetts", "MD": "Maryland", "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri", "MS": "Mississippi", "MT": "Montana",
        "NC": "North Carolina", "ND": "North Dakota", "NE": "Nebraska", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
        "NV": "Nevada", "NY": "New York", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
        "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VA": "Virginia", "WA": "Washington",
        "WI": "Wisconsin", "WV": "West Virginia", "WY": "Wyoming"}
camp24 = A[A.date.dt.year == 2024].groupby("state").co2_mass.sum() * 0.90718474 / 1000   # short tons -> thousand metric tons
E = pd.DataFrame({"state": STATES, "campd_kt": [camp24[s] for s in STATES],
                  "eia_kt": [float(eia.loc[eia.State == NAME[s], "Total"].iloc[0]) for s in STATES]})
E["ratio"] = E.campd_kt / E.eia_kt
E.to_csv(f"{OUT}/eia_comparison_2024.csv", index=False)
res.update(eia_corr=float(np.corrcoef(E.campd_kt, E.eia_kt)[0, 1]), eia_log_corr=float(np.corrcoef(np.log(E.campd_kt), np.log(E.eia_kt))[0, 1]),
           eia_ratio_total=float(E.campd_kt.sum() / E.eia_kt.sum()), eia_ratio_median=float(E.ratio.median()),
           eia_ratio_q1=float(E.ratio.quantile(.25)), eia_ratio_q3=float(E.ratio.quantile(.75)),
           eia_ratio_min=[E.loc[E.ratio.idxmin(), "state"], float(E.ratio.min())], eia_ratio_max=[E.loc[E.ratio.idxmax(), "state"], float(E.ratio.max())],
           eia_states_within_10pct=int(((E.ratio > 0.9) & (E.ratio < 1.1)).sum()))
res.update(hc_share_median=float(COV.hc_share.median()), hc_share_min=[COV.loc[COV.hc_share.idxmin(), "state"], float(COV.hc_share.min())],
           hc_corr_median=float(COV.hc_corr.median()), bal_share_median=float(COV.bal_share.median()),
           bal_share_min=[COV.loc[COV.bal_share.idxmin(), "state"], float(COV.bal_share.min())],
           bal_corr_median=float(COV.bal_corr.median()), bal_corr_min=[COV.loc[COV.bal_corr.idxmin(), "state"], float(COV.bal_corr.min())],
           entries_total=int(COV.entries.sum()), exits_total=int(COV.exits.sum()), facilities_total=int(COV.facilities.sum()),
           states_no_hc=[s for s in STATES if s not in series_hc], reporting_cv_median=float(COV.reporting_fac_cv.median()))
json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
