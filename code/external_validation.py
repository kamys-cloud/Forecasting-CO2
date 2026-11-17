"""External validation: multi-year inventory (EIA, 2019-2024), monthly (EIA-923) and daily (EIA-930) dynamics,
and an audit of facility-days without a CO2 entry. Outputs data/derived/*.csv, data/derived/summary.json and the gap-filled panel
series data/panels/filled (CO2 of partial records and isolated gaps imputed; other quantities as reported)."""
import glob, json, os
import numpy as np, pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import EXTERNAL as P, DERIVED as O, CAMPD, STATE_DAILY as SUP, PANELS
STATES = sorted(f[:-4] for f in os.listdir(SUP))
res = {}
cols = ["state", "name", "fid", "date", "load", "steam", "so2", "co2", "nox", "heat"]
F = pd.concat([pd.read_csv(f).set_axis(cols, axis=1) for f in sorted(glob.glob(os.path.join(CAMPD, "daily-emissions-facility-aggregation-*.csv")))])
F["date"] = pd.to_datetime(F.date); F = F[(F.date >= "2019-01-01") & (F.date <= "2025-12-31")].drop_duplicates(["state", "fid", "date"])
FA = F[F.state.isin(STATES)].copy()

# ---------------------------------------------------------------- 1. audit of facility-days without CO2
FA = FA.sort_values(["fid", "date"])
noco2 = FA.co2.isna()
cls = np.select([noco2 & FA.load.isna() & FA.heat.isna(), noco2 & (FA.load.notna() | FA.heat.notna())], ["no_activity", "partial"], "reported")
FA["cls"] = cls
# isolated gaps inside operating spells: a day without CO2 whose previous and next days at the same facility report positive CO2
g = FA.groupby("fid")
prev_ok = g.co2.shift(1).fillna(0) > 0; next_ok = g.co2.shift(-1).fillna(0) > 0
FA["isolated_gap"] = noco2 & prev_ok & next_ok
ef = FA[FA.co2.notna() & (FA.heat > 0)].groupby("fid").apply(lambda d: d.co2.sum() / d.heat.sum(), include_groups=False).rename("ef")
lr = FA[FA.co2.notna() & (FA.load > 0)].groupby("fid").apply(lambda d: d.co2.sum() / d.load.sum(), include_groups=False).rename("co2_per_mwh")
FA = FA.join(ef, on="fid").join(lr, on="fid")
imp_partial = np.where(FA.cls == "partial", np.where(FA.heat.notna() & (FA.heat > 0), FA.heat * FA.ef,
                                                    np.where(FA.load.notna(), FA.load * FA.co2_per_mwh, 0)), 0)
# isolated gaps: bound by the mean of the two neighbouring days
nb = (g.co2.shift(1).fillna(0) + g.co2.shift(-1).fillna(0)) / 2
imp_gap = np.where(FA.isolated_gap & (FA.cls == "no_activity"), nb, 0)
FA["imp_partial"] = np.nan_to_num(imp_partial); FA["imp_gap"] = np.nan_to_num(imp_gap)
st = FA.groupby("state").agg(records=("co2", "size"), no_activity=("cls", lambda c: (c == "no_activity").sum()), partial=("cls", lambda c: (c == "partial").sum()),
                             isolated_gaps=("isolated_gap", "sum"), co2=("co2", "sum"), imp_partial=("imp_partial", "sum"), imp_gap=("imp_gap", "sum"))
st["share_imputable_partial"] = st.imp_partial / st.co2; st["share_imputable_gaps"] = st.imp_gap / st.co2
st.reset_index().to_csv(f"{O}/missingness_audit.csv", index=False)
res["audit"] = dict(records=int(st.records.sum()), no_activity=int(st.no_activity.sum()), partial=int(st.partial.sum()),
                    isolated_gaps=int(st.isolated_gaps.sum()), share_partial_records=float(st.partial.sum() / st.records.sum()),
                    imputable_share_total=float((st.imp_partial.sum() + st.imp_gap.sum()) / st.co2.sum()),
                    imputable_share_max_state=[st.eval("(imp_partial+imp_gap)/co2").idxmax(), float(st.eval("(imp_partial+imp_gap)/co2").max())],
                    imputable_share_median=float(st.eval("(imp_partial+imp_gap)/co2").median()))
# effect on the state series: correlation between original and gap-filled daily totals
fill = FA.assign(co2f=FA.co2.fillna(0) + FA.imp_partial + FA.imp_gap).groupby(["state", "date"]).agg(orig=("co2", "sum"), filled=("co2f", "sum"))
fill["rel"] = fill.filled / fill.orig - 1
res["audit"]["max_daily_relative_change_median_state"] = float(fill.groupby("state").rel.quantile(0.99).median())
res["audit"]["corr_orig_filled_min"] = float(fill.groupby("state").apply(lambda d: np.corrcoef(d.orig, d.filled)[0, 1], include_groups=False).min())
# gap-filled panel: state-day series of the supplied days with imputed CO2 added
s1 = lambda s: s.sum(min_count=1)
GF = FA.assign(co2f=FA.co2.fillna(0) + FA.imp_partial + FA.imp_gap).groupby(["state", "date"]).agg(
    co2_mass=("co2f", "sum"), so2_mass=("so2", s1), nox_mass=("nox", s1), gross_load_mwh=("load", s1), heat_input=("heat", s1),
    n_facilities=("fid", "count")).reset_index()
os.makedirs(f"{PANELS}/filled", exist_ok=True)
for s in STATES:
    days = pd.read_csv(f"{SUP}/{s}.csv", parse_dates=["date"]).date
    g = GF[GF.state == s].set_index("date").reindex(days).reset_index(); g["state"] = s
    g[["date", "co2_mass", "so2_mass", "nox_mass", "gross_load_mwh", "heat_input", "n_facilities", "state"]].astype({"n_facilities": int}).to_csv(
        f"{PANELS}/filled/{s}.csv", index=False, date_format="%Y-%m-%d")

# ---------------------------------------------------------------- 2. multi-year external inventory (EIA, by producer type)
E = pd.read_excel(f"{P}/raw/emission_annual.xlsx", sheet_name=0)
E.columns = ["year", "state", "producer", "source", "co2", "so2", "nox"]
E = E[(E.source == "All Sources") & E.year.between(2019, 2024) & E.state.isin(STATES)]
tot = E[E.producer == "Total Electric Power Industry"].set_index(["state", "year"]).co2
utl = E[E.producer.isin(["Electric Utility", "IPP NAICS-22 Non-Cogen", "IPP NAICS-22 Cogen"])].groupby(["state", "year"]).co2.sum()
camp = FA.assign(year=FA.date.dt.year).groupby(["state", "year"]).co2.sum() * 0.90718474     # short -> metric tons
M = pd.DataFrame({"campd": camp, "eia_total": tot, "eia_util_ipp": utl}).dropna().reset_index()
M["ratio_total"] = M.campd / M.eia_total; M["ratio_util_ipp"] = M.campd / M.eia_util_ipp
M = M.sort_values(["state", "year"])
for c in ("campd", "eia_total", "eia_util_ipp"): M[c + "_chg"] = M.groupby("state")[c].pct_change()
M.to_csv(f"{O}/eia_multiyear.csv", index=False)
ch = M.dropna(subset=["campd_chg"])
st_ratio = M.groupby("state").agg(ratio_mean=("ratio_total", "mean"), ratio_min=("ratio_total", "min"), ratio_max=("ratio_total", "max"),
                                  ratio_util_mean=("ratio_util_ipp", "mean"))
st_ratio["corr_changes"] = ch.groupby("state").apply(lambda d: np.corrcoef(d.campd_chg, d.eia_total_chg)[0, 1], include_groups=False)
st_ratio.reset_index().to_csv(f"{O}/eia_multiyear_by_state.csv", index=False)
res["eia_multiyear"] = dict(years=[2019, 2024], n_state_years=int(len(M)),
                            median_abs_pct_diff_total=float(100 * (M.ratio_total - 1).abs().median()),
                            median_abs_pct_diff_util_ipp=float(100 * (M.ratio_util_ipp - 1).abs().median()),
                            share_within_10pct_total=float(((M.ratio_total - 1).abs() <= 0.10).mean()),
                            share_within_10pct_util_ipp=float(((M.ratio_util_ipp - 1).abs() <= 0.10).mean()),
                            corr_levels_log=float(np.corrcoef(np.log(M.campd), np.log(M.eia_total))[0, 1]),
                            corr_yoy_changes_pooled=float(np.corrcoef(ch.campd_chg, ch.eia_total_chg)[0, 1]),
                            median_state_corr_changes=float(st_ratio.corr_changes.median()),
                            ratio_sd_within_state_median=float(M.groupby("state").ratio_total.std().median()),
                            aggregate_ratio_by_year={int(y): float(g.campd.sum() / g.eia_total.sum()) for y, g in M.groupby("year")},
                            discrepant={s: dict(ratio_total=float(st_ratio.loc[s, "ratio_mean"]), ratio_util_ipp=float(st_ratio.loc[s, "ratio_util_mean"]),
                                                corr_changes=float(st_ratio.loc[s, "corr_changes"])) for s in ["LA", "NH", "OR", "MI", "MT"]})

# ---------------------------------------------------------------- 3. monthly dynamics (EIA-923 fossil generation) - from phys/generation_mix_shares.csv
gm = pd.read_csv(f"{O}/generation_mix_shares.csv").set_index("state")
res["monthly"] = dict(median_corr_levels=float(gm.corr_campd_load_eia_fossil.median()), min_corr_levels=[gm.corr_campd_load_eia_fossil.idxmin(), float(gm.corr_campd_load_eia_fossil.min())],
                      median_corr_changes=float(gm.corr_campd_load_eia_fossil_changes.median()),
                      min_corr_changes=[gm.corr_campd_load_eia_fossil_changes.idxmin(), float(gm.corr_campd_load_eia_fossil_changes.min())],
                      discrepant={s: float(gm.loc[s, "corr_campd_load_eia_fossil_changes"]) for s in ["LA", "NH", "OR", "MI", "MT"]})

# ---------------------------------------------------------------- 4. daily dynamics against EIA-930 balancing authorities
B = pd.read_csv(f"{P}/eia930_daily.csv", parse_dates=["date"])
B["fossil"] = B[["coal", "gas", "oil"]].sum(axis=1, min_count=1)
allF = F.copy()
match = {"ERCO": ["TX"], "CISO": ["CA"], "NYIS": ["NY"], "ISNE": ["CT", "MA", "NH", "RI", "ME", "VT"], "FPL": ["FL"], "PNM": ["NM"]}
drows = []
for ba, sts in match.items():
    c = allF[allF.state.isin(sts)].groupby("date").agg(load=("load", "sum"), co2=("co2", "sum"))
    b = B[B.ba == ba].set_index("date")
    j = c.join(b[["fossil", "demand"]], how="inner").dropna()
    j = j[(j.fossil > 0) & (j.load > 0)]
    dl, df_ = j.load.diff(), j.fossil.diff()
    # seasonal-anomaly correlation: remove each series' 30-day centred mean
    al = j.load - j.load.rolling(31, center=True, min_periods=15).mean(); af = j.fossil - j.fossil.rolling(31, center=True, min_periods=15).mean()
    drows.append(dict(ba=ba, states="+".join(sts), days=int(len(j)), corr_levels=float(np.corrcoef(j.load, j.fossil)[0, 1]),
                      corr_daily_changes=float(dl.corr(df_)), corr_within_month_anomalies=float(al.corr(af)),
                      corr_co2_demand=float(j.co2.corr(j.demand)), ratio=float(j.load.sum() / j.fossil.sum())))
D = pd.DataFrame(drows); D.to_csv(f"{O}/eia930_daily_validation.csv", index=False)
res["daily"] = D.set_index("ba").to_dict(orient="index")
json.dump(res, open(f"{O}/summary.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float)[:4000])
