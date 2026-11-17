"""Interannual variation of monitored power-sector CO2 (Fig. 9).
(a) national daily emissions, 28-day mean and annual means; (b) year-on-year change decomposed into heat input and aggregate
emission factor (log contributions), with the annual mean Henry Hub price; (c) 2020 relative to 2019 by month for all, coal-like and
gas-like units and for total generation (EIA-923); (d) bias of the fourteen-day forecasts of the national total by evaluation year.
Usage: python fig_interannual.py [OUT_PNG]"""
import os, sys, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import STATE_DAILY, DERIVED, EXTERNAL, FORECASTS, RESULTS, FIGURES, ensure

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ensure(FIGURES), "fig09_interannual.png")
INK, MUTED, GRID = "#1f1f1f", "#55554f", "#e4e4df"
COL = {"Ridge": "#2a78d6", "Fourier ridge": "#6250d6", "GAM": "#eb6834", "HGB": "#1baf7a"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10.5, "axes.titlecolor": INK, "legend.frameon": False})
COVID = (pd.Timestamp("2020-03-15"), pd.Timestamp("2020-06-30")); EVAL0 = pd.Timestamp("2022-07-18")


def tidy(ax, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.8); ax.set_axisbelow(True)


D = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in sorted(glob.glob(f"{STATE_DAILY}/*.csv"))])
nat = D.groupby("date")[["co2_mass", "heat_input"]].sum()
nat = nat[nat.index.year <= 2025]
C = pd.read_csv(f"{DERIVED}/state_daily_components.csv", parse_dates=["date"])
G = pd.read_csv(f"{DERIVED}/monthly_generation_mix.csv")
HH = pd.read_csv(f"{EXTERNAL}/henry_hub_annual.csv").set_index("year").henry_hub_usd_per_mmbtu

fig = plt.figure(figsize=(13.6, 8.6)); gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.2)

# (a) national series
a = fig.add_subplot(gs[0, 0]); y = nat.co2_mass / 1e3
a.plot(y.index, y, color="#c8c8c2", lw=0.5, label="daily")
a.plot(y.index, y.rolling(28, center=True, min_periods=14).mean(), color=INK, lw=1.8, label="28-day mean")
for yr, g in y.groupby(y.index.year):
    a.plot([g.index.min(), g.index.max()], [g.mean()] * 2, color=COL["GAM"], lw=1.8, ls="--", label="annual mean" if yr == 2019 else None)
a.axvspan(*COVID, color="#f0d8cc", lw=0, label="COVID-19 restrictions")
a.axvline(EVAL0, color=MUTED, lw=1, ls=":"); a.text(EVAL0 + pd.Timedelta(days=20), y.max() * 0.97, "evaluation\nperiod →", fontsize=8.4, color=MUTED, va="top")
a.set_ylabel("CO$_2$, 45 states (thousand short tons per day)"); a.set_title("(a) National monitored power-sector CO$_2$", loc="left")
a.legend(fontsize=8, ncol=2, loc="lower left"); tidy(a)

# (b) year-on-year decomposition
b = fig.add_subplot(gs[0, 1]); A = nat.groupby(nat.index.year).sum(); yrs = list(A.index)
rows = []
for y0, y1 in zip(yrs[:-1], yrs[1:]):
    dE = 100 * (A.co2_mass[y1] / A.co2_mass[y0] - 1); dH = 100 * np.log(A.heat_input[y1] / A.heat_input[y0])
    ef0, ef1 = A.co2_mass[y0] / A.heat_input[y0], A.co2_mass[y1] / A.heat_input[y1]; dF = 100 * np.log(ef1 / ef0)
    rows.append((f"{y0}→{y1}", dE, dH, dF, HH.get(y1, np.nan)))
x = np.arange(len(rows))
for i, (_, dE, dH, dF, _) in enumerate(rows):
    lo = 0.0; hi = 0.0
    for v, col, lab in ((dH, COL["Ridge"], "generating activity (heat input)"), (dF, COL["GAM"], "emission factor (fuel mix)")):
        base = hi if v >= 0 else lo
        b.bar(i, v, bottom=base, color=col, width=0.62, label=lab if i == 0 else None)
        if v >= 0: hi += v
        else: lo += v
b.scatter(x, [r[1] for r in rows], color=INK, s=40, zorder=5, label="total CO$_2$ change")
b.set_ylim(top=max(r[1] for r in rows) + 3.5); b.axhline(0, color=MUTED, lw=0.9); b.set_xticks(x); b.set_xticklabels([r[0] for r in rows], fontsize=8.4)
b.set_ylabel("Change from previous year (%)"); b.set_title("(b) Year-on-year change and its decomposition", loc="left"); tidy(b)
b2 = b.twinx(); b2.plot(x, [r[4] for r in rows], color="#8a6a32", ls="--", marker="s", ms=5, lw=1.4, label="Henry Hub gas price")
b2.set_ylabel("Henry Hub ($ per MMBtu)", color="#8a6a32"); b2.tick_params(axis="y", colors="#8a6a32")
for s in ("top",): b2.spines[s].set_visible(False)
h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
b.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

# (c) 2020 relative to 2019 by month
c = fig.add_subplot(gs[1, 0])
C["year"], C["month"] = C.date.dt.year, C.date.dt.month
M = C[C.year.isin([2019, 2020])].groupby(["year", "month"])[["E", "E_coal", "E_gas"]].sum()
Gm = G[G.year.isin([2019, 2020])].groupby(["year", "month"]).Total.sum()
rel = lambda s: 100 * (s.loc[2020] / s.loc[2019] - 1)
mo = np.arange(1, 13)
c.axvspan(3.5, 6.5, color="#f0d8cc", lw=0)
c.plot(mo, rel(M.E).loc[mo], color=INK, marker="o", lw=2, label="CO$_2$ (all units)")
c.plot(mo, rel(M.E_coal).loc[mo], color="#5b4a42", marker="s", lw=1.4, label="CO$_2$, coal-like units")
c.plot(mo, rel(M.E_gas).loc[mo], color="#2a9d8f", marker="^", lw=1.4, label="CO$_2$, gas-like units")
c.plot(mo, rel(Gm).loc[mo], color=COL["Ridge"], marker="D", lw=1.4, ls="--", label="total generation (EIA-923)")
c.axhline(0, color=MUTED, lw=0.9); c.set_xticks(mo); c.set_xticklabels(list("JFMAMJJASOND"))
c.set_ylabel("2020 relative to 2019 (%)"); c.set_title("(c) 2020 by month: before, during and after restrictions", loc="left")
c.legend(fontsize=8, loc="lower right"); tidy(c)

# (d) bias of the fourteen-day national-total forecasts by evaluation year
d = fig.add_subplot(gs[1, 1])
# national total of all state forecasts whose target falls in the year, relative to the observed total of the same state-days
B = pd.read_csv(f"{FORECASTS}/baseline.csv.gz", parse_dates=["target_date"])
F = pd.read_csv(f"{FORECASTS}/fourier_ridge.csv.gz", parse_dates=["target_date"])
X = B.merge(F[["state", "horizon", "target_date", "fourier_ridge"]], on=["state", "horizon", "target_date"])
MS = ("ridge", "fourier_ridge", "gam", "hgb")
tab = [dict(horizon=h, year=yr, **{m: 100 * (g[m].sum() / g.target.sum() - 1) for m in MS})
       for (h, yr), g in X.groupby([X.horizon, X.target_date.dt.year]) if h in (7, 14)]
pd.DataFrame(tab).to_csv(os.path.join(ensure(RESULTS, "benchmark"), "national_bias_by_year.csv"), index=False)
T = X[X.horizon == 14]
bias = {r["year"]: {m: r[m] for m in MS} for r in tab if r["horizon"] == 14}
yrs = sorted(bias); x = np.arange(len(yrs)); w = 0.2
for j, (m, lab) in enumerate([("ridge", "Ridge"), ("fourier_ridge", "Fourier ridge"), ("gam", "GAM"), ("hgb", "HGB")]):
    d.bar(x + (j - 1.5) * w, [bias[yr][m] for yr in yrs], width=w, color=COL[lab], label=lab)
d.axhline(0, color=MUTED, lw=0.9); d.set_xticks(x)
first = T.target_date.min()
d.set_xticklabels([f"{yr} (Jul-Dec)" if yr == first.year and first.month > 1 else str(yr) for yr in yrs])
d.set_ylabel("Bias of 14-day national total (% of observed)"); d.set_title("(d) Forecast bias by evaluation year", loc="left")
d.legend(fontsize=8, ncol=2, loc="lower left"); tidy(d)

fig.savefig(OUT, dpi=220, bbox_inches="tight"); plt.close(fig); print(OUT)
summary = dict(yoy=[dict(change=r[0], total_pct=r[1], heat_input_log_pct=r[2], emission_factor_log_pct=r[3]) for r in rows],
               rel2020_all=rel(M.E).round(2).tolist(), rel2020_coal=rel(M.E_coal).round(2).tolist(), rel2020_gas=rel(M.E_gas).round(2).tolist(),
               rel2020_generation=rel(Gm).round(2).tolist(), bias_14d_national=bias)
json.dump(summary, open(os.path.join(ensure(RESULTS, "benchmark"), "interannual.json"), "w"), indent=1, default=float)
