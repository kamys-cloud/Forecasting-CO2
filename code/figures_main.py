"""Figures 3-7 and Supplementary Figs. S1-S3. Usage: python figures_main.py [fig03 fig04 fig05 fig06 fig07 figS1 figS2 figS3]"""
import sys, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK as V, RESULTS, EXTERNAL, DERIVED, FIGURES, ensure
OB, OF = f"{RESULTS}/benchmark", f"{RESULTS}/factorial"
FIG = ensure(FIGURES)
H = [1, 2, 3, 5, 7, 10, 14]
INK, MUTED, GRID = "#1f1f1f", "#55554f", "#e4e4df"
COL = {"Ridge": "#2a78d6", "GAM": "#eb6834", "HGB": "#1baf7a", "Fourier ridge": "#6250d6", "Persistence": "#8a8a84",
       "SARIMAX": "#3d3d3a", "Fourier-ETS": "#eda100", "Seasonal naive (weekly)": "#b9b9b3", "Prophet": "#c2477e"}
MK = {"Ridge": "o", "GAM": "s", "HGB": "^", "Fourier ridge": "D", "Persistence": "v", "SARIMAX": "P", "Fourier-ETS": "X",
      "Seasonal naive (weekly)": ".", "Prophet": "*"}
REG = {"Northeast": ("#2a78d6", "o"), "Midwest": ("#eb6834", "s"), "South": ("#1baf7a", "^"), "West": ("#eda100", "D")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10.5, "axes.titlecolor": INK, "legend.frameon": False})
RB = json.load(open(f"{OB}/results.json")); RF = json.load(open(f"{OF}/results.json"))
RC = json.load(open(f"{V}/core/results.json")); PC = pd.read_csv(f"{V}/core/panel.csv")


def tidy(ax, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.8); ax.set_axisbelow(True)


def save(fig, name, tight=True):
    if tight: fig.tight_layout()
    fig.savefig(f"{FIG}/{name}", dpi=220, bbox_inches="tight"); plt.close(fig); print(name)


# ---------------------------------------------------------------- persistence and regional differences
def fig03():
    P = PC; g = P.groupby("horizon")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14.2, 4.4))
    a.plot(H, g.persist.median().values, color=COL["Ridge"], marker="o", lw=2, ms=6, label="Median")
    a.plot(H, g.persist.mean().values, color=COL["GAM"], marker="s", lw=2, ms=6, label="Mean")
    a.set_xticks(H); a.set_xlabel("Forecast horizon (days)"); a.set_ylabel("Persistence R$^2$")
    a.set_title("(a) Decay of autoregressive persistence", loc="left"); a.legend(); tidy(a)
    seq = ["#cde2fb", "#a8ccf6", "#86b6ef", "#5f9de9", "#3987e5", "#2a6fc2", "#1c5cab"]
    for h, cc in zip(H, seq):
        x = P[P.horizon == h]; b.scatter(x.persist, x.d_hr, s=22, color=cc, edgecolor="white", lw=0.5, label=f"{h} d", zorder=3)
    cf = np.polyfit(P.persist, P.d_hr, 1); xs = np.array([P.persist.min(), P.persist.max()])
    b.plot(xs, cf[0] * xs + cf[1], color=INK, lw=1.4, zorder=4); b.axhline(0, color=MUTED, lw=0.9, ls=":")
    b.text(0.03, 0.05, f"r = {RC['r_persist_dhr']:.2f}".replace("-", "−"), transform=b.transAxes, fontsize=10, color=INK)
    b.set_xlabel("Persistence R$^2$"); b.set_ylabel("R$^2$(HGB) − R$^2$(ridge)")
    b.set_title("(b) Persistence and the HGB–ridge difference", loc="left")
    b.legend(title="Horizon", fontsize=7.5, title_fontsize=7.5, loc="upper right"); tidy(b, "both")
    rg = P.groupby(["region", "horizon"]).d_hr.mean().unstack()
    for r, (col, mk) in REG.items():
        n = P[(P.region == r) & (P.horizon == 1)].shape[0]
        c.plot(H, rg.loc[r, H].values, color=col, marker=mk, lw=2, ms=6, label=f"{r} ({n} states)")
    c.axhline(0, color=MUTED, lw=0.9, ls=":"); c.set_xticks(H); c.set_xlabel("Forecast horizon (days)")
    c.set_ylabel("Mean R$^2$(HGB) − R$^2$(ridge)"); c.set_title("(c) By U.S. Census region", loc="left"); c.legend(loc="upper left", fontsize=8); tidy(c)
    save(fig, "fig03_persistence.png")


# ---------------------------------------------------------------- ablation
def fig04():
    A = RC["ablation_by_h"]; AC = pd.DataFrame(RB["ablation_components"])
    fig = plt.figure(figsize=(13.4, 7.6)); gs0 = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], hspace=0.42)
    top = gs0[0].subgridspec(1, 2, width_ratios=[1.15, 1], wspace=0.42); gs = gs0[1].subgridspec(1, 4, wspace=0.12)
    a = fig.add_subplot(top[0])
    for lab, k in [("Ridge", "ridge"), ("GAM", "gam"), ("HGB", "hgb")]:
        a.plot(H, [A["baseline"][str(h)][k] for h in H], color=COL[lab], marker=MK[lab], ms=5.5, lw=2, label=f"{lab}, all 27 inputs")
        a.plot(H, [A["without_calendar"][str(h)][k] for h in H], color=COL[lab], marker=MK[lab], ms=5.5, lw=1.6, ls="--", mfc="white",
               label=f"{lab}, without calendar/year group")
    a.set_xticks(H); a.set_xlabel("Forecast horizon (days)"); a.set_ylabel("Mean state-level R$^2$")
    a.set_title("(a) Removing the calendar/year group", loc="left"); a.legend(fontsize=7.4, ncol=1, loc="upper right"); tidy(a)
    names = {"comp_annual": "(c) Both annual pairs removed", "comp_doy": "(d) Day-of-year pair removed",
             "comp_month": "(e) Month pair removed", "comp_year": "(f) Numeric year removed"}
    axs = [fig.add_subplot(gs[i]) for i in range(4)]
    for i, (ax, (v, ttl)) in enumerate(zip(axs, names.items())):
        g = AC[AC.variant == v].set_index("horizon").sort_index()
        for lab, m in [("Ridge", "ridge"), ("GAM", "gam"), ("HGB", "hgb")]:
            y = g[f"{m}_loss"].values; c = np.array(g[f"{m}_loss_ci"].tolist())
            ax.fill_between(g.index, c[:, 0], c[:, 1], color=COL[lab], alpha=0.15, lw=0)
            ax.plot(g.index, y, color=COL[lab], marker=MK[lab], ms=4.5, lw=1.8, label=lab)
        ax.axhline(0, color=MUTED, lw=0.9, ls=":"); ax.set_xticks(H); ax.set_xlabel("Horizon (days)"); ax.set_ylim(-0.02, 0.21)
        ax.set_title(ttl, loc="left", fontsize=9.3); tidy(ax)
        if i: ax.set_yticklabels([])
    axs[0].set_ylabel("R$^2$ loss when removed\n(joint-bootstrap 95% CI)"); axs[0].legend(fontsize=8, loc="upper left")
    # summary panel (top right): mean loss over horizons by model and component
    s = fig.add_subplot(top[1])
    comps = [("comp_annual", "Both annual pairs"), ("comp_doy", "Day-of-year pair"), ("comp_month", "Month pair"), ("comp_year", "Numeric year"), ("comp_cal", "Whole calendar/year group")]
    y0 = np.arange(len(comps))[::-1]
    for j, (lab, m) in enumerate([("Ridge", "ridge"), ("GAM", "gam"), ("HGB", "hgb")]):
        vals = [AC[(AC.variant == v) & (AC.horizon == 14)][f"{m}_loss"].values[0] for v, _ in comps]
        s.barh(y0 + (j - 1) * 0.26, vals, height=0.25, color=COL[lab], label=lab)
    s.set_yticks(y0); s.set_yticklabels([c[1] for c in comps]); s.axvline(0, color=MUTED, lw=0.8)
    s.set_xlabel("R$^2$ loss at the 14-day horizon"); s.set_title("(b) Calendar-component removals at 14 days", loc="left"); s.legend(fontsize=8, loc="center right"); tidy(s, "x")
    save(fig, "fig04_ablation.png", tight=False)


# ---------------------------------------------------------------- seasonal representation x model family
def fig05():
    F = pd.DataFrame(RF["factorial"]).set_index("horizon"); C = pd.DataFrame(RB["classical"]).set_index("horizon")
    PZ = pd.DataFrame(RF.get("prophet", [])).set_index("horizon") if RF.get("prophet") else None
    EQ = pd.DataFrame(RF["equiv"])
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.0, 4.7))
    for lab, col, src in [("Persistence", "persistence", C), ("Seasonal naive (weekly)", "snaive7", C), ("Fourier-ETS", "fets", C),
                          ("Prophet", "prophet", PZ), ("SARIMAX", "sarimax", C), ("Ridge", "ridge", F), ("GAM", "gam", F), ("HGB", "hgb", F),
                          ("Fourier ridge", "fourier_ridge", F)]:
        if src is None: continue
        a.plot(H, src.loc[H, col].values, color=COL[lab], marker=MK[lab], ms=5 if lab != "Prophet" else 7, lw=2 if lab in ("Ridge", "GAM", "HGB", "Fourier ridge", "SARIMAX") else 1.3,
               ls="--" if lab in ("Persistence", "Seasonal naive (weekly)") else (":" if lab == "SARIMAX" else "-"), label=lab, zorder=5 if lab == "Fourier ridge" else 3)
    a.set_xticks(H); a.set_xlabel("Forecast horizon (days)"); a.set_ylabel("Mean state-level R$^2$ (45 states)"); a.set_ylim(0, 0.9)
    a.set_title("(a) Classical, linear, and nonlinear models", loc="left"); a.legend(fontsize=7.4, ncol=2, loc="lower left"); tidy(a)
    styles = [("orig", "Original encodings", "-", "o"), ("fw", "+ weekly Fourier", ":", "x"), ("fa", "+ annual Fourier", "--", "s"),
              ("cs", "+ annual cyclic spline", "-.", "^"), ("fourier", "+ annual and weekly Fourier", (0, (1, 1)), "D")]
    for key, lab, ls, mk in styles:
        g = (lambda m: m) if key == "orig" else (lambda m, k=key: f"{k}_{m}")
        gap = F[g("gam")] - F[g("ridge")]
        b.plot(H, gap.loc[H].values, color=COL["GAM"], ls=ls, marker=mk, ms=4.5, lw=1.7, label=lab)
        gap = F[g("hgb")] - F[g("ridge")]
        b.plot(H, gap.loc[H].values, color=COL["HGB"], ls=ls, marker=mk, ms=4.5, lw=1.4)
    b.axhline(0, color=MUTED, lw=0.9, ls=":"); b.set_xticks(H); b.set_xlabel("Forecast horizon (days)")
    b.set_ylabel("Nonlinear minus ridge, same basis (mean R$^2$)"); b.set_title("(b) Family gap under matched seasonal bases", loc="left")
    b.text(0.02, 0.97, "orange: GAM − ridge   green: HGB − ridge", transform=b.transAxes, fontsize=7.8, va="top", color=MUTED)
    b.legend(fontsize=7.4, loc="upper left", bbox_to_anchor=(0.0, 0.93), title="Seasonal basis (both families)", title_fontsize=7.4, frameon=False); tidy(b)
    offs = np.linspace(-0.32, 0.32, 6)
    for (lab, m, colr, mk, mfc), o in zip([("Ridge", "ridge", COL["Ridge"], "o", "full"), ("GAM", "gam", COL["GAM"], "s", "full"), ("HGB", "hgb", COL["HGB"], "^", "full"),
                                           ("SARIMAX", "sarimax", COL["SARIMAX"], "P", "full"), ("Fourier GAM", "fourier_gam", COL["GAM"], "s", "white"),
                                           ("Fourier HGB", "fourier_hgb", COL["HGB"], "^", "white")], offs):
        g = EQ[(EQ.a == "fourier_ridge") & (EQ.b == m)].set_index("horizon").loc[H]
        y = -g["mean"].values; lo = -g.ci_hi.values; hi = -g.ci_lo.values
        c.errorbar(np.array(H) + o, y, yerr=[y - lo, hi - y], color=colr, marker=mk, mfc=colr if mfc == "full" else "white", ms=4.5, lw=1.3, capsize=2,
                   ls="-" if mfc == "full" else "--", label=f"{lab} − Fourier ridge")
    c.axhspan(-0.01, 0.01, color="#efeee9", zorder=0, label="Equivalence margin (±0.01)")
    c.axhline(0, color=MUTED, lw=0.9, ls=":"); c.set_xticks(H); c.set_xlabel("Forecast horizon (days)")
    c.set_ylabel("Difference in mean R$^2$ (joint-bootstrap 95% CI)"); c.set_title("(c) Relative to the Fourier ridge", loc="left")
    c.legend(fontsize=7.2, loc="lower left"); tidy(c)
    save(fig, "fig05_seasonal_representation.png")


# ---------------------------------------------------------------- seasonal structure: out-of-sample aggregates and fitted responses
def fig06():
    O = pd.read_csv(f"{OF}/oos_seasonal.csv"); Pf = pd.read_csv(f"{OB}/fitted_responses.csv", parse_dates=["date"])
    states = ["TX", "PA", "CA", "ND", "GA", "NM"]
    fig, axes = plt.subplots(4, 3, figsize=(13.0, 12.6), sharex=False)
    lab = {"target": ("Observed", "#9a9a94"), "ridge": ("Ridge", COL["Ridge"]), "fourier_ridge": ("Fourier ridge", COL["Fourier ridge"]),
           "gam": ("GAM", COL["GAM"]), "hgb": ("HGB", COL["HGB"])}
    for i, s in enumerate(states):
        ax = axes[(i // 3) * 2, i % 3]
        g = O[O.state == s]
        for m, (nm, c) in lab.items():
            x = g[g.series == m].sort_values("week")
            ax.plot(x.week, x.value, color=c, lw=2.6 if m == "target" else 1.5, label=nm, zorder=1 if m == "target" else 3,
                    ls=(0, (4, 1.5)) if m == "fourier_ridge" else "-")
        ax.set_xticks([0, 13, 26, 39, 51]); ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Dec"])
        ax.set_title(f"{s} (a) out-of-sample, by target week", loc="left", fontsize=9.5); tidy(ax)
        ax2 = axes[(i // 3) * 2 + 1, i % 3]
        g2 = Pf[Pf.state == s]
        for m, nm, c in [("ridge", "Ridge", COL["Ridge"]), ("fourier_ridge", "Fourier ridge", COL["Fourier ridge"]), ("gam", "GAM", COL["GAM"]),
                         ("hgb", "HGB (partial dependence)", COL["HGB"])]:
            x = g2[g2.model == m]
            ax2.plot(x.date, x.value_pct, color=c, lw=1.5, ls=(0, (4, 1.5)) if m == "fourier_ridge" else "-", label=nm)
        ax2.axvline(pd.Timestamp("2026-01-01"), color=MUTED, lw=0.8, ls=":")
        ax2.set_xticks(pd.to_datetime(["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01", "2026-01-01"])); ax2.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Jan"])
        ax2.set_title(f"{s} (b) model diagnostic: calendar response", loc="left", fontsize=9.5); tidy(ax2)
    for r in (0, 2): axes[r, 0].set_ylabel("Target-week mean, % of annual mean")
    for r in (1, 3): axes[r, 0].set_ylabel("Calendar response, % of mean")
    h1, l1 = axes[0, 0].get_legend_handles_labels(); h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1, [x if x != "HGB" else "HGB (partial dependence in b)" for x in l1], loc="lower center", ncol=5, fontsize=8.6, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{FIG}/fig06_seasonal_structure.png", dpi=200); plt.close(fig); print("fig06")


# ---------------------------------------------------------------- physical sources of the annual cycle
def fig07():
    T = pd.read_csv(f"{DERIVED}/seasonal_decomposition.csv"); Pp = pd.read_csv(f"{DERIVED}/seasonal_profiles_phys.csv")
    GM = pd.read_csv(f"{DERIVED}/generation_mix_shares.csv")
    states = ["TX", "PA", "CA", "ND", "GA", "NM"]
    fig = plt.figure(figsize=(13.6, 8.6)); gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.05], hspace=0.5, wspace=0.26)
    x = np.arange(366)
    for i, s in enumerate(states):
        ax = fig.add_subplot(gs[i // 3, i % 3]); g = Pp[Pp.state == s]
        get = lambda k: g[g.series == k].sort_values("doy").value.values
        E = get("E"); ax.plot(x, (E - 1) * 100, color="#9a9a94", lw=2.6, label="CO$_2$ (all units)")
        ax.plot(x, get("coal") * 100, color="#55554f", lw=1.4, label="coal-unit contribution")
        ax.plot(x, get("gas") * 100, color=COL["GAM"], lw=1.4, label="gas-unit contribution")
        ax.plot(x, (get("fit_degree_days") - 1) * 100, color=COL["Fourier ridge"], lw=1.6, ls="--", label="heating/cooling degree-day fit")
        r = T.set_index("state").loc[s]
        ax.set_xticks([0, 90, 181, 273, 365]); ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Jan"]); ax.axhline(0, color=GRID, lw=0.8)
        ax.set_title(f"({'abcdef'[i]}) {s}: intensity share {100 * max(r.var_share_intensity, 0):.0f}%, degree-day R$^2$ {r.r2_clim_degree_days:.2f}", loc="left", fontsize=9.2); tidy(ax)
        if i % 3 == 0: ax.set_ylabel("% of annual-mean CO$_2$")
    h, l = fig.axes[0].get_legend_handles_labels()
    g1 = fig.add_subplot(gs[2, 0])
    for j, (k, lab, c) in enumerate([("amp_E", "CO$_2$", "#9a9a94"), ("amp_H", "Heat input\n(activity)", COL["Ridge"]), ("amp_EF", "Emission factor\n(intensity)", COL["GAM"])]):
        g1.scatter(np.full(len(T), j) + np.random.default_rng(1).uniform(-0.15, 0.15, len(T)), 100 * T[k], s=14, color=c, alpha=0.8)
        g1.plot([j - 0.25, j + 0.25], [100 * T[k].median()] * 2, color=INK, lw=2)
    g1.set_xticks([0, 1, 2]); g1.set_xticklabels(["CO$_2$", "Heat input\n(activity)", "Emission factor\n(intensity)"], fontsize=8)
    g1.set_ylabel("Seasonal peak-to-trough range (%)"); g1.set_title("(g) Activity versus intensity", loc="left"); tidy(g1)
    g2 = fig.add_subplot(gs[2, 1])
    for k, lab, c in [("r2_clim_fourier1", "one annual harmonic", COL["Ridge"]), ("r2_clim_degree_days", "degree days (HDD, CDD)", COL["Fourier ridge"]), ("r2_clim_fourier3", "three annual harmonics", COL["GAM"])]:
        g2.scatter(np.full(len(T), ["one annual harmonic", "degree days (HDD, CDD)", "three annual harmonics"].index(lab)) + np.random.default_rng(0).uniform(-0.15, 0.15, len(T)),
                   T[k], s=14, color=c, alpha=0.75)
        g2.plot([["one annual harmonic", "degree days (HDD, CDD)", "three annual harmonics"].index(lab) - 0.25, ["one annual harmonic", "degree days (HDD, CDD)", "three annual harmonics"].index(lab) + 0.25],
                [T[k].median()] * 2, color=INK, lw=2)
    g2.set_xticks([0, 1, 2]); g2.set_xticklabels(["One annual\nharmonic", "Degree days\n(HDD, CDD)", "Three annual\nharmonics"], fontsize=8)
    g2.set_ylabel("R$^2$ for the CO$_2$ seasonal climatology"); g2.set_title("(h) What explains the annual cycle", loc="left"); tidy(g2)
    g3 = fig.add_subplot(gs[2, 2])
    MM = pd.read_csv(f"{DERIVED}/monthly_generation_mix.csv"); rr = []
    for st_, m_ in MM.groupby("state"):
        an = lambda c: (m_[c] - m_.groupby("year")[c].transform("mean")).values
        y_, d_, n_ = an("fossil"), an("Total"), an("nonfossil")
        r1 = np.corrcoef(y_, d_)[0, 1] ** 2
        X = np.column_stack([np.ones(len(y_)), d_, n_]); bb = np.linalg.lstsq(X, y_, rcond=None)[0]
        r2_ = 1 - np.sum((y_ - X @ bb) ** 2) / np.sum((y_ - y_.mean()) ** 2); rr.append((st_, r1, r2_ - r1))
    rr = sorted(rr, key=lambda t: -t[1])
    g3.bar(range(len(rr)), [t[1] for t in rr], color=COL["Ridge"], label="demand (total generation)")
    g3.bar(range(len(rr)), [t[2] for t in rr], bottom=[t[1] for t in rr], color=COL["HGB"], label="+ non-fossil generation")
    g3.set_xticks(range(len(rr))); g3.set_xticklabels([t[0] for t in rr], rotation=90, fontsize=5.8); g3.set_ylim(0, 1.05)
    g3.set_ylabel("R$^2$, monthly fossil generation"); g3.set_title("(i) Demand versus generation mix (EIA-923)", loc="left"); g3.legend(fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); tidy(g3)
    R13 = dict(median_r2_demand=float(np.median([t[1] for t in rr])), median_r2_both=float(np.median([t[1] + t[2] for t in rr])),
               states_nonfossil_gt_demand=[t[0] for t in rr if t[2] > t[1]])
    json.dump(R13, open(f"{DERIVED}/genmix_r2.json", "w"))
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=8.4, bbox_to_anchor=(0.5, 1.01))
    fig.savefig(f"{FIG}/fig07_physical_sources.png", dpi=210, bbox_inches="tight"); plt.close(fig); print("fig07")


# ---------------------------------------------------------------- data validation
def figS1():
    M = pd.read_csv(f"{DERIVED}/eia_multiyear.csv"); D = pd.read_csv(f"{DERIVED}/eia930_daily_validation.csv")
    C = pd.read_csv(f"{DERIVED}/coverage_by_state.csv")
    B = pd.read_csv(f"{EXTERNAL}/eia930_daily.csv", parse_dates=["date"])
    fig = plt.figure(figsize=(14.0, 12.4)); gs = fig.add_gridspec(3, 3, hspace=0.48, wspace=0.28, height_ratios=[1, 1, 0.9])
    a = fig.add_subplot(gs[0, :]); st = M.groupby("state").ratio_util_ipp.mean().sort_values().index
    for i, s in enumerate(st):
        r = M[M.state == s].ratio_util_ipp.values
        a.plot([i] * len(r), r, "o", color=COL["Ridge"], ms=3, alpha=0.7); a.plot([i - 0.3, i + 0.3], [r.mean()] * 2, color=INK, lw=1.4)
    a.axhspan(0.9, 1.1, color="#efeee9", zorder=0); a.axhline(1, color=MUTED, lw=0.9, ls=":")
    a.set_xticks(range(len(st))); a.set_xticklabels(st, rotation=90, fontsize=7); a.set_ylabel("CAMPD / EIA (utilities + IPPs)")
    a.set_title("(a) Annual inventory ratio by state, 2019-2024 (dots: years)", loc="left"); tidy(a)
    b = fig.add_subplot(gs[1, 0])
    ch = M.dropna(subset=["campd_chg"])
    b.scatter(100 * ch.eia_total_chg, 100 * ch.campd_chg, s=10, color=COL["GAM"], alpha=0.7)
    lim = [-50, 50]; b.plot(lim, lim, ls="--", color=MUTED, lw=1); b.set_xlim(lim); b.set_ylim(lim)
    b.set_xlabel("EIA year-on-year change (%)"); b.set_ylabel("CAMPD year-on-year change (%)")
    b.text(0.04, 0.9, f"r = {np.corrcoef(ch.campd_chg, ch.eia_total_chg)[0, 1]:.2f}", transform=b.transAxes); b.set_title("(b) Interannual dynamics", loc="left"); tidy(b, "both")
    c = fig.add_subplot(gs[1, 1])
    F = pd.read_csv(f"{DERIVED}/state_daily_components.csv", parse_dates=["date"])
    tx = F[F.state == "TX"].set_index("date").L; e = B[B.ba == "ERCO"].set_index("date"); e = e.coal + e.gas + e.oil
    j = pd.concat([tx, e], axis=1, keys=["campd", "eia"]).dropna().loc["2024-01-01":"2024-12-31"]
    c.plot(j.index, j.campd / 1e3, color=COL["Ridge"], lw=1.1, label="CAMPD gross load (TX)")
    c.plot(j.index, j.eia / 1e3, color=COL["GAM"], lw=1.1, label="EIA-930 fossil net generation (ERCOT)")
    c.set_ylabel("GWh per day"); c.set_title("(c) Daily dynamics, Texas 2024", loc="left"); c.set_ylim(150, 1950); c.legend(fontsize=7.4, loc="upper left", frameon=False); tidy(c)
    c.set_xticks(pd.to_datetime(["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"])); c.set_xticklabels(["Jan", "Apr", "Jul", "Oct"])
    d = fig.add_subplot(gs[1, 2])
    D2 = D[D.ba.isin(["ERCO", "CISO", "NYIS", "ISNE"])]
    xx = np.arange(len(D2)); d.bar(xx - 0.2, D2.corr_levels, 0.4, color=COL["Ridge"], label="daily levels"); d.bar(xx + 0.2, D2.corr_daily_changes, 0.4, color=COL["HGB"], label="day-to-day changes")
    d.set_xticks(xx); d.set_xticklabels([f"{b}\n({s.replace('+', ', ') if len(s) < 8 else 'New England'})" for b, s in zip(D2.ba, D2.states)], fontsize=7.6)
    d.set_ylim(0.9, 1.012); d.set_ylabel("Correlation with EIA-930 fossil generation"); d.set_title("(d) Daily dynamics, 2019-2025", loc="left"); d.legend(fontsize=7.4, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0)); tidy(d)
    e_ = fig.add_subplot(gs[2, :]); C = C.sort_values("bal_share"); x = np.arange(len(C))
    e_.bar(x - 0.2, 100 * C.bal_share, width=0.4, color=COL["Ridge"], label="facilities reporting in all 7 years")
    e_.bar(x + 0.2, 100 * C.hc_share, width=0.4, color=COL["GAM"], label="facilities reporting on ≥95% of days")
    e_.set_xticks(x); e_.set_xticklabels(C.state, rotation=90, fontsize=7); e_.set_ylabel("% of 2019-2025 state CO$_2$")
    e_.set_title("(e) Facility-panel composition", loc="left"); e_.set_ylim(0, 128); e_.legend(fontsize=7.6, loc="upper left", ncol=2, frameon=False); tidy(e_)
    fig.savefig(f"{FIG}/figS1_data_validation.png", dpi=210, bbox_inches="tight"); plt.close(fig); print("figS1")


# ---------------------------------------------------------------- state-by-horizon heatmaps
def figS2():
    P = PC; order = ["Northeast", "Midwest", "South", "West"]
    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#efeee9", "#f09a8a", "#b8282c"])
    blocks = []
    for r in order:
        st = P[(P.region == r) & (P.horizon == 14)].sort_values("d_hr", ascending=False).state.tolist(); blocks.append((r, st))
    n = [len(b[1]) for b in blocks]
    fig = plt.figure(figsize=(7.6, 9.6)); gs = fig.add_gridspec(len(order), 2, height_ratios=n, hspace=0.04, wspace=0.18, bottom=0.17, top=0.96)
    vmax = 0.16
    for j, (col, ttl) in enumerate([("d_hr", "(a) R$^2$(HGB) − R$^2$(ridge)"), ("d_gr", "(b) R$^2$(GAM) − R$^2$(ridge)")]):
        for i, (r, st) in enumerate(blocks):
            ax = fig.add_subplot(gs[i, j])
            M = P[P.state.isin(st)].pivot(index="state", columns="horizon", values=col).loc[st, H].values
            im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_yticks(range(len(st))); ax.set_yticklabels(st if j == 0 else [], fontsize=7)
            ax.set_xticks(range(len(H))); ax.set_xticklabels(H if i == len(order) - 1 else [])
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.tick_params(length=0)
            if j == 0: ax.set_ylabel(r, fontsize=9)
            if i == 0: ax.set_title(ttl, loc="left")
            if i == len(order) - 1: ax.set_xlabel("Forecast horizon (days)")
    cax = fig.add_axes([0.14, 0.07, 0.76, 0.014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("ΔR$^2$  (negative: ridge more accurate; positive: nonlinear model more accurate)", fontsize=8.5); cb.outline.set_visible(False)
    fig.savefig(f"{FIG}/figS2_state_heatmaps.png", dpi=220, bbox_inches="tight"); plt.close(fig); print("figS2")


# ---------------------------------------------------------------- seasonal bias of the fourteen-day forecasts
def figS3():
    O = pd.read_csv(f"{OF}/oos_seasonal.csv"); states = ["TX", "PA", "CA", "ND", "GA", "NM"]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.4), sharex=True)
    for ax, s in zip(axes.flat, states):
        for m, nm in [("ridge", "Ridge"), ("fourier_ridge", "Fourier ridge"), ("gam", "GAM"), ("hgb", "HGB")]:
            x = O[(O.state == s) & (O.series == f"bias_{m}")].sort_values("week")
            ax.plot(x.week, x.value, color=COL[nm], marker=MK[nm], ms=4, lw=1.5, label=nm)
        ax.axhline(0, color=MUTED, lw=0.8, ls=":"); ax.set_title(s, loc="left"); ax.set_xticks(range(1, 13)); tidy(ax)
    for ax in axes[:, 0]: ax.set_ylabel("Mean 14-day forecast error\n(% of annual mean)")
    for ax in axes[1]: ax.set_xlabel("Target month")
    axes[0, 0].legend(fontsize=7.6); save(fig, "figS3_seasonal_bias.png")


for f in (sys.argv[1:] or ["fig03", "fig04", "fig05", "fig06", "fig07", "figS1", "figS2", "figS3"]):
    globals()[f]()
