"""Operational forecasting (Fig. 10). Usage: python figures_operational.py"""
import sys, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS, FIGURES, ensure
FIG = ensure(FIGURES)
H = [1, 2, 3, 5, 7, 10, 14]
INK, MUTED, GRID = "#1f1f1f", "#55554f", "#e4e4df"
CR, CG, CH, CF = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10.5, "axes.titlecolor": INK, "legend.frameon": False})
RO = json.load(open(f"{RESULTS}/operational/results.json"))


def tidy(ax, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.8); ax.set_axisbelow(True)


def fig10():
    M = RO["ops_mean"]; C = pd.DataFrame(RO["ops_contrasts"]); SK = pd.DataFrame(RO["gefs_skill"])
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.0, 5.6))
    def ser(k): return [M[str(h)].get(k, np.nan) for h in H]
    a.plot(H, ser("bench:fourier_ridge"), color="#9a9a94", ls=":", lw=1.6, marker="D", ms=4, label="Fourier ridge, benchmark information")
    a.plot(H, ser("ops:ridge"), color=CR, lw=1.4, marker="o", ms=4, ls="--", label="Ridge, lagged inputs")
    for k, lab, col, mk in [("ops_f:ridge", "Fourier ridge", CF, "D"), ("ops_f:gam", "Fourier GAM", CG, "s"), ("ops_f:hgb", "Fourier HGB", CH, "^")]:
        a.plot(H, ser(k), color=col, lw=1.4, marker=mk, ms=4, ls="-", alpha=0.55, label=lab)
    for k, lab, col, mk in [("ops_fw:ridge", "Fourier ridge + GEFS", CF, "D"), ("ops_fw:gam", "Fourier GAM + GEFS", CG, "s"), ("ops_fw:hgb", "Fourier HGB + GEFS", CH, "^")]:
        a.plot(H, ser(k), color=col, lw=2.2, marker=mk, ms=5, label=lab)
    hs = [h for h in H if not np.isnan(M[str(h)].get("ops_fo:hgb", np.nan))]
    a.plot(hs, [M[str(h)]["ops_fo:hgb"] for h in hs], color=CH, lw=0, marker="^", mfc="white", ms=7, label="Fourier HGB + observed weather")
    a.plot(hs, [M[str(h)]["ops_fo:ridge"] for h in hs], color=CF, lw=0, marker="D", mfc="white", ms=6, label="Fourier ridge + observed weather")
    a.set_xticks(H); a.set_xlabel("Forecast horizon (days)"); a.set_ylabel("Mean state-level R$^2$ (45 states)")
    a.set_title("(a) Operational information set", loc="left"); a.legend(fontsize=7.0, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2); tidy(a)
    off = {"orig": -0.25, "f": 0.0, "fw": 0.25}
    for key, lab, ls in [("lagged inputs only", "lagged inputs", ":"), ("+ Fourier", "+ Fourier", "--"), ("+ Fourier + GEFS", "+ Fourier + GEFS", "-")]:
        for fam, col, mk in [("GAM", CG, "s"), ("HGB", CH, "^")]:
            g = C[C.label == f"{fam} − ridge, {key}"].sort_values("horizon")
            if not len(g): continue
            dx = {"lagged inputs only": -0.25, "+ Fourier": 0.0, "+ Fourier + GEFS": 0.25}[key] + (0.08 if fam == "HGB" else -0.08)
            b.errorbar(g.horizon + dx, g["mean"], yerr=[g["mean"] - g.ci_lo, g.ci_hi - g["mean"]], color=col, marker=mk, ms=4.5, lw=1.3,
                       ls=ls, capsize=2, mfc=col if key != "lagged inputs only" else "white", label=f"{fam} − ridge, {lab}")
    b.axhspan(-0.01, 0.01, color="#efeee9", zorder=0); b.axhline(0, color=MUTED, lw=0.9, ls=":")
    b.set_xticks(H); b.set_xlabel("Forecast horizon (days)"); b.set_ylabel("Difference in mean R$^2$ (95% CI)")
    b.set_title("(b) Nonlinear minus ridge, same information", loc="left"); b.legend(fontsize=7.0, loc="upper left", ncol=1); tidy(b)
    c.plot(SK.h, SK.median_r, color=CF, marker="o", lw=2, label="correlation with observed (median state)")
    c.plot(SK.h, SK.median_skill, color=CG, marker="s", lw=2, label="MSE skill vs climatology (median state)")
    c.axhline(0, color=MUTED, lw=0.9, ls=":"); c.set_xticks(sorted(SK.h)); c.set_ylim(min(-0.1, SK.median_skill.min() - 0.05), 1.0)
    c.set_xlabel("Lead (days)"); c.set_ylabel("Calibrated GEFS degree-day forecast")
    c.set_title("(c) Degree-day forecast skill", loc="left"); c.legend(fontsize=7.6, loc="lower left", bbox_to_anchor=(0.0, 0.12)); tidy(c)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig10_operational.png", dpi=220, bbox_inches="tight"); plt.close(fig); print("fig10")


fig10()
