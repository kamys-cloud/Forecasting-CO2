"""Scheduling figure: (a) ranking performance - share of the perfect-foresight average-intensity reduction captured (crosswalk
intensity; open markers: decisions scored on EIA-reported intensity); (b) average-intensity and modelled marginal saving per MWh
shifted; (c) weekly electrolytic-hydrogen process: emissions reduction relative to a fixed schedule on average and marginal bases."""
import os, sys, json
import numpy as np, pandas as pd
from fig_style import plt, C
import ba_data as bd

OUT = sys.argv[1]; D = os.path.join(bd.ROOT, "results", "scheduling")
Cp = pd.read_csv(os.path.join(D, "scheduling_capture.csv")); Pr = pd.read_csv(os.path.join(D, "process_scheduling.csv"))
rules = [("Same weekday last week", "#bdbdbd"), ("Fourier ridge", "#9ecae1"), ("Fourier ridge + GEFS", C["fourier_ridge"]),
         ("Fourier GAM + GEFS", C["gam"]), ("Fourier HGB + GEFS", C["hgb"])]
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9))
x = np.arange(4); w = 0.16
for k, (r, col) in enumerate(rules):
    d = Cp[Cp.rule == r].set_index("ba").loc[bd.BAS]
    ci = np.array([json.loads(v) for v in d.capture_avg_ci]) * 100
    ax[0].bar(x + (k - 2) * w, 100 * d.capture_avg, w, color=col, label=r, yerr=[100 * d.capture_avg - ci[:, 0], ci[:, 1] - 100 * d.capture_avg],
              error_kw=dict(lw=0.7, capsize=1.5))
    ax[0].scatter(x + (k - 2) * w, 100 * d.capture_eia, facecolor="white", edgecolor="k", s=14, zorder=4, lw=0.7)
ax[0].set_xticks(x); ax[0].set_xticklabels([bd.NAME[b] for b in bd.BAS]); ax[0].axhline(0, color="#444", lw=0.8)
ax[0].set(ylabel="Share of perfect-foresight reduction (%)", title="(a) Ranking of low-intensity days")
ax[0].legend(fontsize=7, loc="upper left", ncol=2); ax[0].set_ylim(-5, 100)
for k, (r, col, lab) in enumerate([("Perfect foresight", "#555555", "Perfect foresight"), ("Fourier ridge + GEFS", C["fourier_ridge"], "Fourier ridge + GEFS")]):
    d = Cp[Cp.rule == r].set_index("ba").loc[bd.BAS]
    ax[1].bar(x + (2 * k - 1.5) * 0.18, d.saving_avg_kg_mwh, 0.18, color=col, alpha=0.45, label=f"{lab}: average")
    ci = np.array([json.loads(v) for v in d.saving_marginal_ci])
    ax[1].bar(x + (2 * k - 0.5) * 0.18, d.saving_marginal_kg_mwh, 0.18, color=col, label=f"{lab}: marginal",
              yerr=[d.saving_marginal_kg_mwh - ci[:, 0], ci[:, 1] - d.saving_marginal_kg_mwh], error_kw=dict(lw=0.7, capsize=1.5))
ax[1].set_xticks(x); ax[1].set_xticklabels([bd.NAME[b] for b in bd.BAS])
ax[1].set(ylabel="Saving (kg CO$_2$ per MWh shifted)", title="(b) Average vs modelled marginal saving"); ax[1].legend(fontsize=7, loc="upper right")
for k, (r, col) in enumerate([("Perfect foresight", "#555555"), ("Fourier ridge + GEFS", C["fourier_ridge"]), ("Fourier GAM + GEFS", C["gam"])]):
    d = Pr[Pr.rule == r].set_index("ba").loc[bd.BAS]
    ax[2].bar(x + (k - 1) * 0.26, d.avg_reduction_pct, 0.26, color=col, alpha=0.45)
    ax[2].scatter(x + (k - 1) * 0.26, d.marginal_reduction_pct, marker="D", color=col, s=22, zorder=4, edgecolor="k", lw=0.5)
ax[2].set_xticks(x); ax[2].set_xticklabels([bd.NAME[b] for b in bd.BAS])
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
h = [Patch(color="#555555", alpha=0.45, label="Perfect foresight"), Patch(color=C["fourier_ridge"], alpha=0.45, label="Fourier ridge + GEFS"),
     Patch(color=C["gam"], alpha=0.45, label="Fourier GAM + GEFS"), Line2D([], [], marker="D", ls="", color="k", label="marginal basis")]
ax[2].legend(handles=h, fontsize=7, loc="upper right")
ax[2].set(ylabel="Reduction vs fixed schedule (%)", title="(c) Electrolytic H$_2$ process (bars: average)")
for a in ax: a.title.set_ha("left"); a.title.set_x(0)
fig.tight_layout(); fig.savefig(OUT); print("saved")
