"""Supplementary figure: relative economic value of raw and calibrated P90 warnings against the cost-loss ratio, on identical held-out
origins (17 July 2023-31 December 2025), for the Fourier ridge, GAM, HGB and persistence at one and fourteen days, and the operational
Fourier ridge with GEFS at seven days."""
import os, sys
import numpy as np, pandas as pd
from fig_style import plt, C
import panels as pn

OUT = sys.argv[1]
W = pd.read_csv(os.path.join(pn.ROOT, "results", "warnings", "warnings_identical_origins.csv"))
CL = [0.05, 0.1, 0.2, 0.3, 0.5]
fig, ax = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
panels = [("benchmark", 1, ["persistence", "fourier_ridge", "gam", "hgb"]), ("benchmark", 14, ["persistence", "fourier_ridge", "gam", "hgb"]),
          ("operational", 7, ["ops_ridge", "opsf_ridge", "opsfw_ridge"])]
LAB = {"persistence": "Persistence", "fourier_ridge": "Fourier ridge", "gam": "GAM", "hgb": "HGB", "ops_ridge": "Ridge, lagged inputs",
       "opsf_ridge": "Fourier ridge", "opsfw_ridge": "Fourier ridge + GEFS"}
COL = {"persistence": C["persistence"], "fourier_ridge": C["fourier_ridge"], "gam": C["gam"], "hgb": C["hgb"], "ops_ridge": C["ridge"],
       "opsf_ridge": C["fourier_ridge"], "opsfw_ridge": C["gefs"]}
for a, (s, h, ms) in zip(ax, panels):
    for m in ms:
        for rule, ls in (("raw", "--"), ("calibrated", "-")):
            r = W[(W.set == s) & (W.horizon == h) & (W.model == m) & (W.rule == rule)].iloc[0]
            a.plot(CL, [r[f"value_{c}"] for c in CL], ls, marker="o", ms=3.5, color=COL[m], label=f"{LAB[m]}, {rule}")
    a.axhline(0, color="#444", lw=0.8); a.set_xscale("log"); a.set_xticks(CL); a.set_xticklabels([str(c) for c in CL]); a.minorticks_off()
    a.set(xlabel="Cost–loss ratio", title=f"({'abc'[panels.index((s, h, ms))]}) {'Operational' if s == 'operational' else 'Benchmark'}, {h} d")
    a.title.set_ha("left"); a.title.set_x(0); a.set_ylim(-0.6, 0.8)
ax[0].set_ylabel("Relative economic value")
for a in ax[1:]: a.legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax[0].text(0.5, -0.3, "solid: calibrated; dashed: raw (models as in b)", transform=ax[0].transAxes, ha="center", fontsize=7)
fig.tight_layout(); fig.savefig(OUT); print("saved")
