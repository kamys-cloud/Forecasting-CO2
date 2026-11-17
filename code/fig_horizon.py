"""Fig. 2: (a) national mean state-level R2 by horizon; (b) HGB - ridge by state and national mean with the joint 28-day
date-block bootstrap 95% interval (the procedure used for all national differences)."""
import os, sys
import numpy as np, pandas as pd
from fig_style import plt, C
import panels as pn
from jboot import Panel

OUT = sys.argv[1]
I = pd.read_csv(os.path.join(pn.ROOT, "results", "inference", "hgb_ridge_by_horizon.csv"))
I = I[I.a == "hgb"].set_index("horizon")
mean = {m: [] for m in ("persistence", "ridge", "gam", "hgb")}
for h in pn.H:
    P = Panel(pn.frames(pn.benchmark(h)), list(mean)); [mean[m].append(np.nanmean(P.r2(m))) for m in mean]
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9))
H = pn.H
for m, mk, lab, ls in (("persistence", "D", "Persistence", "--"), ("ridge", "o", "Ridge", "-"), ("gam", "s", "GAM", "-"), ("hgb", "^", "HGB", "-")):
    ax[0].plot(H, mean[m], ls, marker=mk, color=C[m], lw=2, ms=6, label=lab)
ax[0].set(xticks=H, xlabel="Forecast horizon (days)", ylabel="Mean state-level R$^2$ (45 states)", title="(a) Mean accuracy by model")
ax[0].title.set_ha("left"); ax[0].title.set_x(0); ax[0].legend(loc="upper right")
st = [c for c in I.columns if c.startswith("state_")]
for c in st:
    ax[1].plot(H, I[c].values, color=C["grey"], lw=0.8, zorder=1)
    if I[c].iloc[-1] < -0.01: ax[1].annotate(c[6:], (14.15, I[c].iloc[-1]), fontsize=8, color="#666666", va="center")
ax[1].fill_between(H, I.joint_lo, I.joint_hi, color=C["hgb"], alpha=0.22, lw=0, zorder=2)
ax[1].plot(H, I["mean"], "-o", color="#0d7a55", lw=2.4, ms=6, zorder=3, label="National mean (joint date-block 95% CI)")
ax[1].axhline(0, color="#444444", lw=0.8, ls=":")
ax[1].set(xticks=H, xlabel="Forecast horizon (days)", ylabel="R$^2$(HGB) $-$ R$^2$(ridge)", title="(b) HGB relative to ridge: states (grey) and national mean")
ax[1].title.set_ha("left"); ax[1].title.set_x(0); ax[1].legend(loc="upper left")
fig.tight_layout(); fig.savefig(OUT); print("saved", OUT)
