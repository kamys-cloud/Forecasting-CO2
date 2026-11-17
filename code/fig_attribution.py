"""Conditional attribution figure: out-of-sample incremental R2 of same-day degree days (a) across 45 states without and with the
annual Fourier terms and controls, (b) for the four balancing authorities (controls include same-day wind, solar, hydro, nuclear
and net interchange)."""
import os, sys
import numpy as np, pandas as pd
from fig_style import plt, C
import ba_data as bd

OUT = sys.argv[1]; A = os.path.join(bd.ROOT, "results", "attribution")
S = pd.read_csv(os.path.join(A, "conditional_attribution_states.csv")); B = pd.read_csv(os.path.join(A, "conditional_attribution_ba.csv"))
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
cols = [("inc_D_alone", "Weekday only", "#9ecae1"), ("inc_D_given_F", "+ annual Fourier", "#4292c6"), ("inc_D_given_FC", "+ Fourier, mix,\npersistence", "#08306b")]
rng = np.random.default_rng(0)
for i, (c, lab, col) in enumerate(cols):
    v = S[c].values; ax[0].boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color="k"), patch_artist=True,
                                   boxprops=dict(facecolor=col, alpha=0.45, edgecolor=col))
    ax[0].scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=col, zorder=3)
    ax[0].annotate(f"median {np.median(v):.3f}", (i, max(v) + 0.03), ha="center", fontsize=8)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([c[1] for c in cols]); ax[0].axhline(0, color="#444", lw=0.8, ls=":")
ax[0].set(ylabel="Incremental out-of-sample R$^2$ of HDD, CDD", title="(a) 45 states: degree days added to ...", ylim=(-0.15, 1.0))
ax[0].title.set_ha("left"); ax[0].title.set_x(0)
x = np.arange(len(B)); w = 0.26
for k, (c, lab, col) in enumerate(cols):
    ax[1].bar(x + (k - 1) * w, B[c], w, color=col, label=lab.replace("\n", " "))
ax[1].set_xticks(x); ax[1].set_xticklabels([bd.NAME[b] for b in B.ba]); ax[1].axhline(0, color="#444", lw=0.8)
ax[1].set(title="(b) Balancing authorities (controls incl. renewables)", ylabel="Incremental out-of-sample R$^2$")
ax[1].title.set_ha("left"); ax[1].title.set_x(0); ax[1].legend(fontsize=8, loc="upper right"); ax[1].set_ylim(-0.05, 1.0)
fig.tight_layout(); fig.savefig(OUT); print("saved")
