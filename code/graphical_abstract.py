"""Graphical abstract. Usage: python graphical_abstract.py [OUT_PNG]"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS, FIGURES, ensure

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ensure(FIGURES), "graphical_abstract.png")
INK, MUTED, GRID = "#1f1f1f", "#55554f", "#e4e4df"
CR, CG, CH, CF = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False})
H = [1, 2, 3, 5, 7, 10, 14]
F = pd.DataFrame(json.load(open(f"{RESULTS}/factorial/results.json"))["factorial"]).set_index("horizon").loc[H]
O = json.load(open(f"{RESULTS}/operational/results.json"))["ops_mean"]
CP = pd.read_csv(f"{RESULTS}/scheduling/scheduling_capture.csv")


def tidy(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)


fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.3, 5.3), gridspec_kw=dict(width_ratios=[1.15, 1, 0.9]))
# (1) the long-horizon nonlinear gain disappears with a shared seasonal basis
a.plot(H, F.gam - F.ridge, color=CG, marker="s", lw=2.2, label="GAM − ridge, calendar encodings")
a.plot(H, F.hgb - F.ridge, color=CH, marker="^", lw=2.2, label="HGB − ridge, calendar encodings")
a.plot(H, F.fourier_gam - F.fourier_ridge, color=CG, marker="s", lw=1.6, ls="--", mfc="white", label="GAM − ridge, shared Fourier basis")
a.plot(H, F.fourier_hgb - F.fourier_ridge, color=CH, marker="^", lw=1.6, ls="--", mfc="white", label="HGB − ridge, shared Fourier basis")
a.axhspan(-0.01, 0.01, color="#efeee9", zorder=0); a.axhline(0, color=MUTED, lw=0.9, ls=":")
a.set_xticks(H); a.set_xlabel("Forecast horizon (days)"); a.set_ylabel("Difference in mean state R$^2$ (45 states)")
a.set_title("Seasonal representation, not nonlinearity", loc="left", weight="bold"); a.legend(fontsize=8.4, loc="upper left"); tidy(a)
# (2) weather forecasts available at issue
ser = lambda k: [O[str(h)][k] for h in H]
b.plot(H, ser("ops:ridge"), color=CR, marker="o", lw=1.6, ls="--", label="Ridge, lagged inputs")
b.plot(H, ser("ops_f:ridge"), color=CF, marker="D", lw=1.6, alpha=0.6, label="+ annual Fourier terms")
b.plot(H, ser("ops_fw:ridge"), color=CF, marker="D", lw=2.4, label="+ GEFS degree-day forecasts")
b.set_xticks(H); b.set_xlabel("Forecast horizon (days)"); b.set_ylabel("Mean state R$^2$, inputs available at issue")
b.set_title("Operational forecasts", loc="left", weight="bold"); b.legend(fontsize=8.4, loc="upper right"); tidy(b)
# (3) carbon-aware scheduling on facility-mapped balancing-authority intensity
ba = [("ERCO", "ERCOT"), ("CISO", "CAISO"), ("NYIS", "NYISO"), ("ISNE", "ISO-NE")]
x = np.arange(len(ba)); wd = 0.36
v1 = [100 * CP[(CP.ba == k) & (CP.rule == "Same weekday last week")].capture_avg.iloc[0] for k, _ in ba]
v2 = [100 * CP[(CP.ba == k) & (CP.rule == "Fourier ridge + GEFS")].capture_avg.iloc[0] for k, _ in ba]
c.bar(x - wd / 2, v1, wd, color="#b9b9b3", label="Same weekday last week")
c.bar(x + wd / 2, v2, wd, color=CF, label="Fourier ridge + GEFS")
for xi, v in zip(x + wd / 2, v2): c.text(xi, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9, color=INK)
c.axhline(0, color=MUTED, lw=0.8); c.set_xticks(x); c.set_xticklabels([n for _, n in ba]); c.set_ylim(min(0, min(v1)) - 2, 100)
c.set_ylabel("Share of perfect-foresight reduction (%)"); c.set_title("Low-carbon-day scheduling", loc="left", weight="bold")
c.legend(fontsize=8.4, loc="upper left"); tidy(c)
fig.tight_layout(); fig.savefig(OUT, dpi=300); plt.close(fig); print(OUT)
