"""Analytical workflow (Fig. 1). Usage: python figure_workflow.py [OUT_PNG]"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import FIGURES, ensure
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ensure(FIGURES), "fig01_workflow.png")
INK = "#1f1f1f"; SUB = "#3a3a3a"
C = dict(data="#dbe8f7", feat="#e3f1df", model="#fbe3d1", eval="#e9e0f3", base="#fbf1d0", test="#f6dcdc")
E = dict(data="#7c9cc4", feat="#86ad7f", model="#d39a6c", eval="#9b86bd", base="#c8ad5b", test="#c47b7b")

fig = plt.figure(figsize=(11.0, 6.3))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 110); ax.set_ylim(0, 63); ax.axis("off")


def box(x, y, w, h, kind, title, sub=None, tfs=10.5, sfs=8.6, ty=0.66, sy=0.30):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0",
                                fc=C[kind], ec=E[kind], lw=1.1))
    if sub:
        ax.text(x + w / 2, y + h * ty, title, ha="center", va="center", fontsize=tfs, weight="bold", color=INK)
        ax.text(x + w / 2, y + h * sy, sub, ha="center", va="center", fontsize=sfs, color=SUB, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tfs, weight="bold", color=INK)


def arrow(x1, y1, x2, y2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.0,
                                 color="#555555", connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


# left column (x 2..72)
box(8, 56, 58, 6, "data", "EPA CAMPD daily facility data",
    "CO$_2$ mass, gross load, heat input, SO$_2$, NO$_x$, reporting facilities")
arrow(37, 56, 37, 53.4)
box(8, 47.4, 58, 6, "data", "State–day aggregation on a complete daily calendar",
    "missing days kept missing; 45 states, 2019–2025; audited; checked against EIA data")
arrow(37, 47.4, 37, 44.6)
ax.text(37, 43.0, "Explicit CO$_2$ features; target-reconstructable ratio features excluded",
        ha="center", va="center", fontsize=8.8, style="italic", color=SUB)
box(2.5, 34.2, 21, 6.4, "feat", "Calendar/year", r"$c(t)\in\mathbb{R}^{8}$", sfs=9.5)
box(26.5, 34.2, 21, 6.4, "feat", "CO$_2$ history", r"$l(t)\in\mathbb{R}^{13}$", sfs=9.5)
box(50.5, 34.2, 21, 6.4, "feat", "Operational", r"$o(t)\in\mathbb{R}^{6}$", sfs=9.5)
for x in (13, 37, 61):
    arrow(x, 34.2, 37 + (x - 37) * 0.25, 30.6)
box(14, 25.2, 46, 5.4, "feat", r"$x(t)\in\mathbb{R}^{27}$ and target $y(t+h)$",
    "target dated exactly $t+h$ calendar days, formed before complete-case filtering", tfs=10)
arrow(28, 25.2, 13, 22.4); arrow(37, 25.2, 37, 22.4); arrow(46, 25.2, 61, 22.4)
box(1.0, 13.6, 22.5, 8.4, "model", "Ridge ↔ GAM ↔ HGB",
    "45 states × 7 horizons\n$h\\in\\{1,2,3,5,7,10,14\\}$ days\nsame 27 inputs", ty=0.80, sy=0.36, tfs=9.8, sfs=8.0)
box(25.5, 13.6, 22.5, 8.4, "base", "Family × seasonal basis",
    "3 families × 5 bases (Fourier,\nspline); naive, ETS, SARIMAX, Prophet", ty=0.76, sy=0.33, tfs=9.8, sfs=8.0)
box(50.0, 13.6, 22.5, 8.4, "model", "Ablation and operations",
    "annual components at 7 horizons;\ninputs lagged one day + GEFS\ndegree-day forecasts (ex-ante)", ty=0.80, sy=0.36, tfs=9.8, sfs=8.0)
arrow(12, 13.6, 25, 10.8); arrow(37, 13.6, 37, 10.8); arrow(61, 13.6, 49, 10.8)
box(4, 2.2, 66, 8.6, "eval", "Expanding-window rolling-origin backtest",
    "hyperparameters selected once before the first origin, then frozen\n"
    "refit every 14 calendar days; training and tuning use only outcomes observed by each origin", tfs=10.5)

# right column (x 80..108)
box(82, 52.5, 27, 7.5, "data", "Physical and weather data", "heat input × emission factor; CPC degree\ndays; EIA-923/930; NOAA GEFS forecasts", sfs=8.0)
arrow(66.4, 50.4, 81.6, 55.0)
box(82, 39.6, 27, 8.8, "test", "Diebold–Mariano test", "HLN small-sample correction,\nNewey–West HAC variance")
arrow(95.5, 39.6, 95.5, 35.4)
box(82, 27.6, 27, 7.8, "test", "Holm / BH; TOST equivalence", "per-state tests; ±0.01 R$^2$ margin")
arrow(95.5, 27.6, 95.5, 23.4)
box(82, 13.0, 27, 10.4, "test", "Inference and decisions",
    "date- and year-block bootstrap; TOST;\nnational totals, warnings, carbon-\naware scheduling in 4 grids", ty=0.78, sy=0.36)
arrow(70, 6.5, 81.6, 43.0, rad=0.26)
ax.text(78.4, 26.0, "paired daily errors", rotation=72, fontsize=8.4, color=SUB, ha="center", va="center")
fig.savefig(out, dpi=220)
print("saved", out)
