import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: F401  (layout and optional publication export)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
C = dict(persistence="#808080", ridge="#2774d9", gam="#ee6a33", hgb="#1fa77a", fourier_ridge="#1b3f8b", sarimax="#8e5cc4", gefs="#c0392b",
         grey="#bdbdbd", dark="#333333")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.grid.axis": "y",
                     "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 100})
