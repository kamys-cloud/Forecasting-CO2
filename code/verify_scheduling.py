"""Independent recomputation of the scheduling capture shares (no code shared with ba_data.py or scheduling_eval.py): facility-mapped
intensity rebuilt from data/derived/ba_daily_co2.csv and data/external/eia930_daily.csv, the decision rule re-implemented, and the result
compared with results/scheduling/scheduling_capture.csv. Usage: python verify_scheduling.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DERIVED, EXTERNAL, FORECASTS, RESULTS

X = pd.read_csv(f"{DERIVED}/ba_daily_co2.csv", parse_dates=["date"]); B = pd.read_csv(f"{EXTERNAL}/eia930_daily.csv", parse_dates=["date"])
F = pd.read_csv(f"{FORECASTS}/scheduling_intensity_crosswalk.csv.gz", parse_dates=["date"])
C = pd.read_csv(f"{RESULTS}/scheduling/scheduling_capture.csv")
LAB = {("four_gefs", "ridge"): "Fourier ridge + GEFS", ("four", "ridge"): "Fourier ridge", ("orig", "ridge"): "Ridge", ("four_gefs", "gam"): "Fourier GAM + GEFS"}
bad, n = [], 0
for ba in ["ERCO", "CISO", "NYIS", "ISNE"]:
    g = B[B.ba == ba].set_index("date").netgen.asfreq("D").astype(float)
    g = g.where((g > 0) & (g >= 0.6 * g.rolling(29, center=True, min_periods=5).median()))
    I = (X[X.ba == ba].set_index("date").co2_xw.asfreq("D") * 907.18474 / g)
    origins = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(f"{RESULTS}/scheduling/evaluation_origins.csv").query("ba == @ba").origin))
    A = np.column_stack([I.reindex(origins + pd.Timedelta(days=k)).values for k in range(1, 8)])
    for (v, m), lab in LAB.items():
        f = F[(F.balancing_authority == ba) & (F.variant == v) & (F.model == m)].pivot(index="date", columns="horizon", values="forecast").reindex(origins)
        Fm = f[list(range(1, 8))].values
        ok = ~np.isnan(A).any(1) & ~np.isnan(Fm).any(1)
        ref = C[(C.ba == ba) & (C.rule == lab)].iloc[0]
        # restrict to the common origins used by the evaluation (all rules valid): recompute on rows valid here and compare within 0.01
        ch = np.argmin(np.where(ok[:, None], Fm, 0), 1); r = np.arange(len(origins))
        sav = (A.mean(1) - A[r, ch])[ok]; ora = (A.mean(1) - A.min(1))[ok]
        cap = sav.sum() / ora.sum(); n += 1
        if abs(cap - ref.capture_avg) > 0.01: bad.append((ba, lab, cap, ref.capture_avg))
        print(f"{ba:5s} {lab:22s} recomputed {100*cap:5.1f}%  reported {100*ref.capture_avg:5.1f}%")
print(f"Scheduling verification: {n - len(bad)}/{n} capture shares agree within 1 percentage point")
for b in bad: print("  MISMATCH", b)
