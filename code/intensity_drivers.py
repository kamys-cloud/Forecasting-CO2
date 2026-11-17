"""Same-day drivers of daily average carbon intensity (facility-mapped) in the four balancing authorities over the evaluation period:
variance explained by linear regressions on the wind-and-solar share of generation, on demand, and on both.
Usage: python intensity_drivers.py OUT_CSV"""
import sys
import numpy as np, pandas as pd
import ba_data as bd

rows = []
for ba in bd.BAS:
    I = bd.intensity(ba, "crosswalk"); b = bd.eia930(ba); g = bd.netgen(ba)
    d = pd.DataFrame(dict(I=I, vre=(b.wind + b.solar) / g, dem=b.demand)).loc["2022-07-17":"2025-12-31"].dropna()
    def r2(cols):
        X = np.column_stack([np.ones(len(d))] + [d[c] for c in cols]); e = d.I - X @ np.linalg.lstsq(X, d.I, rcond=None)[0]
        return float(1 - e.var() / d.I.var())
    rows.append(dict(ba=ba, n=len(d), r2_wind_solar=r2(["vre"]), r2_demand=r2(["dem"]), r2_both=r2(["vre", "dem"]), mean_vre_share=float(d.vre.mean())))
T = pd.DataFrame(rows); T.to_csv(sys.argv[1], index=False); print(T.round(3))
