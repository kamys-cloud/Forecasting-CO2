"""Download EIA-930 six-month balance files 2019-2025 and aggregate selected BAs to daily (local date) sums (MWh).
Usage: python get_eia930.py [DIR]      (DIR is normally data/external; downloads go to DIR/raw, output DIR/eia930_daily.csv)"""
import subprocess, os, sys, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import EXTERNAL
P = sys.argv[1] if len(sys.argv) > 1 else EXTERNAL
os.makedirs(f"{P}/raw", exist_ok=True)
BAS = ["ERCO", "CISO", "NYIS", "ISNE", "SOCO", "PJM", "MISO", "SWPP", "FPL", "DUK", "TVA", "BPAT", "PACE", "PNM", "AZPS"]
COLS = {"Demand (MW) (Adjusted)": "demand", "Net Generation (MW) (Adjusted)": "netgen",
        "Net Generation (MW) from Coal (Adjusted)": "coal", "Net Generation (MW) from Natural Gas (Adjusted)": "gas",
        "Net Generation (MW) from Nuclear (Adjusted)": "nuclear", "Net Generation (MW) from All Petroleum Products (Adjusted)": "oil",
        "Net Generation (MW) from Hydropower and Pumped Storage (Adjusted)": "hydro", "Net Generation (MW) from Solar (Adjusted)": "solar",
        "Net Generation (MW) from Wind (Adjusted)": "wind"}
out = []
for y in range(2019, 2026):
    for half in ("Jan_Jun", "Jul_Dec"):
        fn = f"{P}/raw/930_{y}_{half}.csv"
        part = f"{P}/raw/930d_{y}_{half}.csv"
        if os.path.exists(part): out.append(pd.read_csv(part, parse_dates=["date"])); continue
        if not os.path.exists(fn):
            subprocess.run(["curl", "-s", "--max-time", "600", "-o", fn, f"https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{y}_{half}.csv"], check=True)
        hdr = pd.read_csv(fn, nrows=0).columns
        adj = [c for c in hdr if c.endswith("(Adjusted)")]
        pick = {"demand": ["Demand (MW) (Adjusted)"], "netgen": ["Net Generation (MW) (Adjusted)"],
                "coal": [c for c in adj if "from Coal" in c], "gas": [c for c in adj if "from Natural Gas" in c],
                "nuclear": [c for c in adj if "from Nuclear" in c], "oil": [c for c in adj if "Petroleum" in c],
                "hydro": [c for c in adj if "Hydropower" in c or "Pumped Storage" in c],
                "solar": [c for c in adj if "from Solar" in c], "wind": [c for c in adj if "from Wind" in c]}
        need = sorted(set(sum(pick.values(), [])))
        d = pd.read_csv(fn, usecols=["Balancing Authority", "Local Time at End of Hour"] + need, low_memory=False, thousands=",")
        d = d[d["Balancing Authority"].isin(BAS)]
        for c in need: d[c] = pd.to_numeric(d[c], errors="coerce")
        for k, cs in pick.items(): d[k] = d[cs].sum(axis=1, min_count=1) if cs else np.nan
        t = pd.to_datetime(d["Local Time at End of Hour"], format="%m/%d/%Y %I:%M:%S %p") - pd.Timedelta(hours=1)
        d["date"] = t.dt.normalize()
        g = d.groupby(["Balancing Authority", "date"])[list(pick)].agg(["sum", "count"])
        s = g.xs("sum", axis=1, level=1); n = g.xs("count", axis=1, level=1)
        s["hours_demand"] = n["demand"]; s = s.reset_index().rename(columns={"Balancing Authority": "ba"})
        s.to_csv(part, index=False); out.append(s); os.remove(fn); print(y, half, len(s), flush=True)
A = pd.concat(out); A["date"] = pd.to_datetime(A.date); A = A.groupby(["ba", "date"]).sum(min_count=1).reset_index()
A.to_csv(f"{P}/eia930_daily.csv", index=False); print("saved", A.shape)
