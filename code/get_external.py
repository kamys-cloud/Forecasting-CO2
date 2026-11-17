"""Retrieve and parse the public external data (writes into DIR/raw and DIR; DIR is normally data/external).
Usage: python get_external.py DIR [--parse-only]
  NOAA CPC population-weighted daily degree days by state (StatesCONUS, 2019-2025) -> cpc_degree_days.csv (state, date, cdd, hdd)
  EIA generation_monthly.xlsx (EIA-923 net generation by state, producer type and source, monthly)
  EIA emission_annual.xlsx (estimated CO2 by state, producer type and source, annual)
  EIA-930 balance files: run get_eia930.py (six-month files, aggregated to daily by balancing authority)"""
import os, sys, subprocess
import pandas as pd
D = sys.argv[1]; RAW = f"{D}/raw"; os.makedirs(RAW, exist_ok=True)
URL = {f"cpc_{y}_{k}.txt": f"https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/{y}/StatesCONUS.{k}.txt"
       for y in range(2019, 2026) for k in ("Heating", "Cooling")}
URL["generation_monthly.xlsx"] = "https://www.eia.gov/electricity/data/state/generation_monthly.xlsx"
URL["emission_annual.xlsx"] = "https://www.eia.gov/electricity/data/state/emission_annual.xlsx"
if "--parse-only" not in sys.argv:
    for fn, u in URL.items():
        if not os.path.exists(f"{RAW}/{fn}"): subprocess.run(["curl", "-sSL", "--max-time", "600", "-o", f"{RAW}/{fn}", u], check=True)


def read_cpc(fn):
    L = [l.strip() for l in open(fn) if "|" in l]
    hdr = L[0].split("|"); rows = [l.split("|") for l in L[1:]]
    d = pd.DataFrame(rows, columns=hdr).set_index("Region")
    d = d.apply(pd.to_numeric, errors="coerce").T; d.index = pd.to_datetime(d.index, format="%Y%m%d")
    return d.stack().rename_axis(["date", "state"])


out = []
for y in range(2019, 2026):
    h = read_cpc(f"{RAW}/cpc_{y}_Heating.txt"); c = read_cpc(f"{RAW}/cpc_{y}_Cooling.txt")
    out.append(pd.concat([c.rename("cdd"), h.rename("hdd")], axis=1).reset_index())
DD = pd.concat(out)[["state", "date", "cdd", "hdd"]].sort_values(["state", "date"]).astype({"cdd": float, "hdd": float})
DD.to_csv(f"{D}/cpc_degree_days.csv", index=False, date_format="%Y-%m-%d"); print("cpc rows", len(DD))
