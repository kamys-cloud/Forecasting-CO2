"""Download the public inputs of the balancing-authority analyses: EIA-860 Schedule 2 (Plant) files for 2019-2025 and the EIA Hourly
Electric Grid Monitor workbooks of ERCOT, CAISO, NYISO and ISO-NE.
Usage: python get_raw_inputs.py RAW_DIR      (writes RAW_DIR/eia860/2___Plant_Yyyyy.xlsx and RAW_DIR/gridmonitor/*.xlsx)"""
import os, sys, subprocess, zipfile

RAW = sys.argv[1]; E = os.path.join(RAW, "eia860"); G = os.path.join(RAW, "gridmonitor"); os.makedirs(E, exist_ok=True); os.makedirs(G, exist_ok=True)
for y in range(2019, 2026):
    url = (f"https://www.eia.gov/electricity/data/eia860/archive/xls/eia860{y}.zip" if y < 2025 else
           f"https://www.eia.gov/electricity/data/eia860/xls/eia860{y}.zip")
    z = os.path.join(E, f"eia860{y}.zip")
    if not os.path.exists(z): subprocess.run(["curl", "-sSL", "--max-time", "900", "-o", z, url], check=True)
    with zipfile.ZipFile(z) as f: f.extract(f"2___Plant_Y{y}.xlsx", E)
for ba in ("ERCO", "CISO", "NYIS", "ISNE"):
    fn = os.path.join(G, f"{ba}.xlsx")
    if not os.path.exists(fn):
        subprocess.run(["curl", "-sSL", "--max-time", "1800", "-o", fn, f"https://www.eia.gov/electricity/gridmonitor/knownissues/xls/{ba}.xlsx"], check=True)
print("done")
