"""Daily balancing-authority series from the EIA Hourly Electric Grid Monitor workbooks (sheet 'Published Daily Data'), including EIA's
estimated CO2 emissions of generated and consumed electricity (metric tons), 2019-2025.
Usage: python parse_gridmonitor.py RAW_DIR OUT_CSV      (OUT_CSV is normally data/external/gridmonitor_ba_daily.csv)"""
import sys, glob, os
import openpyxl, pandas as pd

raw, out = sys.argv[1], sys.argv[2]
frames = []
for fn in sorted(glob.glob(os.path.join(raw, "*.xlsx"))):
    wb = openpyxl.load_workbook(fn, read_only=True); ws = wb["Published Daily Data"]
    rows = ws.iter_rows(values_only=True); hdr = list(next(rows))
    frames.append(pd.DataFrame(list(rows), columns=hdr)); wb.close()
G = pd.concat(frames, ignore_index=True)
num = lambda c: pd.to_numeric(G[c], errors="coerce")
o = pd.DataFrame({"ba": G["BA"], "date": pd.to_datetime(G["Local date"]), "netgen_mwh": num("Net generation"), "demand_mwh": num("Demand"),
                  "co2_generated_t": num("CO2 Emissions Generated"), "co2_consumed_t": num("CO2 Emissions Consumed"),
                  "consumed_mwh": num("Consumed Electricity"), "co2_coal_t": num("CO2 Emissions: COL"), "co2_gas_t": num("CO2 Emissions: NG"),
                  "co2_oil_t": num("CO2 Emissions: OIL")})
o = o[(o.date >= "2019-01-01") & (o.date <= "2025-12-31")]
o.to_csv(out, index=False); print(o.groupby("ba").size())
