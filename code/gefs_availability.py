"""Archive audit of the GEFS files used for the degree-day forecasts: for every issue day, the S3 Last-Modified time of the
ensemble-mean (geavg) and ensemble-spread (gespr) files of the 12 UTC run at the longest lead used (f354 = 14 days + 18 h).
Bucket noaa-gefs-pds (NOAA Open Data Dissemination; real-time operational GEFS v12 output), path
gefs.YYYYMMDD/12/atmos/pgrb2ap5/{geavg,gespr}.t12z.pgrb2a.0p50.fHHH.
Usage: python gefs_availability.py START END OUT_CSV"""
import sys, time, urllib.request, concurrent.futures as cf
from email.utils import parsedate_to_datetime
import pandas as pd

START, END, OUT = sys.argv[1:4]
BASE = "https://noaa-gefs-pds.s3.amazonaws.com"


def head(url):
    for k in range(4):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=60) as r:
                return parsedate_to_datetime(r.headers["Last-Modified"]).strftime("%Y-%m-%dT%H:%M:%SZ"), int(r.headers.get("Content-Length", 0))
        except Exception as e:
            if k == 3: return f"ERROR {e!r}"[:60], 0
            time.sleep(2 * (k + 1))


def one(t):
    run = t.strftime("%Y%m%d"); row = dict(issue_date=t.strftime("%Y-%m-%d"), run="12 UTC")
    for prod in ("geavg", "gespr"):
        for lead in (24, 354):
            lm, size = head(f"{BASE}/gefs.{run}/12/atmos/pgrb2ap5/{prod}.t12z.pgrb2a.0p50.f{lead:03d}")
            row[f"{prod}_f{lead:03d}_last_modified"] = lm; row[f"{prod}_f{lead:03d}_bytes"] = size
    return row


with cf.ThreadPoolExecutor(16) as ex:
    rows = list(ex.map(one, pd.date_range(START, END)))
D = pd.DataFrame(rows); D.to_csv(OUT, index=False)
for c in [c for c in D.columns if c.endswith("last_modified")]:
    lm = pd.to_datetime(D[c], errors="coerce", utc=True); init = pd.to_datetime(D.issue_date, utc=True) + pd.Timedelta(hours=12)
    lag = (lm - init).dt.total_seconds() / 3600
    print(c, "missing", int(lm.isna().sum()), "hours after init: median", round(lag.median(), 2), "p95", round(lag.quantile(.95), 2), "max", round(lag.max(), 1),
          "share<=12h", round(float((lag <= 12).mean()), 4))
