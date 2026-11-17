"""Population-weighted state degree-day FORECASTS from the archived NOAA GEFS v12 ensemble mean (0.5 deg, AWS open data).
For each run (12 UTC of issue day t, available by about 18 UTC) and target day t+h (h in HS), the target day is the 24 h from 06 UTC
of t+h (approximately the U.S. local day); daily Tmax/Tmin are the max/min of the four 6-h TMAX/TMIN:2 m fields, county degree days
use (Tmax+Tmin)/2 against 65 F, and states are population-weighted over 2020 county centres of population (as CPC does over stations).
Usage: python gefs_degree_days.py OUTDIR CENPOP.txt START END [HS] [WORKERS]       (dates YYYY-MM-DD; HS e.g. 1,2,3,4,5,6,7,10,14)
Output: OUTDIR/YYYYMMDD.csv (state, h, target_date, hdd_f, cdd_f, tmean_f); existing files are skipped."""
import os, sys, io, time, urllib.request, concurrent.futures as cf
import numpy as np, pandas as pd
import eccodes

OUT, CEN, START, END = sys.argv[1:5]
HS = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "1,2,3,4,5,6,7,10,14").split(",")]
WORKERS = int(sys.argv[6]) if len(sys.argv) > 6 else 12
os.makedirs(OUT, exist_ok=True)
BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
ABBR = {"Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
        "Florida": "FL", "Georgia": "GA", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
        "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
        "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
        "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
        "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
        "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}
C = pd.read_csv(CEN, encoding="utf-8-sig"); C = C[C.STNAME.isin(ABBR)].copy(); C["st"] = C.STNAME.map(ABBR)
lat = C.LATITUDE.values.astype(float); lon = (C.LONGITUDE.values.astype(float) + 360) % 360
# 0.5-deg global grid: lat 90 -> -90 (361), lon 0 -> 359.5 (720); bilinear weights at county centres
fi = (90 - lat) / 0.5; fj = lon / 0.5
i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int); di = fi - i0; dj = fj - j0
j1 = (j0 + 1) % 720
W4 = [((1 - di) * (1 - dj), i0, j0), ((1 - di) * dj, i0, j1), (di * (1 - dj), i0 + 1, j0), (di * dj, i0 + 1, j1)]
popw = C.POPULATION.values.astype(float); states = C.st.values


def get(url, rng=None, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=60) as r: return r.read()
        except Exception as e:
            if k == tries - 1: raise
            time.sleep(2 * (k + 1))


def field(run, lead):
    """TMAX and TMIN (2 m, 6-h window ending at `lead`) at county centres, in Kelvin."""
    f = f"{BASE}/gefs.{run}/12/atmos/pgrb2ap5/geavg.t12z.pgrb2a.0p50.f{lead:03d}"
    idx = get(f + ".idx").decode().splitlines()
    out = {}
    for n, line in enumerate(idx):
        parts = line.split(":")
        if parts[3] in ("TMAX", "TMIN") and parts[4] == "2 m above ground":
            start = int(parts[1]); end = int(idx[n + 1].split(":")[1]) - 1 if n + 1 < len(idx) else ""
            msg = get(f, (start, end))
            gid = eccodes.codes_new_from_message(msg)
            v = eccodes.codes_get_values(gid).reshape(eccodes.codes_get(gid, "Nj"), eccodes.codes_get(gid, "Ni")); eccodes.codes_release(gid)
            out[parts[3]] = sum(w * v[i, j] for w, i, j in W4)
    return out


def run_day(t):
    run = t.strftime("%Y%m%d"); fn = f"{OUT}/{run}.csv"
    if os.path.exists(fn): return run, "skip"
    leads = sorted({24 * h + 6 * k for h in HS for k in range(4)})
    try:
        with cf.ThreadPoolExecutor(WORKERS) as ex:
            F = dict(zip(leads, ex.map(lambda L: field(run, L), leads)))
    except Exception as e:
        return run, f"FAIL {e!r}"[:200]
    rows = []
    for h in HS:
        ls = [24 * h + 6 * k for k in range(4)]
        tmax = np.max([F[L]["TMAX"] for L in ls], axis=0); tmin = np.min([F[L]["TMIN"] for L in ls], axis=0)
        tm = ((tmax + tmin) / 2 - 273.15) * 9 / 5 + 32
        df = pd.DataFrame(dict(st=states, w=popw, hdd=np.maximum(65 - tm, 0) * popw, cdd=np.maximum(tm - 65, 0) * popw, tm=tm * popw))
        g = df.groupby("st").sum()
        for s, r in g.iterrows():
            rows.append(dict(state=s, h=h, target_date=(t + pd.Timedelta(days=h)).strftime("%Y-%m-%d"), hdd_f=r.hdd / r.w,
                             cdd_f=r.cdd / r.w, tmean_f=r.tm / r.w))
    pd.DataFrame(rows).to_csv(fn + ".tmp", index=False); os.replace(fn + ".tmp", fn)
    return run, "ok"


if __name__ == "__main__":
    for t in pd.date_range(START, END):
        t0 = time.time(); r, st = run_day(t)
        print(r, st, f"{time.time() - t0:.1f}s", flush=True)
