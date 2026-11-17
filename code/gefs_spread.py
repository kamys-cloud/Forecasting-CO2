"""Population-weighted state ensemble spread of 2-m temperature from the archived NOAA GEFS v12 operational runs
(noaa-gefs-pds, product pgrb2ap5, file gespr = ensemble standard deviation, 0.5 deg, 12 UTC run of issue day t).
For target day t+h (06-06 UTC) the spread of TMP:2 m is taken at 18 UTC of t+h (lead 24h+6) and 06 UTC of t+h+1 (lead 24h+18), the
times nearest the daily maximum and minimum, bilinearly interpolated to the 2020 county centres of population and population-weighted
to states; the daily value is their mean (degrees F).
Usage: python gefs_spread.py OUTDIR CENPOP.txt START END [HS] [WORKERS]
Output: OUTDIR/YYYYMMDD.csv (state, h, target_date, tsd_f); existing files are skipped."""
import os, sys, time, urllib.request, concurrent.futures as cf
import numpy as np, pandas as pd
import eccodes

OUT, CEN, START, END = sys.argv[1:5]
HS = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "1,2,3,5,7,10,14").split(",")]
WORKERS = int(sys.argv[6]) if len(sys.argv) > 6 else 8
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
fi = (90 - lat) / 0.5; fj = lon / 0.5
i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int); di = fi - i0; dj = fj - j0
j1 = (j0 + 1) % 720
W4 = [((1 - di) * (1 - dj), i0, j0), ((1 - di) * dj, i0, j1), (di * (1 - dj), i0 + 1, j0), (di * dj, i0 + 1, j1)]
popw = C.POPULATION.values.astype(float); states = C.st.values


def get(url, rng=None, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=90) as r: return r.read()
        except Exception:
            if k == tries - 1: raise
            time.sleep(3 * (k + 1))


def field(run, lead):
    f = f"{BASE}/gefs.{run}/12/atmos/pgrb2ap5/gespr.t12z.pgrb2a.0p50.f{lead:03d}"
    idx = get(f + ".idx").decode().splitlines()
    for n, line in enumerate(idx):
        parts = line.split(":")
        if parts[3] == "TMP" and parts[4] == "2 m above ground":
            start = int(parts[1]); end = int(idx[n + 1].split(":")[1]) - 1
            gid = eccodes.codes_new_from_message(get(f, (start, end)))
            v = eccodes.codes_get_values(gid).reshape(eccodes.codes_get(gid, "Nj"), eccodes.codes_get(gid, "Ni")); eccodes.codes_release(gid)
            return sum(w * v[i, j] for w, i, j in W4) * 9 / 5          # K -> F (spread)
    raise KeyError("TMP:2 m not found")


def run_day(t):
    run = t.strftime("%Y%m%d"); fn = f"{OUT}/{run}.csv"
    if os.path.exists(fn): return run, "skip"
    leads = sorted({24 * h + k for h in HS for k in (6, 18)})
    try:
        with cf.ThreadPoolExecutor(WORKERS) as ex:
            F = dict(zip(leads, ex.map(lambda L: field(run, L), leads)))
    except Exception as e:
        return run, f"FAIL {e!r}"[:200]
    rows = []
    for h in HS:
        sd = (F[24 * h + 6] + F[24 * h + 18]) / 2
        g = pd.DataFrame(dict(st=states, w=popw, s=sd * popw)).groupby("st").sum()
        for s, r in g.iterrows():
            rows.append(dict(state=s, h=h, target_date=(t + pd.Timedelta(days=h)).strftime("%Y-%m-%d"), tsd_f=r.s / r.w))
    pd.DataFrame(rows).to_csv(fn + ".tmp", index=False); os.replace(fn + ".tmp", fn)
    return run, "ok"


if __name__ == "__main__":
    for t in pd.date_range(START, END):
        t0 = time.time(); r, st = run_day(t)
        print(r, st, f"{time.time() - t0:.1f}s", flush=True)
