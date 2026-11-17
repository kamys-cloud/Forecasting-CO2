"""State-day series from the CAMPD daily facility-aggregation exports, and state eligibility.

Facility records are restricted to 2019-01-01 to 2025-12-31 and de-duplicated on (state, facility, date). A record without a CO2 entry
is treated as absent, not as zero; explicitly reported zeros are kept. Each state-day value is the sum of the available facility values
(missing when no facility reports that quantity); n_facilities is the number of facility records of that state-day. Days on which no
facility reports CO2 are omitted, so they remain missing on the complete daily calendar used by the forecasting code.

Eligibility (fixed before any model comparison): CO2 observed on at least 98% of days, at least five reporting facilities, mean daily
emissions of at least 5,000 short tons, coefficient of variation of daily emissions of at least 0.05, and no gap longer than 14 days.

Usage: python build_state_series.py [CAMPD_DIR] [OUT_DIR]      (defaults: CAMPD from paths.py, OUT_DIR = data/state_daily)
Writes OUT_DIR/<STATE>.csv for the eligible states and OUT_DIR/../derived/state_eligibility.csv."""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CAMPD, STATE_DAILY, DERIVED, ensure

SRC = sys.argv[1] if len(sys.argv) > 1 else CAMPD
OUT = ensure(sys.argv[2] if len(sys.argv) > 2 else STATE_DAILY)
ELIG = os.path.join(os.path.dirname(os.path.abspath(OUT)), "derived", "state_eligibility.csv") if len(sys.argv) > 2 \
    else os.path.join(ensure(DERIVED), "state_eligibility.csv")
COLS = ["state", "name", "fid", "date", "load", "steam", "so2", "co2", "nox", "heat"]
DAYS = pd.date_range("2019-01-01", "2025-12-31")

files = sorted(glob.glob(os.path.join(SRC, "daily-emissions-facility-aggregation-*.csv")))
if not files: sys.exit(f"no CAMPD facility-aggregation exports in {SRC}")
F = []
for f in files:
    d = pd.read_csv(f); assert len(d.columns) == len(COLS), f"unexpected columns in {f}"
    d.columns = COLS; F.append(d)
F = pd.concat(F, ignore_index=True)
F["date"] = pd.to_datetime(F.date)
F = F[(F.date >= DAYS[0]) & (F.date <= DAYS[-1])].drop_duplicates(["state", "fid", "date"])
s1 = lambda s: s.sum(min_count=1)
A = F.groupby(["state", "date"]).agg(co2_mass=("co2", s1), so2_mass=("so2", s1), nox_mass=("nox", s1), gross_load_mwh=("load", s1),
                                      heat_input=("heat", s1), n_facilities=("fid", "count")).reset_index()
A = A[A.co2_mass.notna()]

rows = []
for st, g in A.groupby("state"):
    y = g.set_index("date").co2_mass.reindex(DAYS)
    obs = y.notna(); runs = (~obs).astype(int).groupby(obs.cumsum()).sum()
    nfac = F[(F.state == st) & F.co2.notna()].fid.nunique()
    r = dict(state=st, share_days_observed=float(obs.mean()), facilities=int(nfac), mean_daily_co2=float(y.mean()),
             cv_daily_co2=float(y.std() / y.mean()) if y.mean() > 0 else np.nan, longest_gap_days=int(runs.max()))
    r["eligible"] = bool(r["share_days_observed"] >= 0.98 and r["facilities"] >= 5 and r["mean_daily_co2"] >= 5000
                         and r["cv_daily_co2"] >= 0.05 and r["longest_gap_days"] <= 14)
    rows.append(r)
E = pd.DataFrame(rows).sort_values("state"); ensure(os.path.dirname(ELIG)); E.to_csv(ELIG, index=False)
keep = E[E.eligible].state.tolist()
for st in keep:
    g = A[A.state == st].sort_values("date")
    g[["date", "co2_mass", "so2_mass", "nox_mass", "gross_load_mwh", "heat_input", "n_facilities", "state"]].to_csv(
        os.path.join(OUT, f"{st}.csv"), index=False, date_format="%Y-%m-%d")
print(f"{len(E)} state codes, {len(keep)} eligible; excluded: {', '.join(E[~E.eligible].state)}")
