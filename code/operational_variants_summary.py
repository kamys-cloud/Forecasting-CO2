"""Summarize the perfect-prognosis / forecast-trained comparison: national mean state-level R2 on identical rows (origins and
targets present in every variant) with joint 28-day date-block bootstrap intervals for differences from the perfect-prognosis model.
Usage: python operational_variants_summary.py VARIANT_DIR [VARIANT_DIR ...] OUT_CSV"""
import os, sys, glob
import numpy as np, pandas as pd
from jboot import Panel, boot, ci
import panels as pn

dirs, OUT = sys.argv[1:-1], sys.argv[-1]
rows = []
for h in pn.H:
    base = pn.operational(h)[["state", "date", "target", "opsfw_ridge", "opsfw_gam", "opsfw_hgb"]].rename(
        columns={"opsfw_ridge": "pp:ridge", "opsfw_gam": "pp:gam", "opsfw_hgb": "pp:hgb"})
    cols = ["pp:ridge", "pp:gam", "pp:hgb"]
    for d in dirs:
        for vdir in sorted(glob.glob(os.path.join(d, "*"))):
            v = os.path.basename(vdir); fs = sorted(glob.glob(os.path.join(vdir, f"*_h{h}.csv")))
            if len(fs) < 45: continue
            x = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in fs])
            ms = [m for m in ("ridge", "gam", "hgb") if m in x.columns]
            x = x[["state", "date"] + ms].rename(columns={m: f"{v}:{m}" for m in ms})
            base = base.merge(x, on=["state", "date"], how="inner"); cols += [f"{v}:{m}" for m in ms]
    base = base.dropna()
    P = Panel(pn.frames(base), cols); r2 = {c: float(np.nanmean(P.r2(c))) for c in cols}
    pairs = [(c, "pp:" + c.split(":")[1]) for c in cols if not c.startswith("pp:")]
    bs = boot({"x": P}, lambda rr: np.array([np.nanmean(rr["x"](a) - rr["x"](b)) for a, b in pairs]))
    for c in cols:
        row = dict(horizon=h, variant=c.split(":")[0], model=c.split(":")[1], mean_r2=r2[c], n_rows=len(base))
        if (c, "pp:" + c.split(":")[1]) in pairs:
            k = pairs.index((c, "pp:" + c.split(":")[1])); row["diff_vs_pp"] = r2[c] - r2["pp:" + c.split(":")[1]]
            row["ci_lo"], row["ci_hi"] = ci(bs[:, k])
        rows.append(row)
    print(h, flush=True)
T = pd.DataFrame(rows); T.to_csv(OUT, index=False)
print(T.pivot_table(index=["variant", "model"], columns="horizon", values="mean_r2").round(4).to_string())
