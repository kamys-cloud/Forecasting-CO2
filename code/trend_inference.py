"""Horizon trend of the state-level HGB - ridge difference: OLS of Delta(s,h) on h (315 state-horizon cases) with CR1 standard
errors clustered by state and by Census region, and the exact wild-cluster (Rademacher sign-flip) test over the 2^4 assignments of
the four regions, with the null imposed. With four clusters the identity and the all-flip assignments give |t*| = |t|, so the
smallest attainable two-sided p-value is 2/16 = 0.125.
Usage: python trend_inference.py OUT_CSV"""
import sys, itertools, os
import numpy as np, pandas as pd
from scipy import stats

from paths import ROOT
I = pd.read_csv(os.path.join(ROOT, "results", "inference", "hgb_ridge_by_horizon.csv")); I = I[I.a == "hgb"]
REG = {"Northeast": "CT MA NH RI NY NJ PA", "Midwest": "OH IN IL MI WI MN IA MO ND SD NE KS",
       "South": "DE MD VA WV KY TN NC SC GA FL AL MS AR LA OK TX", "West": "MT WY CO NM AZ UT NV CA OR WA"}
reg = {s: r for r, v in REG.items() for s in v.split()}
rows = [dict(state=c[6:], h=int(h), d=float(v)) for _, r in I.iterrows() for c, v in r.items() if c.startswith("state_") for h in [r.horizon]]
D = pd.DataFrame(rows); D["region"] = D.state.map(reg)
X = np.column_stack([np.ones(len(D)), D.h.values]); y = D.d.values; n, k = X.shape
XtXi = np.linalg.inv(X.T @ X)


def cr1(e, g):
    G = pd.unique(g); meat = sum(np.outer(X[g == c].T @ e[g == c], X[g == c].T @ e[g == c]) for c in G)
    V = len(G) / (len(G) - 1) * (n - 1) / (n - k) * XtXi @ meat @ XtXi; return V, len(G)


b = XtXi @ X.T @ y; e = y - X @ b
out = {}
for cl in ("state", "region"):
    V, G = cr1(e, D[cl].values); t = b[1] / np.sqrt(V[1, 1]); out[f"t_{cl}"] = t; out[f"p_{cl}"] = 2 * stats.t.sf(abs(t), G - 1)
b0 = y.mean(); er = y - b0; regs = sorted(D.region.unique()); ts = []
for signs in itertools.product([-1, 1], repeat=len(regs)):
    w = D.region.map(dict(zip(regs, signs))).values; ys = b0 + er * w
    bs = XtXi @ X.T @ ys; es = ys - X @ bs; V, _ = cr1(es, D.region.values); ts.append(bs[1] / np.sqrt(V[1, 1]))
out["p_wild_region"] = float(np.mean(np.abs(ts) >= abs(out["t_region"]) - 1e-12)); out["slope"] = b[1]
out["n_states_positive_slope"] = int(sum(np.polyfit(g.h, g.d, 1)[0] > 0 for _, g in D.groupby("state")))
pd.DataFrame([out]).to_csv(sys.argv[1], index=False); print(out)
